# 3. SIMD and RMSNorm kernels

> See also: [SIMD lane / block layouts](../../../chisel-ssm/docs/memory_layouts/10_simd.md).
> This page covers **program structure**, not lane order or `[Re×N/2 | Im×N/2]`
> packing.

The SIMD core shares streamer ports with the IS-core: `R7` is operand A,
`R13` is operand B, `W3` is the result. There is no separate enable for the
IS-core's data path; the `MODE` CSR selects between IS-GeMM and SIMD-op
behaviour for the same physical streamer wiring.

Two helper macros do most of the work:

- `set_simd_streamer_csr(R7..., R13..., W3...)` for two-operand ops.
- `set_simd_streamer_no_b(R7..., W3...)` for one-operand ops (`Sqrt`, `Rms`,
  `NoOp`, ...).

`set_simbacore_simd_mode(MODE)` rewrites only the MODE CSR (cheaper than
`set_simbacore_csr` which also writes the five `d*` bound registers — for
SIMD all five `d*` registers are 0 because the SIMD ops use no GeMM loop).

## 3.1 `simd` — coverage of every SIMD op

**Source**: [simd/src/simd.c](../../target/snitch_cluster/sw/apps/simd/src/simd.c).

Two test functions: `test_simd_fp8` and `test_simd_bf16`. Each runs the
SIMD core through every op the data type supports, **switching mode** and
(when output type changes) the **W3 streamer config** in between.

**TCDM allocation** is a flat block per data type:

```
FP8: [ in_a | in_b | add_out | sub_out | mul_out | cmul_out | inprod_out | rms_out | mul_requant_out | noop_requant_out | softshrink_out ]
BF16: [ in_a | in_b | add_out | sub_out | mul_out | cmul_out | inprod_out | rms_out | div_out | sqrt_out | mul_requant_out | noop_requant_out ]
```

Inputs (`in_a`, `in_b`) are DMA'ed once. Each op writes its own dedicated
output slot, so the program does not have to re-DMA between ops.

**Per-op pattern**: when only the writer address changes and the streamer
layout is the same, the program only rewrites `BASE_PTR_WRITER_3_LOW`. When
the *operand layout* changes (one-operand vs two-operand, requant up-cast,
reduce vs elementwise), it re-issues `set_simd_streamer_csr` / `_no_b`.

Concrete sequence for FP8:

| Step | MODE                       | Operand wiring change?            | Notes                                  |
| ---- | -------------------------- | --------------------------------- | -------------------------------------- |
| 1    | `M20_SIMD_CMUL_FP8`        | full re-wire (R7=a, R13=b, W3=cmul) | initial setup                          |
| 2    | `M16_SIMD_ADD_FP8`         | only `W3` rewritten               | reuses R7/R13 from previous step       |
| 3    | `M17_SIMD_SUB_FP8`         | only `W3` rewritten               | "                                      |
| 4    | `M18_SIMD_MUL_FP8`         | only `W3` rewritten               | "                                      |
| 5    | `M19_SIMD_MUL_FP8_REQUANT` | full re-wire (W3 now BF16-sized)  | output type changes FP8→BF16           |
| 6    | `M21_SIMD_INPROD_FP8`      | full re-wire (W3 reduce strides)  | output is a per-channel reduction      |
| 7    | `M22_SIMD_RMS_FP8`         | `_no_b` re-wire                   | one-operand op, reduce output          |
| 8    | `M25_SIMD_NOOP_FP8_REQUANT`| `_no_b` re-wire (W3 BF16)         | up-cast no-op                          |
| 9    | `M26_SIMD_SOFTSHRINK_FP8`  | full re-wire (R13 carries the threshold) | softshrink output                |

The BF16 variant is structurally the same; it adds `DIV` and `SQRT` and
replaces the FP8-only `SOFTSHRINK`/`CMUL` with their BF16 equivalents (note
that the BF16 SIMD has no CMUL).

**Reusing R7/R13 wiring**: the trick the program leans on is that the streamer
config for `R7`/`R13` is **independent of the MODE** for fixed input data type
and op kind. So you only need to rewrite all three CSRs when the operand kind
changes (one-operand vs two-operand vs reduce). For the same op family,
`write_csr(BASE_PTR_WRITER_3_LOW, ...)` is enough.

## 3.2 `rmsnorm` — six-step SIMD pipeline with in-place buffer reuse

**Source**: [rmsnorm/src/rmsnorm.c](../../target/snitch_cluster/sw/apps/rmsnorm/src/rmsnorm.c).

RMSNorm computes, per token:

```
rms[l] = sqrt(Σ_d x[l, d]^2 / D)
out[l, d] = x[l, d] * (1 / rms[l]) * weight[d]
```

The program fuses this into six SIMD ops with **in-place** writes into the
same `rms` buffer. Only `x` and `weight` are DMA'ed in; `d_inverse` and
`ones` are constructed in TCDM by core 0 (one full SIMD lane filled with the
constant by a small CPU loop — this is cheap because the lane is at most 32
elements wide).

**TCDM allocation**:

```
[ x | d_inverse | ones | weight | rms ]
```

**Stage sequence**:

| Step | MODE                  | R7 input         | R13 input              | W3 output | Effect on `rms`/`x`                |
| ---- | --------------------- | ---------------- | ---------------------- | --------- | ---------------------------------- |
| 1    | `M13_SIMD_RMS_BF16`   | `x` (reduce)     | — (no_b)               | `rms`     | `rms[l] = Σ_d x[l,d]^2`            |
| 2    | `M10_SIMD_MUL_BF16`   | `rms`            | `d_inverse` (broadcast)| `rms`     | `rms[l] /= D` (× 1/D)              |
| 3    | `M15_SIMD_SQRT_BF16`  | `rms`            | — (no_b)               | `rms`     | `rms[l] = sqrt(...)`               |
| 4    | `M14_SIMD_DIV_BF16`   | `ones` (broadcast)| `rms`                 | `rms`     | `rms[l] = 1 / rms[l]`              |
| 5    | `M10_SIMD_MUL_BF16`   | `x`              | `rms` (slide over D)   | `x`       | `x[l,d] *= rms[l]`                 |
| 6    | `M10_SIMD_MUL_BF16`   | `x`              | `weight` (stationary)  | `x`       | `x[l,d] *= weight[d]`              |

**Broadcast / stationary tricks**: steps 2 and 4 pass `(int32_t*)zero_ts` as
the R13/R7 temporal stride array (`zero_ts = {0,0,0,0}`), which holds the
operand constant across the temporal loop — that is how a single lane of
`d_inverse` or `ones` broadcasts across all SIMD invocations. Steps 5 and 6
program R13's stride to slide either over L (one `rms[l]` per token) or stay
on `weight[d]` (one weight per channel).

**In-place writes**: `rms` is the only intermediate; it is written in-place
through steps 1–4. The final result lands in `x` (steps 5–6 write back to
the input buffer). No scratch buffer is needed.

## 3.3 Recreate `rmsnorm`

1. Allocate `[ x | d_inverse | ones | weight | rms ]` in TCDM.
2. DMA `x` and `weight` in. Barrier.
3. Core 0 fills `d_inverse` and `ones` lanes (BF16-encode `1/D` and `1.0`).
4. Run the six SIMD ops in the order above. Between steps:
   - When operand shape changes, re-issue `set_simd_streamer_csr` /
     `_no_b`.
   - When only the W3 destination changes, rewrite `BASE_PTR_WRITER_3_LOW`.
   - Always re-set the MODE CSR (`set_simbacore_simd_mode`).
5. Verify `x` against the golden RMSNorm output.
