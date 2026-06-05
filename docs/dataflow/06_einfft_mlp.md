# 6. EinFFT MLP: `einfft` and `einfft-tiled`

> **All pages:**
> [README](README.md) ·
> [1. OS-core kernels](01_oscore_kernels.md) ·
> [2. IS-core kernels](02_iscore_kernels.md) ·
> [3. SIMD / RMSNorm kernels](03_simd_kernels.md) ·
> [4. Mamba main](04_mamba_main.md) ·
> [5. FFT family](05_fft.md) ·
> **6. EinFFT MLP (this page)** ·
> [7. VMamba SS2D](07_vmamba.md) ·
> [8. Performance optimization](08_performance_optimization.md) ·
> [9. Async tiling](09_async_tiling.md)

> Byte layouts of A, B, D for the OS-core:
> [memory_layouts/02](../../../chisel-ssm/docs/memory_layouts/02_gemm_layouts.md)
> and [03 — ConvFormat](../../../chisel-ssm/docs/memory_layouts/03_conv_format.md).
> Algorithmic reference (Scala): `chisel-ssm/src/test/scala/simbalib/EinfftMlpLib.scala`,
> with the OS-core dataflow specialisation in
> `chisel-ssm/src/test/scala/datagen/DataGeneratorEinfftMlp.scala::einfftLinearOS`.

`einfft` (un-tiled, PASS) and `einfft-tiled` (N-axis tiled; `nb_tiles`
constrained by `N_t >= 2`, see §6.2) implement the **2-layer EinFFT
MLP** — the MLP block that follows the EinFFT projection. The math is

```
Layer 1 (with ReLU)
  x_re_1 = ReLU(  x_re · W1_re  −  x_im · W1_im  +  b1_re )
  x_im_1 = ReLU(  x_re · W1_im  +  x_im · W1_re  +  b1_im )

Layer 2 (no activation)
  x_re_2 =       x_re_1 · W2_re  −  x_im_1 · W2_im  +  b2_re
  x_im_2 =       x_re_1 · W2_im  +  x_im_1 · W2_re  +  b2_im
```

All tensors carry a leading **branch axis of size 4** (`nBranches`). Each
"`multiply(x, w)`" is therefore 4 independent `(L, D/4) @ (D/4, D/4)`
matmuls, one per complex sub-branch. We treat each branch independently —
nothing is shared across branches.

For one branch and one complex linear layer, the four matmuls

```
rr = x_re · W_re
ii = x_im · W_im
ri = x_re · W_im
ir = x_im · W_re
```

drive a fused per-side post-processing

```
out_re = ReLU?( rr − ii + b_re )
out_im = ReLU?( ri + ir + b_im )
```

Matmuls run on the **OS-core**; the post-processing — FP8↔BF16 casts,
BF16 SUB/ADD, bias add, ReLU, and final FP8 requantization — runs on the
**SIMD core**. Snitch only sequences DMAs and accelerator launches; there
are **no byte-shuffling C helpers in the program** (see §6.3).

## 6.1 ConvFormat-throughout: zero Snitch-side reorders

The OS-core writes its `D` output in **ConvFormat per branch** (see
[memory_layouts/03](../../../chisel-ssm/docs/memory_layouts/03_conv_format.md)),
and the SIMD streamers walk that buffer linearly (= same byte order)
through widen → SUB/ADD → bias-add → narrow. Crucially, every staging
buffer carries data in **conv-walk byte order**: lane `k` of any SIMD
launch reads byte `2k` (BF16) or byte `k` (FP8) of the buffer, which is
the BF16-of-FP8-of conv-walk position `k`.

This is enabled by three coordinated decisions:

1. **The chisel reference output_*_* is emitted in ConvFormat per branch**
   (`flattenPerBranchConvFormat` in `DataGeneratorEinfftMlp.scala`).
   The SNAX program's TCDM result therefore matches the reference
   byte-for-byte without any reorder pass.
2. **The bias is pre-expanded by `datagen.py`** into the same conv-walk
   order: for each `(branch, conv-walk-position t)` we emit
   `bias[branch][col(t)]`. The SIMD ADD step's R13 reads this linearly,
   so lane `k` of R13 supplies the right `bias[col]` for whatever
   logical `(row, col)` position lane `k` of R7 happens to be processing.
3. **ReLU is folded into the SIMD launch** via a synthesized
   `SIMD_ADD_BF16 + doRelu` mode (`M3_SIMD_ADD_BF16_RELU`, encoded in
   `datagen.py`). No Snitch-side BF16 sign-bit clamp loop.

Without these three, the program would need C helpers to gather
ConvFormat → flat / scatter flat → flatA / broadcast bias / clamp BF16,
which would dominate the wall-clock cycle count (≈45× speed-up was
measured against the original C-helper version on the un-tiled config).

The one piece that ConvFormat-throughout does **not** cover is the
inter-layer chain: layer 1's output is in ConvFormat in TCDM, but layer
2's OS-core A input expects flatA. Rather than reorder on-chip, the
chisel reference emits `x_2_real/imag` (= layer 1's quantized output in
flatA per branch) as a separate file; the SNAX program DMAs it in from
L3 alongside `x_real/imag`. Each layer therefore reads its own pristine
flatA input from L3 and there is no inter-layer reformat step.

## 6.2 Why two variants

**`einfft`** is the un-tiled reference implementation. Each branch is
walked serially; the full per-branch weight matrix `(D/4, D/4)` is
DMA'd into TCDM before the four OSGEMMs run. With `L=16, D=192`
(`dPerB=48`) the per-branch weight is 2304 B, easily resident.

**`einfft-tiled`** tiles the N (= D/4 output-channel) axis at the
**OS-core level only**. Per (layer, branch):

- The four 4-OSGEMM-per-tile loops fire with `N_t = N_full / nb_tiles`
  and the weight DMA is double-buffered against the next tile's compute.
- Each tile's 4 OSGEMMs write a TILE-sized ConvFormat scratch
  (`rr/ii/ri/ir`, each `L × dPerB_tile`), and the SIMD widen / SUB /
  ADD-with-bias / narrow chain runs **immediately for that tile** (per
  side, at tile bounds) before the next tile's OSGEMM. Output tiles are
  spilled to L3 as they are produced. Consequently the scratch, the BF16
  staging, and the output buffers only ever hold ONE tile — the TCDM
  footprint scales as `~1/nb_tiles` in those buffers, which is what lets
  large `(L, dModel)` fit (see §6.5).
- The earlier "SIMD degrades at short per-tile bounds" caveat applies
  only to *tiny absolute* bounds (~12 fp8 cycles, the `L=16` default).
  At realistic params the per-tile bound is far above that and the fuse
  is correct; validated at `64/384/2`, `192/384/2`, and `384/384/2`
  (the last ~648 KiB with the old full-buffer scheme — formerly OOM).

This program PASSES at `nb_tiles = 1` (per-tile fuse degenerates to the
un-tiled `einfft` chain, plus a one-iteration tile loop and one-tile
weight ping-pong).

### `nb_tiles` constraint: `N_t >= 2`

Each per-tile OSGEMM must walk `N_t = N_full / nb_tiles >= 2`. The
datagen rejects shapes that would violate this. With `dPerB = 48`
(`dModel=192`), `N_full = 2`, so only `nb_tiles = 1` is possible;
`dModel >= 384` (`N_full = 4`) allows up to `nb_tiles = 2`. See
[chisel-ssm/docs/memory_layouts/02_gemm_layouts.md §2.4](../../../chisel-ssm/docs/memory_layouts/02_gemm_layouts.md)
for the underlying OSGEMM `N >= 2` requirement.

## 6.3 Un-tiled dataflow (`einfft`)

The whole program is a flat `for layer in {0,1}` × `for branch in 0..3`
double loop in `main.c`. There is no tile loop. Per (layer, branch):

```
DMA-in W_re_branch, W_im_branch (full per-branch weight matrices)
4 × OSGEMM (re-bind R0/R1/W0 base ptrs only between launches):
   rr = x_re @ W_re          (ConvFormat scratch)
   ir = x_im @ W_re
   ii = x_im @ W_im
   ri = x_re @ W_im
for side in {real, imag}:
   widen pos (FP8 ConvFormat → BF16, NOOP_FP8_REQUANT)  pos→bf16_a
   widen neg                                            neg→bf16_b
   bf16 binop (SUB for real, ADD for imag)              bf16_a (±) bf16_b → bf16_a
   bf16 ADD with bias broadcast (R13 reads bias_*_bcast)
     layer 1: mode = SIMD_ADD_BF16 + doRelu (M3_SIMD_ADD_BF16_RELU)
     layer 2: mode = SIMD_ADD_BF16
                                                       bf16_a + bias → bf16_a
   narrow BF16 → FP8 NOOP_BF16_REQUANT                 bf16_a → out_full_branch
```

`x_re/x_im` are FULL inputs (all 4 branches concatenated), pre-loaded
once at boot. Layer 2's `x_re/x_im` (= `x_2_real/imag`) are also
pre-loaded at boot; the four bias buffers (pre-expanded over L into
conv-walk order) are pre-loaded too. Only the per-branch weight tiles
are DMA'd inside the branch loop.

All SIMD streamer strides — for widen, BF16 binop, narrow — come from
`datagen.py` as `M3_R7_*`, `M3_R13_*`, `M3_W3_*` constants. The C side
calls `set_simd_streamer_csr / set_simd_streamer_no_b` with those
constants and then `write_csr(BASE_PTR_*, ...)` to rebind only the
moving base pointers between launches.

The four OS-core scratches (`rr / ii / ri / ir`) and the two BF16
staging buffers (`bf16_a`, `bf16_b`) are per-branch — i.e., overwritten
each iteration. The four `output_{1,2}_{real,imag}` buffers in TCDM
stay FULL (all 4 branches) for the whole program; the SIMD narrow at
the end of each (layer, branch) writes 768 contiguous FP8 bytes into
the matching branch slot, which is exactly the ConvFormat-per-branch
layout the reference expects.

## 6.4 Tiled dataflow (`einfft-tiled`, partial)

For one `(L, D/4) @ (D/4, D/4)` matmul in OS-core terms:

| Axis  | What it is        | Mapping in `MatmulDims`        |
| ----- | ----------------- | ------------------------------ |
| `M`   | sequence length   | `dim0 = L`                     |
| `K`   | reduction         | `dim1 = D/4`                   |
| `N`   | output channels   | `dim2 = D/4`                   |

Two structural constraints push the tiling onto **N**:

1. **OS-core has no psum read-back.** K-tiling would require an
   external accumulator across launches; the hardware doesn't expose
   one (rule 1 in the [OS-core doc](01_oscore_kernels.md)).
2. **M-tiling would force per-tile activation loads.** Each output
   tile needs the *full* activation, so M-tiling loses the
   "stationary activations" reuse and turns activations into a
   streamed input.

The tiled version then pipelines `nb_tiles` weight DMAs against the
per-tile compute, with each weight tile reused across two matmuls
(`W_re` across rr + ir, `W_im` across ii + ri). This is the same axis
[`osgemm-tiled`](01_oscore_kernels.md) picks.

The tiled program applies the ConvFormat-throughout strategy at the
OS-core level (per-tile OSGEMM writing a TILE-sized ConvFormat scratch)
and runs the SIMD fuse per tile at tile bounds, so scratch / BF16
staging / output stay one tile in size. It works correctly when
`N_t = N_full / nb_tiles >= 2`. At the default-shape `dModel=192`,
`dPerB=48`, `N_full=2`, that caps `nb_tiles` at 1; see §6.2 for the
underlying HW limitation and the datagen assertion that enforces it.

## 6.5 What stays FULL, what's per-branch

| Tensor                              | Size (FP8 bytes)         | Lifecycle                                                                  |
| ----------------------------------- | ------------------------ | -------------------------------------------------------------------------- |
| `x_re[0..3]`, `x_im[0..3]`          | `4 · L · (D/4)`          | **FULL** (all 4 branches), loaded once at boot                             |
| `x_2_re[0..3]`, `x_2_im[0..3]`      | `4 · L · (D/4)`          | **FULL**, layer-2 input (= quantized layer-1 ref); loaded once at boot     |
| `b_{1,2}_{re,im}_bcast[0..3]`       | `4 · L · (D/4) · BF16`   | **FULL** pre-expanded biases (conv-walk order); loaded once at boot        |
| `l1_re/im[0..3]`, `l2_re/im[0..3]`  | `4 · L · (D/4)` each     | **FULL**, SIMD narrow writes into them in ConvFormat per branch            |
| `W_re_branch`, `W_im_branch`        | `(D/4)²` each            | **per-branch**, re-DMA'd at every branch iteration                         |
| `rr / ii / ri / ir`                 | `L · (D/4)`              | **per-branch scratch** (un-tiled); **per-TILE** `L · (D/4)/nb_tiles` in `einfft-tiled` |
| `bf16_a`, `bf16_b`                  | `L · (D/4) · BF16` each  | **per-branch staging** (un-tiled); **per-TILE** in `einfft-tiled`           |

In `einfft-tiled` the scratch, BF16 staging, and output buffers are
TILE-sized (`/nb_tiles`) because the fuse runs per tile, and x / x2 /
mini-bias / outputs are L3-staged and DMA'd per (layer, branch) rather
than all held resident — see the app's `memory_model.py` for the exact
resident footprint. Raising `nb_tiles` (subject to `N_t >= 2`) shrinks
the per-tile buffers, so larger `(L, dModel)` fit; the floor is the
FULL per-branch `x_re/x_im` (reused by every N-tile) plus mini-bias.

Total live TCDM (un-tiled, `L=16, D=192`):
≈ 36 KiB (most of it the 16 FULL FP8 / BF16 broadcast buffers).

## 6.6 Layer chaining

Both layer 1 and layer 2 read their `x_re / x_im` input as flatA per
branch from L3. Layer 1 reads `x_real / x_imag`; layer 2 reads
`x_2_real / x_2_imag` (= the FP8 quantization of layer 1's output,
emitted in flatA by `einfftLinearOS`).

This sidesteps the inter-layer ConvFormat → flatA conversion that would
otherwise have to live on the C side, at the cost of one extra L3 → TCDM
DMA at boot. Layer 1's TCDM output stays in ConvFormat and is verified
against the ConvFormat reference; it is not re-read by layer 2.

## 6.7 SIMD pipeline per side

A per-side fuse is 5 SIMD launches. All streamer configs are emitted by
`datagen.py`:

| Step | Mode                       | R7 base   | R13 base                        | W3 base       |
| ---- | -------------------------- | --------- | ------------------------------- | ------------- |
| 1    | `NOOP_FP8_REQUANT`         | `pos`     | —                               | `bf16_a`      |
| 2    | `NOOP_FP8_REQUANT`         | `neg`     | —                               | `bf16_b`      |
| 3    | `SUB_BF16` / `ADD_BF16`    | `bf16_a`  | `bf16_b`                        | `bf16_a`      |
| 4    | `ADD_BF16` (+ doRelu in L1) | `bf16_a` | `b_*_bc_branch` (conv-walk order) | `bf16_a`      |
| 5    | `NOOP_BF16_REQUANT`        | `bf16_a`  | —                               | `out_branch`  |

Only step 1 needs a full `set_simd_streamer_no_b` (the previous launch
was the OSGEMM, with different streamers active); steps 2–5 mostly
rebind base pointers via `write_csr(BASE_PTR_*_LOW, ...)` and write
the new `MODE` CSR. The mode for step 4 in layer 1 is the synthesized
value `M3_SIMD_ADD_BF16_RELU = SIMD_ADD_BF16 | doRelu` emitted by
datagen — the chisel `SimbaCoreMode.isKnown` table does not contain it,
but the HW bits decode the same way the recognised modes do (the
"`Unknown mode`" warning in the sim is purely informational).

## 6.8 Verification

`DataGeneratorEinfftMlp` runs `einfftLinearOS` (per-matmul
FP8 requantization between the OS-core math and the BF16 fuse) and
emits the reference outputs in **ConvFormat per branch** so the SNAX
program can verify TCDM in place. It also emits `x_2_real/imag` in
flatA per branch for layer 2's input. Sample-based checks
(`check_result_sample`) on all four `output_*` buffers are sufficient
because the byte order in TCDM matches the reference exactly.

## 6.9 Dual-core variant: `einfft-tiled-is-osgemm`

`einfft-tiled` runs all four per-side matmuls serially on the **OS-core**.
`einfft-tiled-is-osgemm` instead splits them **by complex side across both GEMM
cores**, running them concurrently under the `IS_OSGEMM` kernel mode (the same
mode [`is-osgemm-tiled`](09_async_tiling.md) uses to run one OS-core and one
IS-core matmul in a single launch). Generator:
`chisel-ssm/.../datagen/DataGeneratorEinfftMlpIsOs.scala`.

- **REAL side → OS-core** (`rr = x_re·W_re`, `ii = x_im·W_im`): FP8 ConvFormat,
  fused exactly like `einfft` (widen → SUB → bias-add → narrow).
- **IMAG side → IS-core** (`ri = x_re·W_im`, `ir = x_im·W_re`): raw BF16
  `flattenCD`, NO_REQUANT — so the intermediates are *not* re-quantized to FP8
  (strictly more accurate than the OS-only variant).

Two `IS_OSGEMM` calls per (layer, branch), both cores busy each call:
`A: OS=rr ∥ IS=ri`, then `B: OS=ii ∥ IS=ir` — halving the 4-matmul GEMM time.

### Layout duplication

The two cores read different layouts, so each activation/weight is DMA'd in
**both**: `x` as `flatA` (OS A) and `_conv` ConvFormat (IS A, see
[memory_layouts/02 §2.5](../../../chisel-ssm/docs/memory_layouts/02_gemm_layouts.md));
weights in OS B-layout (`N_M_K`) and IS B-layout (`_is`, `K_M_N`). The imag
output reference is plain `flattenCD`: the final narrow reads the IS-core psum
`P` directly and writes `out_im` in the same psum byte order, which matches
`flattenCD` for `M = seqLen/seqLenUnroll ≥ 2` (the `M=1` single-seq-tile case
mis-orders that banked read-back, hence the `M ≥ 2` constraint below).

### Folding the imag add + bias into the IS-core psum

The imag post-processing (`out_im = ri + ir + b_im`) is folded into the
IS-core's native `D = A·B + C` accumulation across the two calls, exactly like a
2-step K-reduction:

- **Call A** seeds the psum with `C = b_im` (DMA'd into the psum buffer) and adds
  `ri` → psum holds `ri + b_im`.
- **Call B** reads that psum as `C` and adds `ir` → psum holds `ri + ir + b_im`.

So the IS-core does the complex add and the bias for free; the SIMD core only
does the final **ReLU + FP8 narrow** (one launch, layer 1 uses a synthesized
`NOOP_BF16_REQUANT | doRelu` mode). This removes two SIMD launches and the
psum-zeroing per branch, and drops the resident `ir`/zero/bias buffers — which
also shrinks the footprint enough for `L=512, dModel=192` to fit.

### Constraints

- **No N-tiling.** The `IS_OSGEMM` shared CSRs make the OS-core's N axis and the
  IS-core's K (reduction) axis the *same* `dInner` CSR; tiling N would wrongly
  shrink the IS reduction. Everything stays resident, so the app is
  footprint-bound (`dModel=192` fits up to `L=512`).
- `dModel ≥ 192` (OSGEMM needs `N = dPerB/Nu ≥ 2`).
- `seqLen ≥ 2·seqLenUnroll` (`M ≥ 2`): the folded narrow reads the IS-core psum
  directly, and the degenerate single-seq-tile case (`M=1`) mis-orders it.

### When it wins

EinFFT is SIMD-fuse-bound at small `dModel`, so the GEMM parallelism only helps
once the matmuls are large enough to matter. The win grows with `dModel`
(GEMM scales `~dPerB²`, SIMD `~dPerB`): roughly break-even at the tiny default,
~**1.4–1.5×** fewer accelerator (simbacore) cycles and a similar wall-clock
improvement at `dModel ≥ 384` or `L ≥ 512`. Verification is sample-based on all
four `output_*` buffers; the imag side may show a couple of
quantization-noise mismatches (the folded BF16 accumulation order differs from
the reference by ≤1 LSB at ReLU/rounding boundaries).
