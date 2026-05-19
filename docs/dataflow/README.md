# SNAX Program Dataflow Reference

> **All pages:**
> **README (this page)** ·
> [1. OS-core kernels](01_oscore_kernels.md) ·
> [2. IS-core kernels](02_iscore_kernels.md) ·
> [3. SIMD / RMSNorm kernels](03_simd_kernels.md) ·
> [4. Mamba main (Phase 1 + Phase 2)](04_mamba_main.md) ·
> [5. FFT family](05_fft.md)

This folder documents, for every program under
[target/snitch_cluster/sw/apps/](../../target/snitch_cluster/sw/apps/), **how the
program orchestrates the SimbaCore**: the order in which tensors are DMA'ed
into TCDM, the sequence of accelerator invocations, which streamer ports each
stage uses, and how each program is tiled when there is a tiled variant. The
goal is that the information in these pages is sufficient to **recreate any of
the programs from scratch**, without re-reading the source.

These docs do **not** describe the byte-level memory layouts of individual
tensors. Those live in
[../../../chisel-ssm/docs/memory_layouts/](../../../chisel-ssm/docs/memory_layouts/README.md)
(`flattenA`, `flattenB`, `ConvFormat`, `SUCFormat`, `xProj`, `splitDeltaWeight`,
`bankTranspose`, etc.). The two reference sets are complementary:

| Question                                              | Reference                                          |
| ----------------------------------------------------- | -------------------------------------------------- |
| *In what order do I DMA things in?*                   | This folder.                                       |
| *Which `set_simbacore_csr` MODE do I call, in what order?* | This folder.                                       |
| *Which streamer ports get hooked up to which TCDM pointer?* | This folder.                                       |
| *Along which axis is a tiled variant cut, and why?*   | This folder.                                       |
| *Which buffer survives between phases, and which is overlaid?* | This folder.                                       |
| *What does the byte layout of A / B / D look like in memory?* | [memory_layouts/](../../../chisel-ssm/docs/memory_layouts/README.md) |
| *How does `ConvFormat` encode columns into 16-byte chunks?* | [memory_layouts/03](../../../chisel-ssm/docs/memory_layouts/03_conv_format.md) |
| *What does `interleaveColsXProj` do?*                 | [memory_layouts/05](../../../chisel-ssm/docs/memory_layouts/05_xproj_format.md) |

## Index

1. [OS-core kernels](01_oscore_kernels.md) — `osgemm`, `osgemm-tiled`. Single GeMM through the OS-core, with optional dInner tiling and a 3-stage DMA/compute/DMA pipeline.
2. [IS-core kernels](02_iscore_kernels.md) — `isgemm`, `isgemm-tiled`. Bias-loaded GeMM through the IS-core, with K-axis tiling for the tiled variant (because the IS-core requant fires on the last K iteration).
3. [SIMD / RMSNorm kernels](03_simd_kernels.md) — `simd` (op coverage), `rmsnorm` (six-step SIMD pipeline reusing one buffer in place).
4. [Mamba main (Phase 1 + Phase 2)](04_mamba_main.md) — `main`, `main-tiled`, `main-full`, `suc`. The full two-phase Mamba block with intra-Phase 2 multi-core dataflow (OS-core + Switch-core + SU-core + IS-core all in flight), and dInner tiling in `main-tiled`.
5. [FFT family](05_fft.md) — `fft` (2-way partitioned EinFFT: GeMM + CMul + reorder + GeMM), `fft-tiled` (dModel-tiled Phase A + K-tiled Phase B, with L3 spill of the intermediate), `fft-3way` (3-way partitioned EinFFT with five accelerator stages and zero inter-stage software reorders).

## Conventions used across all programs

These conventions are reused everywhere; later pages assume them rather than
re-state them.

**The standard program skeleton** is:

```
1. Allocate TCDM pointers from snrt_l1_next().
2. cluster barrier.
3. (DM core only) DMA each L3 input down into its TCDM slot, then dma_wait_all.
4. cluster barrier.
5. (Compute core 0 only) for each stage:
       set_<role>_streamer_csr(...)          // hook up R/W ports to TCDM pointers
       set_simbacore_csr(MODE, d0, d1, d2, d3, d4)   // set the loop bounds
       start_simbacore_and_streamers(...)
       wait_simbacore_and_streamer()
6. (Compute core 0 only) check_result_sample / check_result_all against the golden tensor.
7. cluster barrier.
```

**Streamer ports** (defined in [streamer_csr_addr_map.h](../../target/snitch_cluster/sw/snax/simbacore/include/streamer_csr_addr_map.h)):

| Port    | Direction | Typical role                                                                       |
| ------- | --------- | ---------------------------------------------------------------------------------- |
| `R0`    | read      | OS-core input A                                                                    |
| `R1`    | read      | OS-core weight B                                                                   |
| `R2`    | read      | Switch-core (xProj) input                                                          |
| `R3`    | read      | Switch-core weight 1 (conv weight in Phase 1, delta-weight 1 in Phase 2)           |
| `R4`    | read      | Switch-core bias (conv bias in Phase 1, dt bias in Phase 2)                        |
| `R5`    | read      | Switch-core weight 2 (delta-weight 2)                                              |
| `R6`    | read      | SU-core `A`                                                                        |
| `R7`    | read      | SU-core `BC` *and* SIMD operand A                                                  |
| `R8`    | read      | SU-core `D`                                                                        |
| `R9`    | read      | SU-core `x`                                                                        |
| `R10`   | read      | SU-core `z` = OS-core out (Phase 2 forwarding; `R10_en=1` connects OS→SU on-chip)  |
| `R11`   | read      | IS-core input from SU-core `y` (Phase 2) / IS-core input A (standalone IS GeMM, FFT) |
| `R12`   | read      | IS-core weight B                                                                   |
| `R13`   | read      | IS-core psum read-back (bias / accumulator) *and* SIMD operand B                   |
| `W0`    | write     | OS-core output `z`                                                                 |
| `W1`    | write     | Switch-core output (`conv_out` in Phase 1)                                         |
| `W2`    | write     | SU-core output `y`                                                                 |
| `W3`    | write     | IS-core output *and* SIMD output                                                   |

**Per-port CSR fields**: `*_ss` (spatial stride), `*_tb` (temporal bounds),
`*_ts` (temporal strides), `*_en` (enable). The `set_*_streamer_csr` helpers
in [snax-simbacore-lib.h](../../target/snitch_cluster/sw/snax/simbacore/include/snax-simbacore-lib.h)
just wrap `write_csr` calls in a friendlier signature; a tiled re-launch only
needs to rewrite the four `BASE_PTR_*_LOW` CSRs plus optionally `MODE`.

**SimbaCore CSR loop bounds**: `set_simbacore_csr(MODE, d0, d1, d2, d3, d4)`.
The semantics of `d0..d4` depend on the MODE — see
[memory_layouts/07_mode_reference.md](../../../chisel-ssm/docs/memory_layouts/07_mode_reference.md)
for the exact mapping. Per-program calls are listed below with the values they
pass.

**Tiling re-uses the per-tile bounds**: in every tiled variant in this repo,
the streamer bounds and `set_simbacore_csr` are configured **once** with
*per-tile* values. Inside the per-tile loop we only rewrite the four
`BASE_PTR_*_LOW` CSRs (the moving ones) and, where requant timing matters,
the `MODE` CSR. This keeps the per-tile overhead to a handful of CSR writes.

**Pipeline pattern**: the canonical pipelined loop in this repo runs for
`nb_tiles + (NB_STAGES − 1)` iterations and decides per iteration `i` which
stages to fire (`transfer_in` for `i ∈ [0, nb_tiles)`, `compute` for `i ∈ [1,
nb_tiles + 1)`, `transfer_out` for `i ∈ [2, nb_tiles + 2)` if present). DMA
work runs on the DM core; compute on core 0. Each iteration ends with
`snrt_dma_wait_all()` and a `cluster_hw_barrier()`.

**Last-tile requant trick** (used in every accumulating-tiled program):
the IS-core requantizes on its *last K iteration*. When K-axis tiling is used,
each tile *thinks* it is the last K iteration. To get the correct result, non-
final tiles run in a `*_NO_REQUANT` mode that keeps the psum in BF16, and the
final tile switches to the normal mode to apply the requant on the fully
accumulated psum. `R13` (psum read-back) and `W3` (psum write) hit the *same
TCDM address* on every tile, which is how the accumulator is kept in TCDM
across tiles. See [iscore_n_tiling memory note](../../../.claude/projects/-esat-micas-lapserv11-users-rgeens-snax-cluster/memory/iscore_n_tiling.md)
for why this works on K but not on N.

**Hard cardinal rules** (also enforced by the memory_layouts docs):

- Loop order is fixed in hardware (OS=`N_M_K`, IS=`K_M_N`, Switch=`N_M_K`).
- IS-core output bytes are always `accType`-sized (BF16). FP8 requant output
  occupies the low byte of each BF16 slot.
- Inter-stage byte reorders must be folded into streamer programming. Scalar
  reorders over TCDM via Snitch are too slow and not used in any of the
  programs in this folder.
