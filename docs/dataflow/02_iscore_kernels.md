# 2. IS-core kernels

> **All pages:**
> [README](README.md) ·
> [1. OS-core kernels](01_oscore_kernels.md) ·
> **2. IS-core kernels (this page)** ·
> [3. SIMD / RMSNorm kernels](03_simd_kernels.md) ·
> [4. Mamba main](04_mamba_main.md) ·
> [5. FFT family](05_fft.md) ·
> [6. EinFFT MLP](06_einfft_mlp.md) ·
> [7. VMamba SS2D](07_vmamba.md) ·
> [8. Performance optimization](08_performance_optimization.md) ·
> [9. Async tiling](09_async_tiling.md) ·
> [12. SUC async](12_suc_async.md) ·
> [14. RMSNorm tiled](14_rmsnorm_tiled.md) ·
> [20. Bank-conflict-free double GEMM](20_double_gemm_conflict_free.md) ·
> [21. Conv downsample (im2col GEMM)](21_conv_downsample.md)

> Byte layouts of A, B, C, D: [memory_layouts/02](../../../chisel-ssm/docs/memory_layouts/02_gemm_layouts.md).

The IS-core computes `D = C + A · B` and reads its own psum back on every
iteration. This is what enables in-place accumulation across tiles.

## `isgemm`

A single-shot IS-core launch, run twice (with FP8 requant, and without).

| Tensor | Role                                                                  |
| ------ | --------------------------------------------------------------------- |
| `A`    | activations (in)                                                      |
| `B`    | weights (in)                                                          |
| `C/D`  | bias (in) and output, sharing the same TCDM buffer                    |

## `isgemm-tiled`

The **K (reduction) axis** is tiled.

| Tensor | Role                | Lifecycle                                                                                  |
| ------ | ------------------- | ------------------------------------------------------------------------------------------ |
| `A`    | activations (in)    | **tiled** — per-tile K-slice, ping-pong'd                                                  |
| `B`    | weights (in)        | **tiled** — per-tile K-slice, ping-pong'd                                                  |
| `C`    | bias (in)           | **shared** — preloaded once into the `C/D` buffer; serves as tile-0's input                |
| `D`    | output (= `C/D`)    | **FULL accumulator** — same TCDM address every tile, psum updated in place                 |

**Why K, not N.** The IS-core applies its FP8 requant on the *last K
iteration* of the inner loop. N-tiling makes every N-tile see a "last K
iteration" and applies the requant to partial sums — corrupting output.
K-tiling keeps the requant tied to the real last iteration, which is the
final tile. Non-final tiles run in a no-requant mode (psum stays in BF16);
the final tile applies the requant on the fully accumulated psum.

**Pipeline.** Two stages: DMA-in the next A-tile and B-tile while computing
the previous tile. Compute stages cannot overlap with each other because
they share the accumulator.

## `is-osgemm` and `is-osgemm-tiled`

Two independent GEMMs run concurrently in a single `IS_OSGEMM` launch: an
OS-core `D = A · B` (streamers `R0`/`R1` → `W0`, ConvFormat output) and an
IS-core `D = C + A · B` (streamers `R11`/`R12`/`R13` → `W3`, psum read-back).
They share nothing but the TCDM fabric, so their only coupling is bank
contention — see [20. Bank-conflict-free double GEMM](20_double_gemm_conflict_free.md)
for the skip-128 partition that removes it.

`is-osgemm-tiled` tiles both GEMMs along `dInner`, which is the OS-core's
output `N` axis (each tile = different output columns) and the IS-core's
reduction `K` axis (each tile accumulates into the `CD` buffer). Non-final
tiles run `IS_OSGEMM_NO_REQUANT`; the final tile runs `IS_OSGEMM`, applying
the IS-core requant on the last K iteration (see "Why K, not N" above). A
three-stage pipeline double-buffers DMA-in / compute / DMA-out.

The async variant (`is-osgemm-tiled-async`) streams both rings at once — see
[9. Async tiling](09_async_tiling.md).
