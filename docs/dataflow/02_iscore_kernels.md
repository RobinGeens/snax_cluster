# 2. IS-core kernels: `isgemm` and `isgemm-tiled`

> **All pages:**
> [README](README.md) ·
> [1. OS-core kernels](01_oscore_kernels.md) ·
> **2. IS-core kernels (this page)** ·
> [3. SIMD / RMSNorm kernels](03_simd_kernels.md) ·
> [4. Mamba main](04_mamba_main.md) ·
> [5. FFT family](05_fft.md) ·
> [6. EinFFT MLP](06_einfft_mlp.md) ·
> [7. VMamba SS2D](07_vmamba.md) ·
> [8. Performance optimization](08_performance_optimization.md)

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
