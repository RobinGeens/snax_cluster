# 1. OS-core kernels: `osgemm` and `osgemm-tiled`

> **All pages:**
> [README](README.md) ·
> **1. OS-core kernels (this page)** ·
> [2. IS-core kernels](02_iscore_kernels.md) ·
> [3. SIMD / RMSNorm kernels](03_simd_kernels.md) ·
> [4. Mamba main](04_mamba_main.md) ·
> [5. FFT family](05_fft.md) ·
> [6. EinFFT MLP](06_einfft_mlp.md) ·
> [7. VMamba SS2D](07_vmamba.md) ·
> [8. Performance optimization](08_performance_optimization.md) ·
> [9. Async tiling](09_async_tiling.md)

> Byte layouts of A, B, D: [memory_layouts/02](../../../chisel-ssm/docs/memory_layouts/02_gemm_layouts.md).

The OS-core computes `D = A · B` with no input bias. The output lands in
ConvFormat.

## `osgemm`

A single-shot OS-core launch.

| Tensor | Role            |
| ------ | --------------- |
| `A`    | activations (in)|
| `B`    | weights (in)    |
| `D`    | output          |

## `osgemm-tiled`

The same GeMM, but the **output (N) axis** is tiled.

| Tensor | Role                 | Lifecycle                                                          |
| ------ | -------------------- | ------------------------------------------------------------------ |
| `A`    | activations (in)     | **shared** — preloaded once, reused across all tiles               |
| `B`    | weights (in)         | **tiled** — per-tile slice, ping-pong'd                            |
| `D`    | output               | **tiled** — per-tile slice, ping-pong'd, reassembled into a FULL buffer by DMA-out |

**Why N, not K or M.** The OS-core has no psum read-back, so K-tiling would
need accumulation logic the hardware doesn't provide. M-tiling would force
per-tile A-tile loads and lose the "shared A" reuse. N-tiling fits cleanly
because each tile's output is independent.

**Pipeline.** Three stages overlap across tiles: DMA-in the next B-tile,
compute the current tile, DMA-out the previous tile's D-slice into the
reassembled buffer.
