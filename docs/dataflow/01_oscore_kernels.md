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
> [9. Async tiling](09_async_tiling.md) ·
> [12. SUC async](12_suc_async.md) ·
> [14. RMSNorm tiled](14_rmsnorm_tiled.md) ·
> [20. Bank-conflict-free double GEMM](20_double_gemm_conflict_free.md) ·
> [21. Conv downsample (im2col GEMM)](21_conv_downsample.md)

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
| `D`    | output               | **tiled** — per-tile slice, ping-pong'd, reassembled by DMA-out into a contiguous FULL TCDM buffer that emulates an off-chip (L3) destination (L1→L1 DMA; the L3 `D` is the golden reference), not L3 itself |

**Why N, not K or M.** The OS-core has no psum read-back, so K-tiling would
need accumulation logic the hardware doesn't provide. M-tiling would force
per-tile A-tile loads and lose the "shared A" reuse. N-tiling fits cleanly
because each tile's output is independent.

**Pipeline.** The three stages (DMA-in `B`, compute, DMA-out `D`) overlap
across tiles via the [Multi-stage pipeline](README.md#recurring-patterns) pattern.
