# 3. SIMD, RMSNorm and BatchNorm kernels

> **All pages:**
> [README](README.md) ·
> [1. OS-core kernels](01_oscore_kernels.md) ·
> [2. IS-core kernels](02_iscore_kernels.md) ·
> **3. SIMD / RMSNorm kernels (this page)** ·
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

> SIMD lane / block layouts: [memory_layouts/10](../../../chisel-ssm/docs/memory_layouts/10_simd.md).

## `simd`

Coverage test. Each SIMD op is invoked on the same input pair, writing into
its own output slot so no re-DMA is needed between ops. Two test functions
cover the BF16 and FP8 op sets:

- BF16: `Add`, `Sub`, `Mul`, `Div`, `Sqrt`, `InProd`, `Rms`, `MulRequant`
  (BF16→FP8), `NoopRequant` (BF16→FP8). BF16 SIMD has no `CMul`.
- FP8: `CMul`, `Add`, `Sub`, `Mul`, `InProd`, `Rms`, `MulRequant`
  (FP8→BF16), `NoopRequant` (FP8→BF16), `SoftShrink`.

Ops that share an operand shape and output type only need the SIMD mode
swapped between launches.

**Tensors:** `in_a` and `in_b` (preloaded), one dedicated output slot per
op.

## `rmsnorm`

RMSNorm is fused into five SIMD launches with the intermediate `rms` buffer
updated in place each step. The sequence per token is:

1. `Rms`: `rms ← Σ x²`
2. `Mul`: `rms *= 1/D` (D is broadcast from a one-lane constant in TCDM)
3. `Rsqrt`: `rms ← 1 / sqrt(rms)` (one launch; the unit runs sqrt then reciprocal per element)
4. `Mul`: `x *= rms` (the rms scalar slides over D, one per token)
5. `Mul`: `x *= weight` (weight is held stationary)

| Tensor      | Role                              | Lifecycle                                                  |
| ----------- | --------------------------------- | ---------------------------------------------------------- |
| `x`         | input activations and final output| DMA'd in; modified in place in steps 4–5                   |
| `weight`    | RMSNorm scale                     | DMA'd in once                                              |
| `rms`       | per-token denom (intermediate)    | written in step 1, updated in place through step 3         |
| `d_inverse` | broadcast constant `1/D`          | built in TCDM by the compute core (no DMA)                 |

The "broadcast a one-lane constant" pattern in step 2 and the "slide
one operand over D while the other walks the matrix" pattern in steps 4 and
5 are both expressed by configuring the streamer's per-port temporal
strides, not by software loops.

## `batchnorm`

Stand-alone folded BatchNorm + ReLU: `out = ReLU(x · scale + shift)` with a
per-channel `scale` and `shift`. This is the SegFormer ConvModule tail that
follows the 1×1 conv (which is just an OS-core GeMM — the `osgemm` app); the
two are separate apps, not a fused kernel.

A single fused BF16 pass (`SIMD_FMA_HOLD_BF16_RELU`): `out ← ReLU(x · scale + shift)`.
The FmaHold b-operand stream delivers one interleaved `(scale, shift)` lane-block
pair per channel (R13 walk `[2, D]` over the `[D][2][lanes]` buffer); the core
latches the pair for the channel's `n_acc = seqLen/lanes` outputs, so the pass
runs at the full one-output-per-cycle rate with almost no R13 traffic.

| Tensor       | Role                                    | Lifecycle                                    |
| ------------ | --------------------------------------- | -------------------------------------------- |
| `x`          | input activations (`seqLen × channels`) | DMA'd in                                     |
| `scaleshift` | per-channel scale/shift, interleaved    | DMA'd in once; one pair read per channel     |
| `out`        | output                                  | written by the single pass                   |
