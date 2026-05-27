# 3. SIMD and RMSNorm kernels

> **All pages:**
> [README](README.md) ·
> [1. OS-core kernels](01_oscore_kernels.md) ·
> [2. IS-core kernels](02_iscore_kernels.md) ·
> **3. SIMD / RMSNorm kernels (this page)** ·
> [4. Mamba main](04_mamba_main.md) ·
> [5. FFT family](05_fft.md) ·
> [6. EinFFT MLP](06_einfft_mlp.md) ·
> [7. VMamba SS2D](07_vmamba.md) ·
> [8. Performance optimization](08_performance_optimization.md)

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

RMSNorm is fused into six SIMD launches with the intermediate `rms` buffer
updated in place each step. The sequence per token is:

1. `Rms`: `rms ← Σ x²`
2. `Mul`: `rms *= 1/D` (D is broadcast from a one-lane constant in TCDM)
3. `Sqrt`: `rms ← sqrt(rms)`
4. `Div`: `rms ← 1 / rms`
5. `Mul`: `x *= rms` (the rms scalar slides over D, one per token)
6. `Mul`: `x *= weight` (weight is held stationary)

| Tensor      | Role                              | Lifecycle                                                  |
| ----------- | --------------------------------- | ---------------------------------------------------------- |
| `x`         | input activations and final output| DMA'd in; modified in place in steps 5–6                   |
| `weight`    | RMSNorm scale                     | DMA'd in once                                              |
| `rms`       | per-token denom (intermediate)    | written in step 1, updated in place through step 4         |
| `d_inverse` | broadcast constant `1/D`          | built in TCDM by the compute core (no DMA)                 |
| `ones`      | broadcast constant `1.0`          | built in TCDM by the compute core (no DMA)                 |

The "broadcast a one-lane constant" pattern in steps 2 and 4 and the "slide
one operand over D while the other walks the matrix" pattern in steps 5 and
6 are both expressed by configuring the streamer's per-port temporal
strides, not by software loops.
