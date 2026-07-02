# 21. Conv downsample (im2col GEMM, reduction-tiled)

> **All pages:**
> [README](README.md) ·
> [1. OS-core kernels](01_oscore_kernels.md) ·
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
> **21. Conv downsample (im2col GEMM) (this page)**

App: [`conv-downsample`](../../target/snitch_cluster/sw/apps/conv-downsample/).

The SiMBA inter-stage **downsample** (patch-embed) is a dense 3×3, stride-2, pad-1
`Conv2d`. Mapped to a matmul via im2col it becomes

```
D(L_out, Cout) = A(L_out, K) · B(K, Cout) + bias      with  K = 9 · Cin
```

where `A` is the im2col matrix (one row per output position, one column per
`(cin, ki, kj)` tap) and `B` is the conv weight reshaped to `(K, Cout)`. 

No new device kernel: conv-downsample reuses isgemm-tiled's main.c and streamer programming; only the data generator differs.

The data is a genuine convolution, not random: the Scala generator
`DataGeneratorConvDownsample` builds a random image + 3×3 weights + bias,
materializes the im2col `A` and reshaped `B`, and computes the golden with the same
hardware-faithful adder-tree `matmul` as the other GEMMs. The Scala conv model
itself (`VMambaLib.conv2dStrided`) is checked against the Python/Simba
`torch.nn.Conv2d` reference by the `conv2dStrided` test in `VMambaLibTest`.

The app computes only the conv (the downsample's trailing RMSNorm is a separate
`rmsnorm-tiled` pass).
