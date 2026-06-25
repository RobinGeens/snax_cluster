# 21. Conv downsample (im2col GEMM, reduction-tiled)

App: [`conv-downsample`](../../target/snitch_cluster/sw/apps/conv-downsample/).

The SiMBA inter-stage **downsample** (patch-embed) is a dense 3×3, stride-2, pad-1
`Conv2d`. Mapped to a matmul via im2col it becomes

```
D(L_out, Cout) = A(L_out, K) · B(K, Cout) + bias      with  K = 9 · Cin
```

where `A` is the im2col matrix (one row per output position, one column per
`(cin, ki, kj)` tap) and `B` is the conv weight reshaped to `(K, Cout)`. 

No new app, just a new data generator and the existing `isgemm-tiled` app.

The data is a genuine convolution, not random: the Scala generator
`DataGeneratorConvDownsample` builds a random image + 3×3 weights + bias,
materializes the im2col `A` and reshaped `B`, and computes the golden with the same
hardware-faithful adder-tree `matmul` as the other GEMMs. The Scala conv model
itself (`VMambaLib.conv2dStrided`) is checked against the Python/Simba
`torch.nn.Conv2d` reference by the `conv2dStrided` test in `VMambaLibTest`.

The app computes only the conv (the downsample's trailing RMSNorm is a separate
`rmsnorm-tiled` pass).
