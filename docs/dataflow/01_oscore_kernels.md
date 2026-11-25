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

`D = A · B` on the OS-core alone (`A`: `dim0 × dim1` activations, `B`: `dim1 × dim2` weights, FP8,
ConvFormat output), with every operand read from L3 as few times as the TCDM allows. It is the
energy counterpart of the GEMM programs that are tuned for latency:

- `double-gemm-conflict-free` ([20](20_double_gemm_conflict_free.md)) splits the tokens over the
  OS-core and the IS-core, and the IS-core costs more energy per MAC;
- `isgemm-tiled` ([2](02_iscore_kernels.md)) and `conv-downsample` ([21](21_conv_downsample.md)) run
  on the IS-core;
- all three are called on a few tokens at a time, so they reload the whole weight matrix from L3 on
  every call.

The batch runs of this app fill the `Latency_energy` tab of the evaluation sheet.

### Schedule

One call is `nb_l_tiles` activation chunks (along `M`) times `nb_tiles` weight tiles (along `N`),
one OS-core launch per (chunk, tile) pair. Each launch takes the full `K`, because the OS-core has
no psum read-back. Chunk and tile sizes are as equal as possible (the first `total % n` parts get one
block more), so neither count has to divide `dim0 / 16` or `dim2 / 24`; `main.c` rewrites the
streamer bounds for the smaller remainder launches.

`a_stationary` picks the loop order:

| `a_stationary` | Outer loop (resident, one buffer) | Inner loop (double-buffered) | L3 reads |
| :---: | --- | --- | --- |
| 0 | weight tiles | activation chunks | `B` once, `A` `nb_tiles` times |
| 1 | activation chunks | weight tiles | `A` once, `B` `nb_l_tiles` times |

A streamed operand with a single chunk or tile is loaded once and kept, so with
`nb_tiles = nb_l_tiles = 1` the call is one launch with every operand read once. The memory model
rejects a single streamed part with several resident ones (the resident buffer would be reloaded
before every launch); swap `a_stationary` instead.

The [multi-stage pipeline](README.md#recurring-patterns) runs over the launches in loop order
(iteration `i`):

```
DMA in the streamed operand of launch i | compute launch i-1 | DMA D of launch i-2 to L3
```

A new resident chunk or tile is loaded after the last launch that reads the old one. These loads are
the call's only stalls (about 6 kcc for a 384 KiB weight tile).

### TCDM layout

`[ resident | streamed 0 | streamed 1 | D 0 | D 1 ]`, every buffer 64 B aligned, sized for the
largest chunk and tile. Layouts:

- `A` chunk: `[M-block][K][16]`;
- `B` tile: `[N-block][K][24]`;
- `D`: `[N-block][M-block][16 × 24]`.

Each `D` tile is written to its place in the call's `D` in L3 with one 2-D DMA (one row of
`chunk_blocks` tiles per `N`-block).

### Choosing the parameters

Pick the order and the tile counts that minimise L3 traffic while the peak
([`memory_model.py`](../../target/snitch_cluster/sw/apps/osgemm-tiled/data/memory_model.py)) stays
within the TCDM budget:

- **Weight-stationary (0)**: for many tokens with weights that fit in a few tiles (projections,
  downsamples and SegFormer layers at high resolution);
- **Activation-stationary (1)**: for few tokens with large weights (stage-D layers at 49 tokens).

`datagen.py` appends the call's L3 traffic to `data.h` (`// L3 DMA per run: read X B, write Y B`).
This is the count that memsim reports with `MEMSIM_L3_BYTES=1`.

A layer with more tokens than one call holds is run as `J` calls on `dim0 = tokens / J`. A
weight-stationary run over the whole layer would read `B` only once, so the energy model counts the
weights once per layer, not once per call.

### Verified

vsim, 25 sampled outputs each, 0 errors; memsim L3 bytes equal the memory model:

| Params (`dim0/dim1/dim2`, `nb_tiles`/`nb_l_tiles`, order) | Simbacore cc | TCDM peak | L3 read / write |
| --- | ---: | ---: | ---: |
| 48/64/168, 3/1, activation | 1,443 | 19 KiB | 13,824 / 8,064 B |
| 64/56/192, 3/3, weight | 1,961 | 12 KiB | 21,504 / 12,288 B |
| 32/2048/528, 3/2, weight | 93,339 | 454 KiB | 1,277,952 / 16,896 B |
| 80/384/264, 1/1, weight | 21,842 | 150 KiB | 132,096 / 21,120 B |
