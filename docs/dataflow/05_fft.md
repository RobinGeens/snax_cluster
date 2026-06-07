# 5. FFT family: `fft`, `fft-tiled`, `fft-3way`, `fft-3way-tiled`

> **All pages:**
> [README](README.md) ·
> [1. OS-core kernels](01_oscore_kernels.md) ·
> [2. IS-core kernels](02_iscore_kernels.md) ·
> [3. SIMD / RMSNorm kernels](03_simd_kernels.md) ·
> [4. Mamba main](04_mamba_main.md) ·
> **5. FFT family (this page)** ·
> [6. EinFFT MLP](06_einfft_mlp.md) ·
> [7. VMamba SS2D](07_vmamba.md) ·
> [8. Performance optimization](08_performance_optimization.md) ·
> [9. Async tiling](09_async_tiling.md)

> Byte layouts (weights, twiddles, intermediates, the `[d][l]`-col-major
> reorder output): [memory_layouts/09](../../../chisel-ssm/docs/memory_layouts/09_fft.md).

All FFT programs implement an **EinFFT-style partitioned DFT**: a length-`L`
complex DFT is run as a sequence of small IS-core matmuls interleaved with
SIMD Hadamards (CMul) and SIMD reorders (Noop deinterleave of re/im). The
streamer chain is fully on-chip — there are no software byte reorders
between stages.

**The no-software-reorder trick.** The data generator emits inputs in a
byte layout chosen so each SIMD-Noop deinterleave output is *already* in
the format the next IS-core partition consumes. This is why no stage needs
a software byte loop between accelerator launches.

## `fft` (2-way, `L = L1 · L2`)

Four accelerator stages in order:

1. **partition 1** — IS-core matmul against `weight1`.
2. **hadamard** — SIMD CMul against the twiddle table.
3. **reorder** — SIMD Noop that deinterleaves re/im into a partition-2-ready
   layout.
4. **partition 2** — IS-core matmul against `weight2`. Produces the FFT
   output.

The first IS-core launch uses an output bank-transposed mode (its FP8 result
goes through the bank transposer so the downstream SIMD CMul can read it via
the banked port); the final IS-core launch uses the plain mode.

| Tensor                 | Role                                  |
| ---------------------- | ------------------------------------- |
| `weight1`, `weight2`   | partition GeMM weights (in)           |
| `in`                   | DFT input (in)                        |
| `twiddles`             | hadamard twiddles (in)                |
| `partition1_out`       | stage-1 output (intermediate)         |
| `hadamard_out`         | stage-2 output (intermediate)         |
| `hadamard_reordered`   | stage-2B reorder output (intermediate)|
| `partition2_out`       | FFT output                            |

## `fft-tiled` (2-way, tiled)

Tiles use different axes per phase, because the two phases have different
constraints:

- **Phase A** (partition 1 + hadamard + reorder) is **dModel-tiled**. Each
  tile DMAs a `dModel`-slice of the input in, runs the three accelerator
  stages on per-tile TCDM slots, and DMAs the per-tile reorder output up to
  an L3 spill buffer.
- **Phase B** (partition 2) is **K-tiled**, mirroring `isgemm-tiled`. The
  `partition2_out` psum stays FULL in TCDM and accumulates across tiles;
  non-final tiles run in no-requant mode, final tile applies the requant.

The L3 spill is needed because the reorder output of all of Phase A would
not fit in TCDM at once.

| Tensor                            | Phase | Lifecycle                                                          |
| --------------------------------- | ----- | ------------------------------------------------------------------ |
| `weight1`, `weight2`              | A / B | **shared** — preloaded once, sliced per tile via base-pointer move |
| `twiddles`                        | A     | **shared** — depends only on `(l1, l2)`, not `d`                   |
| `in_tile`                         | A     | **tiled** — per-tile dModel-slice (single-buffered)                |
| `partition1_out_tile`             | A     | **tiled** — per-tile psum slot                                     |
| `hadamard_out_tile`               | A     | **tiled** — per-tile intermediate                                  |
| `had_reord_a_tile`                | A     | **tiled** — per-tile reorder output                                |
| `hadamard_reordered_l3`           | A→B   | **L3 spill** — assembled across Phase A tiles, consumed in Phase B |
| `had_reord_b_ktile`               | B     | **tiled** — per-K-tile B slot                                      |
| `partition2_out`                  | B     | **FULL accumulator** — psum across K-tiles, final FFT output       |

## `fft-3way` (3-way, `L = L1 · L2 · L3`)

Five accelerator stages in order:

1. **partition 1** — IS-core (bank-transposed output).
2. **hadamard 1** — SIMD CMul against `twiddles1`.
3. **reorder 1** — SIMD Noop deinterleave.
4. **partition 2** — IS-core (bank-transposed output).
5. **hadamard 2** — SIMD CMul against `twiddles2`.
6. **reorder 2** — SIMD Noop deinterleave.
7. **partition 3** — IS-core (plain mode). Produces the FFT output.

All buffers fit in TCDM at once; no L3 spill.

| Tensor                                 | Role                                  |
| -------------------------------------- | ------------------------------------- |
| `weight1`, `weight2`, `weight3`        | partition GeMM weights (in)           |
| `in`                                   | DFT input (in)                        |
| `twiddles1`, `twiddles2`               | hadamard twiddles (in)                |
| `partition{1,2,3}_out`                 | stage outputs (intermediates / final) |
| `hadamard{1,2}_out`                    | CMul outputs (intermediates)          |
| `hadamard{1,2}_packed`                 | reorder outputs (intermediates)       |

## `fft-3way-tiled` (3-way, **outer dModel-tile**)

`dModel` is the independent batch axis of every partition (`N_i = dModel·…`,
weights/twiddles broadcast over `d`). So instead of tiling each phase
differently, the **whole un-tiled `fft-3way` kernel is run once per dModel-slice**
of `dM = dModel/nb_tiles_A` channels. Every intermediate buffer shrinks to
`1/nb_tiles_A` and the 5-step pipeline runs entirely in TCDM per slice — no
inter-phase L3 spill. This replaced the earlier per-phase regime (Phase A
dModel-tiled, B un-tiled, C K-tiled), whose *full* Phase-B/C buffers
(`partition{2,3}_out` are full BF16 activations = `~32 KiB · dModel`) blew past
the 512 KiB TCDM for any realistic `dModel` (e.g. 3072 KiB at `dModel=96`).

Per outer slice (`nb_tiles_A` = number of slices, must divide `dModel`):

1. DMA the slice's `dM` input channels in (`dft_in` is d-major, so a slice is one
   contiguous chunk).
2. Run the 5-step kernel (partition1 → hadamard1 → reorder1 → partition2 →
   hadamard2 → reorder2 → partition3) on `dM`-sized buffers — descriptors are the
   un-tiled `fft-3way` descriptors with `dModel → dM`.
3. **Scatter** `partition3_out` into its d-slice of the full L3 output. The output
   is flattened `K_M_N` (`d` outer in the column axis, `Mu = seqLenUnroll`), so a
   d-slice is `M_3 = 2·L3/seqLenUnroll` strided row-blocks, **not** one contiguous
   chunk — a 2-D DMA (`repeat = M_3 = out_nblk`, `dst_stride = out_block_full`,
   `src_stride = out_block_slice`) places it correctly. The block sizes are the
   **real (FP8) row-block bytes** (`out_block_{full,slice} = {dModel,dM}·L1·L2·seqLenUnroll`),
   **not** `length_partition3_out / M_3`: `partition3_out`'s final type is FP8, which
   the IS-core zero-pads to the BF16 psum footprint (`padWithZeros`), so each
   per-slice buffer is `[real | zeros]` and the padded length is 2× the real data.
   Dividing the padded length by `M_3` makes each block straddle two `m2` row-blocks
   and scrambles `d` vs `m2` — that was the original 22/25-failure bug.

The full output is assembled in L3 and verified there (scalar reads).

**TCDM footprint / `nb_tiles_A`.** The slice runs through a **2-slot ping-pong**
(slotA/slotB, each = the largest per-slice buffer `partition_out` BF16); peak ≈
`always_live + 2·align64(2·L·dM·2)`. At `L=4096` this fits TCDM for `dM ≤ 12`, so
**`nb_tiles_A ≥ ceil(dModel/12)`** (e.g. `dModel=96 → nb_tiles_A=8, dM=12`,
~396 KiB). `nb_tiles_A=1` (no slicing) is exact but only fits small `dModel`.

| Tensor | Lifetime | Notes |
| --- | --- | --- |
| `weight1/2/3`, `twiddles1/2` | all slices | **shared** — preloaded once, broadcast over `d` |
| `in`, `partition1_out`, `hadamard1_out/_packed`, `partition2_out`, `hadamard2_out/_packed`, `partition3_out` | per slice | `dM`-sized, ping-pong'd through 2 slots |
| `output` (full `partition3_out`) | L3 | assembled by the per-slice 2-D scatter |

**Verification.** Only the final `partition3_out` is byte-checked (±1-LSB). Slicing
is **exact**: the per-slice kernel is the un-tiled `fft-3way` kernel run on `dM`
channels, and every quantization (BF16 partitions, FP8 hadamards/output) is
per-element — no scale depends on the `dModel` grouping — so a slice's bytes equal
the same channels of the full-`dModel` run for any `nb_tiles_A`. (An earlier note
here blamed a "coarser per-`dM` requant scale"; that was wrong. The 22/25 failures
were the scatter zero-pad bug above, not quantization.)

The small residual (`dModel=96` `dM=12` → ~2/25) is the FFT DFT kernel's intrinsic
HW-vs-Scala-model FP8 rounding floor: many FFT outputs are near-zero and land on
FP8 rounding boundaries, so a few round-to-zero or flip ±1 LSB. The 2-way
`fft-tiled` shows the same floor (~1/50); `einfft` (no DFT-matmul rounding) hits
0/100. It is independent of `nb_tiles_A`, so prefer the largest `dM` that fits.
