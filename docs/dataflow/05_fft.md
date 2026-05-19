# 5. FFT family: `fft`, `fft-tiled`, `fft-3way`, `fft-3way-tiled`

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

## `fft-3way-tiled` (3-way, tiled)

Three different tiling regimes, one per phase, because each phase's tiling
constraint differs:

- **Phase A** (partition 1 + hadamard 1 + reorder 1): **dModel-tiled**.
  Partition 1 has `K_1 = 1`, so N-axis tiling is safe — there is no
  requant-on-last-K hazard. The per-tile reorder output is spilled to L3.
- **Phase B** (partition 2 + hadamard 2 + reorder 2): **un-tiled**.
  K-tiling partition 2 would interleave a transposed-output final tile with
  no-requant non-final tiles whose write orders are incompatible, corrupting
  the FULL psum. So Phase B runs in one go; the partition 1 output is DMA'd
  back from L3 first, and the Phase B output is spilled to L3 to free TCDM
  for Phase C.
- **Phase C** (partition 3): **K-tiled**. Canonical accumulating-tiled
  pattern. The final partition is in plain (non-transposed) mode, so all
  tiles write in the same order — coherent psum.

| Tensor                                          | Phase | Lifecycle                                                  |
| ----------------------------------------------- | ----- | ---------------------------------------------------------- |
| `weight1`, `weight2`, `weight3`                 | all   | **shared** — preloaded once                                |
| `twiddles1`, `twiddles2`                        | all   | **shared** — depend only on `(l1, l2, l3)`, not `d`        |
| `in_tile`, `partition1_out_tile`, `hadamard1_out_tile`, `hadamard1_packed_a_tile` | A | **tiled** — per-tile dModel-slice |
| `hadamard1_packed_l3`                           | A→B   | **L3 spill** — assembled across Phase A tiles              |
| `hadamard1_packed_full`, `partition2_out`, `hadamard2_out`, `hadamard2_packed` | B | **FULL** — un-tiled in Phase B (with TCDM overlay) |
| `hadamard2_packed_l3`                           | B→C   | **L3 spill** — Phase B output, freed before Phase C        |
| `hadamard2_packed_b_ktile`                      | C     | **tiled** — per-K-tile B slot                              |
| `partition3_out`                                | C     | **FULL accumulator** — psum across K-tiles, final FFT output|

**Verification.** Only the final output (`partition3_out`) is byte-checked.
The intermediate `partition2_out` differs at byte positions the downstream
streamer doesn't read (a side-effect of the L3 round-trip), but the live
positions match and the final output is correct to the same FP8 quantization
noise level as the un-tiled `fft-3way`.
