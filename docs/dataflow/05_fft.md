# 5. FFT family: `fft`, `fft-tiled`, `fft-3way`, `fft-3way-tiled(-async)`, `fft-4way(-tiled-async)`

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
> [9. Async tiling](09_async_tiling.md) ·
> [12. SUC async](12_suc_async.md) ·
> [14. RMSNorm tiled](14_rmsnorm_tiled.md) ·
> [20. Bank-conflict-free double GEMM](20_double_gemm_conflict_free.md) ·
> [21. Conv downsample (im2col GEMM)](21_conv_downsample.md)

> Byte layouts (weights, twiddles, intermediates, the `flattenB`
> reorder output): [memory_layouts/09](../../../chisel-ssm/docs/memory_layouts/09_fft.md).

All FFT programs implement an **EinFFT-style partitioned DFT**: a length-`L`
complex DFT is run as a sequence of small IS-core matmuls interleaved with
SIMD Hadamards (CMul) and SIMD reorders (Noop deinterleave of re/im). The
streamer chain is fully on-chip — there are no software byte reorders
between stages (the one exception is `fft-4way`'s reorder2 `m3↔m4`
transpose, a scalar C loop in the un-tiled validation harness — see that
section).

**The no-software-reorder trick.** The data generator emits inputs in a
byte layout chosen so each SIMD-Noop deinterleave output is *already* in
the format the next IS-core partition consumes. This is why no stage needs
a software byte loop between accelerator launches (again excepting
`fft-4way`'s reorder2 transpose).

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

(Naming: `L1`/`L2` are the FFT butterfly dimensions, `L = L1·L2` — not the TCDM
scratchpad, which is also colloquially called "L1".)

Tiles use different axes per phase, because the two phases have different
constraints:

- **Phase A** (partition 1 + hadamard + reorder) is **dModel-tiled**. Each
  tile DMAs a `dModel`-slice of the input in, runs the three accelerator
  stages on per-tile TCDM slots, and spills the per-tile reorder output up to
  an L3 spill buffer. The on-chip reorder emits col-major per d-tile; the
  spill writes it into the L3 buffer in `flattenB` (K-tile-major) order — the
  layout the Phase B IS-core reads (see below).
- **Phase B** (partition 2) is **K-tiled**, mirroring `isgemm-tiled`. The
  `partition2_out` psum stays FULL in TCDM and accumulates across tiles;
  non-final tiles run in no-requant mode, final tile applies the requant.

The L3 spill is needed because the reorder output of all of Phase A would
not fit in TCDM at once.

**`hadamard_reordered` L3 layout (`flattenB`).** The partition-2 IS-core reads
its `B` input (the `2·L2 × L1·dModel` stacked re/im matrix) in `flattenB`
(`K_M_N`) order: the `2·L2` contraction rows are split into `2·L2/seqLenUnroll`
K-tiles laid out **outermost**, each followed by its full `L1·dModel` N-sweep.
When `L2 == seqLenUnroll` each re/im half is exactly one K-tile and `flattenB`
collapses to plain col-major reals-then-imags, so the Phase A spill is a single
contiguous 1D DMA. When `L2 > seqLenUnroll` the col-major reorder output no
longer matches `flattenB`, so the Phase A spill **scatters** each
`seqLenUnroll`-element block `reals[m·L2 + k2·seqLenUnroll … ][d]` into its
`flattenB` block `(k2, d, m)` (a `k2 × dModel_tile` loop of block-granular 2D
DMAs). Phase B then reads each K-slice as a contiguous L3 chunk. This keeps the
finicky on-chip reorder streamers (`R7_2B`/`W3_2B`) L2-agnostic and folds the
K-tile reorder into the already-present spill DMA.

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

Seven accelerator stages in order:

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
differently, the **whole `fft-3way` kernel is run once per dModel-slice**
of `dM = dModel/nb_tiles_A` channels. Every intermediate buffer shrinks to
`1/nb_tiles_A` and the pipeline runs entirely in TCDM per slice — no
inter-phase L3 spill. (A per-phase regime — Phase A dModel-tiled, B un-tiled, C
K-tiled — would keep *full* Phase-B/C buffers (`partition{2,3}_out` are full BF16
activations = `~32 KiB · dModel`) that blow past the 512 KiB TCDM for any realistic
`dModel`, e.g. 3072 KiB at `dModel=96`.)

Per outer slice (`nb_tiles_A` = number of slices, must divide `dModel`):

1. For each l3-tile, DMA-gather the tile's input + twiddles into contiguous
   tile-local buffers (for `nb_l3 = 1` this is the whole `dM`-channel slice).
2. Run the 5-step kernel (partition1 → hadamard1 → reorder1 → partition2 →
   hadamard2 → reorder2) per l3-tile, assembling the full `[re|im]` H2; then run
   **partition3** as a K-tiled (over `nb_l3`) accumulation reading that H2.
   Descriptors are the un-tiled `fft-3way` descriptors with `dModel → dM`,
   `L3 → l3_tile`.
3. **Scatter** `partition3_out` into its d-slice of the full L3 output. The output
   is flattened `K_M_N` (`d` outer in the column axis, `Mu = seqLenUnroll`), so a
   d-slice is `M_3 = 2·L3/seqLenUnroll` strided row-blocks, **not** one contiguous
   chunk — a 2-D DMA (`repeat = M_3 = out_nblk`, `dst_stride = out_block_full`,
   `src_stride = out_block_slice`) places it correctly. The block sizes are the
   **real (FP8) row-block bytes** (`out_block_{full,slice} = {dModel,dM}·L1·L2·seqLenUnroll`),
   **not** `length_partition3_out / M_3`: `partition3_out`'s final type is FP8, which
   the IS-core zero-pads to the BF16 psum footprint (`padWithZeros`), so each
   per-slice buffer is `[real | zeros]` and the padded length is 2× the real data.
   Dividing the padded length by `M_3` would make each block straddle two `m2` row-blocks
   and scramble `d` vs `m2`.

The full output is assembled in L3 and verified there (scalar reads).

**TCDM footprint.** Stages 1-4 use **tile-local** buffers (one l3-tile at a time):
gathered `in_tile`/`tw1_tile`/`tw2_tile`, a BF16 per-tile psum `P_tile`
(`slot_size_tile = 2·Lt·dM·2`, reused by gemm1/2), and two FP8 hadamard scratch
buffers `H1`/`H2` (each `slot_size_tile/2`). The assembled full **H2**
(`2·L·dM` FP8) is resident from stages 1-4 into partition3. Partition3's psum
**P3** (full BF16, `2·L·dM·2`) overlays the now-dead stages-1-4 scratch, so
`peak = weights + full_H2 + max(stages-1-4 scratch, P3)`. The footprint shrinks with
`nb_tiles_A` (smaller `dM`) and with `nb_l3` (smaller per-tile scratch). The
build's memory model asserts `P3 ≤ stages-1-4 scratch` (P3 must fit the overlay).

| Tensor | Lifetime | Notes |
| --- | --- | --- |
| `weight1/2/3` | all slices | **shared** — preloaded once, broadcast over `d` |
| `in_tile`, `tw1_tile`, `tw2_tile` | per l3-tile | gathered from `dft_in`/`twiddles` |
| `P_tile` (partition psum) | per l3-tile | BF16 `slot_size_tile`; gemm1→cmul1, gemm2→cmul2 |
| `H1` (CMul out), `H2` (reorder out) | per l3-tile | FP8 `slot_size_tile/2` each |
| `H2_full` (`[re|im]` partition-3 input) | stages 1-4 → partition3 | FP8 `2·L·dM`, assembled on-chip |
| `P3` (partition-3 psum) | partition3 | BF16 `2·L·dM·2`, overlays dead stage scratch |
| `output` (full `partition3_out`) | L3 | assembled by the per-slice 2-D scatter |

**Latency hiding (DMA + CSR).** Right-sizing the FP8 scratch frees the headroom for
two overlaps that the old 2-slot ping-pong could not afford:

- **DMA behind compute.** Once a partition psum has been consumed by its CMul, the DM
  core re-zeros `P` for the next gemm *during* the following reorder (NOOP) step, and
  prefetches the next slice's input in the same window. Only the per-slice output
  scatter stays on the critical path. (The gemms read-accumulate the psum via R13/W3
  on the same `P` pointer, so each gemm needs a zeroed target — `K_1=1` but
  `K_2=K_3=2`, so the zero is not removable, only hideable.)
- **CSR setup behind the accelerator.** The streamer latches its config at `start`
  (verified: programming a step's CSRs — including disabling the *running* gemm's
  R11/R12 ports — mid-run does not disturb it), so each step's ~50 streamer-CSR writes
  are issued while the previous step still runs. The 6-write simbacore MODE CSR stays
  serial (issued right before each `start`).

These cut the Snitch overhead from **~136% to ~107%** (`dModel=96`: 585k→512k cycles),
byte-identical output. The residual ~107% is the per-invocation **streamer fill/drain**
(W3 writer + bank-transposer) across the 56 small accelerator invocations
(8 slices × 7 steps): the streamer is busy ~1.9× the compute core, and that drain does
**not** overlap — consecutive steps have true data deps *and* share the R13/W3 ports, so
no streaming-chain is possible — and shrinks only with fewer invocations (capped by the
512 KiB TCDM at `dM=12`). Reducing it further is an RTL-level concern (faster
transposer/W3), not a SW one.

**Verification.** Only the final `partition3_out` is byte-checked (±1-LSB). Slicing
is **exact**: the per-slice kernel is the un-tiled `fft-3way` kernel run on `dM`
channels, and every quantization (BF16 partitions, FP8 hadamards/output) is
per-element — no scale depends on the `dModel` grouping — so a slice's bytes equal
the same channels of the full-`dModel` run for any `nb_tiles_A`.

The small residual (`dModel=96` `dM=12` → ~2/25) is the FFT DFT kernel's intrinsic
HW-vs-Scala-model FP8 rounding floor: many FFT outputs are near-zero and land on
FP8 rounding boundaries, so a few round-to-zero or flip ±1 LSB. The 2-way
`fft-tiled` shows the same floor (~1/50); `einfft` (no DFT-matmul rounding) hits
0/100. It is independent of `nb_tiles_A`, so prefer the largest `dM` that fits.

## `fft-3way-tiled-async` (l3-streamed, for long sequences)

`fft-3way-tiled` keeps every partition tile (so `LxD_tile`) resident, for a total of  `~9·L·D_tile` + `twiddles1 (2·L)`
This app streams the `L3` axis to fit much longer sequences

The key structural fact: `m3` (the `L3` axis) is a batch factor for partitions 1&2
(they run independently per `m3`) but the contraction axis of partition 3. So:

1. Stages 1–4 run per `l3`-tile (`l3_tile` channels of `m3`, must be a multiple of
   `seqLenUnroll`). Their input + twiddles are DMA-gathered from DRAM into contiguous
   tile-local buffers, on which the unchanged `L3=l3_tile` descriptors are correct.
2. Each tile's reordered output is staged to L3 with its re/im halves split into the
   full-K order. Partition 3 contracts the complex `m3` axis, whose K dimension is
   stacked `[re(all m3) | im(all m3)]`. A tile produces `[re(tile m3) | im(tile m3)]`, so
   concatenating tiles would scramble K (`[re t0|im t0|re t1|im t1]`). Instead the noop2
   output's re half goes to the L3 re region (`+ lt·h2_half`) and its im half to the im
   region (`+ h2_im_region + lt·h2_half`), so L3 ends up as the correct
   `[re t0|re t1|…|im t0|im t1|…]`.
3. Partition 3 N-tiles its output batch (`N_3 = dM·L1·L2`) into `nb_ntile` chunks. Each
   chunk gathers its full-K `H2` `N`-run back from L3 into a small TCDM buffer, then
   K-tiles the contraction into `nb_l3` chunks of `K=2·L3_padded/nb_l3` and accumulates. The K-tiling is required because the
   IS-core hangs if a single `ISGEMM_SQ` is given the full `K=2·L3_padded` once `2·L3 > ~32`
   (e.g. `K=96` at `L3=32`). Because both `weight3` and the gathered `H2` are now in `[re|im]`
   order, contiguous K-chunks pair correctly. Then scatter the chunk to the L3 output.

So only one `l3`-tile of stages-1–4 scratch, one gathered `N`-tile of `H2`, and one `N`-tile
of the partition-3 psum are ever resident. Partition 3 runs only after stages 1–4 finish for a
slice, so its buffers (`h2_ntile`, `P3`) overlay the by-then-dead stages-1–4 scratch (in
`src/fft.c`, `ptr_h2ntile = ptr_in`). Peak ≈ `weights + max(in_tile + tw1_tile + tw2_tile +
P_tile + 2·H_tile, h2_ntile + P3_ntile)` (all in `memory_model.py`).

## `fft-3way-L8` (3-way with a factor-8 axis, `L3 < seqLenUnroll`)

`fft-3way-tiled` needs every `Lx` a multiple of `seqLenUnroll` (=16). `fft-3way-L8` adds an
`L3 < 16` axis (e.g. `seqLen = 2048 = 16·16·8`, `L3 = 8`) by **padding `m3`**: stages 1–2 run
at `L3t = seqLenUnroll = 16` with the real `L3` m3 gathered into pre-zeroed buffers, so the
`(16-L3)` pad m3 stay zero through the linear DFT+twiddle stages and the real results land at
m3 `0..L3-1`. Then the real `L3` partition-3 `B` is extracted from the `L3t=16` reorder2 output
and partition 3 runs at the real `L3` (a plain GEMM, fine at any size). `nb_l3 = 1` in this mode.

Two `L3 < seqLenUnroll`-specific points (both no-ops for `L3 ≥ 16`, so the other 3-way apps are
unchanged):

1. **twiddles2 padding.** `twiddles2` is written via `simdInterleaveRealImag`, which groups 16
   lanes; with only `L3` m3 per `k2` it would pack `16/L3` different `k2` into one group and the
   per-`k2` gather would mis-read it. So the generator pads each `k2` row's m3 to `seqLenUnroll`
   (zeros) before interleaving — one `k2` per group again. (`twiddles1`/`hadamard1` are unaffected
   because tw1's index mixes m3 with m2.)
2. **B extract.** At `L3t=16` partition-3 sees `K=2`, so reorder2 emits `B` region-split
   `[re region | im region]`, each a per-`N` block of `L3t` m3 (`L3` real + pad), `N`-stride
   `L3t`. The real `B` (`K=1`) is `[N][L3 re, L3 im]` interleaved (`N`-stride `2·L3`). The extract
   copies the first `L3` re-m3 (re region) and `L3` im-m3 (im region at `+Lt·dM`) into each packed
   real `N`-group.

Passes the FP8 floor at `L3=8`. Only `L3` can take the small factor: `L1` must be a multiple of
`seqLenUnroll`, and `L2 < 16` hangs (the gemm2 `B`-reorder needs an `m2`-block of 16).

## `fft-4way` (4-way, `L = L1·L2·L3·L4`)

Generalises `fft-3way` to four sub-DFTs with three twiddles between them: `partition1 →
hadamard1 → reorder1 → partition2 → hadamard2 → reorder2 → partition3 → hadamard3 → reorder3 →
partition4`. Partitions 1–3 are bank-transposed (`ISGEMM_SQ_TRANSPOSE`, each feeds the next
SIMD CMul); partition4 is plain (host-read output). Index conventions and twiddles are the
recursive extension of the 3-way (see `memory_layouts/09_fft.md` §9.15).

**Contraction-order chip layout.** The single design idea that makes this work: lay out the
trailing axes `(m2, m3, m4)` in *contraction order* — `m2` innermost, then `m3`, then `m4` —
in the on-chip byte layout (the host `dft_in` is pre-shuffled by datagen, the chip never
shuffles). Then partition2 contracts `m2`, partition3 contracts `m3`, partition4 contracts
`m4`, and each contracted axis is the innermost of the trailing group at its stage. Stages 1–2
keep both `m3,m4` in the trailing, so their chip layout uses the combined `jc = m4·L3 + m3`
(m3 inner) which maps to the math index `jm = m3·L4 + m4`; stages 3–4 have only `m4` left.

**reorder2 needs a transpose.** reorder1 and reorder3 are plain block-swap deinterleaves (the
contracted axis is already innermost, so the SIMD NOOP just splits re/im). reorder2 is the
exception: partition3 contracts `m3`, but after partition2/cmul2 the data has `m4` innermost, so
`m3` must be gathered to the K-axis — an `m3↔m4` transpose. The bank-based streamers cannot do
this (a bank is a fixed run of contiguous bytes; a transpose needs a strided scatter). The
un-tiled `fft-4way` does it as a scalar C loop (it is a kernel-validation harness, not a
performance target); the async variant folds it into the L3-staging DMA (which *can* scatter),
exactly as the 3-way async stages H2 with its re/im split.

`fft-4way` keeps every buffer resident, so it only fits small `dModel` at the minimum valid
`L = 8192` (`L1=16, L2=L3=L4=8`); use it to validate the kernel. Residual sim errors are the
FFT FP8 rounding floor (near-zero outputs landing on FP8 boundaries), cascaded across four
partitions — same nature as the 3-way's `~2/25` floor.

## `fft-4way-tiled-async` (l3-streamed, for `L = 16⁴` and beyond)

The dModel-tile alone cannot fit `L = 65536`: it only slices `dM` and leaves the L-proportional
buffers (`twiddles1 ≈ 2·L`, the BF16 partition psum `≈ 4·L·dM`, two FP8 hadamards `≈ 2·L·dM`)
intact — peak `≈ 717 KiB > 512 KiB` even at `dM=1`. The async streams `m3` (the partition-3
contraction) per `l3`-tile, but the whole partition-3 input stays TCDM-resident — no L3 spill.
Stages 1–2 run per `l3`-tile on gathered tile-local buffers: the tile's input and twiddles are
DMA-gathered from DRAM so the `m3`-block lands in K-position for partition 3. reorder2
deinterleaves cmul2's re/im (DMA gather for `L3 == 16`, block-swap SIMD NOOP for `2·L3 ≤ 16`)
and writes the tile's contribution into a persistent TCDM buffer `packed3`. Once all `l3`-tiles
have accumulated into `packed3`, partition3 runs **once** as a single full-K (`2·L3_padded`)
`ISGEMM_SQ_TRANSPOSE`, and `cmul3 → reorder3 → partition4` (contracts `m4`, already innermost)
run **once** over the full buffer before the output scatter. Stage-3/4 N-tiling (`nb_ntile > 1`)
is asserted unimplemented.

**reorder2 / reorder3 regimes.** Both deinterleave the cmul re/im into the next partition's
K-blocks, in HW, picked by the contracted axis length:
- axis `== 16` (`= seqLenUnroll`): re/im split into K-blocks via two DMA gathers (`Lx`-byte
  chunks at `2·Lx` source stride).
- `2·axis ≤ 16` (`K = 1`): the split degenerates to the block-swap SIMD NOOP (reuses
  reorder1's `R7_2B/W3_2B`), writing the N-major next-partition input directly.

**Factorisation limits.** `L1` is a multiple of 16; `L3 ≤ 32` (`2·L3 ≤ 64`, one square tile), l3-tiled at `l3_tile=16`; `L2 ≤ 16` (`L2 > 16` is unsupported).

