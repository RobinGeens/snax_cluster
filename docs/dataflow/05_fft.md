# 5. FFT family: `fft`, `fft-tiled`, `fft-3way`

> See also:
> [memory_layouts/09 — partitioned DFT layout](../../../chisel-ssm/docs/memory_layouts/09_fft.md)
> for the byte layouts of `weight1/2/3`, `twiddles`, the partition outputs,
> and the `hadamard_reordered` `[d][l]`-col-major layout. The `09_fft.md`
> page also documents *why* the deinterleave-step output of stage 2 is
> already a valid partition-2 input.

All three FFT programs implement an **EinFFT-style partitioned DFT**: a
length-`L` complex DFT is run as a sequence of small IS-core matmuls and SIMD
Hadamards. The streamer chain is fully on-chip — there are **no software
byte loops** between stages.

## 5.1 `fft` — 2-way partitioned EinFFT (L = L1 · L2)

**Source**: [fft/src/fft.c](../../target/snitch_cluster/sw/apps/fft/src/fft.c).

**TCDM allocation**:

```
[ weight1 | weight2 | in
| partition1_out          (IS-core out from step 1; FP8 after requant+transpose)
| twiddles
| hadamard_out            (SIMD CMUL output; packed re/im, interleaved every 16 elts)
| hadamard_reordered      (SIMD NOOP output; re/im split into one block each)
| partition2_out          (IS-core out from step 3; the FFT output) ]
```

**DMA-in**: `weight1`, `weight2`, `in`, `twiddles` once at startup.
`partition1_out` and `partition2_out` are zero-init via DMA from
`snrt_zero_memory_ptr()` (because they are IS-core psums and must not
contain stale BF16 values).

**Stage sequence** (all run on core 0, no barriers between stages):

| Step | Operation                       | MODE                    | Streamer ports                   | Shape args                                    |
| ---- | ------------------------------- | ----------------------- | -------------------------------- | --------------------------------------------- |
| 1    | partition 1 (IS-core GeMM)      | `M7_ISGEMM_SQ_TRANSPOSE`| `R11=weight1`, `R12=in`, `W3=partition1_out` | `2L1, 1, L1_padded, 1, dModel*L2`             |
| 2    | hadamard (SIMD CMUL_FP8)        | `M20_SIMD_CMUL_FP8`     | `R7=partition1_out`, `R13=twiddles`, `W3=hadamard_out` | `0,0,0,0,0`                                   |
| 2B   | re/im deinterleave (SIMD NOOP_FP8) | `M23_SIMD_NOOP_FP8`  | `R7=hadamard_out`, `W3=hadamard_reordered` (no_b) | `0,0,0,0,0`                                   |
| 3    | partition 2 (IS-core GeMM)      | `M6_ISGEMM_SQ`          | `R11=weight2`, `R12=hadamard_reordered`, `W3=partition2_out` | `2L2, 1, 2*L2_padded, 1, dModel*L1`           |

The fact that step 3 reads `hadamard_reordered` directly (no software
shuffle) is the central trick — the datagen-side layout is set up so that
SIMD-NOOP's output is already in the format partition-2's `R12` walks. See
[memory_layouts/09](../../../chisel-ssm/docs/memory_layouts/09_fft.md) for the
exact byte arithmetic.

**Why two `set_simd_streamer_*` helpers**: step 2 needs both R7 and R13
(operand pair for CMUL); step 2B only reads one operand (NOOP), so it uses
`set_simd_streamer_no_b`.

**Why `M7_ISGEMM_SQ_TRANSPOSE` for partition 1 but `M6_ISGEMM_SQ` for
partition 2**: the IS-core's output bank-transposer puts the FP8 result into
the layout the downstream SIMD step needs. Partition 1 feeds the SIMD CMul,
which needs the bank-transposed layout. Partition 2 is the final step (its
output is consumed only by the host check), so the plain `ISGEMM_SQ` is
sufficient.

## 5.2 `fft-tiled` — Phase A dModel-tiled, Phase B K-tiled, L3 spill

**Source**: [fft-tiled/src/fft.c](../../target/snitch_cluster/sw/apps/fft-tiled/src/fft.c).

**Goal**: cut TCDM peak by keeping only one tile's working set live at a
time. Two different tile axes are used because the two phases have different
constraints:

- **Phase A (steps 1 + 2 + 2B)**: tiled along **dModel**. `in` in L3 is
  stored as `[L2 · dModel][L1]` col-major with d outer in the N axis, so a
  dModel-slice is a contiguous L3 chunk. Each tile DMAs its slice in, runs
  partition1 + hadamard + reorder on that slice, then DMAs the reorder
  output up to **L3** into the matching `[d][l]`-col-major position.
- **Phase B (step 3)**: tiled along **K** (same reason as
  [isgemm-tiled](02_iscore_kernels.md#22-isgemm-tiled--k-axis-tiled-accumulating)).
  The IS-core requantizes its last-K iteration; tiling N would put a fake
  last-iter in every N-tile and corrupt the requant. So we tile K, and
  accumulate the psum in place via R13/W3 hitting the same FULL TCDM
  address; non-final K-tile runs in `M30_ISGEMM_SQ_NO_REQUANT`; the final
  one in `M6_ISGEMM_SQ`.

**L3 buffer**: `hadamard_reordered_l3 = snrt_l3alloc(M6_length_hadamard_reordered)`.
This holds the intermediate that Phase A produces and Phase B consumes; it
does not fit in TCDM at once.

**TCDM allocation** (Phase B overlays Phase A's working region):

```
Always live:                weight1 | weight2 | twiddles_tiled
Phase A working region:     in_tile | partition1_out_tile | hadamard_out_tile | had_reord_a_tile
Phase B (overlay):          had_reord_b_ktile | partition2_out (FULL psum)
```

(All slots are 64-byte aligned via `align64`.)

**Phase A loop** (`nb_tiles` iterations, single-buffered):

```
For tile = 0..nb_tiles-1:
  DMA-in:
    - in_tile <- M6_dft_in + tile * length_in_tile
    - zero-init partition1_out_tile (IS-core psum)
    - (defensive) zero-init hadamard_out_tile, had_reord_a_tile
  Barrier.
  Compute (3 stages):
    - step 1: partition1 with M7_ISGEMM_SQ_TRANSPOSE, N bound = M6_N_1_tile
    - step 2: hadamard CMUL with twiddles (shared across tiles — twiddles depend
              only on (l1, l2), not on d)
    - step 2B: SIMD-NOOP deinterleave
  Barrier.
  DMA-out:
    - reals: tile * M6_phaseA_dma_per_tile_per_part bytes
              from had_reord_a_tile to hadamard_reordered_l3
    - imags: same, offset by length_hadamard_reordered/2 on dst
             and length_hadamard_reordered_tile_re on src
  Barrier.
```

Phase A's compute is **not** pipelined with its own DMA (single-buffered);
DMA-out has to finish before the next tile's DMA-in to free the working
slots, and the working set is small enough that the latency is dominated by
compute.

**Phase B setup** (after Phase A barrier):

- Zero-init `partition2_out` (FULL psum) once.
- `set_isgemm_streamer_csr` once with `R11=weight2`, `R12=had_reord_b_ktile`,
  `W3=partition2_out`.
- `set_simbacore_csr(M6_ISGEMM_SQ, 2*L2, 1, M6_dInner_2_tile, 1, dModel*L1)`.

**Phase B loop** (`nb_tiles` iterations):

```
For tile = 0..nb_tiles-1:
  DMA-in:
    - had_reord_b_ktile <- hadamard_reordered_l3 + tile * length_had_reord_ktile
      (tile 0 = reals K-macro, tile 1 = imags K-macro)
  Barrier.
  Compute:
    - write_csr(BASE_PTR_READER_11_LOW, weight2 + tile * length_weight2_ktile)
    - write_csr(BASE_PTR_READER_12_LOW, had_reord_b_ktile)
    - write_csr(MODE, (last) ? M6_ISGEMM_SQ : M30_ISGEMM_SQ_NO_REQUANT)
    - fire and wait
  Barrier.
```

The W3 base (`partition2_out`) does **not** move — it is the FULL accumulator.

## 5.3 `fft-3way` — 3-way partitioned EinFFT (L = L1 · L2 · L3)

**Source**: [fft-3way/src/fft.c](../../target/snitch_cluster/sw/apps/fft-3way/src/fft.c).

The 3-way variant is a five-accelerator-stage version of `fft`. Three IS-core
GeMMs are interleaved with two CMUL+NOOP SIMD pairs. **No tiling**; all
buffers fit in TCDM.

**TCDM allocation**:

```
[ weight1 | weight2 | weight3
| in
| partition1_out  | twiddles1 | hadamard1_out | hadamard1_packed
| partition2_out  | twiddles2 | hadamard2_out | hadamard2_packed
| partition3_out ]
```

`hadamard{1,2}_packed` is the SIMD-NOOP deinterleave output. As in `fft`,
**no software reorder** is needed between stages: the datagen emits `dft_in`
/ `twiddles{1,2}` / `partition{1,2}_expected` in a "NEW" byte layout where
the inner `(L2·L3)` index is `col = d·L2·L3 + m3·L2 + m2` rather than the
standard `m2·L3 + m3`. Under that layout, the SIMD-NOOP output's 16-byte
chunks are 16 stride-1 `m2` values for one `(k1, m3, d)` cell — which is
exactly partition 2's K-tile format. The same trick applies between
partition 2 and partition 3.

**DMA-in**: `weight1`, `weight2`, `weight3`, `in`, `twiddles1`, `twiddles2`,
plus zero-init for `partition{1,2,3}_out`. Done once at startup.

**Stage sequence**:

| Step | Operation        | MODE                     | R7/R11    | R13/R12     | W3              | Shape args                            |
| ---- | ---------------- | ------------------------ | --------- | ----------- | --------------- | ------------------------------------- |
| 1    | partition 1      | `M7_ISGEMM_SQ_TRANSPOSE` | `weight1` | `in`        | `partition1_out`| `2L1, 1, L1_padded, 1, dModel·L2·L3`  |
| 2    | hadamard 1       | `M20_SIMD_CMUL_FP8`      | `partition1_out` | `twiddles1` | `hadamard1_out` | `0,0,0,0,0`                           |
| 2B   | reorder 1        | `M23_SIMD_NOOP_FP8`      | `hadamard1_out` (no_b) | — | `hadamard1_packed` | `0,0,0,0,0`                           |
| 3    | partition 2      | `M7_ISGEMM_SQ_TRANSPOSE` | `weight2` | `hadamard1_packed` | `partition2_out` | `2L2, 1, 2·L2_padded, 1, dModel·L1·L3` |
| 4    | hadamard 2       | `M20_SIMD_CMUL_FP8`      | `partition2_out` | `twiddles2` | `hadamard2_out` | `0,0,0,0,0`                           |
| 4B   | reorder 2        | `M23_SIMD_NOOP_FP8`      | `hadamard2_out` (no_b) | — | `hadamard2_packed` | `0,0,0,0,0`                           |
| 5    | partition 3      | `M6_ISGEMM_SQ`           | `weight3` | `hadamard2_packed` | `partition3_out` | `2L3, 1, 2·L3_padded, 1, dModel·L1·L2` |

Steps 1–2B and 3–4B use `M7_ISGEMM_SQ_TRANSPOSE` for the IS-core (the bank-
transposed output is what the downstream SIMD step needs); step 5 is the
final IS-core invocation and uses `M6_ISGEMM_SQ`.

There is one **`cluster_hw_barrier()`** between step 4B and step 5 in the
source (out of caution); the other inter-stage transitions are pure
back-to-back `wait_simbacore_and_streamer()` followed by the next
`start_simbacore_and_streamers()`.

## 5.4 Recreate the FFT programs

### `fft` (2-way)

1. Allocate the 8 TCDM buffers listed in §5.1.
2. DMA `weight1`, `weight2`, `in`, `twiddles` in; zero-init both partition
   psum buffers.
3. Run steps 1, 2, 2B, 3 with the streamer / MODE pairs from §5.1's table.
   No barriers needed between stages (single core).
4. Verify `partition1_out`, `hadamard_out`, `partition2_out` against golden.

### `fft-tiled` (dModel + K tiled, L3 spill)

1. L3-alloc `hadamard_reordered_l3` (16 KiB).
2. Allocate TCDM as in §5.2.
3. Preload `weight1`, `weight2`, `twiddles_tiled` (shared across tiles).
4. **Phase A** loop (`nb_tiles` iters): per-tile DMA-in `in_tile` + zero
   psum slots → 3 compute stages → DMA-out to `hadamard_reordered_l3`.
5. Zero-init `partition2_out`. Configure Phase B streamer + simbacore CSRs
   once.
6. **Phase B** loop (`nb_tiles` iters): per-tile DMA-in K-slice; rewrite
   `BASE_PTR_READER_11/12_LOW` + `MODE` (final vs no-requant); fire.
7. Verify `partition2_out`.

### `fft-3way` (no tiling)

1. Allocate the TCDM buffers listed in §5.3.
2. DMA all 6 weight/in/twiddles tensors; zero-init the 3 partition psums.
3. Run steps 1, 2, 2B, 3, 4, 4B, 5 with the streamer / MODE pairs from §5.3.
4. Verify `partition{1,2,3}_out` and `hadamard{1,2}_out` against golden.
