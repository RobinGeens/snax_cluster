# 2. IS-core kernels: `isgemm` and `isgemm-tiled`

> See also: [byte layouts for IS-core A, B, C, D](../../../chisel-ssm/docs/memory_layouts/02_gemm_layouts.md).
> The IS-core's input A is in **ConvFormat** when it comes from the Switch-
> core (the Mamba use case); for standalone `isgemm` and FFT it is fed
> directly via `R11`.

The IS-core computes `D[dim0, dim2] = C[dim0, dim2] + A[dim0, dim1] · B[dim1, dim2]`
in `K_M_N` loop order. The IS-core **reads its own psum back** through `R13`
and writes to `W3`, which is what enables in-place accumulation across tiles.

## 2.1 `isgemm` — single-shot, with and without requant

**Source**: [isgemm/src/isgemm.c](../../target/snitch_cluster/sw/apps/isgemm/src/isgemm.c).

This program runs the same GeMM twice — once with FP8 requant
(`M4_ISGEMM`) and once without (`M5_ISGEMM_NO_REQUANT`) — to show both modes.

**TCDM allocation**:

```
[ A (M4_addr_a) | B (M4_addr_b) | CD (M4_addr_cd) ]
```

`CD` is a single buffer that serves as both the *input bias* C and the
*output* D. The IS-core reads C via `R13` and writes D via `W3` to the
**same address**. Output bytes are always BF16-sized; in requant mode the
FP8 value lives in the low byte of each BF16 slot with the high byte zeroed.

**DMA-in**: A, B, and C (loaded into `CD`).

**Compute**: a single accelerator invocation per mode.

| Variant                 | MODE                    | Streamer ports         | `set_simbacore_csr` args |
| ----------------------- | ----------------------- | ---------------------- | ------------------------ |
| `test_isgemm`           | `M4_ISGEMM`             | `R11=A`, `R12=B`, `W3=CD` | `dim0, 1, dim1, 1, dim2` |
| `test_isgemm_no_requant`| `M5_ISGEMM_NO_REQUANT`  | same                   | same                     |

Both are launched with `start_simbacore_and_streamers(M4_R10_en, 0, M4_R11_en, 0)`.
`R13` is implicitly active because the IS-core always reads its psum bias.

## 2.2 `isgemm-tiled` — K-axis tiled, accumulating

**Source**: [isgemm-tiled/src/main.c](../../target/snitch_cluster/sw/apps/isgemm-tiled/src/main.c).

**What is tiled**: the **K axis** (`dInner`, the IS-core's reduction axis).
A and B are tiled and double-buffered; C/D is the **full** psum accumulator
and is allocated once.

**Why K and not N**: the IS-core applies its FP8 requant on the *last K
iteration* of the inner K loop. Tiling N would make every N-tile think it
saw a last K iteration, applying the requant on partial sums and producing
half-of-N garbage on subsequent tiles. K-tiling keeps the requant tied to the
real last iteration, which is the *final tile*. The non-final tiles use
`M5_ISGEMM_NO_REQUANT` (psum stays in BF16); the final tile uses `M4_ISGEMM`
(requant fires on the fully accumulated psum). See the
[iscore_n_tiling memory note](../../../.claude/projects/-esat-micas-lapserv11-users-rgeens-snax-cluster/memory/iscore_n_tiling.md)
for the failure mode of N-tiling.

**TCDM allocation**:

```
[ A ping | A pong | B ping | B pong | CD (full) ]
```

**DMA-in protocol**:

- Bias `M4_C` is preloaded into `CD` once before the loop (it serves as the
  tile-0 input).
- Per tile: DMA A-tile `i` and B-tile `i` into the ping/pong slot.

**One-time setup**:

```c
set_isgemm_streamer_csr(ptr_a[0], R11_ss/tb/ts,
                        ptr_b[0], R12_ss/tb/ts,
                        ptr_cd,   W3_ss/tb/ts);
set_simbacore_csr(M4_ISGEMM, dim0, 1, M4_dInner_tile, 1, dim2);
```

`R13` is enabled with `M4_R11_en` and reads `ptr_cd` (no separate enable for
R13's base pointer — it tracks W3's). The CD buffer is shared by R13/W3 on
every tile, which is the accumulator mechanism.

**2-stage pipeline** (per iteration `i ∈ [0, nb_tiles + 1)`):

| Iteration window     | Stage                                                       | Core    |
| -------------------- | ----------------------------------------------------------- | ------- |
| `i ∈ [0, nb_tiles)`  | DMA-in A-tile + B-tile `i` (ping/pong)                      | DM core |
| `i ∈ [1, nb_tiles+1)`| Compute tile `i−1`: `CD += A_tile · B_tile`                 | core 0  |

We do **not** overlap compute with compute (only DMA-in with compute). The
compute stages are serial because they share the CD accumulator.

**Per-tile CSR updates**:

```c
write_csr(BASE_PTR_READER_11_LOW, ptr_a[cbuf]);
write_csr(BASE_PTR_READER_12_LOW, ptr_b[cbuf]);
write_csr(MODE, (tile == nb_tiles - 1) ? M4_ISGEMM : M5_ISGEMM_NO_REQUANT);
start_simbacore_and_streamers(M4_R10_en, 0, M4_R11_en, 0);
wait_simbacore_and_streamer();
```

The two moving base pointers and the `MODE` CSR are the only things that
change per tile. CD's W3 base pointer, the loop bounds (with `M4_dInner_tile`),
and the streamer strides are configured once.

## 2.3 Recreate the program

To recreate `isgemm-tiled` from scratch:

1. Allocate `A0 | A1 | B0 | B1 | CD` in TCDM.
2. Preload C into CD once. Barrier.
3. Configure the IS-core streamer **once** with `M4_dInner_tile` as the K
   bound, `ptr_cd` as W3's base.
4. Loop `nb_tiles + 1` iterations. In iteration `i`:
   - If `i < nb_tiles`, DMA A-tile + B-tile `i` into `ptr_a[i%2]`, `ptr_b[i%2]`.
   - If `i ≥ 1`, rewrite `BASE_PTR_READER_11/12_LOW` to the previous tile's
     buffer, set `MODE` to `NO_REQUANT` unless this is the final tile, fire
     the accelerator, wait.
   - End the iteration with `dma_wait_all + cluster_hw_barrier`.
5. After the loop, CD holds the final FP8 result (low byte of each BF16 slot).
