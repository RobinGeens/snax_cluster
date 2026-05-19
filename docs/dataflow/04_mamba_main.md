# 4. Mamba main: `main`, `main-tiled`, `main-full`, `suc`

> See also:
> [memory_layouts/07 — per-mode reference](../../../chisel-ssm/docs/memory_layouts/07_mode_reference.md),
> [04 — SUCFormat](../../../chisel-ssm/docs/memory_layouts/04_suc_format.md),
> [05 — xProj format](../../../chisel-ssm/docs/memory_layouts/05_xproj_format.md),
> [06 — delta-weight split](../../../chisel-ssm/docs/memory_layouts/06_delta_weight.md).
>
> The byte layouts of every Phase 1 / Phase 2 buffer are documented there.
> This page covers **program orchestration** — which DMAs, which streamer
> ports, which modes, which buffers survive between phases.

The Mamba main program runs the SSM block as two SimbaCore invocations:

- **Phase 1** (`M1_PHASE1`): a "side" path that produces `conv_out` (= Phase
  2's `x`) and an IS-core output `xProj` (= Phase 2's `dt_in`/`BC`).
- **Phase 2** (`M2_PHASE2`): the main path. Inside one Phase-2 invocation,
  the OS-core, Switch-core, SU-core, and IS-core **all run in flight** with
  on-chip forwarding. The Phase-2 invocation reads `x` and `dt_in` from
  Phase 1's outputs.

The `R10_en` and `R11_en` flags in `start_simbacore_and_streamers` control
the on-chip forwarding wires that bypass TCDM:

- `R10_en` connects OS-core out → SU-core `z`.
- `R11_en` connects SU-core `y` → IS-core input A.

In Phase 2 both are enabled; SU-core sees streaming `z` from the OS-core
without a TCDM round-trip, and the IS-core sees streaming `y` from the
SU-core. Each invocation also has *start counters* (`R10_start_cnt`,
`R11_start_cnt`) controlling when the downstream core begins consuming.

## 4.1 `main` — non-tiled, two sequential phases (Phase 1 → Phase 2)

**Source**: [main/src/main.c](../../target/snitch_cluster/sw/apps/main/src/main.c),
[main/src/helper.c](../../target/snitch_cluster/sw/apps/main/src/helper.c).

Three entry points are defined: `test_phase1`, `test_phase2`, and
`test_phase1_and_2`. The fused one is what runs in `main`.

**TCDM layout (Phase 1 + Phase 2, packed in TCDM)**:

```
Phase 1 region:
  [ oscore_in (M1_addr_oscore_in)
  | oscore_weight_P1
  | conv_weight | conv_bias
  | conv_out  (M1_addr_conv_out)
  | iscore_weight_P1
  | iscore_out_P1 (psum buffer; BF16)  ]

Phase 2 region (immediately above iscore_out_P1):
  [ oscore_weight_P2
  | z (= OS-core output)
  | dt_in       == iscore_out_P1   (same TCDM address — REUSED)
  | BC          == dt_in + M2_dt_to_BC_offset
  | dt_weight_1 | dt_weight_2 | dt_bias
  | x           == conv_out         (same TCDM address — REUSED)
  | A | D | y
  | iscore_weight_P2
  | iscore_out_P2 ]
```

The two cross-phase shares are the heart of the design:

- **`dt_in == iscore_out_P1`**: the Phase 1 IS-core writes its FP8 `xProj`
  output into the psum buffer. Phase 2 reads it directly as Switch-core
  input (`R2` in P2).
- **`x == conv_out`**: the Phase 1 Switch-core writes `conv_out` (in
  ConvFormat); Phase 2 reads it as SU-core `x` (`R9`).

**Phase 1 DMA-in** (DM core, before P1 launch):

| Buffer            | L3 source            |
| ----------------- | -------------------- |
| `oscore_in`       | `M1_oscore_in`       |
| `oscore_weight`   | `M1_oscore_weight`   |
| `conv_weight`     | `M1_conv_weight`     |
| `conv_bias`       | `M1_conv_bias`       |
| `iscore_weight`   | `M1_iscore_weight`   |
| `iscore_out` (= bias preload) | `M1_iscore_bias` |

Note `iscore_out` is preloaded with the **bias** into the psum buffer. The
IS-core reads it as the tile-0 C input.

**Phase 1 streamer wiring** (`set_streamer_phase1` in helper.c):

| Port  | Buffer            | Role                                |
| ----- | ----------------- | ----------------------------------- |
| `R0`  | `oscore_in`       | OS-core input A                     |
| `R1`  | `oscore_weight`   | OS-core weight B                    |
| `R3`  | `conv_weight`     | Switch-core weight                  |
| `R4`  | `conv_bias`       | Switch-core bias                    |
| `R12` | `iscore_weight`   | IS-core weight                      |
| `R13` | `iscore_out`      | IS-core psum read (bias)            |
| `W1`  | `conv_out`        | Switch-core output                  |
| `W3`  | `iscore_out`      | IS-core psum write                  |
| `R2/R5..R11`, `W0/W2` | — | disabled                            |

`start_simbacore_and_streamers(M1_R10_en, 0, M1_R11_en, 0)`. The `set_simbacore_csr`
call uses `M1_PHASE1`, `seqLen, dModel, dInner, dtRank, xProjDim`.

**Phase 2 DMA-in** (between P1 and P2, optionally **overlapped** with P1
compute — the in-tree code intentionally drops the post-P1 barrier so the DM
core can DMA P2 weights while core 0 runs P1):

| Buffer                  | L3 source             |
| ----------------------- | --------------------- |
| `oscore_weight_P2`      | `M2_oscore_weight`    |
| `dt_weight_1`           | `M2_dt_weight_1`      |
| `dt_weight_2`           | `M2_dt_weight_2`      |
| `dt_bias`               | `M2_dt_bias`          |
| `A`                     | `M2_suc_A`            |
| `D`                     | `M2_suc_D`            |
| `iscore_weight_P2`      | `M2_iscore_weight`    |
| `iscore_out_P2` (bias)  | `M2_iscore_bias`      |

`x`, `dt_in`, `BC`, and `oscore_in` are **not** DMA'ed — `oscore_in` lives
through both phases, and the others are P1 outputs already in TCDM.

**Phase 2 streamer wiring** (`set_streamer_phase2`):

| Port  | Buffer              | Role                                  |
| ----- | ------------------- | ------------------------------------- |
| `R0`  | `oscore_in`         | OS-core input A                       |
| `R1`  | `oscore_weight`     | OS-core weight B                      |
| `R2`  | `dt_in`             | Switch-core input (xProj from P1)     |
| `R3`  | `dt_weight_1`       | Switch-core weight 1                  |
| `R4`  | `dt_bias`           | Switch-core bias                      |
| `R5`  | `dt_weight_2`       | Switch-core weight 2                  |
| `R6`  | `A`                 | SU-core A                             |
| `R7`  | `BC`                | SU-core BC                            |
| `R8`  | `D`                 | SU-core D                             |
| `R9`  | `x`                 | SU-core x (= `conv_out` from P1)      |
| `R10` | `z`                 | SU-core z (also OS-core out address)  |
| `R11` | `y`                 | IS-core input A (= SU-core y)         |
| `R12` | `iscore_weight`     | IS-core weight                        |
| `R13` | `iscore_out`        | IS-core psum read (bias)              |
| `W0`  | `z`                 | OS-core write to TCDM (also forwarded to SU-core via R10) |
| `W2`  | `y`                 | SU-core write to TCDM (also forwarded to IS-core via R11) |
| `W3`  | `iscore_out`        | IS-core psum write                    |

`start_simbacore_and_streamers(M2_R10_en, M2_R10_start_cnt, M2_R11_en, M2_R11_start_cnt)` —
both forwarding wires are on. `set_simbacore_csr` uses `M2_PHASE2,
seqLen, dModel, dInner, dtRank, dModel`.

## 4.2 `main-tiled` — dInner-tiled, double-buffered, fused P1 + P2

**Source**: [main-tiled/src/main.c](../../target/snitch_cluster/sw/apps/main-tiled/src/main.c).

**Tile axis**: `dInner`. The IS-core's K-axis-tiling property
(see [§2.2](02_iscore_kernels.md#22-isgemm-tiled--k-axis-tiled-accumulating))
carries over to both phases: the psum accumulator (`iscore_out_P{1,2}`) stays
**FULL** in TCDM across tiles, and R13/W3 hit the same address each tile.

**What is full vs ping-pong**:

| Phase | FULL (shared / once-loaded)                          | Ping-pong per tile                                                              |
| ----- | ---------------------------------------------------- | ------------------------------------------------------------------------------- |
| P1    | `oscore_in`, `iscore_out_P1`, `conv_out`             | `oscore_weight_P1`, `conv_weight`, `conv_bias`, `iscore_weight_P1`              |
| P2    | `oscore_in`, `iscore_out_P2`, `dt_in`, `x`, `z`, `y` | `oscore_weight_P2`, `dt_weight_1`, `dt_weight_2`, `dt_bias`, `A`, `D`, `iscore_weight_P2` |

`z` and `y` are FULL but written **with a per-tile slot** (i.e., per tile,
W0/W2 base pointers move by `M{2}_length_{z,y}_tile`).

**TCDM layout** (Phase 2's ping-pong region *overlays* Phase 1's, because P1
fully completes before P2 starts):

```
[ oscore_in
| iscore_out_P1 (== dt_in for P2)
| conv_out      (== x for P2)
| z (P2 W0 dest)
| y (P2 W2 dest)
| iscore_out_P2
| pingpong_base_ptr  →  P1 region (4 tiles × 2 buffers) → reused by P2 region (7 tiles × 2 buffers)
]
```

**Alignment**: several streamer ports (R0/R1/R12/R13/W3) need
**32-byte aligned** base addresses (sparse interconnect access granularity is
4 banks). When per-tile sizes are not multiples of 32 (e.g. `nb_tiles=8`
makes `conv_bias` 24 B per tile), the cascade misaligns later slots. The
code uses a `_ALIGN64(p)` macro for every ping-pong pointer to also match
the AXI DMA burst width.

**Phase 1 tile loop** (`nb_tiles + 1` iterations, 2-stage pipeline):

```
i ∈ [0, nb_tiles)     : DMA-in tile i (oscore_weight, conv_weight, conv_bias, iscore_weight)
i ∈ [1, nb_tiles + 1) : compute tile i-1
```

**Per-tile compute CSR rewrites**:

```c
write_csr(BASE_PTR_READER_1_LOW,  ptr_oscore_weight_P1[cbuf]);
write_csr(BASE_PTR_READER_3_LOW,  ptr_conv_weight[cbuf]);
write_csr(BASE_PTR_READER_4_LOW,  ptr_conv_bias[cbuf]);
write_csr(BASE_PTR_READER_12_LOW, ptr_iscore_weight_P1[cbuf]);
write_csr(BASE_PTR_WRITER_1_LOW,  ptr_conv_out + tile * M1_length_conv_out_tile);
write_csr(MODE, (tile == nb_tiles - 1) ? M1_PHASE1 : M28_PHASE1_NO_REQUANT);
```

The `MODE` flips between `M28_PHASE1_NO_REQUANT` (non-final: keep psum in
BF16, R13 next tile re-reads it) and `M1_PHASE1` (final: requant + transpose
to produce ConvFormat FP8 output).

**Between phases**: `iscore_out_P2` is preloaded with its bias. Then P2's
streamer is configured **once** with per-tile bounds (`M2_dInner_tile`).

**Phase 2 tile loop** (`nb_tiles + 1` iterations): same pipeline, more
per-tile base-pointer rewrites because more buffers are tiled:

```c
write_csr(BASE_PTR_READER_1_LOW,  ptr_oscore_weight_P2[cbuf]);
write_csr(BASE_PTR_READER_3_LOW,  ptr_dt_weight_1[cbuf]);
write_csr(BASE_PTR_READER_4_LOW,  ptr_dt_bias[cbuf]);
write_csr(BASE_PTR_READER_5_LOW,  ptr_dt_weight_2[cbuf]);
write_csr(BASE_PTR_READER_6_LOW,  ptr_A[cbuf]);
write_csr(BASE_PTR_READER_8_LOW,  ptr_D[cbuf]);
write_csr(BASE_PTR_READER_9_LOW,  x_tile_ptr);    // x + tile * length_x_tile
write_csr(BASE_PTR_READER_10_LOW, z_tile_ptr);    // z + tile * length_z_tile
write_csr(BASE_PTR_READER_11_LOW, y_tile_ptr);    // y + tile * length_y_tile
write_csr(BASE_PTR_READER_12_LOW, ptr_iscore_weight_P2[cbuf]);
write_csr(BASE_PTR_WRITER_0_LOW,  z_tile_ptr);
write_csr(BASE_PTR_WRITER_2_LOW,  y_tile_ptr);
write_csr(MODE, (tile == nb_tiles - 1) ? M2_PHASE2 : M29_PHASE2_NO_REQUANT);
```

The IS-core psum (R13/W3) base is the only one that does **not** move per
tile — it is the accumulator.

## 4.3 `main-full` — same compute as `main`, different data generator

`main-full` (no `src/`; only a `data/` and `Makefile`) reuses the `main`
compute path. It exists to drive a fuller / different parameter sweep through
the data generator. From a dataflow perspective it is identical to §4.1.

## 4.4 `suc` — SU-core stand-alone (debug / utilization probe)

**Source**: [suc/src/main.c](../../target/snitch_cluster/sw/apps/suc/src/main.c).

This program runs **only** the SU-core path of Phase 2, with the OS-core
output (`z`) pre-loaded directly from the golden reference. It exists to
probe the SU-core in isolation and to demonstrate the bank-conflict failure
mode on the `BC` port.

**TCDM allocation**: same Phase-2 layout as in `main`, with `z` pre-loaded
from `M2_oscore_expected` instead of being produced by the OS-core.

**DMA-in**: `z` (= the OS-core's golden output), `dt_in`, `dt_weight_1`,
`dt_weight_2`, `dt_bias`, `x`, `A`, `D`. The IS-core inputs (`iscore_weight`,
`iscore_out`) are **not** loaded — `R12/R13/W3` are disabled.

**Streamer wiring** (`set_streamer_suc_only`): same Phase-2 ports as in
`main`, but `R0/R1` (OS-core), `R12/R13` (IS-core inputs), `W0` (OS-core out),
and `W3` (IS-core out) are disabled. `R10` reads `z` directly from TCDM
instead of receiving it from the OS-core forwarding.

**Compute**: a single invocation with `M27_SUC_ONLY` and
`start_simbacore_and_streamers(M2_R10_en, 0, 0, 0)` (`R11_en = 0` because the
SU-core's output is not consumed downstream by an IS-core in this mode).

**Intentional bank conflict**: the program **forces an incorrect `BC` spatial
stride** (`ss_BC_test[] = {16}`) to demonstrate the bank-conflict throughput
hit. The result check is intentionally skipped — the test only verifies it
runs and prints the cycle count.

## 4.5 Recreate `main-tiled`

1. Compute TCDM offsets for the full region (oscore_in, iscore_out_P1,
   conv_out, z, y, iscore_out_P2) and a 64-byte-aligned ping-pong region.
2. Preload `oscore_in` and `iscore_out_P1` (bias). Barrier.
3. **Phase 1**:
   - `set_streamer_phase1` with tile-0 ping pointers.
   - `set_simbacore_csr(M1_PHASE1, seqLen, dModel, M1_dInner_tile, dtRank, xProjDim)`.
   - Loop `nb_tiles + 1`. Per iteration: DMA-in the 4 tiled P1 tensors; if
     `i ≥ 1`, rewrite 4 R base pointers + W1 base pointer + MODE
     (`M28_PHASE1_NO_REQUANT` or `M1_PHASE1` on the last tile), then fire and
     wait.
4. Preload `iscore_out_P2` (bias). Barrier.
5. **Phase 2**:
   - `set_streamer_phase2` with tile-0 ping pointers.
   - `set_simbacore_csr(M2_PHASE2, seqLen, dModel, M2_dInner_tile, dtRank, dModel)`.
   - Loop `nb_tiles + 1`. Per iteration: DMA-in the 7 tiled P2 tensors; if
     `i ≥ 1`, rewrite ~12 base pointers (including the moving z/y per-tile
     slots) + MODE (`M29_PHASE2_NO_REQUANT` or `M2_PHASE2` on the last tile),
     then fire and wait with both forwarding flags on.
6. Verify `z`, `y`, `iscore_out_P2` against golden.
