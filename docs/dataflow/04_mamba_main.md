# 4. Mamba main: `main`, `main-tiled`, `main-full`, `suc-only`

> Byte layouts of every Phase 1 / Phase 2 buffer:
> [memory_layouts/07 — per-mode reference](../../../chisel-ssm/docs/memory_layouts/07_mode_reference.md),
> [04 — SUCFormat](../../../chisel-ssm/docs/memory_layouts/04_suc_format.md),
> [05 — xProj format](../../../chisel-ssm/docs/memory_layouts/05_xproj_format.md),
> [06 — delta-weight split](../../../chisel-ssm/docs/memory_layouts/06_delta_weight.md).

The Mamba block runs as two SimbaCore launches:

- **Phase 1**: a "side" path that produces `conv_out` and an IS-core output
  `xProj` (the `(dt, B, C)` projection).
- **Phase 2**: the main path. Inside one Phase-2 launch, the OS-core, Switch-
  core, SU-core, and IS-core all run concurrently with on-chip forwarding:
  OS-core out streams directly into the SU-core, and SU-core out streams
  directly into the IS-core. Neither hop round-trips through TCDM.

**Cross-phase buffer sharing.** Phase 1's outputs are Phase 2's inputs:

- Phase 1 IS-core out (`xProj`) is consumed as Phase 2 Switch-core input
  (`dt_in`).
- Phase 1 Switch-core out (`conv_out`) is consumed as Phase 2 SU-core input
  (`x`).

No re-DMA between phases; the buffers stay at the same TCDM address.

## Tensors

### Phase 1

| Tensor          | Role                                     | Origin             |
| --------------- | ---------------------------------------- | ------------------ |
| `oscore_in`     | OS-core input A                          | DMA from L3        |
| `oscore_weight` | OS-core weight B                         | DMA from L3        |
| `conv_weight`   | Switch-core weight                       | DMA from L3        |
| `conv_bias`     | Switch-core bias                         | DMA from L3        |
| `iscore_weight` | IS-core weight                           | DMA from L3        |
| `iscore_bias`   | IS-core bias (loaded into the psum slot) | DMA from L3        |
| `conv_out`      | Switch-core output                       | produced in P1     |
| `iscore_out`    | IS-core output `xProj`                   | produced in P1     |

### Phase 2

| Tensor              | Role                                  | Origin                                |
| ------------------- | ------------------------------------- | ------------------------------------- |
| `oscore_in`         | OS-core input A                       | **shared from P1** (no re-DMA)        |
| `oscore_weight`     | OS-core weight B                      | DMA from L3                           |
| `dt_in` / `BC`      | Switch-core input (`xProj`)           | **shared from P1** (= `iscore_out`)   |
| `dt_weight_1/2`     | Switch-core weights                   | DMA from L3                           |
| `dt_bias`           | Switch-core bias                      | DMA from L3                           |
| `x`                 | SU-core input                         | **shared from P1** (= `conv_out`)     |
| `A`, `D`            | SU-core state matrices                | DMA from L3                           |
| `iscore_weight`     | IS-core weight                        | DMA from L3                           |
| `iscore_bias`       | IS-core bias                          | DMA from L3                           |
| `z`                 | OS-core output (forwarded to SU-core) | produced in P2                        |
| `y`                 | SU-core output (forwarded to IS-core) | produced in P2                        |
| `iscore_out`        | IS-core final output                  | produced in P2                        |

## `main` and `main-full`

Both run Phase 1 → Phase 2 untiled. They differ only in the parameter set
their data generator produces. While Phase 1 computes, the DM core
overlaps the DMA-in of Phase 2's tensors.

## `main-tiled`

Both phases are **dInner-tiled**. The IS-core psum buffers stay FULL in
TCDM and accumulate across tiles in place (non-final tiles in no-requant
mode, final tile applies the requant — same trick as `isgemm-tiled`).
Phase 1 completes fully, then Phase 2 starts. Phase 2 reuses Phase 1's
ping-pong region for its own ping-pong, since Phase 1 is done with it.

| Lifecycle                       | Phase 1 tensors                                              | Phase 2 tensors                                                                                |
| ------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| **shared / FULL**               | `oscore_in`, `iscore_out_P1_psum`, `iscore_out_P1_final`     | `oscore_in`, `dt_in` (= P1 `iscore_out_P1_final`), `iscore_out_P2`                             |
| **tile-sized in TCDM, FULL in L3** | `conv_out`                                                | `x` (= P1 `conv_out`), `z`, `y`                                                                |
| **tiled, ping-pong**            | `oscore_weight`, `conv_weight`, `conv_bias`, `iscore_weight` | `oscore_weight`, `dt_weight_1`, `dt_weight_2`, `dt_bias`, SU-core `A` and `D`, `iscore_weight` |

### Memory-saving tricks

- **`x` (= `conv_out`) staged through L3.** TCDM holds a 2-slot
  ping-pong; the FULL tensor lives in L3. P1 spills each `conv_out` tile to
  L3 two iterations after it is written (3-stage pipeline: weight-in / kernel
  / spill); P2 DMA-ins `x_tile[i]` from L3 alongside loading weight tile `i`.
- **`z` and `y` staged through L3.** Tile-sized ping-pong in TCDM (W0/W2
  write, R10/R11 read within the same P2 kernel call), then DMA'd out to L3
  after the kernel. Verification reads from L3.
- **`iscore_out_P1_psum` overlays `iscore_out_P2`.** The P1 psum buffer is
  dead after P1's final K-tile (R13/W3 switch to `iscore_out_P1_final` on the
  last tile); `iscore_out_P2` is only used in P2. They never coexist, so they
  share one address. Saves `min(length_iscore_P1, length_iscore_P2)` bytes.
- **`iscore_out_P1_final` is a separate FULL buffer.** Required by the
  K-tile + TRANSPOSE-final fix (see `main_tiled_p1_split_buffer_fix` memory).
  `iscore_out_P1_psum` holds the standard-layout BF16 psum accumulating
  across K (R13 reads, W3 writes on non-final tiles, no-requant mode).
  `iscore_out_P1_final` receives the final tile's W3 in TRANSPOSE mode and
  holds the requantized output in the bank-transposed layout that P2's R2/R7
  `dt`+`BC` readers expect. Without the split, the final tile's transposed
  W3 writes element (m,n) to byte F(n,m) and would otherwise clobber
  `psum_partial(n,m)` before IS-core reads it.

**Required parameter constraint.** `nb_tiles` MUST equal
`dInner / dInnerUnroll` so K\_i=1 per kernel. With K\_i>1, the final kernel
needs intra-kernel W3→R13 feedback that the split-W3 buffer breaks.

### Possible further tricks (not yet implemented)

- L-tile `iscore_out_P2` by splitting P2 into a new "PHASE2\_NO\_ISCORE" mode
  (chisel-ssm change) + IS-core-only kernel calls. Frees the `L*dModel*2`
  buffer.

## `suc-only`

Stand-alone SU-core run, used to probe the SU-core in isolation. The OS-core
output `z` is preloaded from the golden reference instead of being produced
upstream; the IS-core stage is disabled. Used to demonstrate the bank-
conflict throughput hit when the `BC` operand is given an incorrect spatial
stride.
