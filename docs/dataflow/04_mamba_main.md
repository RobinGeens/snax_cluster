# 4. Mamba main: `main`, `main-tiled`, `main-full`, `suc`

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

| Lifecycle               | Phase 1 tensors                                       | Phase 2 tensors                                                                                              |
| ----------------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **shared / FULL**       | `oscore_in`, `iscore_out` (psum accumulator)          | `oscore_in`, `dt_in` (= P1 `iscore_out`), `x` (= P1 `conv_out`), `iscore_out` (psum accumulator)             |
| **FULL with per-tile slot** | `conv_out` (W1 writes a different slice each tile) | `z`, `y` (W0/W2 write a different slice each tile)                                                            |
| **tiled, ping-pong**    | `oscore_weight`, `conv_weight`, `conv_bias`, `iscore_weight` | `oscore_weight`, `dt_weight_1`, `dt_weight_2`, `dt_bias`, SU-core `A` and `D`, `iscore_weight`           |

## `suc`

Stand-alone SU-core run, used to probe the SU-core in isolation. The OS-core
output `z` is preloaded from the golden reference instead of being produced
upstream; the IS-core stage is disabled. Used to demonstrate the bank-
conflict throughput hit when the `BC` operand is given an incorrect spatial
stride.
