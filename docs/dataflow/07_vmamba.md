# 7. VMamba SS2D: `vmamba`, `vmamba-tiled`

> Byte layouts of every Phase 1 / Phase 2 buffer:
> [memory_layouts/07 — per-mode reference](../../../chisel-ssm/docs/memory_layouts/07_mode_reference.md),
> [03 — ConvFormat](../../../chisel-ssm/docs/memory_layouts/03_conv_format.md),
> [05 — xProj format](../../../chisel-ssm/docs/memory_layouts/05_xproj_format.md),
> [06 — delta-weight split](../../../chisel-ssm/docs/memory_layouts/06_delta_weight.md).
>
> VMamba-specific dataflow reference:
> [chisel-ssm/docs/vmamba_dataflow.md](../../../chisel-ssm/docs/vmamba_dataflow.md).

VMamba's SS2D (Selective State Space for 2D) applies the Mamba selective scan
along K=4 cross-scan directions over a 2D spatial grid (H, W) where
`seqLen = H * W`.  The full SS2D forward pass is:

```
input (L, dModel)
  │
  ├── Phase 1: osCore(in_proj) → conv → IsCore(x_proj)
  │     produces conv_out (x) and xProj (dt, B, C)
  │
  ├── software: cross-scan x, z into K=4 directional sequences
  │
  ├── Phase 2 (× K directions):
  │     osCore → z_k, SwitchCore(dt_proj) → delta_k,
  │     SUC(x_k, z_k, delta_k, A, B_k, C_k, D) → y_k,
  │     IsCore(out_proj) → output_k
  │
  ├── software: cross-merge y_0..y_3 → y
  │
  └── RMSNorm + final output projection
```

Both `vmamba` and `vmamba-tiled` implement the **base case**: a single Phase 1
→ Phase 2 pass (one direction, no cross-scan/merge).  This validates the
hardware path at VMamba-sized parameters (`seqLen = H*W`).

## Relationship to the `main` / `main-tiled` programs

The hardware execution is the same two-phase split:

- **Phase 1** (PHASE1 mode): osCore (in-projection) → Switch-core (conv1d +
  SiLU) → IS-core (x_proj). Produces `conv_out` and `iscore_out` (= `dt+B+C`).
- **Phase 2** (PHASE2 mode): osCore (z-projection) → Switch-core (dt_proj) →
  SU-core (selective scan) → IS-core (out_proj). All four cores run
  concurrently with on-chip forwarding.

**Cross-phase buffer sharing** is the same: Phase 1's `conv_out` is Phase 2's
SU-core `x`, and Phase 1's `iscore_out` is Phase 2's Switch-core `dt_in`.
No re-DMA between phases.

The difference from `main` / `main-tiled` is in the data generator:
`DataGeneratorVMamba` (chisel-ssm) generates the data, and the algorithmic
parameters include `H`, `W`, `K` alongside the standard `seqLen`, `dModel`,
`dInner`, `dtRank`.

## Tensors

### Phase 1

| Tensor          | Role                                     | Origin      |
| --------------- | ---------------------------------------- | ----------- |
| `oscore_in`     | OS-core input A (= model input)          | DMA from L3 |
| `oscore_weight` | OS-core weight B (in-projection x-branch)| DMA from L3 |
| `conv_weight`   | Switch-core weight (conv1d kernel)       | DMA from L3 |
| `conv_bias`     | Switch-core bias                         | DMA from L3 |
| `iscore_weight` | IS-core weight (x_proj)                  | DMA from L3 |
| `iscore_bias`   | IS-core bias (loaded into psum slot)     | DMA from L3 |
| `conv_out`      | Switch-core output (= P2 SU-core `x`)   | produced in P1 |
| `iscore_out`    | IS-core output `xProj` (= P2 `dt_in`)   | produced in P1 |

### Phase 2

| Tensor           | Role                                  | Origin                              |
| ---------------- | ------------------------------------- | ----------------------------------- |
| `oscore_in`      | OS-core input A                       | **shared from P1** (no re-DMA)      |
| `oscore_weight`  | OS-core weight B (z-branch projection)| DMA from L3                         |
| `dt_in` / `BC`   | Switch-core input (`xProj`)           | **shared from P1** (= `iscore_out`) |
| `dt_weight_1/2`  | Switch-core weights (split)           | DMA from L3                         |
| `dt_bias`        | Switch-core bias                      | DMA from L3                         |
| `x`              | SU-core input                         | **shared from P1** (= `conv_out`)   |
| `A`, `D`         | SU-core SSM parameters                | DMA from L3                         |
| `iscore_weight`  | IS-core weight (out-projection)       | DMA from L3                         |
| `iscore_bias`    | IS-core bias                          | DMA from L3                         |
| `z`              | OS-core output → SU-core              | produced in P2                      |
| `y`              | SU-core output → IS-core              | produced in P2                      |
| `iscore_out`     | IS-core final output                  | produced in P2                      |

## `vmamba`

Untiled Phase 1 → Phase 2.  All tensors stay FULL in TCDM. While Phase 1
computes, the DM core overlaps the DMA-in of Phase 2's tensors. Structurally
identical to `main`.

## `vmamba-tiled`

Both phases are **dInner-tiled**, following the same strategy as `main-tiled`
(see [04_mamba_main.md §main-tiled](04_mamba_main.md#main-tiled)). All tiling
tricks, buffer reuse, and pipeline staging from `main-tiled` apply unchanged:

| Lifecycle                       | Phase 1 tensors                                          | Phase 2 tensors                                                                                |
| ------------------------------- | -------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **shared / FULL**               | `oscore_in`, `iscore_out_P1_psum`, `iscore_out_P1_final` | `oscore_in`, `dt_in` (= P1 `iscore_out_P1_final`), `iscore_out_P2`                            |
| **tile-sized in TCDM, FULL in L3** | `conv_out`                                            | `x` (= P1 `conv_out`), `z`, `y`                                                               |
| **tiled, ping-pong**            | `oscore_weight`, `conv_weight`, `conv_bias`, `iscore_weight` | `oscore_weight`, `dt_weight_1`, `dt_weight_2`, `dt_bias`, SU-core `A` and `D`, `iscore_weight` |

### Memory-saving tricks (inherited from `main-tiled`)

1. **`conv_out` (= P2 `x`) staged through L3.** TCDM holds a 2-slot
   ping-pong; the FULL tensor lives in L3. P1 spills each tile to L3 two
   iterations after it is written; P2 DMA-ins `x_tile[i]` from L3 alongside
   loading weight tile `i`.
2. **`z` and `y` staged through L3.** Tile-sized ping-pong in TCDM (W0/W2
   write, R10/R11 read within the same P2 kernel call), then DMA'd out to L3.
3. **`iscore_out_P1_psum` overlays `iscore_out_P2`.** The P1 psum buffer is
   dead after P1's final K-tile; `iscore_out_P2` is only used in P2. They
   share one address.
4. **`iscore_out_P1_final` is a separate FULL buffer.** Required by the
   K-tile + TRANSPOSE-final fix: the final tile's transposed W3 writes
   element (m,n) to byte F(n,m) and would otherwise clobber partial psums.

### 3-stage pipeline

Both phases run a `nb_tiles + 2` iteration loop:

- **Iter i** (DM core): DMA-in tile `i` weights into ping-pong slot `i%2`.
- **Iter i** (compute core 0): run kernel(s) for tile `i−1`.
- **Iter i** (DM core, after barrier): DMA-spill tile `i−2` output to L3.

### Phase 1 kernel variants

- **Bulk** (tiles 0 .. nb_tiles−2): K_i K-steps in `PHASE1_NO_REQUANT`.
  Accumulates psum in-place.
- **FinalLead** (final tile, only if K_i > 1): K_i−1 K-steps in
  `PHASE1_NO_REQUANT`. Continues accumulating psum.
- **FinalStep** (absolute last K-step): 1 K-step in `PHASE1` (requant +
  transpose). Reads psum from `iscore_out_P1_psum`, writes transposed result
  to `iscore_out_P1_final`.

### Phase 2 kernel

One kernel per DMA tile (K_i K-steps each). Non-final tiles use
`PHASE2_NO_REQUANT`; the final tile uses `PHASE2` (HW gates requant to the
absolute-final K-step via `isCoreOutIsFinal`). Only the MODE CSR changes
between tiles; base pointers are updated via `write_csr`.

### Tested configurations

| H  | W  | dModel | dInner | nb_tiles | L1 usage   | Result       |
|----|-----|--------|--------|----------|------------|--------------|
|  4 |  4  |   48   |   96   |    2     |  ~30 KiB   | PASS 0/125   |
| 14 | 16  |  192   |  384   |   16     | 246 KiB    | 12/125 (FP8 rounding) |

The 12 errors at L=224 D=384 are all off-by-one or off-by-two in the FP8
quantization level — expected numerical noise from the long selective scan
(L=224 timesteps) and wide IS-core reduction (D=384), not correctness bugs.
Phase 1 outputs are exact at all tested sizes.

## VMamba-specific extensions (not yet implemented)

The full SS2D requires these additions on top of the base Phase 1 + Phase 2:

1. **Conv2d instead of conv1d.** The VMamba block uses a 2D depthwise
   convolution (kH × kW kernel, same-padding) on the x-branch, not the 1D
   conv with `dConv=4` that the Switch-core implements. A software fallback or
   new hardware mode is needed.

2. **K-direction loop.** After Phase 1 produces `conv_out` and `z` (or their
   inputs), cross-scan permutes them into K=4 directional sequences. Phase 2
   then runs K times — once per direction — each time loading the appropriate
   cross-scanned slice of `x` and `z`. The SwitchCore weights, `A`, and `D`
   are shared across directions; `dt_in` (`dt+B+C` from x_proj) may need to
   be per-direction if x_proj weights differ across directions.

3. **Cross-merge.** After all K selective scans, the per-direction outputs
   `y_0..y_3` are inverse-permuted and summed in software to produce the
   merged `y` of shape `(L, dInner)`.

4. **RMSNorm + output projection.** A SIMD RMSNorm on the merged `y`, followed
   by an IS-core output projection `y @ W_out → (L, dModel)`.
