# 7. VMamba SS2D: `vmamba`

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

The current `vmamba` program implements the **base case**: a single Phase 1 →
Phase 2 pass without the K-direction loop or cross-scan/merge. This is
structurally identical to the `main` program and validates the hardware path
at VMamba-sized parameters (`seqLen = H*W`).

## Relationship to the `main` program

The hardware execution is the same two-phase split as `main`:

- **Phase 1** (PHASE1 mode): osCore (in-projection) → Switch-core (conv1d +
  SiLU) → IS-core (x_proj). Produces `conv_out` and `iscore_out` (= `dt+B+C`).
- **Phase 2** (PHASE2 mode): osCore (z-projection) → Switch-core (dt_proj) →
  SU-core (selective scan) → IS-core (out_proj). All four cores run
  concurrently with on-chip forwarding.

**Cross-phase buffer sharing** is the same: Phase 1's `conv_out` is Phase 2's
SU-core `x`, and Phase 1's `iscore_out` is Phase 2's Switch-core `dt_in`.
No re-DMA between phases.

The difference from `main` is in the data generator: `DataGeneratorVMamba`
(chisel-ssm) generates the data, and the algorithmic parameters include `H`,
`W`, `K` alongside the standard `seqLen`, `dModel`, `dInner`, `dtRank`.

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
