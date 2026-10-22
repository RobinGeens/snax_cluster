# 7. VMamba SS2D: `vmamba`, `vmamba-tiled`

> **All pages:**
> [README](README.md) ·
> [1. OS-core kernels](01_oscore_kernels.md) ·
> [2. IS-core kernels](02_iscore_kernels.md) ·
> [3. SIMD / RMSNorm kernels](03_simd_kernels.md) ·
> [4. Mamba main](04_mamba_main.md) ·
> [5. FFT family](05_fft.md) ·
> [6. EinFFT MLP](06_einfft_mlp.md) ·
> **7. VMamba SS2D (this page)** ·
> [8. Performance optimization](08_performance_optimization.md) ·
> [9. Async tiling](09_async_tiling.md) ·
> [12. SUC async](12_suc_async.md) ·
> [14. RMSNorm tiled](14_rmsnorm_tiled.md) ·
> [20. Bank-conflict-free double GEMM](20_double_gemm_conflict_free.md) ·
> [21. Conv downsample (im2col GEMM)](21_conv_downsample.md)

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
`seqLen = H * W`.

## Full SS2D pipeline (`vmamba`)

The `vmamba` program runs the complete SS2D forward pass on hardware:

```
cross_scan(input) → K=4 directional sequences
  │
  ├── Per direction k (K=4 times):
  │     Phase 1 (PHASE1): osCore(in_proj) → SwitchCore(conv1d+SiLU) → IS-core(x_proj)
  │       produces conv_out (= SUC x) and iscore_out (= dt+B+C in xProj format)
  │     Phase 2 (PHASE2): osCore(z) → SwitchCore(dt_proj) → SUC → IS-core(out_proj)
  │       produces z, y_k (SUC out), iscore_out_k (per-direction output)
  │
  ├── Cross-merge (SIMD ADD): sum inverse-permuted y_0..y_3 → y_merged
  │
  └── RMSNorm (7-step SIMD chain): widen → RMS → ÷D → 1/√ → ×(1/√) → ×weight → narrow
```

Each direction runs the **exact same Phase 1 + Phase 2 as the `main` program**
(using `helper.c` for Phase 1 streamers and `set_streamer_phase2` for Phase 2).
The only per-direction differences are:
- `oscore_in`: cross-scanned input for direction k
- `iscore_weight`: per-direction x_proj weight W_xproj[k]

### Hardware stages

| Stage | HW mode | Cores used | Per-dir? | Inputs | Outputs |
|-------|---------|-----------|----------|--------|---------|
| Phase 1 | PHASE1 | osCore + SwitchCore + IS-core | × K | cross_scan(input), W_x, conv_w, W_xproj[k] | conv_out, dt+B+C |
| Phase 2 | PHASE2 | osCore + SwitchCore + SUC + IS-core | × K | input, W_z, dt+B+C, W_dt, A, D, W_out | z, y_k, iscore_out_k |
| Cross-merge | SIMD_ADD_FP8 | SIMD | × 3 passes | y_invperm[0..3] | y_merged |
| RMSNorm | 8 SIMD modes | SIMD | × 1 | y_merged, norm_weight | y_norm |

### Cross-phase buffer sharing

Same as `main`: Phase 1's `conv_out` is Phase 2's SU-core `x`, and Phase 1's
`iscore_out` is Phase 2's Switch-core `dt_in`. No re-DMA between Phase 1 and
Phase 2 within a direction.

### Cross-scan

The raw input is cross-scanned into K=4 directional sequences before the
per-direction loop. The cross-scan permutations (identity, transpose, reverse,
reverse-transpose) are pre-computed in the data generator and loaded from L3.

### Cross-merge

After all K directions, the per-direction SUC outputs are inverse-permuted
(undoing the cross-scan) and summed via 3 SIMD ADD (FP8 element-wise add)
passes. The inverse-permuted y_k are pre-computed in the data generator.

### RMSNorm

A 7-step SIMD chain on the merged y (adapted from the `rmsnorm` program):

1. **Widen** FP8 → BF16 (SIMD_NOOP_FP8_REQUANT)
2. **RMS** Σ(x²) per token (SIMD_RMS_BF16)
3. **÷ D** multiply by 1/dInner (SIMD_MUL_BF16)
4. **1/√** inverse square root (SIMD_RSQRT_BF16)
5. **Normalize** x × (1/√) (SIMD_MUL_BF16)
6. **Scale** × weight (SIMD_MUL_BF16)
7. **Narrow** BF16 → FP8 (SIMD_NOOP_BF16_REQUANT)

### Tensors

#### Phase 1 (per direction)

| Tensor          | Role                                     | Origin      |
| --------------- | ---------------------------------------- | ----------- |
| `oscore_in`     | OS-core input (cross-scanned for dir k)  | DMA from L3 (per-dir) |
| `oscore_weight` | OS-core weight (in-projection x-branch)  | DMA from L3 (shared) |
| `conv_weight`   | Switch-core weight (conv1d kernel)       | DMA from L3 (shared) |
| `conv_bias`     | Switch-core bias                         | DMA from L3 (shared) |
| `iscore_weight` | IS-core weight (x_proj, per-direction)   | DMA from L3 (per-dir) |
| `iscore_bias`   | IS-core bias (loaded into psum slot)     | DMA from L3 (shared) |
| `conv_out`      | Switch-core output (= P2 SU-core `x`)   | produced in P1 |
| `iscore_out`    | IS-core output `xProj` (= P2 `dt_in`)   | produced in P1 |

#### Phase 2 (per direction)

| Tensor           | Role                                  | Origin                              |
| ---------------- | ------------------------------------- | ----------------------------------- |
| `oscore_in`      | OS-core input (same as P1)            | **shared from P1** (no re-DMA)      |
| `oscore_weight`  | OS-core weight (z-branch projection)  | DMA from L3 (shared)                |
| `dt_in` / `BC`   | Switch-core input (`xProj`)           | **shared from P1** (= `iscore_out`) |
| `dt_weight_1/2`  | Switch-core weights (split)           | DMA from L3 (shared)                |
| `dt_bias`        | Switch-core bias                      | DMA from L3 (shared)                |
| `x`              | SU-core input                         | **shared from P1** (= `conv_out`)   |
| `A`, `D`         | SU-core SSM parameters                | DMA from L3 (shared)                |
| `iscore_weight`  | IS-core weight (out-projection)       | DMA from L3 (shared)                |
| `iscore_bias`    | IS-core bias                          | DMA from L3 (shared)                |
| `z`              | OS-core output → SU-core              | produced in P2                      |
| `y`              | SU-core output → IS-core              | produced in P2                      |
| `iscore_out`     | IS-core final output                  | produced in P2                      |

### Tested configurations

| H | W | dModel | L | D | Binary | L1 | P1 (4 dirs) | P2 (4 dirs) | Post-merge errors | Status |
|---|---|--------|---|---|--------|-----|-------------|-------------|-------------------|--------|
| 4 | 4 | 48 | 16 | 96 | 117 KiB | 53 KiB | 3,164 cc | 12,440 cc | 34/50 (FP8) | PASS |
| 4 | 8 | 48 | 32 | 96 | 129 KiB | 69 KiB | 5,784 cc | 24,328 cc | 35/50 (FP8) | PASS |
| 8 | 8 | 48 | 64 | 96 | 153 KiB | 101 KiB | 11,016 cc | 48,076 cc | 32/50 (FP8) | PASS |
| 4 | 4 | 96 | 16 | 192 | 234 KiB | 125 KiB | 6,512 cc | 24,616 cc | 36/50 (FP8) | PASS |
| 4 | 8 | 96 | 32 | 192 | 254 KiB | 151 KiB | — | — | — | TIMEOUT |

Per-direction Phase 1 + Phase 2 checks (z, SUC y, iscore_out) are exact (0
errors) at all sizes. Post-merge errors are from cross-merge FP8 summation (4
directions) + 8-step SIMD RMSNorm chain — inherent to FP8 arithmetic.

Cross-merge and RMSNorm take 6 cc and 16 cc respectively (trivial vs. the
Phase 1 + Phase 2 compute).

### Scaling limits (untiled)

The untiled `vmamba` binary embeds per-direction x_proj weights
(`iscore_weight_K` = K × dInner × xProjDim bytes) which dominate the ELF
size. At dModel=96 (xProjDim=152, dInner=192), the per-direction weights
alone are K × 192 × 152 = 117 KiB. The practical L3 limit is ~256 KiB,
restricting the untiled version to L ≤ 64 at dModel=48 or L ≤ 16 at
dModel=96.

### Cross-scan implementation

Cross-scan is done in the Scala data generator: `VMambaLib.crossScan` builds
the K=4 directional sequences and `DataGeneratorVMamba` writes them K-stacked
as `oscore_in_K`. Both `vmamba` and `vmamba-tiled` DMA the pre-flattened
per-direction slice (`M2_oscore_in_K + k * dir_size_oscore_in`) from L3 into
TCDM at the top of each direction; there is no scalar reorder on the Snitch
compute core.

For a real multi-layer deployment, the previous layer's output in L3 would be
DMA'd into TCDM per direction, with the cross-scan permutation folded into
the DMA source addressing (tile-level stride changes). No K copies would
exist at any point — just one L3 output buffer and one TCDM input buffer
reused per direction.

## `vmamba-tiled`

Both phases are **dInner-tiled** following `main-tiled`, run inside the full K=4
per-direction loop, followed by cross-merge (SIMD ADD) and the 8-step RMSNorm
chain — the same end-to-end SS2D pipeline as `vmamba`, with each direction's
Phase 1 and Phase 2 tiled over dInner. See
[04_mamba_main.md §main-tiled](04_mamba_main.md#main-tiled) for tiling details.

### Tested configurations

| H  | W  | dModel | dInner | nb_tiles | L1 usage | Result |
|----|-----|--------|--------|----------|----------|--------|
|  4 |  4  |   48   |   96   |    2     | ~30 KiB  | PASS 0/125 |
| 14 | 16  |  192   |  384   |   16     | 246 KiB  | 12/125 (FP8 rounding) |

## Future work

1. **Conv2d → conv1d replacement.** The user will replace the 2D depthwise
   conv with conv1d (same as SiMBA) so the Switch-core can run it natively.
   Currently conv1d is used in the reference model.

2. **DMA-based cross-scan.** Replace the scalar `cross_scan_flattena()` with
   DMA 2D strided transfers that implement the tile-level permutation during
   the L3 → TCDM copy. Requires per-tile source address computation based on
   the (H, W) grid and cross-scan direction.

3. **vmamba-tiled per-direction.** Extending the tiled version to run K=4
   directions for large workloads (L=224+, D=384+).
