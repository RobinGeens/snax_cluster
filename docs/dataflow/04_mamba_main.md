# 4. Mamba main: `main`, `main-tiled`, `main-full`, `suc-only`

> **All pages:**
> [README](README.md) ·
> [1. OS-core kernels](01_oscore_kernels.md) ·
> [2. IS-core kernels](02_iscore_kernels.md) ·
> [3. SIMD / RMSNorm kernels](03_simd_kernels.md) ·
> **4. Mamba main (this page)** ·
> [5. FFT family](05_fft.md) ·
> [6. EinFFT MLP](06_einfft_mlp.md) ·
> [7. VMamba SS2D](07_vmamba.md) ·
> [8. Performance optimization](08_performance_optimization.md)

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

| Lifecycle                          | Phase 1 tensors                                              | Phase 2 tensors                                                                                |
| -------------------------------    | ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| **shared / FULL**                  | `oscore_in`, `iscore_out_P1`                                 | `oscore_in`, `dt_in` (= P1 `iscore_out_P1`), `iscore_out_P2`                                   |
| **tile-sized in TCDM, FULL in L3** | `conv_out`                                                   | `x` (= P1 `conv_out`), `z`, `y`                                                                |
| **tiled, ping-pong**               | `oscore_weight`, `conv_weight`, `conv_bias`, `iscore_weight` | `oscore_weight`, `dt_weight_1`, `dt_weight_2`, `dt_bias`, SU-core `A` and `D`, `iscore_weight` |

### Memory-saving tricks

- **`x` (= `conv_out`) staged through L3.** TCDM holds a 2-slot
  ping-pong; the FULL tensor lives in L3. P1 spills each `conv_out` tile to
  L3 two iterations after it is written (3-stage pipeline: weight-in / kernel
  / spill); P2 DMA-ins `x_tile[i]` from L3 alongside loading weight tile `i`.
- **`z` and `y` staged through L3.** Tile-sized ping-pong in TCDM (W0/W2
  write, R10/R11 read within the same P2 kernel call), then DMA'd out to L3
  after the kernel. Verification reads from L3.

**Why `iscore_out_P1` uses a single buffer (no psum/final split).**
The BankTransposer is gated on `isCoreOutIsFinal`
([MambaCore.scala:370](../../../chisel-ssm/src/main/scala/mambacore/MambaCore.scala#L370)):
intermediate K-steps accumulate in standard layout, and the transposer
only fires on the hardware's internal final K-step. This is the same
mechanism the untiled `main` program relies on. Within a single
`M1_PHASE1` invocation with K_i K-steps, the first K_i-1 steps
write standard-layout psums via W3 and the last step writes the
transposed+requantized output — so R13 and W3 can safely share one
buffer.

**Kernel-call layout.** `datagen.py` emits two streamer bound
configurations (one per phase):

| Config | K-steps per kernel | Mode | W3 destination | Used for |
|---|---|---|---|---|
| **P1 tile** (`M1_R*_tb`) | `K_i` | `M28_PHASE1_NO_REQUANT` / `M1_PHASE1` | `iscore_out_P1` | every P1 tile (mode flips for the final tile) |
| **P2 tile** (`M2_R*_tb`) | `K_i` | `M29_PHASE2_NO_REQUANT` / `M2_PHASE2` | `iscore_out_P2` | every P2 tile (mode flips for the final tile) |

Total kernel calls per phase: P1 = `nb_tiles`, P2 = `nb_tiles`.
Both phases follow the same pattern: non-final tiles run in NO_REQUANT
mode, the final tile writes `MODE = M*_PHASE*` before starting. HW's
`isCoreOutIsFinal` gate applies requant (and bank-transpose in P1) to
just the last K-step of the final tile's invocation.

`nb_tiles` is a free choice (must divide `dInner / dInnerUnroll`), trading off
DMA-tile size against kernel-launch overhead.

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

## Enabling large seqLen

`main-tiled` cannot be expanded to large seqLen because it assumes `iscore_out_P1` (size `L*xProjDim`) and `iscore_out_P2` (size `L*dModel`) remain in TCDM.

To mitigate this, we have 3 options:

1. Split P1 and P2 in half over seqLen. However, this is completely impossible because it requires the intermediate SSM 
states to be read out and restored during the second half. P1 and P2 must always be fully completed.

2. Tile the IS-core output dimension (xProjDim). However, since this tensor is a BF16 psum, we need to load all 
xProjDim-tiles in and out for every psum accumulation iteration (K in total). The full tensor needs to be transferred 2*K times.
Since xProjDim is small, it should be possible to fully hide this latency by overlapping it with compute.
A second complication is that in `main-tiled`, IS-core bank-transposes its output (via the BankTransposer on the final K-step).
If we tile in xProjDim, the bank-transpose still works within each tile, but the downstream consumer must
handle per-tile transposed chunks. Implemented in `main-tiled-dtRank`

3. Tile the whole P1 in seqLen. This simply means we do the full P1 kernel call for every seqLen tile. The downside is that
we loose the weight-reuse on the projection weight matrices, and we need to load in the full weight matrices for every seqLen tile.
This costs extra energy, but the latency should be manageable by overlapping it with compute. The same bank-transpose complication applies here.
To be implemented in `main-tiled-seqLen`.


## `main-tiled-dtRank`

Tiles `iscore_out_P1` in xProjDim.

## `main-tiled-seqLen`

Not implemented yet

