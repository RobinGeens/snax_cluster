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
> [8. Performance optimization](08_performance_optimization.md) ·
> [9. Async tiling](09_async_tiling.md)

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
upstream; the IS-core stage is disabled. Used to demonstrate the bank-conflict throughput hit it `BC`.
The regular (correct) memory layout gives bank conflicts it `BC` because the spatial stride of BC is 16 banks, so
every two elements come from the same bank. In this app, we overwrite the stride with an incorrect one, just to verify
utilization.

## Enabling large seqLen

`main-tiled` cannot be expanded to large seqLen because it assumes `iscore_out_P1` (size `L*xProjDim`), `iscore_out_P2` (size `L*dModel`) and `oscore_in` (size `L*dModel`) remain in TCDM.

### IS-core psum sizes

To mitigate the IS-core output sizes, we have 3 options:

A1. **Split SSM in L:** Split P1 and P2 in half over seqLen. However, this is completely impossible because it requires the
   intermediate SSM states to be read out and restored during the second half. P1 and P2 must always be fully
   completed.

A2. **Async tiling on psums:** Tile the IS-core output dimension N (xProjDim in P1, dModel in P2). The outer loop is still the dInner tiles,
   inner loop is the N-tiles. However, since this tensor is a BF16 psum, we need to load all N-tiles (i.e. the whole
   psum matrix) in and out for every reduction dimension tile K (tile in dInner, i.e. `n_tiles`).
Feasibility check:
- The full psum tensor needs to be transferred 2*n_tiles times -> data = 2*n_tiles*2*L*N bytes.
- The total compute time L*dInner*N / (dInnerUnroll*dModelUnroll) = L*dInner*N / 384
- Required bandwidth > 1536*n_tiles / dInner
-> Certainly feasible.

A second complication is that in `main-tiled`, P1 IS-core bank-transposes its output, both via the bank-transpose and
by scattering the transposed output over the full-size buffer. If we tile in xProjDim, the full-size buffer is no
longer available and the destination address is in L3, not TCDM. For this reason, we omit the P1 scattering and
write to L3 contiguously.
We assume the address reordering is done by the DMA engine or off-chip.

A third complication is that we need to support async tiling in some way. We cannot simply re-launch Simbacore, because we
cannot control the IS-core and other cores independently, and all streamers are fired as one. See docs/dataflow/09_async_tiling.md.

A3. **P1** Split whole phase in L. This simply means we do the full P1 kernel call for every seqLen tile. The downside
   is that we lose the weight-reuse on the projection weight matrices, and we need to load in the full weight
   matrices for every seqLen tile, costing extra energy. However, the weight matrices are significantly smaller that the input tensor, so this is still cheaper than the alternative. The same bank-transpose complication applies here.
Not implemented yet.

A4. Do IS-gemm in a separate program. 
**P2**: We would tile the SUC outputs directly to L3. This way, we loose the SUC/IS-core 
overlap. However, the overlap is diminishing anyways, because 1) we have a safe-to-start delay, which can be 50% of the
compute time for small dInner tiles; and 2) for smaller dModel, the IS-core computation is negligible compared to the SUC
computation. What we win is that we can compute the output projection on OS-core and IS-core in parallel.
**P1**: For xProj, we would lose more because here all cores are nicely coupled. However, the xProj IS-core tiles are smaller,
yet we will have to pace the transfers at the OS-core side (slower), so we have more compute available. The ration 
DMA/compute time becomes xProjDim/dModel. Not a problem if we split in L (A3).

### OS-core input sizes

B1. Tile the OS-core input tensor in seqLen.

B2. **P1** split whole phase in L.

B3. **P2** Do the OS-gemm in a separate program. How much we loose depends on how much we can let OS-core and SUC overlap.


### Combined strategy

- Split P1 in L (A3+B2)
- Exclude IS-gemm from P2, so SUC writes to L3 directly, and we can compute the out proj on os/is-gemm (A4)
- Async-tile the OS-core input (B1)

## `main-tiled-A2`

Not implemented yet.

## `main-tiled-B1`

Implements option B1.



