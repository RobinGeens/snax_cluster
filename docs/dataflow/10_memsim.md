# memsim — functional + cycle-accurate memory simulator

`memsim` runs a compiled SNAX app `.elf` and predicts its **memory-transfer timing**
(bank conflicts, DMA latency/bandwidth, FIFO/streamer behaviour, accelerator
pacing) orders of magnitude faster than the RTL `vsim`, while also carrying real
data so memory *layouts* can be checked. It is a standalone host C++17 tool —
no podman, no license. Source: [`target/snitch_cluster/sim/`](../../target/snitch_cluster/sim).

```
make -C target/snitch_cluster/sim                       # build (host g++), seconds
./bin/snitch_cluster.memsim sw/apps/<app>/build/<app>.elf   # drop-in for the .vsim
```
It emits the same `Simbacore elapsed`/`Snitch elapsed` lines the apps print (so
`scripts/batch_run_report.py` scrapes it identically) plus a PASS/FAIL terminator.

### Engine activity timeline

`--timeline <file>` dumps one CSV row per active cycle window (`engine,start,end,ideal`) for
the five engines `OSCORE`/`ISCORE`/`SUC`/`switchCore`/`DMA` plus a `TCDM` traffic row. The
windows come straight from the busy-cycle model in `run_invocation`: in Phase 2 osCore→SUC→isCore
tile the accelerator window in series (the inter-stage `fill_*` handoffs show as gaps), in Phase 1
osCore / switchCore-conv / isCore overlap from the same start, and DMA windows come from
`dma_submit`. For an engine row, `ideal` is the shortest possible cycle count for that block's
compute (its MAC-group count at peak, conflict- and drain-free); for a `TCDM` row it is instead
the TCDM word-access count over the window (every enabled streamer port issues `∏ t_bound × NCH`
8-byte accesses per invocation; each DMA beat is 8 words on each TCDM-resident side).

[`sim/plot_timeline.py`](../../target/snitch_cluster/sim/plot_timeline.py) runs memsim and
plots it — one row per engine, cycle count on the x-axis (so you see *when* and *how long* each
module runs), each row labelled with its average **hardware utilization = ideal / actual**: of
the cycles a module was active, how much real work it did. A conflict-free array is ~100%; the
SUC's dt_BC bank conflict pulls it to ~57% at bc_pad=0 (~100% at bc_pad=4); a GEMM's output
drain and the DMA first-beat latency also drop it below 100%. A final **TCDM BW** row draws the
combined streamer+DMA word demand as a 0–100% line (peak = 32 banks × 1 word/cyc); it is an
invocation-average, so in Phase 2 the serialized stages are averaged together rather than shown
as per-stage peaks.

```
./bin/snitch_cluster.memsim sw/apps/<app>/build/<app>.elf --timeline tl.csv  # CSV only
python sim/plot_timeline.py sw/apps/<app>/build/<app>.elf                     # run + PNG
```

(SIMD/FFT passes drive no engine row — the SimdCore is outside the perf-counted MambaCore.)

## Architecture

Two layers co-simulate; the C program drives both:

```
 .elf ─> [interp: dual-hart RV32IMA(+FP)] ─events+data─> [SimWorld: timing + functional datapath]
          hart0 = compute CSRs / gauges     <─queries──    TCDM data, DMA, streamers, accelerator
          hart1 = DMA                                       gauges/busy advance over SIM TIME
                  └────────── scheduler: min-cycle interleave, 2-hart barrier ──────────┘
```

- **Front end** (`interp.cpp`, `sched.cpp`): a functional dual-hart RV32IMA(+FP)
  interpreter. It boots the ELF from `e_entry` (runs crt0: TLS/BSS/CLS init,
  atomics, allocators), so `snrt_l1_next`/`snrt_l3alloc` return the exact
  addresses the app feeds into streamer/DMA descriptors. The only HW-facing ops
  it surfaces to the world: the xDMA opcodes (`dmsrc/dmdst/dmstr/dmrep/dmcpyi/
  dmstati`), SNAX streamer/SimbaCore CSRs, the `0x7C2` barrier, `fence`, and
  `mcycle`. htif (`tohost`/`fromhost`) carries `printf` and the exit code.
- **Co-sim** (the load-bearing trick): the C paces on model state via blocking
  polls (`while(read_csr(SIMBACORE_BUSY))`, `while(read_csr(R10_DELAY_GAUGE)<cnt)`,
  `snrt_dma_wait_all`). The scheduler steps whichever hart has the smaller cycle;
  a poll spins, advancing that hart's cycle, until the world's time-based
  busy/gauge/DMA value flips — so the polls resolve at the right *simulated* cycle
  with no special-casing. The HW barrier rendezvous syncs both harts to `max`.
- **Timing world** (`world.cpp`): models the accelerator busy duration, the DMA,
  the gauges, and the TCDM bank conflict. Carries real bytes (DMA does the copy;
  data is correct immediately, timing is tracked separately).

## Accelerator busy-cycle model (= SIMBACORE_PERFORMANCE_COUNTER)

The MambaCore perf counter ticks every cycle `globalState != sIDLE`. Per
invocation (dims are the *per-tile* values the app writes: `D_INNER` = `M*_dInner_tile`):

```
M_i  = seqLen / 16            osN = K_i = dInner_tile / 24
osCore = M_i · osN · dModel             (input-feed bound)
isCore = M_i · dFinal · K_i             (dFinal = xProjDim in P1, dModel in P2)
conv   = seqLen · dInner_tile / 4       (Phase 1 switchCore conv1d; convUnroll=delaySU=4)
matmul = seqLen · dInner_tile · dtRank / (4 · dtRankUnroll)   (Phase 2 switchCore dt projection)
SUC    = bc · seqLen · dInner_tile      (Phase 2 only; bc = bank-conflict factor)
```
- **Phase 1** (osCore → switchCore conv → isCore, pipelined): `max(osCore, conv, isCore) + fill_p1`.
- **Phase 2** (osCore → SUC → isCore, **overlapped** by the safe-to-start gauges): the SUC starts at
  `r10_cnt/r10_total · osCore + fill_osc`, the isCore that far into the SUC; busy =
  `max(osCore, SUC, isCore)`. Full-serialize start_cnts collapse it to `osCore + SUC + isCore + fill_p2`
  (the conservative `main` default); a low-paced tiled app overlaps and finishes sooner.
  The switchCore dt projection (m_switchCoreMode=Matmul, `dt_delta = dt·Wᵀ + bias`) runs
  concurrently — it streams `dt_delta` into the SUC scan — so it is recorded as an overlapping
  trace bar from the invocation start (`matmul` cycles) but does not enter the SUC-bound
  critical-path sum. `matmul` is the MambaCore outer loop (`dInner_tile/convUnroll` windows,
  MambaCore.scala) × the SwitchCore matmul input phase (`seqLen · dtRank/dtRankUnroll`).

**Which stages run is decoded from the MODE bitfield, not the streamer ports.** MODE is a
packed `SimbaCoreCtrlBundle` ([SimbaCoreMode.scala], width 20, first field = MSB): bit19
`en_osCore`, bit18 `en_suCore`, bit17 `en_isCore`. `run_invocation` gates each duration on
these bits — `osCore` iff bit19, `SUC` (the per-cycle conflict sim) iff bit18, `isCore` iff
bit17 — because the streamer ports (R6/R7…) are reused across modes. Inferring "SUC active"
from the ports (the old `phase2_ = ports_[6/7].enabled`) applied the mamba selective-scan
duration to *every FFT SIMD pass* (einfft's SIMD modes enable R6/R7 but have all 3 core bits
0) → a 12.5× SimbaCore over-prediction. Three cases:
- **SUC (bit18)**: strict-serialize `osCore→SUC→isCore` as above.
- **OSGEMM (osCore only)**: **cycle-stepped** (`cyc_gemm`) — the array consumes one group from
  each input reader (R0=A, R1=B) per cycle and emits a tile every K_i steps into a writer that
  drains via W0; busy + output drain are produced by stepping, no `osc_dur`/fill. einfft OSGEMM = 2353
  (2305 array + 48 drain) vs vsim ~2388. **Each port arbitrates on its OWN Fabric** — the TCDM
  interconnect is SPARSE (SparseConfig residue-pins each port to its own bank routing), so ports
  contend only within themselves (gran≥lanes ⇒ no self-conflict; only gran-1 R7 does), not across
  each other. Putting R0/R1/W0 on one shared 32-bank Fabric (all start at bank 0) gave a false
  +21% cross-port stall; per-port Fabric fixed it (and is why the single-port SUC model was already
  right). Only DMA superbank preemption is shared cross-port.
- **ISGEMM (isCore only)**: **cycle-stepped** (`cyc_gemm`, n_readers=0 + W3) — same as OSGEMM;
  the isCore array runs `M_i·dFinal·(dInner/24)` (conflict-free, 1/cycle) and the W3 drain is stepped.
- **chained PHASE1 (both cores + switchCore conv)**: osCore → switchCore → isCore stream as a
  pipeline; globalState (the perf counter) spans until all three finish, so busy = `max(osCore,
  conv, isCore)` + `fill_p1`. The switchCore conv1d (m_switchCoreMode=Conv) runs
  `seqLen·dInner/convUnroll` (convUnroll=delaySU=4, SwitchCore.scala) — it is the bottleneck when
  dModel is small (osCore/isCore cheap) and **must be in the max** (omitting it was a 12 % under-count
  on dModel-small P1 apps). `fill_p1` scales with the seqLen-tile count (see Constants).
- **IS_OSGEMM (both cores, no conv)**: `max(osCore,isCore)` + flat `fill_is_osgemm` (different
  pipeline, no switchCore).

**Where cycle-stepping matters, it is applied; where it doesn't, the formula is the cycle count.**
Conflict (SUC dt_BC → `cyc_suc_duration`) and output drain (OSGEMM/ISGEMM → `cyc_gemm`) are stepped
per cycle. Conflict-free compute (osCore/isCore arrays, gran≥lanes per the sparse interconnect)
runs at 1 MAC-group/cycle, so `M_i·N·K` is exactly its busy count. The remaining `fill_*` constants
are inter-stage handoff pipeline depths (gauge→TCDM-commit); deriving them fully from RTL
register-stage counts (see Constants table) is an open item.
- **SIMD (no core bit)**: busy = the gating streamer's beat count (SimdCore streams one
  bank-group/cycle), and **`perf_=0`** — the perf counter is in MambaCore but the SimdCore is
  *outside* it, so a SIMD pass adds to Total (wall-clock) but not to SimbaCore.

einfft-tiled (seqLen 392, dModel 192): Total **1,952,230 → 259,830** vs vsim 287,239 (−9.5%),
SimbaCore **1,907,264 → 147,712** vs 152,826 (−3.3%); main/main-tiled unchanged. The residuals
are honest, not padded: SimbaCore −3.3% = un-modelled OSGEMM output-drain; Total −9.5% = under-
modelled per-invocation overhead across 224 invocations (streamer drain past `SIMBACORE_BUSY`,
HW-barrier release latency, scalar/poll cost) — a general ISS-fidelity gap, not a fit constant.

The gauges advance linearly over their stage windows: `R10` (osCore tiles)
reaches `M_i·osN` at osCore-done; `R11` (SUC elements) reaches `seqLen·dInner_tile`
at SUC-done; `ISCORE_TILE_CNT` over the isCore window. These are exactly the
thresholds the P2 `start_simbacore_and_streamers` polls wait on.

## TCDM bank conflict (the bc_pad effect)

**Computed per-cycle** (`cyc.cpp`, `cyc_suc_duration`), from the RTL. The derivation has two halves:

1. **Demand = 1 BC group per SUC output, independent of dInner_tile.** The SUC
   ([`StateUpdateCore.scala`](../../../chisel-ssm/src/main/scala/mambacore/StateUpdateCore.scala))
   runs `for i in D/delaySU: for j in L: for k in delaySU`, emitting one `out_y`
   per iteration (`seqLen·dInner_tile` outputs) and pulling a fresh `in_vecB`/
   `in_vecC` (one `Vec(N)` each = 4 R7 groups) every `delaySU` iterations. The R7
   AGU ([`AddressGenUnit.scala`](../../hw/chisel/src/main/scala/snax/readerWriter/AddressGenUnit.scala))
   has one `ProgrammableCounter` per temporal dim and pushes a TCDM read for
   **every** temporal point — including a **stride-0 dim, which re-reads the same
   address rather than reusing a buffer** (the AGU has only an address FIFO, no data
   cache). So `∏ temporal_bounds = seqLen·dInner_tile` reads = exactly one BC group
   per output. The schedule's "reuse in D" is *logical* reuse in the SUC math; it is
   **not** a TCDM-bandwidth saving — R7 re-reads B/C for every `i`.

2. **Delivery rate = bank count, from real arbitration.** R7 is a gran-1 port
   (`sparse_interconnect_config` `[4,1]`) whose 4 spatial lanes CAN collide — unlike the
   residue-pinned gran-4 *reader* ports R1/R12/R13 (`[4,4]`), which cannot self-collide.
   (The output writers W0–W3 are `[3,1]` = gran-1, so they *can* in principle conflict;
   that is the bc_pad concern on the IS-core side and is out of scope for this R7-only model.)
   Unpadded dt_BC strides `s=[128,256]` → lanes hit banks `{0,16,0,16}` → the per-bank
   arbiter grants **0.5 group/cyc**; `bc_pad` strides `s=[160,320]` → banks
   `{0,20,8,28}` → **1.0 group/cyc**. The SUC stalls when its depth-4 BC data FIFO
   (hjson `fifo_depth[7]=4`) empties; the deeper address buffer lets R7 issue ahead so
   the +1cc TCDM latency is hidden.

**Geometry (`test/suc_grid_test.cpp`):** the group-synchronous fabric gives a
SUC busy of `banks_factor · seqLen · dInner_tile`, `banks_factor = 2.0` for bc_pad=0 and
`1.0` for bc_pad=4, **flat across dInner_tile**. So the fabric decides *whether* there is
a conflict (2-bank vs 4-bank) from the real strides, independent of dInner_tile.

**Magnitude.** The conflict magnitude comes out of the same per-cycle
sim, because the R7 reader is modelled faithfully per-lane (not group-synchronously). Per
[Reader.scala](../../hw/chisel/src/main/scala/snax/readerWriter/Reader.scala)/
[DataRequestor.scala](../../hw/chisel/src/main/scala/snax/readerWriter/DataRequestor.scala):
the AGU pushes one group (4 lane addresses) per temporal step into per-lane address FIFOs
(depth = `bufferDepth` = 4) when *all* lanes have room, but each lane's DataRequestor then
issues its head address **independently** (`in.addr.ready := tcdmReq.fire`, per channel) and
pops on a bank grant; the response path buffers a further `responser(4) + dataBuffer(4) = 8`
reads per lane. So a lane on a free bank reads up to ~8 steps (2 refreshes) ahead while a
lane on the conflicted bank finishes the current refresh — and since the next refresh sits
on **different** banks (the dim1 temporal stride is +1 bank), those ahead-reads overlap the
current refresh's tail. The result, straight from arbitration: bc_pad=0 → **1.75×**,
bc_pad=4 → **1.0×**, flat across dInner_tile. The SUC *compute*-pipeline fill (~20 cc) is
modelled separately (the safe-to-start `commit_pipe_y`).

**vsim perf-counter split (the measurement).** main-tiled pad0, one P2 tile, reading
`SIMBACORE_PERFORMANCE_COUNTER` at the R10 (osCore-done) and R11 (SUC-done) gauge
crossings: `osc=4869  suc=8091  isc=4726  total=17686`. vs the analytic MAC counts
(osc=4608, isc=4608, suc compute=4608): osCore carries a **+261** handoff fill, isCore
**+118**, and the SUC is **8091 = 1.756×** (not 2.0×). The per-cycle per-lane sim produces
the SUC at **8069 = 1.751×** with no factor (the −22 is the SUC compute-pipeline fill); the
osc/isc fills remain labelled handoff/pipeline residuals (see the constants table).

## DMA model

Single channel, 64 B/beat. First transfer of a chain pays a first-beat latency;
back-to-back transfers pipeline (bus-bound on beats). `dma_busy` is 1 until the
running completion time; `snrt_dma_wait_all` spins until then. Calibrated from the
real per-cycle DMA trace by [`calib/calibrate_dma.py`](../../target/snitch_cluster/sim/calib/calibrate_dma.py):
**L3↔TCDM first-beat ≈ 19cc**, ≈1 cc/beat. (The "~250cc" in older notes is the
full issue→descriptor-FIFO→roundtrip chain, not the backend first-beat.)

## Performance optimizations (wall-clock)

The apps apply the optimizations in [08_performance_optimization.md](08_performance_optimization.md);
the model reproduces the two that move Snitch (wall-clock) time:

- **CSR pre-loading.** Each tile asserts `SIMBACORE_START` and *then*, while the accelerator
  is busy, writes the **next** tile's base-ptr/stride/bound CSRs (5 in P1, 12 in P2) before
  the `SIMBACORE_BUSY` poll. Those writes overlap compute, so their offload latency is hidden.
  The model charges `SNAX_CSR_WRITE_COST` only when the write is on the critical path:
  `World::snax_write_serializes(csr, at)` returns false (free) for a streamer-config write
  (`[960,1158]`) issued while `at < accel_end_` (core busy), and true (charge) for MODE/dim/
  START writes and for any config write issued while idle (first-tile setup, post-poll
  re-launch). Without this, the preloads serialize onto the critical path at small
  `dInner_tile`, which is exactly what the optimization removes. On main-tiled this frees
  ~3.5k cc (Total 746,907 → 743,432). The accelerator perf counter (`Simbacore elapsed`) is
  untouched — this is a Snitch-only effect.
- **DMA latency hiding (double-buffering).** Modeled *implicitly*: the model already advances
  hart0 to `accel_end_` (compute) and hart1 to `dma_busy_until_` (DMA) and the barrier
  rendezvous takes the max, so the app's own `DMA latency hiding: P1=ok/STALL` diagnostic
  (computed in-app from `snrt_mcycle` reads) comes out of the model verbatim — no extra
  modeling. Faithfulness hinges on the DMA chain (`dma_submit`): load+spill transfers
  serialize on one channel (program order through `dma_busy_until_`) and back-to-back
  transfers pay the first-beat once (chain pipelines), so the stall margin is
  `compute_dur − (first_beat + Σbeats)`.

The other doc'd optimizations need no model change: safe-to-start gauges are the strict
serialize already modeled (and swept, below); inline start/wait is the same CSR writes,
now correctly free during the busy window; printf removal and data alignment don't affect
the interpreted instruction stream.

> **Needs a vsim to calibrate** (flagged, not yet done): (i) the *on-critical-path* START
> write cost (currently the lone `SNAX_CSR_WRITE_COST=12`, validated only against large-tile
> Total — a small `dInner_tile` build makes the preload cost a measurable Total fraction);
> (ii) whether the DMA chain pays a per-descriptor reissue cost beyond the first beat — find
> the smallest `dInner_tile` where vsim's `DMA latency hiding` flips to `STALL` and confirm
> the model flips at the same boundary.
> (iii) ~~the safe-to-start latency `L`~~ **done** — the boundary is computed by a
> per-element commit-vs-read schedule; vsim verifies it (R10=2 exact,
> R11=5397 vs 5400). See the safe-to-start section. Only the small first-element commit-pipe
> (`s2s_lat_z_=3`, `s2s_lat_y_=20`) is a constant; the K_i tile-quantization scales with shape.

## Constants: derived vs residual

The model derives from RTL/architecture; vsim only *verifies*. Each constant in `world.hpp`/
`snax_csr.hpp` is labelled **DERIVED** (an RTL register-stage count or geometry, replace the
number when it changes) or **RESIDUAL** (a genuine effect not register-countable in this checkout —
kept as a clearly-labelled minimal constant, never dressed up as a derivation):

| constant | value | class | basis |
|---|---:|---|---|
| `SNAX_CSR_WRITE_COST` | **1** | DERIVED | `csrw` rd=x0 → no scoreboard wait; offload spill ready same-cycle (`spill_register_flushable.sv`), ReqRspManager config-write `req_ready` combinational (`ReqRspManager.sv:161-166`). The earlier **12 had no RTL basis** — removing it widened the honest Snitch residual (it was masking un-modelled barrier/icache overhead). |
| SUC bank conflict | **per-cycle (no constant)** | DERIVED | The 1.75× magnitude (1.0× for bc_pad=4) comes from the per-lane per-cycle sim — per-lane address FIFOs (depth 4) + 8-deep response read-ahead + the +1-bank dim1 refresh shift. The SUC compute-pipeline fill (~20 cc) is the safe-to-start `commit_pipe_y` = delayTotal 17 (StateUpdateCore.scala:135) + W2 commit ~3, used only in the safe-to-start schedule. |
| `dma_first_beat_l3` | **19** | DERIVED (env-tinged) | iDMA backend req→AR→first-R→first-W (`axi_dma_backend.sv`) + 3 FE stages; **11 of the 16** is the simulated L3+xbar latency (environment constant, not iDMA depth). |
| `fill_osc_p2` | **261** | mostly RESIDUAL | osCore datapath is only ~2 cc (array `pipelineDelay`=1 [osCorePiping mulPost=1, adderless] + output ser 1 [Array.scala:206]); the other **~259 is an osCore→SUC handoff** (W0 drain + R6 first-fetch + ElasticFIFO `delayBtoC=6` [SimbaCore.scala:129] + SUC warmup) the vsim split mis-attributes to the osCore stage. data-dependent FSM rendezvous + TCDM queueing, not a static depth. |
| `fill_isc_p2` | **118** | PARTIAL | ~56 DERIVED (48 input-s2p warmup [`isCoreSerDes=3072/64`, MambaCoreParams.scala:85] + 5 array `pipelineDelay=log2⌈24⌉` [Array.scala:140] + 3 acc/io) + ~62 RESIDUAL (C/bias init + repeaterA refill bubbles + FSM rendezvous, irreducible). |
| `fill_p1` | **M_i·31 + 180** | RESIDUAL (seqLen-dep) | PHASE1 pipeline fill: per-seqLen-tile inter-stage handoff bubble (31/M_i) + constant 3-stage lead-in/drain (180). Fit to P1-tiled-D vsim at M_i=8 (fill 428) and M_i=49 (fill 1686), cross-checked vs main-tiled M_i=12 (~560). Replaces the earlier flat 562, which only matched M_i=12 and undershot large seqLen by ~31/M_i (the source of the P1 SimbaCore gap). IS_OSGEMM (no conv) keeps a flat `fill_is_osgemm=562`. |
| `LOAD_USE_STALL` | **3** | DERIVED | Snitch consumer of an in-flight load stalls: `register_core_req`(1) + TCDM `Latency`=1 (RegisterTCDMCuts=0) + `register_core_rsp`(1). Charged only to the FIRST consumer (single-pending, NumOutstandingLoads=1); hoisted loads → no stall. |
| `MUL_STALL` / `DIV_STALL` | **3 / 32** | DERIVED / envelope | MUL = offload_req(1)+mul-stage(1)+offload_rsp(1). DIV/REM = serdiv `2+div_shift+2`, worst-case `div_shift=32` (exact `div_shift` operand-dependent → residual). |
| `HW_BARRIER_RELEASE_COST` | **2** | DERIVED | `snitch_barrier.sv` arrival latch(1) + `snitch.sv` csr_stall release(1), beyond the barrier CSR's own 1 cc. NrCores-independent. |
| icache-miss refill | — | RESIDUAL (not added) | L1/L2/AXI refill latency = environment-dependent, not derivable from the core RTL. This is the bulk of the remaining Snitch-Total gap; deliberately NOT fudged. |

The bank-conflict geometry (which port, 2 vs 4 banks) and magnitude both come from the per-cycle
per-lane fabric. The residuals above (`fill_osc_p2` handoff, `fill_p1` seqLen-dependence) are
data-dependent (FSM rendezvous + TCDM queueing). Deriving the constants from RTL rather than
matching the total widens the Snitch residual slightly (CSR 12→1 dropped ~3.6 k cc → −2.5 % vs
vsim) — that gap is genuine un-modelled overhead (HW barrier cost, icache, scalar stalls).

## Validation (real vsim builds)

main-tiled (seqLen=192 dModel=384 dtRank=24 nb_tiles=32), built both pad0 and pad4:

| metric | memsim | vsim | error |
|---|---:|---:|---:|
| **Simbacore total (pad0)** | **731,360** | **730,775** | **+0.08%** |
| **Snitch Total (pad0)** | **743,432** | **758,648** | **−2.0%** |
| **P2 per-tile (pad0)** | **17,685** | **17,686** | exact |
| **— osc / suc / isc split** | **4869 / 8090 / 4726** | **4869 / 8091 / 4726** | measured |
| **Simbacore total (pad4)** | **623,072** | **623,161** | **−0.01%** |

(the per-tile osc/suc/isc split is the vsim perf-counter measurement the model is checked
against; the bc_pad=0 conflict comes from the per-cycle fabric, so pad4 — no conflict — and
pad0 use the same model with no separate factor.)

main-tiled-oscore (async oscore_in ring; seqLen=192 dModel=384 nb_tiles=16
nb_l_tiles=12 nb_slots=2 bc_pad=4):

| metric | memsim | vsim | error |
|---|---:|---:|---:|
| **Simbacore total** | **606,448** | **612,289** | **−0.95%** |
| **Snitch Total** | **620,158** | **645,001** | **−3.9%** |

`nop.elf` boots and exits 0. The Total residual (~2–4%) is uncounted SW overhead
(barrier hardware cost, icache, scalar stalls); the async residual also includes
ring-overlap effects not yet fully modeled. The pad0 Snitch Total moved −1.5%→−2.0%
when CSR pre-loading was modeled as overlapped (the ~3.5k cc of preload writes are now
correctly hidden under compute, as in HW, rather than over-charged onto the critical
path) — so the residual is now purely the uncounted overhead above, not a CSR artifact.

## Verdict & the app's FP self-check

The model does timing + integer/layout, **not** the bf16/fp8 datapath, so it does not
fill the apps' FP output buffers. Each app's own `check_result` (its `… ref = N` /
`N/M errors` lines) therefore can never pass under memsim — it is **suppressed by default**
(`MEMSIM_SHOW_APP_CHECK=1` to see it) and is **never the model's verdict**. The integer
LAYOUT/BIST below **always runs** (no flag needed; `--timing-only` skips it); the verdict is
`[SUCCESS]` iff it passes and the program completed (a deadlock fails). Earlier this was the
source of a "the BIST reports all errors" confusion — those were the app's FP compare, not
the model's check.

## Layout verification (`--verify`)

Two complementary layout checks run under `--verify` (on by default):

**(a) General AGU structural audit (`verify_layout`, ALL apps, golden-free, located).** Runs on
every invocation of any app and reports *how many* layout/AGU errors and *where* (port + TCDM
address range), with three pure-address checks: **bounds** — each enabled streamer's address extent
(base + temporal + spatial reach) must stay inside the 512 KiB TCDM `[0x10000000,0x10080000)`;
**producer→consumer containment** — a reader that strictly overlaps a writer must read only within
what that writer produced (reading beyond = the consumer pulls addresses the producer never wrote →
corruption); and **writer no-alias (permutation)** — every writer emits each *distinct* output word
once, so two iterations that differ in a *moving* (non-zero-stride) dim hitting the same word =
output elements overwrite each other = a permuted-order bug the bounds check cannot see (same extent,
different per-iteration address). This is the **"subtle wrong permutation within a correctly-bounded
buffer"**: enumerate each writer's full emitted address sequence and flag the first word written
twice, located by `W<n> ALIASES word <addr> (emit #k)`. Stride-0 dims are *collapsed* first — a
stationary axis is the legitimate accumulation pass (e.g. the isCore K-reduction, where the output
address holds and the array commits once on the final step), not a bug. It catches/locates gross
AGU bugs (OOB, overrun) **and** in-bounds output permutations across every app. Reported as `AGU
layout audit: N located error(s) over M invocation(s)`; correct apps = 0 (verified 0 aliases on all
29 built apps), and `MEMSIM_LAYOUT_FAULT=1` self-tests inject a broken AGU (OOB reader) and a
synthetic equal-stride writer to prove both checks are live (`LOCATED`). **Remaining gap:** a wrong
stride *magnitude* that collapses an axis to one address (a stride wrongly 0) is treated as
accumulation, so a *bijective-but-wrong* order needs the FP/golden round-trip below (mamba only).

**(b) osCore GEMM integer cross-check (`verify_oscore`, mamba only, FP-aware).** An **integer**
cross-check of the osCore GEMM (`z = A·B`, A=oscore_in/flattenA, B=oscore_weight, z=W0/ConvFormat). It
carries real data (DMA + streamers already move bytes), gathers A/B as int8, runs
osCore GEMM (`z = A·B`, A=oscore_in/flattenA, B=oscore_weight, z=W0/ConvFormat). It
carries real data (DMA + streamers already move bytes), gathers A/B as int8, runs
the integer matmul through the documented layouts, and checks:
1. **R0 input AGU** implements the documented `flattenA` (k-stride=Mu, m-tile-stride
   =K·Mu, Mu-contiguous seq lanes).
2. **W0 output AGU** writes z contiguously in ConvFormat.
3. **producer→consumer**: the SUC z-reader R10 reads the buffer the osCore z-writer
   W0 wrote.
4. **layout-sensitivity**: a deliberately permuted gather changes the integer
   result (proving the check discriminates — not a no-op).
Drives PASS/FAIL + exit code in `--verify` mode (independent of the app's FP
`check_result`, which integer compute won't match). Verified PASS on main-tiled and
main-tiled-oscore.

### Timing-coupled BIST (SUC dt_BC delivery)

`--verify` also runs a BIST (`cyc_suc_bist`) on the headline timing path: the SUC's
BC data flows through the per-cycle fabric + depth-4 data FIFO, and a group's data is
valid only **after** its +1cc TCDM response lands. It asserts three falsifiable
properties so that **a wrong delay surfaces as an error**:
1. **complete** — the SUC consumes every BC group with no deadlock (an under-delivered
   / dropped group would stall it forever).
2. **delay-respect** — with the correct +1cc latency the consumer never reads an
   un-landed (poison) slot. The landed-value ring is the single occupancy ledger, so a
   producer/consumer desync would surface as a poison read.
3. **delay-sensitivity** — when the model is perturbed so the consumer treats a
   not-yet-landed read as ready (a delay modelled one cycle *too short*), it reads
   poison and the BIST catches it (`too-short-delay caught=YES`). This is the proof
   that the check has teeth, not a tautology.

Separately, `test/suc_grid_test.cpp` is an asserting regression that the per-cycle
conflict factor equals the **bank count** (≈2.0× for bc_pad=0, ≈1.0× for bc_pad=4),
flat across dInner_tile ∈ {24,48,96} — the genuinely falsifiable proof the faithful
model holds. Build: `g++ -std=c++17 -O2 -Isrc src/cyc.cpp test/suc_grid_test.cpp -o /tmp/suc_grid`.

## FP32 functional datapath (correctness + safe-to-start substrate)

To verify memory layout AND tune the safe-to-start delay, the model recomputes each
Mamba kernel in **FP32** through the *actual* streamer AGU layouts and compares to the
app's existing FP8 goldens (`M2_oscore/suc/iscore_expected`) — interpret golden bytes as
**FP8_ALT (e5m2)** → FP32, normalize a global requant scale (median model/golden), compare
with a generous tolerance (a wrong layout/stale-read gives wrong *operands* → large
residual; a layout-fault self-test confirms the check discriminates). Decoders: `fp.hpp`.
Gated by `MEMSIM_DATAPATH=1` (full-tensor checks need the non-tiled `main` app).

Status: **ALL kernels validated on main** — osCore 6144/6144, isCore 3072/3072, SUC
6065/6144 (98.7%, scale 0.91; the residual is FP32 vs the BF16/FP16 softplus/silu/exp LUTs).
The model doesn't compute P1, so the SUC sources x (`M2_suc_x`) and dt/B/C (`M2_dt_BC`) from
the P1-output goldens; A/D/weights are DMA'd inputs (TCDM valid). `cmp_fp32_golden` flags a
scale-collapse (model→0) so a degenerate "0≈0" can't false-pass.

**Scope: tiled apps.** Full FP32 datapath validation targets the **non-tiled `main`** by
design, not a limitation to fix. A tiled app (e.g. main-tiled, dInner-tiled) computes one
slice per invocation and accumulates the IS-core output **in place across tiles** (the psum
buffer stays FULL, non-final tiles use no-requant — see
[04_mamba_main.md](04_mamba_main.md)), so a single invocation's output is not directly
comparable to the complete golden. Crucially, the layout formulas (flattenA/B, ConvFormat,
dt_BC) are **app- and tile-independent**: every tile runs the *identical* AGU program, only
the base pointer differs (CSR-preloaded). So the integer osCore layout check that runs on the
first P2 tile of *any* app, plus the full FP32 datapath on `main`, already cover every layout
the tiled apps use. Extending FP32 checks to all N tiles would only re-exercise the same
formulas at different base pointers (a per-tile base-ptr bug is already caught by the integer
round-trip and the timing) — deliberately not built.

**Safe-to-start sweep (printed on every P2 run).** Computes the minimum `start_cnt` for each gate that
overlaps a stage without reading stale data, via a per-element commit-vs-read schedule. Releasing a
consumer when the upstream gauge reaches `start_cnt` overlaps the producer: each output element is
committed by the producer at `commit_cyc` and read by the consumer at `reader_start + read_off`; if
`read < commit` the consumer reads an element the producer hasn't written yet (stale). The producer
commit is tile-granular (osCore writes z in N_M_K tile order: element (m,n) commits at
`(tile(m,n)+1)·K_i + commit_pipe`); the consumer read uses the real AGU order (SUC reads z in
SUCFormat). **R11 (y, SUC→isCore) must use the real write/read AGU reorder**, not a same-order
rate model: the SUC writes y *scattered* (W2, port 16) as it scans while the isCore reads it
*linearly* (R11, port 11), so a linear read pulls a low address the scattered write commits late.
Both orders are enumerated from the decoded W2/R11 streamers (dim0 innermost); a word read at step
`j` is stale if it precedes the commit of the word W2 wrote there. The same-order model missed this
and under-predicted catastrophically (~25 where vsim needs thousands; the reorder gives 3549 for
main-tiled dModel=384, and reproduces main's 5398≈5397). The smallest `start_cnt` with zero stale
reads is the optimum. The only constant is
`commit_pipe`, the RTL commit pipeline of the first output element (z: array + W0 ≈3 cc; y: SU-core
output datapath `delayTotal=17` [StateUpdateCore.scala:135: delayPath1 3 + delayNewState 4 +
delayStateC 7 + delayDxhC 2 + delayY 1] + W2 commit ≈3 = ≈20 cc — not `delayBtoC`, which is B-C
input sync). The model prints one line per gate: `optimal start_cnt=X/total (start_cnt=X-1 has N
stale element(s))`.

On main (seqLen=64, dInner=96): **R10 (z, osCore→SUC) = 2/16**, **R11 (y, SUC→isCore) = 5397/6144**.
The app's `get_safe_to_start_delay` ships 5 and 6144 (full-serialize) — both conservative, leaving
~890 cc of P2 overlap per invocation unused (R10 ~144 cc; R11 ~744 cc), which the sweep surfaces.

**vsim check.** The thresholds are confirmed by sweeping the app's runtime
`M2_R10_start_cnt`/`M2_R11_start_cnt` (rebuild SW only, run vsim) for the smallest value that still
produces correct output. The HW is deterministic: a safe start_cnt reproduces the quant-noise floor
exactly; a too-low one reads stale data that hangs the scan (a sharp boundary — catch the hang with a
timeout, don't wait on it). The vsim sweep was coarse (R10: 2 safe / 1 hangs; R11: 5400 safe / 5300
hangs), and the model's 2 and 5397 sit inside those brackets. Validated on main; the K_i
tile-quantization scales with shape (it is in the schedule). Validated layout formulas:

- **osCore** `z=A·B`: A=`oscore_in`=`flattenA(N_M_K)`=`((m/16)·K+k... )` → `(m/16)*(K*16)+k*16+(m%16)`;
  B=`oscore_weight`=`flattenB(N_M_K, Ku=1, Nu=24)` = `((n/24)·K+k)·24+(n%24)`; out=ConvFormat.
- **isCore** `z=y·W`: A=y (ConvFormat), B=`iscore_weight`=`flattenB(K_M_N, Ku=24, Nu=1)` =
  `((k/24)·N+o)·24+(k%24)`; out=`flattenCD(K_M_N)` = `((m/16)·N+o)·16+(m%16)`.
- **ConvFormat** (z, y, x) `convfmt(m,n)`: `flat=((m/16)*N+...`; verified == `temporalToSpatialIdxConvFormat`
  inverse (Mu=16, colsPerTile=24, conv=4).
- **dt_BC** (P1 isCore out; dt=cols[0,24), B/C interleaved cols[24,152)) byte for logical
  `xProj[m][col]` = `padMatrices(bankTranspose(flattenCD(interleaveCols)))`:
  `flat=((m/16)*152+col)*16+(m%16); tile=flat/16, wt=flat%16;`
  `byte=(tile/8)*paddedMat + (wt/2)*16 + (wt%2)*8 + (tile%8)`, `paddedMat=128+bc_pad_banks*8`.
  Interleave: B[n]→col `24+(n/16)*32+(n%16)`, C[n]→col `24+(n/16)*32+16+(n%16)`.
- **switchCore** `dt_delta=dt·Wᵀ+bias` (delta, pre-softplus): `dt_weight_1/2`=`splitDeltaWeight`
  (Ku=6, convUnroll=4, dConv=4): for `W[R][D]`, `i=R/6,r=R%6,j=D/4,conv=D%4`; if `r<4`:
  `w1[((j*nTilesRow+i)*4+conv)*4 + r]` else `w2[((j*nTilesRow+i)*4+conv)*2 + (r-4)]`,
  `nTilesRow=dtRank/6`. bias=`dt_bias[d]` (fp8).
- **SUC recurrence** (`MambaLib.selectiveScan`): `dsp=softplus(dt_delta); for i: for d: for n:
  h[d][n]=h[d][n]·exp(A[d][n]·dsp[i][d]) + (B[i][n]·dsp[i][d])·x[i][d]; y[i][d]=silu(z[i][d])·(Σ_n h[d][n]·C[i][n] + x[i][d]·D[d])`.
  A=`suc_A` row-major `[dInner][dState]`; D=`suc_D[d]`; x=ConvFormat; z=osCore golden/W0;
  out vs `M2_suc_expected` (ConvFormat), TOL≈0.40 ≤10% out (FP32 vs BF16/FP16 LUTs).
  Open: confirm dt vs B/C base (R2 vs R7 / `M2_dt_to_BC_offset`) and `bc_pad_banks` source.

## Uncertainty / scope

- The SUC dt_BC bank conflict is modelled by the per-cycle per-lane fabric (`cyc.cpp`:
  Fabric + CycReader + `cyc_suc_duration`), driven by R7's real strides; its magnitude
  (~1.75× at bc_pad=0, 1.0× at bc_pad=4) comes out of the arbitration, with no conflict
  constant. The conflict-free GEMM readers (R1/R12/R13, residue-pinned gran-4) cannot
  self-collide, so their compute formula is the exact cycle count. **Known gaps:** the
  gran-1 output writers W0/W3 (the IS-core bc_pad concern) are not modelled; the `fill_p1/p2`
  handoff latencies are data-dependent residuals, not static depths; and the fabric's
  `dma_owned` superbank-preemption mask exists but is not yet driven by a per-cycle DMA beat
  engine (it would affect the async/DMA-overlap cases).
- Phase 2 overlaps via the safe-to-start gauges. The app polls the R10 (osCore z) / R11 (SUC y)
  gauges and releases the SUC / isCore once they reach `M2_R10_start_cnt` / `M2_R11_start_cnt`
  (read from the ELF, `set_s2s`). The model starts each downstream stage at its gauge fraction of
  the upstream window (`t_suc_start = r10_cnt/r10_total · osCore + fill`, likewise isCore off the
  SUC) and ends at `max(osCore, SUC, isCore)`. Full-serialize start_cnts (== gauge totals, the
  conservative `main` default) reduce this to strict osCore→SUC→isCore, so `main` is unchanged; a
  tiled app that paces the gauges low overlaps the stages, which strict-serialize over-predicted by
  ~40% (main-tiled dModel=384, r10=11/r11=25: P2 454k→296k vs vsim 318k). The safe-to-start sweep
  still reports the *minimum safe* thresholds independently (see below).
- Accelerator modes: Mamba P1/P2, OSGEMM/ISGEMM (cycle-stepped), and SIMD/FFT/einFFT
  (busy = streamer beat count).
- Layout verification: the golden-free AGU audit (bounds + producer→consumer + writer
  no-alias) runs on every app; an FP32 datapath cross-check (osCore/isCore/SUC vs the FP8
  goldens) runs on the non-tiled `main` under `MEMSIM_DATAPATH`.
