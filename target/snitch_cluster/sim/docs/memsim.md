# memsim — functional + cycle-accurate memory simulator

`memsim` runs a compiled SNAX app `.elf` and predicts its memory-transfer timing
(bank conflicts, DMA latency/bandwidth, FIFO/streamer behaviour, accelerator pacing)
orders of magnitude faster than the RTL `vsim`, while carrying real data so memory
*layouts* can be checked. It is a standalone host C++17 tool — no podman, no license.
Source: [`target/snitch_cluster/sim/`](..).

```
make -C target/snitch_cluster/sim                            # build (host g++), seconds
./bin/snitch_cluster.memsim sw/apps/<app>/build/<app>.elf    # drop-in for the .vsim
```

It emits the same `Simbacore elapsed`/`Snitch elapsed` lines the apps print (so
`scripts/batch_run_report.py` scrapes it identically) plus a PASS/FAIL terminator.

## Architecture

Two layers co-simulate; the C program drives both.

```
 .elf ─> [interp: dual-hart RV32IMA(+FP)] ─events+data─> [SimWorld: timing + functional datapath]
          hart0 = compute CSRs / gauges     <─queries──    TCDM data, DMA, streamers, accelerator
          hart1 = DMA                                       AccelEngine steps per cycle over SIM TIME
                  └────────── scheduler: min-cycle interleave, 2-hart barrier ──────────┘
```

- **Front end** (`interp.cpp`, `sched.cpp`): a functional dual-hart RV32IMA(+FP)
  interpreter. It boots the ELF from `e_entry` (runs crt0: TLS/BSS/CLS init, atomics,
  allocators), so `snrt_l1_next`/`snrt_l3alloc` return the exact addresses the app feeds
  into streamer/DMA descriptors. The HW-facing ops it surfaces to the world: the xDMA
  opcodes (`dmsrc/dmdst/dmstr/dmrep/dmcpyi/dmstati`), SNAX streamer/SimbaCore CSRs, the
  `0x7C2` barrier, `fence`, and `mcycle`. htif (`tohost`/`fromhost`) carries `printf` and
  the exit code.
- **Co-sim** (the load-bearing trick): the C paces on model state via blocking polls
  (`while(read_csr(SIMBACORE_BUSY))`, `while(read_csr(R10_DELAY_GAUGE)<cnt)`,
  `snrt_dma_wait_all`). The scheduler steps whichever hart has the smaller cycle; a poll
  spins, advancing that hart's cycle, until the world's state flips — so the polls resolve
  at the right *simulated* cycle with no special-casing. The HW barrier rendezvous syncs
  both harts to `max`.
- **Timing world** (`world.cpp`): decodes the streamer CSRs into per-port AGUs, decodes
  the MODE bitfield, configures the single `AccelEngine` for the invocation, and steps it
  one cycle at a time from `advance_to`. Carries real bytes (DMA does the copy; data is
  correct immediately, timing is tracked separately).

## The cycle-accurate engine (`cyc_engine.{hpp,cpp}`)

One resident per-cycle co-simulator — the `AccelEngine` — steps every SimbaCore MODE on
**one shared 32-bank Fabric**, with the DMA beat engine stepped in the *same* loop. There
are no phases and no closed-form stage durations: it is one machine configured differently
per MODE. `run_invocation` decodes the MODE, `configure()`s the engine from the per-port
AGUs, and `advance_to()` calls `step()` until the engine completes; `accel_end_`, the perf
counter, and the gauges are then read from its result.

Each reader/writer arbitrates on the one `Fabric` every cycle and the DMA beat engine
grounds the bank(s) it touches the same cycle, so contention, FIFO-slack DMA-hiding,
inter-stage overlap, and pipeline fill/drain all *emerge* from stepping — none is a fixed
constant. The per-word producer-commit (W0/W2 grant) and consumer-read (R10/R11 grant)
cycles are exact, so the safe-to-start sweep returns the exact minimum start_cnt.

### MODE decode selects the units

Which stages run is decoded from the **MODE bitfield, not the streamer ports**. MODE is a
packed `SimbaCoreCtrlBundle` (`SimbaCoreMode.scala`, width 20, first field = MSB): bit19
`en_osCore`, bit18 `en_suCore`, bit17 `en_isCore`, bits[9:8] `switchCoreMode`.
`configure()` gates each unit on these bits (`cyc_engine.cpp` lines 79-104) because the
streamer ports (R6/R7…) are reused across modes. Keying off the ports instead applies the
mamba selective-scan duration to *every* FFT SIMD pass (einfft's SIMD modes enable R6/R7
with all three core bits 0 → a 12.5× SimbaCore over-prediction).

`configure()` selects one of five steppers (`kind_`):

| `kind_` | MODE | wiring |
|---|---|---|
| 1 | P2 (osCore+SUC+isCore) | R0,R1→osCore→W0(z); R7(dt_BC),R10(z)→SUC scan→W2(y); R11(y),R12(w),R13(psum)→isCore→W3; switchCore R2,R3,R5 co-active |
| 2 | P1 / IS_OSGEMM (osCore+isCore, no SUC) | R0,R1→osCore→W0/W1 → switchCore conv R3,R4 → R12,R13→isCore→W3, streamed on-chip |
| 3 | OSGEMM (osCore only) | R0,R1→array(K=dModel)→W0 |
| 4 | ISGEMM (isCore only) | array(K=osN)→W3 drain |
| 5 | SIMD (no core bit) | any enabled reader/writer ports → implicit SimdCore → writers |

A SIMD pass (`kind_=5`) ticks wall-clock but **not** the perf counter (`perf_=0`): the
SimdCore sits outside the perf-counted MambaCore, so a SIMD pass adds to Total but not to
SimbaCore (`world.cpp` `advance_to`).

### How a stepped cycle works

Each cycle, every active port `propose()`s its next lane addresses to the `Fabric`; the
`Fabric` arbitrates one winner per bank (priority-round-robin, DMA-owned banks grant
nobody); granted ports `commit()`. A reader lands its granted reads one cycle later (+1cc
TCDM latency) and a group is consumable only once all its lanes have landed. A writer pops
a group once all its lanes are granted. The array/scan control logic pops inputs and pushes
outputs subject to FIFO back-pressure (a full output FIFO stalls the core → the input
readers), and the gauges (`g_r10_`, `g_r11_`, `g_iscore_`) advance as tiles/elements
retire. Conflict-free readers (residue-pinned gran-4 GEMM ports R1/R12/R13) never
self-collide, so the array is fed at 1 MAC-group/cycle; the SUC's gran-1 R7 dt_BC reader
*can* collide, and its conflict emerges from arbitration (see below).

## Accelerator busy-cycle model (ideal reference)

The MambaCore perf counter (`SIMBACORE_PERFORMANCE_COUNTER`) ticks every cycle
`globalState != sIDLE`. These MAC-count formulas are the *ideal*, conflict- and drain-free
cycle counts — the gauge totals and the timeline plot's `ideal` line. The engine's stepped
busy count equals them for conflict-free work and exceeds them where a bank conflict,
output drain, or pipeline fill applies. Dims are the *per-tile* values the app writes
(`D_INNER` = `M*_dInner_tile`):

```
M_i  = seqLen / 16            osN = K_i = dInner_tile / 24
osCore = M_i · osN · dModel                       (input-feed bound)
isCore = M_i · dFinal · K_i                        (dFinal = xProjDim in P1, dModel in P2)
conv   = seqLen · dInner_tile / 4                  (Phase 1 switchCore conv1d; convUnroll=delaySU=4)
matmul = seqLen · dInner_tile · dtRank / (4 · dtRankUnroll)   (Phase 2 switchCore dt projection)
SUC    = bc · seqLen · dInner_tile                 (Phase 2 only; bc = bank-conflict factor)
```

`run_invocation` computes `osc_dur_`/`isc_dur_` from these for the timeline `ideal` and the
gauge totals (`g_r10_total_ = M_i·osN`, `g_r11_total_ = seqLen·dInner_tile`,
`g_iscore_total_ = M_i·dFinal·K_i`). The stepped windows come from the engine.

- **Phase 2** (osCore → SUC → isCore): the app releases the SUC when the R10 (osCore z)
  gauge reaches `M2_R10_start_cnt`, and the isCore when the R11 (SUC y) gauge reaches
  `M2_R11_start_cnt` (shipped via `SNAX_DELAYED_START_R10/R11`). With full-serialize
  start_cnts (== gauge totals, `main`'s conservative default) the stepper reduces to strict
  osCore→SUC→isCore; a tiled app that paces them low overlaps the stages, and the overlapped
  busy count emerges from stepping. The switchCore dt projection (`switchCoreMode=Matmul`,
  `dt_delta = dt·Wᵀ + bias`) runs co-active — it streams `dt_delta` into the SUC scan, which
  can't scan past it, so the SUC is switchCore-paced. Its cycle count `sw_cyc` (the `matmul`
  formula above) is passed to `configure()`.
- **Phase 1 / IS_OSGEMM** (both cores, no SUC): osCore → switchCore conv → isCore stream as
  an on-chip pipeline; the perf counter spans until all three finish, so the slowest stage,
  the lead-in/drain, and cross-stage bank contention all emerge from stepping the shared
  fabric. The conv1d (`switchCoreMode=Conv`) runs `seqLen·dInner/4` and is the bottleneck
  when dModel is small (osCore/isCore cheap). IS_OSGEMM is the same pipeline with the conv
  stage absent.
- **OSGEMM / ISGEMM**: the array consumes one input group per cycle (conflict-free readers)
  and the writer output drain is stepped, so busy + drain fall out with no fill constant.

The gauges advance over their stage windows: `R10` (osCore tiles) reaches `M_i·osN` at
osCore-done, `R11` (SUC elements) reaches `seqLen·dInner_tile` at SUC-done, `ISCORE_TILE_CNT`
over the isCore window. These are exactly the thresholds the P2
`start_simbacore_and_streamers` polls wait on.

## TCDM bank conflict (the bc_pad effect)

The SUC dt_BC bank conflict comes out of the same per-cycle fabric (`cyc.cpp`,
`cyc_suc_duration`, and inside the engine's R7 hand-path), with no conflict constant. The
derivation has two halves.

1. **Demand = 1 BC group per SUC output, independent of dInner_tile.** The SUC
   (`StateUpdateCore.scala`) runs `for i in D/delaySU: for j in L: for k in delaySU`,
   emitting one `out_y` per iteration (`seqLen·dInner_tile` outputs) and pulling a fresh
   `in_vecB`/`in_vecC` (one `Vec(N)` each = 4 R7 groups) every `delaySU` iterations. The R7
   AGU pushes a TCDM read for **every** temporal point — including a **stride-0 dim, which
   re-reads the same address rather than reusing a buffer** (the AGU has an address FIFO, no
   data cache). So `∏ temporal_bounds = seqLen·dInner_tile` reads = exactly one BC group per
   output. The schedule's "reuse in D" is *logical* reuse in the SUC math, not a
   TCDM-bandwidth saving.
2. **Delivery rate = bank count, from real arbitration.** R7 is a gran-1 port
   (`[4,1]`) whose 4 spatial lanes can collide — unlike the residue-pinned gran-4 reader
   ports R1/R12/R13 (`[4,4]`), which cannot self-collide. Unpadded dt_BC strides `s=[128,256]`
   → lanes hit banks `{0,16,0,16}` (2 distinct banks); `bc_pad` strides `s=[160,320]` → banks
   `{0,20,8,28}` (4 distinct banks). The reader is modelled per-lane, not group-synchronously:
   each lane's `DataRequestor` issues its head address independently (`Reader.scala`,
   `DataRequestor.scala`), retrying on a bank conflict while lanes on free banks read ahead
   (bounded by the addr FIFO = 4 outstanding, and the response-side read-ahead = responser(4)
   + dataBuffer(4) = 8). Because the next refresh sits on **different** banks (the dim1
   temporal stride is +1 bank), those ahead-reads overlap the current refresh's tail. The
   result, straight from arbitration: bc_pad=0 → **≈1.75×**, bc_pad=4 → **1.0×**, flat across
   dInner_tile.

**vsim measurement.** main-tiled pad0, one P2 tile, reading `SIMBACORE_PERFORMANCE_COUNTER`
at the R10 (osCore-done) and R11 (SUC-done) gauge crossings: `osc=4869 suc=8091 isc=4726
total=17686`; the SUC is `8091 = 1.756×` the analytic compute (4608). The per-cycle sim
produces the SUC at ≈1.751× with no factor; `test/suc_grid_test.cpp` asserts the factor
equals the bank count (≈1.0× at bc_pad=4, ≈1.75× at bc_pad=0), flat across
dInner_tile ∈ {24,48,96}.

## DMA model (`DmaEngine`, `cyc.cpp`)

The DMA is a resident per-cycle beat engine, stepped on the same clock and same `Fabric` as
the accelerator streamers (`dma_engine_on_ = true` for every invocation). It drains queued
cluster-DMA transfers one 64 B beat at a time; a 64 B beat = 8 banks = one TCDM superbank,
and the `mem_wide_narrow_mux` (fixed priority, DMA wins) grounds the 8 narrow streamer
`q_ready` on the superbank the DMA presents a beat to. Each beat advances the address by
64 B (+1 superbank), so a streaming DMA walks the 4 superbanks. Each transfer becomes
drainable only at its submit cycle (`at_rel`), so a DMA issued mid-invocation collides only
from when the SW actually launched it. `dma_submit` enqueues the transfer live
(`engine_.dma_enqueue`), so DMA↔compute contention and the FIFO-slack latency-hiding both
emerge from arbitration — no per-app `dma_cycles` parameter.

`period` = cycles the backend holds a superbank per beat; for L3↔TCDM it is 1
(`beat_period_l3 = 1`, calib `dma_cost_per_beat_cc=1.0`). `MEMSIM_DMA_PERIOD=<p>` overrides
it (`=0` disables the engine, debug only). The chain-level first-beat latency (below) is a
separate `dma_submit` cost.

`dma_submit` also tracks the chain completion time: single channel, 64 B/beat; the first
transfer of a chain pays a first-beat latency, back-to-back transfers pipeline (bus-bound on
beats). `dma_busy` is 1 until the running completion time; `snrt_dma_wait_all` spins until
then. Calibrated by `calib/calibrate_dma.py` from the real per-cycle DMA trace:
**L3↔TCDM first-beat = 19cc**, TCDM↔TCDM first-beat = 10cc, ≈1 cc/beat.

## Performance optimizations (wall-clock)

The apps apply the optimizations in [08_performance_optimization.md](../../../../docs/dataflow/08_performance_optimization.md);
the model reproduces the two that move Snitch (wall-clock) time.

- **CSR pre-loading.** Each tile asserts `SIMBACORE_START` and *then*, while the
  accelerator is busy, writes the next tile's base-ptr/stride/bound CSRs (5 in P1, 12 in P2)
  before the `SIMBACORE_BUSY` poll. `World::snax_write_serializes(csr, at)` returns false
  (free) for a streamer-config write (`[960,1158]`) issued while `at < accel_end_` (core
  busy), and true (charge) for MODE/dim/START writes and any config write issued while idle
  (first-tile setup, post-poll re-launch). The accelerator perf counter is untouched — this
  is a Snitch-only effect.
- **DMA latency hiding (double-buffering).** Modelled implicitly: the engine steps the DMA
  live on the shared fabric while compute runs, so a double-buffered app's DMA hides under
  compute via the reader FIFO slack, and the app's own `DMA latency hiding: P1=ok/STALL`
  diagnostic (computed in-app from `snrt_mcycle`) comes out of the model verbatim.

The other doc'd optimizations need no model change: the safe-to-start gauges are the same
release logic the engine steps (and sweeps, below); inline start/wait is the same CSR
writes, free during the busy window; printf removal and data alignment don't affect the
interpreted instruction stream.

## Constants: derived vs residual

The model derives from RTL/architecture; vsim only *verifies*. Each constant lives in
`world.hpp`/`snax_csr.hpp` and is labelled derived (an RTL register/FIFO depth or geometry,
cited) or residual (a data-dependent effect not register-countable, kept as a clearly-labelled
minimal constant). The inter-stage handoff bubbles, pipeline fill/drain, and the safe-to-start
stale-read boundary carry no constant — they all emerge from the per-cycle engine.

| constant | value | class | basis |
|---|---:|---|---|
| `SNAX_CSR_WRITE_COST` | 1 | derived | `csrw` rd=x0 → no scoreboard wait; offload spill ready same cycle (`spill_register_flushable.sv`); ReqRspManager config-write `req_ready` combinational (`ReqRspManager.sv:161-166`). |
| SUC bank conflict | per-cycle (no constant) | derived | ≈1.75× (1.0× at bc_pad=4) emerges from the per-lane fabric — addr FIFO 4, response read-ahead 8, +1-bank dim1 refresh shift. |
| per-port streamer FIFO depth | see array | derived | `snax_streamer_depth` (`cyc.hpp`), verbatim from `cfg/snax_simbacore_cluster.hjson`: readers `[8,3,8,1,6,3,2,4,1,6,7,2,6,4]`, writers `[4,8,3,4]`. Heterogeneous; a uniform depth over-hides contention on shallow ports. |
| `dma_first_beat_l3` | 19 | derived (env-tinged) | iDMA backend req→AR→first-R/W (`axi_dma_backend.sv`) + 3 FE stages; 11 of the 16 is the simulated L3+xbar latency (environment constant, not iDMA depth). |
| `dma_first_beat_tcdm` | 10 | derived | TCDM↔TCDM first-beat. |
| `LOAD_USE_STALL` | 3 | derived | consumer of an in-flight load stalls: `register_core_req`(1) + TCDM Latency 1 (`RegisterTCDMCuts=0`) + `register_core_rsp`(1). Charged only to the first consumer (single-pending, `NumOutstandingLoads=1`). |
| `MUL_STALL` / `DIV_STALL` | 3 / 32 | derived / envelope | MUL = offload_req(1)+mul-stage(1)+offload_rsp(1). DIV/REM = serdiv `2+div_shift+2`, worst-case `div_shift=32` (exact value operand-dependent → residual). |
| `HW_BARRIER_RELEASE_COST` | 2 | derived | `snitch_barrier.sv` arrival latch(1) + `snitch.sv` csr_stall release(1), beyond the barrier CSR's own 1 cc. NrCores-independent. |
| icache-miss refill | — | residual (not added) | L1/L2/AXI refill latency = environment-dependent, not derivable from the core RTL. The bulk of the remaining Snitch-Total gap; deliberately not fudged. |

## Validation (real vsim builds)

main-tiled (seqLen=192 dModel=384 dtRank=24 nb_tiles=32), built both pad0 and pad4:

| metric | memsim | vsim | error |
|---|---:|---:|---:|
| Simbacore total (pad0) | 731,360 | 730,775 | +0.08% |
| Snitch Total (pad0) | 743,432 | 758,648 | −2.0% |
| P2 per-tile (pad0) | 17,685 | 17,686 | exact |
| — osc / suc / isc split | 4869 / 8090 / 4726 | 4869 / 8091 / 4726 | measured |
| Simbacore total (pad4) | 623,072 | 623,161 | −0.01% |

The per-tile osc/suc/isc split is the vsim perf-counter measurement the model is checked
against; the bc_pad=0 conflict comes from the per-cycle fabric, so pad4 (no conflict) and
pad0 use the same model with no separate factor.

main-tiled-oscore (async oscore_in ring; seqLen=192 dModel=384 nb_tiles=16 nb_l_tiles=12
nb_slots=2 bc_pad=4):

| metric | memsim | vsim | error |
|---|---:|---:|---:|
| Simbacore total | 606,448 | 612,289 | −0.95% |
| Snitch Total | 620,158 | 645,001 | −3.9% |

einfft-tiled (seqLen 392, dModel 192): Total 259,830 vs vsim 287,239 (−9.5%), SimbaCore
147,712 vs 152,826 (−3.3%). `nop.elf` boots and exits 0. The Simbacore residual is
un-modelled OSGEMM output-drain; the Total residual (~2–4%) is uncounted SW overhead (HW
barrier cost, icache-miss refill, scalar stalls) plus, on async configs, ring-overlap
effects not fully modelled.

## Correctness verdict and layout verification

The model does timing + integer/layout, **not** the bf16/fp8 datapath, so it does not fill
the apps' FP output buffers. Each app's own `check_result` (its `ref = N` / `N/M errors`
lines) therefore cannot pass under memsim — it is **suppressed by default**
(`MEMSIM_SHOW_APP_CHECK=1` to see it) and is **never the model's verdict**. The integer
layout/BIST check below always runs (no flag; `--timing-only` skips it); the verdict is
`[SUCCESS]` iff it passes and the program completed (a deadlock fails).

Two complementary layout checks run under `--verify` (on by default).

**(a) General AGU structural audit (`verify_layout`, all apps, golden-free, located).**
Runs on every invocation of any app with three pure-address checks:
- **bounds** — each enabled streamer's address extent must stay inside the 512 KiB TCDM
  `[0x10000000, 0x10080000)`.
- **producer→consumer containment** — a reader that strictly overlaps a writer must read
  only within what that writer produced (reading beyond = corruption).
- **writer no-alias (permutation)** — enumerate each writer's full emitted address sequence
  and flag the first word written twice (`W<n> ALIASES word <addr>`). Stride-0 dims are
  collapsed first (a stationary axis is the legitimate accumulation pass, e.g. the isCore
  K-reduction). This catches an in-bounds output permutation the bounds check cannot see.

Reported as `AGU layout audit: N located error(s) over M invocation(s)`; correct apps = 0.
`MEMSIM_LAYOUT_FAULT=1` injects a broken AGU (OOB reader) and a synthetic equal-stride
writer to prove both checks are live. Remaining gap: a stride wrongly 0 collapses an axis to
one address and reads as accumulation, so a bijective-but-wrong order needs the golden
round-trip below (mamba only).

**(b) osCore GEMM integer cross-check (`verify_datapath`, mamba only, FP-aware).** An
integer cross-check of the osCore GEMM (`z = A·B`, A=oscore_in/flattenA, B=oscore_weight,
z=W0/ConvFormat), carrying real data. It checks: R0 input AGU implements the documented
`flattenA`; W0 writes z contiguously in ConvFormat; the SUC z-reader R10 reads the buffer
W0 wrote; and a deliberately permuted gather changes the integer result (proving the check
discriminates). Runs once on the first P2 invocation; drives PASS/FAIL in `--verify` mode.

### Timing-coupled BIST (SUC dt_BC delivery)

`--verify` also runs a BIST (`cyc_suc_bist`, `cyc.cpp`) on the headline timing path: the
SUC's BC data flows through the per-cycle fabric + depth-4 data FIFO, and a group's data is
valid only after its +1cc TCDM response lands. It asserts three falsifiable properties:
1. **complete** — the SUC consumes every BC group with no deadlock.
2. **delay-respect** — with the correct +1cc latency the consumer never reads an un-landed
   (poison) slot.
3. **delay-sensitivity** — when the model is perturbed so the consumer treats a
   not-yet-landed read as ready (a delay modelled one cycle too short), it reads poison and
   the BIST catches it (`too-short-delay caught=YES`).

## FP32 functional datapath (`MEMSIM_DATAPATH`)

To verify memory layout and the safe-to-start substrate, the model recomputes each Mamba
kernel in FP32 through the *actual* streamer AGU layouts and compares to the app's FP8
goldens (`M2_oscore/suc/iscore_expected`): interpret golden bytes as FP8_ALT (e5m2) → FP32,
normalize a global requant scale (median model/golden), compare with a generous tolerance (a
wrong layout/stale-read gives wrong operands → large residual). Decoders: `fp.hpp`. Gated by
`MEMSIM_DATAPATH=1`.

All kernels validate on `main`: osCore 6144/6144, isCore 3072/3072, SUC 6065/6144 (98.7%,
scale 0.91; the residual is FP32 vs the BF16/FP16 softplus/silu/exp LUTs). The model doesn't
compute P1, so the SUC sources x (`M2_suc_x`) and dt/B/C (`M2_dt_BC`) from the P1-output
goldens; A/D/weights are DMA'd inputs. `cmp_fp32_golden` flags a scale-collapse (model→0) so
a degenerate "0≈0" can't false-pass.

**Scope: the non-tiled `main`.** A tiled app computes one slice per invocation and
accumulates the IS-core output in place across tiles (non-final tiles use no-requant, see
[04_mamba_main.md](../../../../docs/dataflow/04_mamba_main.md)), so one invocation's output
is not directly comparable to the complete golden. The layout formulas (flattenA/B,
ConvFormat, dt_BC) are app- and tile-independent — every tile runs the identical AGU program
with only the base pointer differing — so the integer osCore layout check on the first P2
tile of any app, plus the full FP32 datapath on `main`, already cover every layout the tiled
apps use.

Validated layout formulas:

- **osCore** `z=A·B`: A=`oscore_in`=`flattenA(N_M_K)` → `(m/16)*(K*16)+k*16+(m%16)`;
  B=`oscore_weight`=`flattenB(N_M_K, Ku=1, Nu=24)` = `((n/24)·K+k)·24+(n%24)`; out=ConvFormat.
- **isCore** `z=y·W`: A=y (ConvFormat), B=`iscore_weight`=`flattenB(K_M_N, Ku=24, Nu=1)` =
  `((k/24)·N+o)·24+(k%24)`; out=`flattenCD(K_M_N)` = `((m/16)·N+o)·16+(m%16)`.
- **ConvFormat** `convfmt(m,n)`: verified == `temporalToSpatialIdxConvFormat` inverse
  (Mu=16, colsPerTile=24, conv=4).
- **dt_BC** (P1 isCore out; dt=cols[0,24), B/C interleaved cols[24,152)):
  `flat=((m/16)*152+col)*16+(m%16); byte=(tile/8)*paddedMat + (wt/2)*16 + (wt%2)*8 +
  (tile%8)`, `paddedMat=128+bc_pad_banks*8`. Interleave: B[n]→col `24+(n/16)*32+(n%16)`,
  C[n]→col `24+(n/16)*32+16+(n%16)`.
- **switchCore** `dt_delta=dt·Wᵀ+bias`: `dt_weight_1/2`=`splitDeltaWeight` (Ku=6,
  convUnroll=4, dConv=4); bias=`dt_bias[d]` (fp8).
- **SUC recurrence** (`MambaLib.selectiveScan`): `dsp=softplus(dt_delta);
  h[d][n]=h[d][n]·exp(A·dsp) + (B·dsp)·x; y=silu(z)·(Σ_n h·C + x·D)`.

## Safe-to-start sweep (printed on every P2 run)

There is no analytic safe-to-start model. The engine steps osCore→z→SUC→y→isCore on the
fabric at fixed `(r10, r11)` gates (gating the SUC on `g_r10 ≥ r10` and the isCore on
`g_r11 ≥ r11`), and `result().stale_z`/`stale_y` count reads of a word its producer had not
yet committed (the exact per-cycle write/read schedules). A binary search finds the smallest
gate with zero stale reads — the minimum safe `start_cnt`. Because the write and read use the
real W2/R11 AGU reorders (the SUC writes y scattered while the isCore reads it linearly), a
same-order rate model would under-predict catastrophically.

On `main` (seqLen=64, dInner=96): **R10 (z, osCore→SUC) = 2/16**, **R11 (y, SUC→isCore) =
5377/6144**. The app's `get_safe_to_start_delay` ships 5 and 6144 (full-serialize) — both
conservative, leaving P2 overlap unused, which the sweep surfaces. An app start_cnt that
releases a consumer before its producer committed is flagged unsafe and fails the run (the
predicted gross iscore_out/SUC-y wrong-output fault).

**vsim check.** Sweeping the app's runtime `M2_R10_start_cnt`/`M2_R11_start_cnt` (rebuild SW
only, run vsim) for the smallest value that still produces correct output: R10 2 safe / 1
hangs; R11 5400 safe / 5300 hangs. The model's 2 and 5377 sit inside those brackets. The K_i
tile-quantization scales with shape (it is in the schedule).

## Engine activity timeline (`--timeline`)

`--timeline <file>` dumps one CSV row per active cycle window (`engine,start,end,ideal`) for
`OSCORE`/`ISCORE`/`SUC`/`switchCore`/`DMA` plus a `TCDM` traffic row, straight from the
engine's real per-stage windows. For an engine row, `ideal` is the shortest possible cycle
count for that block's compute (its MAC-group count at peak, conflict- and drain-free); for a
`TCDM` row it is the TCDM word-access count over the window. `plot_timeline.py` runs memsim
and plots it — one row per engine, cycle count on the x-axis, each row labelled with its
average hardware utilization = ideal/actual (a conflict-free array ≈100%; the SUC's dt_BC
conflict ≈57% at bc_pad=0, ≈100% at bc_pad=4). It also plots the per-cycle streamer-FIFO
occupancy (`<name>.fifo.csv`) and the safe-to-start hazard-vs-gate sweep (`<name>.s2s.csv`).

```
./bin/snitch_cluster.memsim sw/apps/<app>/build/<app>.elf --timeline tl.csv  # CSV only
python sim/plot_timeline.py sw/apps/<app>/build/<app>.elf                    # run + PNG
```

## Environment variables

- `MEMSIM_DEBUG` — run summary; `MEMSIM_ACC` — per-invocation accelerator durations;
  `MEMSIM_DMA` — first DMA transfers; `MEMSIM_ENGDBG` — engine layout/gauge/DMA trace;
  `MEMSIM_BANKHIST` — per-bank contention histogram.
- `MEMSIM_SHOW_APP_CHECK` — un-suppress the app's FP `check_result`.
- `MEMSIM_DATAPATH` — FP32 datapath vs goldens (non-tiled `main`).
- `MEMSIM_LAYOUT_FAULT` — audit liveness self-test.
- `MEMSIM_DMA_PERIOD=<p>` — override the DMA beat period (`=0` disables the engine).
- `MEMSIM_STREAMER_DEPTHS=<18 ints>` — override the per-port FIFO depths (R0..R13,W0..W3)
  for what-if sweeps without a rebuild; malformed values are ignored with a warning.

## Address remap / state carry

- The AGU XOR bank swizzle (`ADDR_REMAP_INDEX`, doc 22) is applied at every address→bank
  site (`agu_swz`: CycReader/CycWriter, `cyc_suc_duration`, engine `r7bank`), so swizzled
  apps see the de-correlated bank pattern, not the padded/conflicted one.
- SUC-only state carry (13_suc_carry.md): R13 streams `in_state` and W3 streams `out_state`
  concurrently with the scan (proportional pacing, R13 one group ahead), contending on the
  shared fabric. The residual serial exposure at small `L_tile` is not modelled → suc-carry
  C/D sit ≈ −7…−11 % vs vsim.
- IS_OSGEMM (R11 enabled): the isCore GEMM steps independently of the osCore stream and R11
  gates the array like any operand.

## Uncertainty / scope

- The SUC dt_BC bank conflict is modelled by the per-cycle per-lane fabric; the conflict-free
  GEMM readers (R1/R12/R13, residue-pinned gran-4) cannot self-collide, so their compute
  formula is the exact cycle count. **Known gaps:** the gran-1 output writers W0/W3 (the
  IS-core bc_pad concern) are not separately modelled; the icache-miss refill and other
  environment-dependent scalar overheads are the bulk of the remaining Snitch-Total gap and
  are deliberately not fudged.
- FIFO-depth what-ifs are trustworthy for the scan/P2 path (vsim-checked ≈ +1 %), but the
  model is **optimistic about shallow R0/R1/W0** under concurrent dual-GEMM + DMA overlap
  (vsim: dgcf ≈ +5 %, einfft-dcf ≈ +75 % at R0=2/R1=2/W0=1 where the model predicts ~0) —
  vsim-check any config that thins those ports below the current depths.
- Accelerator modes covered: Mamba P1/P2, OSGEMM/ISGEMM, and SIMD/FFT/einFFT (all stepped by
  the single engine on the shared fabric).
- Layout verification: the golden-free AGU audit runs on every app; the FP32 datapath
  cross-check runs on the non-tiled `main` under `MEMSIM_DATAPATH`.
