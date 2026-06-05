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
SUC    = bc · seqLen · dInner_tile      (Phase 2 only; bc = bank-conflict factor)
```
- **Phase 1** (osCore + isCore, isCore overlapped): `osCore + fill_p1`.
- **Phase 2** (osCore → SUC → isCore, **serialized** by the safe-to-start gauges
  whose thresholds equal the full osCore/SUC windows): `osCore + SUC + isCore + fill_p2`.

The gauges advance linearly over their stage windows: `R10` (osCore tiles)
reaches `M_i·osN` at osCore-done; `R11` (SUC elements) reaches `seqLen·dInner_tile`
at SUC-done; `ISCORE_TILE_CNT` over the isCore window. These are exactly the
thresholds the P2 `start_simbacore_and_streamers` polls wait on.

## TCDM bank conflict (the bc_pad effect)

The dt_BC read (SUC BC reader **R7**, a 2×2 spatial group) is **reused (stride-0)
across dInner**, so it is a per-seqLen-position demand. The SUC is therefore
either compute-bound or BC-read-bound:

> `SUC = seqLen · max(dInner_tile, W_bc / banks)`

- `banks` = the number of distinct TCDM banks R7's 4 lanes hit, computed per
  invocation from R7's **real captured strides** (`bank = addr[7:3]`): 2 with an
  unpadded dt_BC matrix stride (`bc_pad_banks=0`), 4 when bc_pad spreads them.
- `W_bc` = calibrated per-position BC demand (≈83).

This single model reproduces all three observed cases from the real strides + the
tile size, with **no hardcoded penalty and no flag**:
- **main-tiled pad0** (banks=2, dInner_tile=24): `max(24, 83/2≈42)=42` → BC-bound → conflict.
- **main-tiled pad4** (banks=4, dInner_tile=24): `max(24, 83/4≈21)=24` → compute-bound → no conflict.
- **oscore pad4** (banks=2, dInner_tile=**48**): `max(48, 42)=48` → compute-bound → no conflict.

The crucial insight: the *same* 2-bank R7 pattern conflicts in main-tiled (small
tiles) but not in oscore (big tiles hide the BC stall). bc_pad is an effect that
only bites when the BC read can't be absorbed by the per-tile compute.

## DMA model

Single channel, 64 B/beat. First transfer of a chain pays a first-beat latency;
back-to-back transfers pipeline (bus-bound on beats). `dma_busy` is 1 until the
running completion time; `snrt_dma_wait_all` spins until then. Calibrated from the
real per-cycle DMA trace by [`calib/calibrate_dma.py`](../../target/snitch_cluster/sim/calib/calibrate_dma.py):
**L3↔TCDM first-beat ≈ 19cc**, ≈1 cc/beat. (The "~250cc" in older notes is the
full issue→descriptor-FIFO→roundtrip chain, not the backend first-beat.)

## Calibrated constants

In `world.hpp` / `snax_csr.hpp` (tunable; ideally move to `calib/constants.json`):
`fill_p1 = 562`, `fill_p2 = 477` (lumps SU-core pipeline fill), `w_bc = 83`
(per-position BC demand for the bank-conflict model), `dma_first_beat_l3 = 19`,
`SNAX_CSR_WRITE_COST = 12` (offload latency, affects Total only). Each affects a
distinct, separable part of the result.

## Validation (real vsim builds)

main-tiled (seqLen=192 dModel=384 dtRank=24 nb_tiles=32), built both pad0 and pad4:

| metric | memsim | vsim | error |
|---|---:|---:|---:|
| **Simbacore total (pad0)** | **730,688** | **730,775** | **−0.03%** |
| **Snitch Total (pad0)** | **742,792** | **758,648** | **−2.1%** |
| **Simbacore total (pad4)** | **623,072** | **623,161** | **−0.01%** |
| **Snitch Total (pad4)** | **635,902** | **651,046** | **−2.3%** |
| **bc_pad 0−4 delta** | **107,616** | **107,614** | exact |

main-tiled-oscore (async oscore_in ring; seqLen=192 dModel=384 nb_tiles=16
nb_l_tiles=12 nb_slots=2 bc_pad=4):

| metric | memsim | vsim | error |
|---|---:|---:|---:|
| **Simbacore total** | **606,448** | **612,289** | **−0.95%** |
| **Snitch Total** | **620,158** | **645,001** | **−3.9%** |

`nop.elf` boots and exits 0. The Total residual (~2–4%) is uncounted SW overhead
(barrier hardware cost, icache, scalar stalls); the async residual also includes
ring-overlap effects not yet fully modeled.

## Layout verification (`--verify`)

`./bin/snitch_cluster.memsim <elf> --verify` runs an **integer** cross-check of the
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

## Uncertainty / scope

- `fill_p1/p2` and `w_bc (83)` are calibrated on this app/config; `w_bc` depends on
  the BC matrix geometry and may differ at other dims. A fully *emergent* per-cycle
  32-bank arbiter (incl. the wide/narrow DMA-vs-streamer superbank preemption) would
  remove the calibrated `w_bc` — the planned upgrade for exact cross-config timing.
- Accelerator modes covered: P1/P2 Mamba (M1/M28/M2/M29). FFT/einFFT/SIMD modes
  need their own rate formulas.
- Layout verification covers the **osCore GEMM** (real integer cross-check). The
  switchCore conv/matmul, SU-core selective scan, and isCore GEMM+requant+transpose
  are specified (operand routing + layouts) and are the documented next kernels to
  add the same int-gather→compute→scatter cross-check for.
