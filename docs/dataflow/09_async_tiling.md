# 9. Async tiling of a shared tensor

> **All pages:**
> [README](README.md) ·
> [1. OS-core kernels](01_oscore_kernels.md) ·
> [2. IS-core kernels](02_iscore_kernels.md) ·
> [3. SIMD / RMSNorm kernels](03_simd_kernels.md) ·
> [4. Mamba main](04_mamba_main.md) ·
> [5. FFT family](05_fft.md) ·
> [6. EinFFT MLP](06_einfft_mlp.md) ·
> [7. VMamba SS2D](07_vmamba.md) ·
> [8. Performance optimization](08_performance_optimization.md) ·
> **9. Async tiling (this page)** ·
> [12. SUC async](12_suc_async.md) ·
> [14. RMSNorm tiled](14_rmsnorm_tiled.md) ·
> [20. Bank-conflict-free double GEMM](20_double_gemm_conflict_free.md) ·
> [21. Conv downsample (im2col GEMM)](21_conv_downsample.md)

When a streamer's tensor is shared across many kernel calls (or across many
K-steps inside one call), it can dominate the TCDM budget. Real tiling is often
blocked: a downstream stage carries cross-tile state, or the accelerator expects
to see the full tensor in a single kernel.

**Async tiling** keeps the *logical* view of the full tensor but stores only a
small **ring of `nb_slots` slots** of it in TCDM. The streamer wraps over the
ring; the DM core moves slots to/from L3 in parallel with compute. The
accelerator never sees a tiled tensor and is never restarted.


Because the accelerator cannot be stopped while this runs, the refill DMA must
*always* complete before the streamer re-reads a slot. The dependency is
one-directional: the DMA waits on the accelerator's progress (it refills a slot
only after that slot has been consumed), but the accelerator never waits on the
DMA. Two ways to enforce that:

**Option 1 (used here): pace the DMA refill with an output-progress gauge.**
The HW exposes a snitch-readable counter that ticks per producer output unit
(e.g. `R10_DELAY_GAUGE` for the OS-core). Snitch 0 polls it to learn when a slot
has been consumed and is safe to refill. Relies on the DMA being faster than
compute.

**Option 2: stall the accelerator through the streamer bounds.**
Put smaller bounds on the streamer that serves the async tile; the accelerator
auto-stalls until that streamer is re-enabled (no kernel restart, no algorithm
CSR change). Re-enable only once the DMA is done. Requires the streamer to have
an individual start signal.

## Implementation concepts

The input-side ring is implemented in `osgemm-tiled-async` (minimal single-osCore
GEMM) and `main-tiled-oscore` (the full P1 with `oscore_in` ring-tiled).

**1. Make the streamer wrap with no HW change.**
Streamer AGUs only support non-negative strides, so a literal modular
wrap-around address is impossible. But an **outer loop with stride 0** re-walks
the inner address range from `base` again. Allocate `nb_slots` *adjacent* slots
in TCDM (they must abut — the wrap walks them contiguously), set the wrap (outer)
loop bound to `nb_l / nb_slots` with stride 0, and the streamer keeps reading
from the same `nb_slots`-slot region while iterating through all `nb_l` tiles.

**2. Sync the refill with the output-progress gauge.**
When the streamer finishes reading a slot, the gauge has crossed a known
threshold — snitch 0 polls it, hits a barrier, the DM core refills that slot with
the next tile, and the streamer re-reads the slot later in the same kernel.
Because the barrier only syncs the snitch cores, the accelerator never stalls;
compute and DMA overlap.

## Refill schedule

Number the slot visits inside a kernel call `r = 0, 1, …`. Slot `r % nb_slots`
must hold tile `r % nb_l` at visit `r`. After visit `r`, that slot is next
re-read at visit `r + nb_slots` and needs tile `(r + nb_slots) % nb_l` — that is
**refill r**, fired once the gauge exceeds `(r+1) * gauge_step`. Running the full
`nb_l` (× K-steps) refills also folds the cross-kernel reset of the slots back to
their starting tiles into the same schedule.

## Sizing the ring (lead margin)

The gauge measures the producer's *output*, which lags its *input read* by the
accelerator's pipeline depth. So when refill `r` is authorized (output tile `r`
done), the streamer's input read has already advanced past tile `r`. The refill
must still finish before that slot's next read at visit `r + nb_slots`, so the
real slack is `nb_slots` tile-reads *minus* that output-vs-input lag.

Because the per-tile DMA time is comparable to one tile of compute (not far
smaller), **`nb_slots = 2` leaves almost no margin and tears** (the read pointer
catches the in-flight refill for the tail of a slot). A few slots of lead
(typically ≥ 3–4) are needed so the DMA reliably wins the race.

The margin is best read as an *absolute time budget*, not a slot count: a slot is
re-read after `nb_slots · (L_tile / Mu) · dModel` cycles of compute, and the
refill (≈ fixed `~250 cc` L3→TCDM latency + the byte movement) must fit inside it.
So `nb_slots`, `L_tile` and `dModel` trade off freely — at a large enough `dModel`,
even `nb_slots = 2` has enough cycles to hide the latency (verified: a config that
tore at `dModel = 192` passes unchanged at `dModel = 384`). `nb_slots` is just the
cheapest knob when `dModel`/`L_tile` are fixed by the workload. There is no
input-read gauge, so the refill still cannot be triggered *earlier* than the output
gauge allows. This only holds while the DMA is faster than compute per byte;
otherwise no slot count closes the gap.

## Critical sequencing requirement

The refill loop **must begin polling the gauge immediately after the producer is
started**, while the gauge is still ≈ 0. Any work between starting the producer
and the first poll — a print, a batch of slow CSR writes, a blocking wait on an
unrelated gauge — lets the producer run ahead unobserved. By the time the loop
polls, the gauge is already high, every poll passes at once, the refills burst,
and the producer ends up reading slots that were never refilled in time →
**silently wrong output**. If the producer's start is entangled with another
blocking gauge-wait (e.g. a dependent reader's delayed start), sequence that wait
*after* the refill loop, or merge it into the loop — never before the first poll.

## Constraints

- `nb_l` divisible by `nb_slots` (the wrap outer-loop count `nb_l / nb_slots`
  must be an integer).
- The per-tile size must divide the producer's output-tile granularity, so
  `gauge_step` is an integer.
- `nb_slots ≥ 3` in practice (see *Sizing the ring*); `nb_slots = nb_l`
  degenerates to "whole tensor resident, no refill".

## Scaling laws 

Async tiling targets larger sequence lengths, not model sizes. At larger `L` for a
given `dModel`, the SUC takes longer than the GEMM cores. This relaxes the bandwidth
requirement further: the DMA refill is paced on the SUC throughput, not the GEMM
throughput.

## Output-side ring (PSUM)

The same mechanism applies to isCore. Instead of ringing a shared
input, keep the full IS-core PSUM (`seqLen × dModel`) in L3 and slide the ring
through TCDM, paced by the IS-core output-tile gauge `ISCORE_TILE_CNT` instead of
`R10_DELAY_GAUGE`. Implemented in `isgemm-tiled-async` (minimal single-isCore
GEMM) and `main-tiled-iscore` (the full P1 with `iscore_out_P1` ring-tiled). Two
differences from the input case:

- **Two transfers per ring visit.** Because K (dInner) is the outer loop, every
  L-tile's running psum is revisited each K-step, so a visit must **spill** the
  slot to L3 *and* **reload** the next tile (the input ring refills only). The ring
  is still balanced — per visit the DMA moves `2 × slot_size` in ≈ one tile period —
  so the lead-margin rule from *Sizing the ring* still holds, with a tighter budget.
- **K must be tiled one step per invocation (`K_t′ = 1`).** Only then does the
  IS-core sweep every L-tile once, in order, within one invocation — which is what
  lets the ring slide mid-invocation. With `K_t′ > 1` the inner K-repeat re-touches
  all tiles each step, forcing the full psum resident. The SW loop over invocations
  *is* the K reduction.

### The final tile rides the ring too, assume requant + transpose off-chip

The `NO_REQUANT` accumulation tiles ride the ring cleanly: BF16 in, BF16 out, the W3 is a
straight psum write, so the output gauge and the committed write stay in lockstep. A requant final tile
does not fit this ring, for two independent reasons:

- **FP8-vs-BF16 rate mismatch.** The requant output is FP8 — half the BF16 psum the ring
  slots are sized for — so the array/W3 fills the ring at a different rate than the BF16
  output gauge (`ISCORE_TILE_CNT`) ticks. The gauge-paced spill, calibrated for the BF16
  accumulation, mis-captures the tail L-tiles (dense check: the first ~6 of 8 L-tiles
  clean, the last ~2 garbage).
- **Transpose scatter.** A requant and transpose*final tile additionally runs the
  stateful `BankTransposer` (an 8-bank scatter) which desyncs on the stride-0 rewind when
  L-tiles are small. Disabling only the transpose still leaves the FP8/BF16 rate mismatch.

## Both rings at once (dual-core, double-pacing)

`IS_OSGEMM` runs the OS-core and IS-core concurrently in one kernel, so both rings
can be live together: the OS-core A **input** ring (refill, gauge `R10_DELAY_GAUGE`)
and the IS-core PSUM **output** ring (spill+reload, gauge `ISCORE_TILE_CNT`), both
sweeping the same `seqLen` L-tiling and serviced by the one DM core. (Implemented in
`is-osgemm-tiled-async`. `K_t′ = 1` forces `nb_inv = dInner / dInnerUnroll`, with the
OS-core producing one `dInnerUnroll` N-slice per invocation; mode stays `NO_REQUANT`
so the OS-core requantizes its FP8 D while the IS-core PSUM rides the ring as BF16.)

The two cores generally run at different speeds, so the refill loop needs **two
independent cursors, each advanced by its own gauge** — *double-pacing*. Each pass:
poll both gauges, refill whichever ring(s) have a consumed slot (one, the other, or
both), advance only those cursors. Do **not** drive both rings from one shared cursor
gated on both gauges ("pace-on-slower"): that makes the *faster* core's refill wait
on the slower core, so the faster core wraps onto a slot that is still un-refilled and
tears. (Measured: pace-on-slower tears the OS-core input ring while the IS-core
passes; switching to double-pacing fixes it.)

Double-pacing removes the gauge-coupling tax but exposes a second one: a **single DM
core/DMA engine serves both rings**, so the faster core's bursty refills can queue
ahead of the slower core's transfers and delay them — pure DMA-bandwidth contention,
not gauge timing. The *lead-margin* rule above absorbs it; in practice the combined
case needs one more slot than each ring would alone (e.g. `nb_slots = 3` where an
isolated ring passes at 2), or equivalently a larger `dModel`. Diagnosis rule of
thumb: **input ring tears → gauge coupling (decouple the cursors); output ring tears
while input passes → DMA contention (deepen `nb_slots`).**

### Overlapping the osCore and SUC (P2-async)

`P2-async-OS-no-IS` runs the fused `M33` (osCore → switchCore → SUC) with two input
rings live at once — `oscore_in` (gauge `R10_DELAY_GAUGE`) and `BC` (gauge
`R11_DELAY_GAUGE`, the SUC output count) — from one double-paced loop,
so the osCore and SUC pipeline: the SUC's z-reader is released at the safe-to-start
threshold `M2_R10_start_cnt` and trails the osCore.

How much they overlap is set entirely by that threshold. The SUC reads z in a scrambled
per-`(seqLenUnroll × dInnerUnroll)` tile order, so it must trail the osCore by ~1 such
window: `safe_to_start = ceil(max(seqLen_tiles, suc_delta) · 1.2)`, clamped to
`gemm_total = seqLen_tiles · n_24L` (`get_safe_to_start_delay`), where
`n_24L = dInner_tile / dInnerUnroll`. Therefore:

- **`n_24L == 1`** → the clamp = the full osCore window → the SUC can only start once the
  osCore is done → **no overlap** (the loop degenerates cleanly to sequential). This is
  the regime a large-`seqLen` config is forced into when the 512 KiB budget caps
  `dInner_tile` at `dInnerUnroll` (e.g. `seqLen=3136, dModel=96 → nb_tiles=8`).
- **`n_24L ≥ 2`** → the SUC overlaps the later windows; overlap fraction ≈
  `(n_24L − 1.2) / n_24L` (n_24L=2 ≈ 40 %, 4 ≈ 70 %, 8 ≈ 85 %).

The value of that overlap scales with the osCore's share of the tile,
`osCore / SUC ≈ dModel / (384 · bc)` (L-independent; `bc ≈ 1.756` for the real 2-bank
conflict): `dModel=96` → osCore ~12.5 % of the tile (overlap buys ≤ ~4 %); `dModel=384`
→ ~36 % (~1.4×); balanced 50/50 near `dModel ≈ 674` (~2×). So overlap is worth pursuing
only when `dModel` is large **and** `dInner_tile` is large enough for `n_24L ≥ 2` — the
two pull against the footprint budget, so it is a moderate-`seqLen` feature.

The osCore array back-pressures cleanly, so a BC-refill DMA that transiently grounds the
osCore W0 (z-write) superbank only stalls it, never drops z — `oscore_in` refill already
overlaps W0 with clean z. The only real care is the standard double-pacing rule above:
the two-barrier handshake (publish `do_os/do_bc`, then "safe to recompute") plus enough
`oscore_in` slots to absorb the handshake latency.
