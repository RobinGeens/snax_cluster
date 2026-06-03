# Async tiling of a shared tensor

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
(typically ≥ 3–4) are needed so the DMA reliably wins the race. There is no
input-read gauge, so the refill cannot be triggered earlier than the output
gauge allows — the slot count is the only knob for margin.

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

We need async tiling to support larger sequence lengths (not model sizes). If L becomes larger for given Dmodel, the
SUC will start taking longer than the GEMM cores. This relaxes our bandwidth requirements even further: we need to pace
the DMA refill based on the SUC throughput, not the GEMM throughput.

## Output-side ring (PSUM)

The same mechanism applies to isCore. Instead of ringing a shared
input, keep the full IS-core PSUM (`seqLen × dModel`) in L3 and slide the ring
through TCDM, paced by the IS-core output-tile gauge `ISCORE_TILE_CNT` instead of
`R10_DELAY_GAUGE`. Two differences from the input case:

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
