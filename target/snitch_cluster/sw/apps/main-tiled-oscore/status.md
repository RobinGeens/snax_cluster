# main-tiled-oscore — async oscore_in refill

**Status (2026-06-01): FIXED.** nb_slots=4 → 5/125 errors (quantization noise: z full-D 2/24576, 4 `iscore_out_P2` samples >1 LSB). Both phases' refill paces correctly.

## Goal
Keep `oscore_in` (the activation x, re-read in full by the osCore for every dInner tile) as a small TCDM ring of `nb_slots × (L_tile×dModel)` instead of the full `seqLen×dModel`. The R0 streamer walks the whole sequence by wrapping over the slots (stride-0 outer loop, `M*_R0_tb_lTile`); the DM core refills slots from L3 during compute, paced by polling `R10_DELAY_GAUGE` (osCore output-tile count). 1 refill per output tile keeps the ring `nb_slots` ahead of the read.

## Root cause (why it was failing)
The refill loop polls `while(R10 < (r+1)*gauge_step)` and refills 1 slot per osCore output tile. **For the pacing to work, the loop must start polling R10 _immediately_ after `start_simbacore`, while R10 ≈ 0.** Anything slow between `start` and the first poll lets the osCore run ahead; then R10 is already at its max (8) on the first poll, every poll passes at once, the refills burst, and the osCore has *already* read the slots → the refilled L-tiles are stale. Symptom: refilled tiles ~91% wrong (z 11300/24576), preloaded tiles perfect.

Two independent triggers of the same "delay before the first poll" bug:

| Phase | Delay between start and first poll | Fix |
|---|---|---|
| **P1** | `printf("Starting oscore_in refill…")` (slow `sys_write`, > whole osCore invocation) | **delete the printf** |
| **P2** | `start_simbacore_and_streamers(..., M2_R10_start_cnt=8, ...)` internally does `while(R10<8); DELAYED_START_READER_10=1` for the SUC z-reader — **blocks core 0 until R10=8** | **non-blocking start** (`_set_streamer_start();_set_simbacore_start();`) + **defer the SUC delayed-start release to AFTER the refill loop** (R10 is already 8 there = the same release point) |

(The printf was in every failing version incl. the `.bak`, so it was a cause the whole time. P2 had no printf but still broke — proving the root cause is the *delay*, not the printf specifically.)

### Caveat for future tuning
The P2 fix defers the SUC release because `M2_R10_start_cnt == nb_l` (release point == refill-loop end). **If `R10_start_cnt < nb_l`** (SUC meant to start mid-z-proj), deferring would start the SUC late — you'd have to **merge** the delayed-start into the refill loop (release the moment R10 crosses `start_cnt`). There's a code comment at the P2 deferred-start to flag this.

## Residual: slot count is a real lead-margin (NOT bandwidth)
Even with correct pacing, R10 (output count) trails R0 (input read) by ~1 L-tile (streamer FIFO depth 8 + array latency), and the per-L-tile DMA refill time is **comparable to** one tile of compute. So:
- **nb_slots=2**: paces correctly (R10 1→8) but still tears ~11% (42/125, z 2072/24576) — refill gets only ~1 tile of real lead, compute's read-pointer catches the DMA's write-pointer at the tail of each slot. **Not usable.**
- **nb_slots=4**: ~2–3 tiles of real lead → DMA always finishes first → clean. **Use this.**
- nb_slots=3: untested (likely the minimum; would save 1 × L_tile×dModel = 1.5 KB).

[RG] I do not agree with this assessment: 1 tile is 96cc, and the FIFO depth is only 8. This does not explain the incorrect lead time. TODO

This is a lead-margin, not a rate/bandwidth problem (the DMA has absolute TCDM priority, so it never starves; it even slows the accelerator's reads, widening the window). There is **no input-read gauge** (only R10/R11 = osCore/SUC *output*), so the refill can't be triggered earlier than `R10≥r+1` safely → the slot-count lead is how you buy margin.

## Key facts established
- `R10_DELAY_GAUGE` = `osCoreTileCnt.io.tick := osCore.io.data.out_d.fire` (MambaCore.scala:202) — pure osCore output count. No `in_a.fire` counter exists.
- R0 wrap/strides/scalars are **byte-identical** between osgemm-tiled-async (M3) and the full app P1/P2 (M1/M2): `tb=[96,4,2,1]`, `ts=[16,1536,0,0]`, gauge_step=1. So the bug was never the streamer bounds.
- TCDM = 32 banks × 64b = 256 B/cc; DMA = 512b/cc = 8 banks; `PriorityRoundRobinArbiter`, **DMA has absolute priority**.
- Hang fix prerequisite: `snax_intf_translator` id-FIFO (stale CSR-read response id) — separate, already fixed.

## Hypotheses tried (and discarded)
1. R10-poll deadlock → was the intf_translator id bug (fixed).
2. DMA-engine contention (1 iDMA queue) → refill loop issues no other DMAs.
3. Bank-bandwidth / TCDM over-subscription starves DMA → DMA has absolute priority; not rate-bound.
4. R10 lags R0 under conv backpressure (needs an input gauge / more slots) → **wrong root cause**; the real issue was the start→poll delay (sequencing), with slot-count only a secondary lead-margin.

## Verified parameters (current version)

Fixed dims for all rows: `dModel=96, dInner=192, dtRank=24, nb_tiles=8, seqLenUnroll=16, dInnerUnroll=24` (K_i=1). Compare FP8 with ±1-LSB tolerance.

**main-tiled-oscore** (full P1+P2 pipeline):

| seqLen | nb_l_tiles | L_tile | nb_slots | result | run |
|---|---|---|---|---|---|
| 128 | 8 | 16 | **4** | **refill CORRECT — 5/125, z 2/24576 (quant noise)** | betgtqnly |
| 128 | 8 | 16 | 2 | INSUFFICIENT — 42/125, z 2072/24576 (~11% tear on refilled tiles) | b31aoss2l |
| 128 | 8 | 16 | 3 | untested (likely the minimum) | — |

**osgemm-tiled-async** (isolated osCore GEMM, direct D verify), same dims:

| dim0 (seqLen) | nb_l | nb_slots | result |
|---|---|---|---|
| 128 | 8 | 4 | PASS (refilled tiles correct) — verified earlier |
| 128 | 8 | 2 | torn (62% / 34%) |
| 256 | 16 | 4 | HUNG (separate, unresolved) |

**Caveats / status of the verified result:**
- The **async refill logic is functionally correct at nb_slots=4** (z full-D clean, 2/24576; the 4 residual `iscore_out_P2` sample mismatches are >1-LSB requant noise). nb_slots=2 is genuinely insufficient (lead-margin), nb_slots≥4 needed.
- **Reliable full-run completion is currently gated by a separate, intermittent P2 SUC hang** (R11 not advancing at a later P2 tile = the known `vmamba_tiled_p2_tile1_hang`). `betgtqnly` completed because the (now-removed) R10 probe's extra cycles dodged it; the clean build hangs at P2 tile ~3 (run `byxpnette`). So the refill *logic* is verified; *every-run* completion is not yet guaranteed.
- **HW/build:** `snax_intf_translator.sv` id-FIFO fix RESTORED (it had been reverted) + clean full re-vlog (Errors: 0) → the P1 R10-poll hang is resolved and P1 fully completes. The fix is **uncommitted** — it must be committed to persist (the committed version is the broken one).

## Current config & open items
- Config: `seqLen=128, dModel=96, dInner=192, dtRank=24, nb_tiles=8, nb_l_tiles=8, nb_slots=4` → L_tile=16, dInner_tile=24, K_i=1. (nb_slots=4 = **half** the 8-tile matrix → real refilling of L4–L7, not "full matrix".)
- Open: (a) confirm at production dims (larger seqLen/dModel/dInner — needs TCDM budget check, iscore_out_P1/P2 are FULL); (b) test nb_slots=3 for the true minimum; (c) at seqLen≥256 (nb_l>8) confirm 4 slots sustains over many refills (osgemm-tiled-async hit a separate hang at nb_l=16 to chase).
