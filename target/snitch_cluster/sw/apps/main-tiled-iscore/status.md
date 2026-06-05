# main-tiled-iscore — status

Async L-tiling of the isCore output psum. The full psum lives in L3; only an `nb_slots`-slot ring is
resident in TCDM, spilled and reloaded paced by `ISCORE_TILE_CNT`. Design, sizing and the commit-gauge
requirement are documented in `docs/dataflow/09_async_tiling.md` (output-side ring). Precondition:
`N_kern = 1` (`nb_tiles = dInner/dInnerUnroll`), so each kernel sweeps the L-tiles once.

## Current state (2026-06-05)

Phase 1 is done and verified; Phase 2 is the next step.

- `RUN_PHASE2 = 0` in `main.c`: only Phase 1's isCore output (dt+BC) is ringed; Phase 2 is skipped.
  The full P1 psum is spilled to L3 and checked directly there.
- Root cause of the long-standing "gross eviction" bug found and fixed in RTL: the gauge ticked on the
  pre-requant array output instead of the committed W3 write. Fix is a commit-side counter
  `isCoreCommitCnt` in chisel-ssm `MambaCore.scala` (see the doc and the `iscore_gauge_counts_prerequant`
  memory note). This needs a chisel-ssm commit+push for durability.

### Verified configs (P1 testbed, seqLen / dModel, nb_l_tiles / nb_slots → iscore_out errors)

| seqLen/dModel | nb_l / nb_slots | residency | result |
|---|---|---|---|
| 128 / 192 | 4 / 4 | full | pass 0/50 |
| 128 / 192 | 4 / 2 | half (evict) | 9 wrong |
| 128 / 192 | 8 / 4 | half (evict) | 1 wrong |
| 128 / 384 | 8 / 4 | half (evict) | pass 0/50 |
| 256 / 192 | 16 / 8 | half (evict) | pass 0/50 |

Takeaways: full residency (`nb_slots = nb_l`) always passes. Eviction needs slack on the requant final
tile, and more slots and/or larger dModel both supply it (the transposer makes P1's final-tile W3
bursty; see the doc). At seqLen=128 a half-residency ring at dModel=192 still tears by one (8/4) and
dModel=384 clears it; at seqLen=256 the half-residency ring (16/8, more and smaller slots) already
passes at dModel=192.

### Largest seqLen

The ring removes the isCore psum from the seqLen-scaling TCDM footprint (constant ring + full psum in
L3). The new ceiling is `oscore_in` (still full, `seqLen·dModel`): roughly seqLen≈384 at dModel=192 in
128 KiB TCDM, about 2× the pre-ring limit. For truly large seqLen, the osCore input also has to be ringed
(`main-tiled-oscore`).

## Next: Phase 2

P2's final tile requants but does not transpose (`PHASE2`: `en_isCoreRequant=1`, `en_isCoreTranspose=0`),
so it shares the same gauge root cause (now fixed) and its W3 should be less bursty than P1's — the
eviction ring may need less slack. Open items for P2: the R10/R11 safe-to-start pacing, and fitting the
full P1+P2 footprint (reduce seqLen so `oscore_in` + `dt_in` + the rings fit, or ring the osCore input too).

## Process notes

- The DM core (hart 1) cannot read CSRs; only core 0 may. Do not pace the refill loop on `snrt_mcycle`.
- The log only prints at phase boundaries, so a quiet log is not a hang; confirm via the `.dasm` traces.
- chisel-ssm scala edits must also be mirrored into `.bender/git/checkouts/chisel-ssm-<hash>/` and the
  generated `SimbaCore.sv` regenerated (`sbt "runMain simbacore.SimbaCoreEmitter"`); commit+push for durability.
