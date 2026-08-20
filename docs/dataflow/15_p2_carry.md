# 15. p2-carry — sequence-tiled Mamba P2 with state carry, fused output projection

> **All pages:**
> [README](README.md) ·
> [4. Mamba main](04_mamba_main.md) ·
> [9. Async tiling](09_async_tiling.md) ·
> [12. SUC async](12_suc_async.md) ·
> [13. SUC carry](13_suc_carry.md) ·
> **15. p2-carry (this page)**

`p2-carry` runs the whole Mamba Phase 2 for a long `seqLen` as one program, replacing the two-launch
pipeline of [`P2-async-OS-no-IS`](09_async_tiling.md) (osCore input projection + SUC, async `BC`/`oscore_in`
rings) **plus** a separate output-projection GEMM. It does this by sequence-tiling P2 and handing the SSM
hidden state across L-tiles with the [suc-carry](13_suc_carry.md) mechanism, so the async rings are gone.

The premise that once forced the split — *"Split SSM in L is impossible because it requires the intermediate
SSM states to be read out and restored"* ([04](04_mamba_main.md) option A1) — no longer holds: state carry
through the isCore streamer ports (R13/W3) is exactly that read-out/restore.

## Two phases

**Phase A: P2 without the isCore, L-tiled with state carry.** Per L-tile the accelerator runs a
`PHASE2_NO_ISCORE` launch (osCore + switchCore + SUC), double-buffered:

- osCore projects `oscore_in[lt] → z[lt]` (W0), read back by the SUC as its gate input (R10) within the
  launch, paced by the R10 safe-to-start.
- switchCore + SUC produce `y[lt]` (W2); the hidden state loads from R13 and saves to W3.
- The mode word is `PHASE2_NO_ISCORE` OR-ed with the two state MSBs (`sw_sucStateLoad`/`sw_sucStateSave`):
  save on the first tile, load+save on the middle tiles, load on the last — the same save/carry/load schedule
  as [suc-carry](13_suc_carry.md).
- The DM core gathers the next tile (`oscore_in`, `dt`, `BC`, `x`) and spills the previous tile's `z`/`y`
  (convFormat) to L3 while the accelerator computes the current tile.

Because the state serialises the L-tiles, Phase A runs at the SUC rate with the osCore hidden underneath —
the same osCore↔SUC overlap `P2-async` had, but with `BC` fetched from L3 **once** instead of the
`dInner/delaySU`× re-read the ring pays.

**Phase B: output projection, K-tiled.** `out = y · iscore_weight (+ bias)` is an `ISGEMM` reducing over
`dInner`. Following [isgemm-tiled](02_iscore_kernels.md), the `out` psum stays **full-resident** in TCDM and
accumulates in place while `y` and the weight are streamed one contiguous `K`-tile (a `dInner` slice — `K`
is convFormat-outermost, so each slice is contiguous) at a time; non-final `K`-tiles run
`ISGEMM_NO_REQUANT`, the last applies the FP8 requant. The isCore input reader R11 is released at start (no
on-chip producer to pace it). Footprint is then `out`(full) + one `K`-tile of `y`/weight, so the whole
program fits wherever the full `out` psum (`seqLen·dModel·BF16`) fits; if even that does not, the program
**fails loudly**. `nb_op_k_tiles` (default 4) sets the number of `K`-tiles.

Tiling the **reduction axis `K`** is the supported isCore tiling — tiling the output `M`/`N` is not
(an M-tiled projection produces wrong output, because the isCore is built to accumulate over `K`, not to
emit a partial output range).

## Why not fuse the projection into the Phase-A launch

The SUC state carry and the isCore output projection both need the R13/W3 ports, so they cannot share a
launch — hence the projection is a separate phase (Phase B) after all L-tiles, reading `y` back from L3.
This loses the intra-launch SUC↔isCore overlap that full P2 had, which [04](04_mamba_main.md) option A4
already argues is marginal (safe-to-start dominates for small dInner tiles; the isCore is negligible for
small dModel).

## Footprint

**Phase A** is flat in `seqLen`: only `L_tile`-sized `oscore_in`/`dt`/`BC`/`x`/`z`/`y` slots
(double-buffered) plus the resident weights, `A`, `D`, and the full `dInner` state buffer — so the
state-carry P2 scales to long sequences the same way [suc-carry](13_suc_carry.md) and
[suc-async](12_suc_async.md) do. **Phase B** holds the full `out` psum plus one `K`-tile of `y`/weight (see
above). The binding constraint is therefore the `out` psum (`seqLen·dModel·BF16`) and — in Phase A — the
resident osCore weight (`dModel·dInner`); a very large `dModel` (e.g. the Simba-L CS block, `dModel=528`,
`dInner=1056`) exceeds the 512 KiB TCDM and stays on the tiled-`dInner` [P2-async](09_async_tiling.md).

## Mode

`PHASE2_NO_ISCORE_STATE_{SAVE,CARRY,LOAD}` (`SimbaCoreMode.scala`) = `PHASE2_NO_ISCORE` plus the state MSBs,
mirroring `SUC_ONLY_STATE_*`. The isCore is disabled in these modes, so its R13/W3 ports are free for the
state exactly as in the SUC-only carry modes.
