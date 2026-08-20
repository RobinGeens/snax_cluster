# suc-carry — sequence-tiled SUC with on-chip state carry

> **All pages:**
> [README](README.md) ·
> [4. Mamba main](04_mamba_main.md) ·
> [9. Async tiling](09_async_tiling.md) ·
> [12. SUC async](12_suc_async.md) ·
> **13. SUC carry (this page)** ·
> [14. RMSNorm tiled](14_rmsnorm_tiled.md)

The stand-alone SUC kernel over a long `seqLen`, kept flat in TCDM footprint like
[suc-async](12_suc_async.md) but without the async `BC` ring. The sequence is split into
`nb_l_tiles` synchronous, plain L-tile invocations that hand the SSM hidden state to one another, so each
`BC` L-tile is fetched from L3 **once** (re-read on-chip across the `broadcast` channel-group passes)
instead of the `dInner/delaySU`-times L3 re-read the ring pays.

Two mechanisms make this cheaper than the ring:

- **State carry** keeps the hidden state on-chip between invocations, so a tile can resume where the
  previous one stopped.
- **Double-buffering** overlaps each tile's `BC`/`x`/`z` gather and the previous tile's `y` spill with the
  accelerator's compute, so the per-tile DMA does not sit on the critical path.

## State carry through the isCore streamer ports

The SUC exposes `in_state`/`out_state` (one `dState`-wide hidden-state vector per channel). In the
SUC-only modes the isCore is idle, so its `in_c` (R13) and `out_d` (W3) streamer ports are free and are
**re-used to move the state** to and from a TCDM `state_buf`:

- `sw_sucStateLoad` routes `in_state` from R13 (else the state resets to zero on `in_state`).
- `sw_sucStateSave` routes `out_state` to W3 (else the final state is dropped).

Both bits are two MSBs of the mode word, so existing encodings are unchanged; with both `0` the datapath
is identical to plain `M27_SUC_ONLY`. Three modes set them (`SimbaCoreMode.scala`):
`SUC_ONLY_STATE_SAVE` (`M34`, first tile — reset, save), `SUC_ONLY_STATE_CARRY` (`M35`, middle — load,
save), `SUC_ONLY_STATE_LOAD` (`M36`, last — load, drop). A state vector is `dState·accType` bits and
streams over the `serialWidthD`-wide port in `dState·accType / serialWidthD` beats; `state_buf` holds the
full `dInner` state in channel order, and R13 leads W3 by one channel group so each slot is read before it
is overwritten (in-place carry, no extra buffer).

## Dataflow

`nb_tiles == 1`: every invocation runs the full `dInner` over one L-tile (`L_tile = seqLen/nb_l_tiles`).

- **`BC` — resident per L-tile.** One L-tile of `BC` is gathered to TCDM and re-read `broadcast =
  dInner/delaySU` times from there (stride-0 outer loop). L3 traffic for `BC` is `1×`.
- **`x`/`z`/`y` — per-tile, ConvFormat-resident.** The slot holds one L-tile in ConvFormat
  (`[col-block][window][subtile]`, see [04](04_mamba_main.md) / [03](03_conv_format.md)), so each col-block
  is contiguous in both L3 and TCDM: gather/spill is **one 2D DMA of `dInner/dInnerUnroll` col-block bursts**
  per tensor, not one per broadcast index. The streamer does the ConvFormat → SUCFormat transpose in its
  address generator (col-block-major, subtile-within-col-block, then windows), so the DM core stays off the
  critical path even at small `L_tile` (where the per-broadcast-index gather would otherwise issue
  `broadcast` tiny strided transfers and become issue-bound).
- **`dt` — full-resident, unpacked.** `dt` is small and L-independent enough to stay resident; only its
  L-tile base offset advances per invocation.
- **`A`/`D`/weights** are resident (L-independent).

The default `M2_*` streamer bounds are the full-`dInner` P2 config; this app emits per-tile temporal
bounds (`M2_*_tb_sync` / `_ts_sync`) that read one resident L-tile, plus the R13/W3 state bounds.

## Double-buffering

`BC`/`x`/`z`/`y` use two TCDM slots. Per L-tile the accelerator computes on the current slot while the DM
core gathers the next tile and spills the previous tile's `y` into the other slot; a barrier closes each
tile. Compute (state carry serialises the tiles) is the long pole; because the per-tile gather/spill is a
handful of contiguous DMAs it stays a small fraction of it, so the tiles run back-to-back at the compute
rate and the DMA is hidden — including the large-`dInner` sizes where footprint forces a small `L_tile`
(many L-tiles).

## TCDM layout

One 64B-aligned chain; the peak is the sum of the resident operands plus the two rings:

```
dt → dt_w1 → dt_w2 → dt_bias → A → D → state_buf
   → BC[0] → x[0] → z[0] → y[0] → BC[1] → x[1] → z[1] → y[1]
```

Footprint is flat in `seqLen` (set by `L_tile`, not `seqLen`), so it scales to long sequences the same way
[suc-async](12_suc_async.md) does, while moving `dInner/delaySU`× less `BC` from L3.
