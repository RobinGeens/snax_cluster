# Async tiling of a shared input

When a streamer's input tensor is shared across many kernel calls (or across
many K-steps inside one call), it can dominate the TCDM budget. Real tiling
of the input is often blocked: a downstream stage carries cross-tile state,
or HW expects to see the full tensor in a single kernel.

**Async tiling** keeps the *logical* view of the full tensor but stores only
a small ping-pong of it in TCDM. The streamer wraps over the ping-pong; the
DM core refills slots in parallel with compute.

In short, we do tiling, but don't restart simbacore every time. the DMA needs to complete the tiling before simbacore actually reaches these values.
We cannot stop simbacore when doing this, so the DMA must ALWAYS complete in time.
The other way around: if the DMA is too fast, it will start moving data before it is completed. so we must still let the DMA wait on Simbacore (but not the other way around).

**Option 1** We need to use some hardware gauge to see how far Simbacore actually is in its computation, and let the DMA wait on that. That way, the DMA knows when it is safe to start again (using the assumption that DMA is always faster than compute) One such gauge is the R10_DELAY_GAUGE, which is incremented by the OS-core every time it streams out a tile.

**Option 2** We can stall Simbacore through the streamers: by putting smaller bounds on the streamers that serve the "async" tile, Simbacore will automatically stop until the streamers start again. We do not have to re-start Simbacore or change the algorithm CSR's (so no dModel_tile). We can then make sure that we only re-enable these streamers when the DMA is completed. This is only possible if the Streamer has an individual start signal, which is the case for the IS-core reader.

## Implementation concepts

**1. Make the streamer wrap with no HW change.**
Streamer AGUs only support non-negative strides, so a literal modular
wrap-around address is impossible. But an **outer loop with stride 0**
re-walks the inner address range from `base` again. Allocate two adjacent
slots in TCDM, set the next-outer loop bound to `nb_tiles/2` with stride 0,
and the streamer keeps reading from the same 2-slot region while iterating
through all tiles.

**2. Sync the refill with an existing output-progress counter.**
The HW already exposes a snitch-readable counter that ticks per producer
output unit (e.g. `R10_DELAY_GAUGE` for the OS-core). When the streamer
finishes reading slot *s*, the counter has crossed a known threshold —
snitch 0 polls it, hits a barrier, the DM core refills slot *s* with the
next tile, and the streamer can re-read slot *s* later in the same kernel.

Because the barrier only syncs the snitch cores, the accelerator never
stalls. Compute and DMA overlap.

## Refill schedule

Number the slot visits inside a kernel call `r = 0, 1, …`. Slot `r % 2`
must hold tile `r % nb_tiles` at visit `r`. After visit `r`, the same slot
is next re-read at visit `r+2` and needs tile `(r+2) % nb_tiles` — that is
**refill r**, fired once the counter exceeds `(r+1) * gauge_step`. Doing
`K_visits` refills folds the cross-kernel slot reset back to tile 0/1 into
the schedule.

## Constraints

`nb_tiles` even and ≥ 2 (so the wrap dim is integer). The per-tile size
must divide the producer's output-tile granularity so that `gauge_step` is
an integer.

## Implementation order

1. **Serialized** (debuggable first cut): snitch 0 polls → barrier → DM core
  issues DMA *and waits* → barrier → snitch 0 resumes polling. Compute is
  overlapped with DMA, but the *next* refill can't be issued until the
  current DMA drains.
2. **True async**: drop the DMA wait inside the loop; let refills queue up
  and only `snrt_dma_wait_all` at kernel-end. Correctness condition: each
  DMA must drain before the streamer re-reads its slot (slack window = one
  tile of streamer cycles). If it doesn't, the streamer reads stale bytes
  and you get **silently wrong output** — no hang to warn you. Always
  validate against the serialized version.
