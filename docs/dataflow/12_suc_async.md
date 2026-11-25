# suc-async — multi-operand async ring on the isolated SUC

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
> [9. Async tiling](09_async_tiling.md) ·
> **12. SUC async (this page)** ·
> [14. RMSNorm tiled](14_rmsnorm_tiled.md) ·
> [20. Bank-conflict-free double GEMM](20_double_gemm_conflict_free.md) ·
> [21. Conv downsample (im2col GEMM)](21_conv_downsample.md)

`suc-async` runs the stand-alone SUC kernel (`M27_SUC_ONLY`), dInner-tiled, with **four** operands
kept in async TCDM rings instead of full-`seqLen` buffers: the `BC` input, the `x` and `z` inputs,
and the `y` output. `dt` stays full-`seqLen` resident.

The per-dInner weights/bias, `A` and `D` are tile-sized. The result is a TCDM footprint that is flat
in `seqLen` apart from the resident `dt` — the point of the rings. See [09](09_async_tiling.md) for
the ring mechanism itself (stride-0 wrap, gauge pacing, lead margin); this page only covers what is
specific to running several operands off one loop.

Resident `dt` dominates the budget at large `seqLen` — e.g. 960 KiB at `seqLen=32768`.

## One loop, one gauge, four operands

All rings are serviced by the single DM core from one refill loop (`suc_ring_loop`), paced by
one gauge — `R11_DELAY_GAUGE`, the SUC output count. Number the slot visits `r = 0, 1, …` with
`r = (subtile r/nb_l_tiles, L-tile r%nb_l_tiles)`. Per visit the DM core does four transfers:

- **gather `x`** for the future visit `r + nb_slots`,
- **gather `z`** for the future visit `r + nb_slots`,
- **spill `y`** for the just-finished visit `r` (to its ConvFormat slot in L3),
- **refill `BC`** for the future visit.

`x`/`z` are per-dInner-tile, so their future gather is guarded by `r + nb_slots < visits`; `y` is
fully spilled to L3, so there is no separate post-loop spill stage. The first `nb_slots` slots of
every ring are preloaded before the SUC is started, and the loop begins polling the gauge immediately
after the (non-blocking) start — the critical-sequencing rule from [09](09_async_tiling.md).

## Two slot granularities on the same schedule

The visit count (`nb_l_tiles · broadcast`) and gauge step (`L_tile · delaySU`) are shared by all
four rings, but the slots are **not** the same shape:

- **`BC` — full-L-tile slot.** The SUC re-reads `BC` identically across the
  `broadcast = dInner_tile / delaySU` channel-group passes, so a slot holds a whole L-tile and the
  refill only ever cycles L-tiles (idempotent across the broadcast passes — exactly the
  single-input ring of [09](09_async_tiling.md)).

- **`x`/`z`/`y` — compact per-subtile slot.** In ConvFormat the `convUnroll`-column subtile sits
  **outer** to the L sweep, so each subtile-pass touches **distinct** columns (unlike `BC`, which is
  re-read). So each `(L-tile, subtile)` is its own compact slot of `win_per_l_tile` subtile-windows
  (`xzy_l3_offset` maps a visit to its L3 chunk: col-block-major `(col-block, subtile)` decomposition
  plus the L-tile stride). The compact slot is one subtile, not a whole L-tile window — this is why
  the `x`/`z`/`y` loop structure differs from the `BC` loop.

## TCDM layout

One sequentially-64B-aligned chain, all live during the per-dInner-tile compute loop (so the peak
is their sum):

```
dt (full seqLen) → BC ring (nb_slots) → dt_w1 → dt_w2 → dt_bias → A → D
    → x ring (nb_slots) → z ring (nb_slots) → y ring (nb_slots)
```

`BC` and `x`/`z`/`y` are `nb_slots` slots each. (`memory_model.py` sizes and reports this and aborts
on TCDM overflow.)

## Lead margin scales with the operand count

Four transfers per visit is several× the DMA of the `BC`-only ring, so the *Sizing the ring* budget
from [09](09_async_tiling.md) is tighter: the per-visit DMA must still fit inside
`nb_slots · tile_period`. The law is unchanged — tear iff `nb_slots × tile_period` is below the
~250 cc L3→TCDM latency — only the per-visit byte movement is larger, so small tiles need more
slots while large tiles still hide it at `nb_slots = 2`. Measured at `dModel = 96` unless noted:

| `seqLen` / `nb_l_tiles` / `nb_slots` | `L_tile` | result |
|---|---|---|
| 512 / 8 / 4 | 64  | pass |
| 512 / 8 / 2 | 64  | tears (`y` wrong) |
| 512 / 16 / 2 | 32 | tears |
| 3136 / 4 / 2 | 784 | pass (`dModel = 48`, the default params) |

So at small `L_tile` (= 64) the four-operand ring needs `nb_slots = 4`, whereas the big-tile config
(`L_tile = 784`) holds at `nb_slots = 2` — the standard `nb_slots`/`L_tile`/`dModel` trade-off, just
shifted up by the extra per-visit DMA.
