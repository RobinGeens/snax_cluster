# 23. Add tiled (bidirectional-scan merge)

> **All pages:**
> [README](README.md) ·
> [3. SIMD / RMSNorm kernels](03_simd_kernels.md) ·
> [7. VMamba SS2D](07_vmamba.md) ·
> [13. SUC carry](13_suc_carry.md) ·
> [14. RMSNorm tiled](14_rmsnorm_tiled.md) ·
> **23. Add tiled (this page)**

App: [`add-tiled`](../../target/snitch_cluster/sw/apps/add-tiled/).

Element-wise FP8 add `out = a + b` of two tensors too large for TCDM. It merges the two scan
directions of a bidirectional Mamba block (VideoMamba: `out_proj(y_fwd + flip(y_bwd))`) between the
per-direction SUC runs and the shared output projection. As in the VMamba cross-merge
([7](07_vmamba.md)), the flip of the backward output is a data-movement permutation and not part of
this kernel: the add sees both tensors in the same layout.

The add is element-wise, so the layout of `a` and `b` does not matter as long as they share it: both
are flat byte vectors of `nb_tiles` tiles of `numElem` elements (a multiple of 64 B).
Golden data comes from the SIMD generator (`DataGeneratorSIMD`, its FP8 add) for one tile; its
`numElem` and `n_acc` (unused here) are the app's data-shape params. Every tile streams that same
tile from L3, because the generator's run time grows quadratically with `numElem` (it indexes
linked lists for its complex and inner-product outputs); the TCDM pipeline and the DMA traffic are
those of `nb_tiles` distinct tiles.

## Tiling and double-buffering

Two `a` slots and two `b` slots are held in TCDM and ping-ponged, as in
[RMSNorm tiled](14_rmsnorm_tiled.md). While the SIMD core adds tile *n* in slot `n&1`, the DM core
DMAs tile *n−1*'s result out and tile *n+1*'s `a` and `b` in, all through slot `(n+1)&1`. The sum is
written in place into the `a` slot.

Each tile is one `SIMD_ADD_FP8` launch: R7 streams `a`, R13 streams `b` and W3 writes the sum, one
32-lane beat per port per cycle. The `b` slots sit at a `+32 B` offset so R13 does not hit the bank
group R7 reads in the same cycle.

Per tile the DMA moves three tiles (two in, one out) against one SIMD pass over one tile, so the
kernel is DMA-bound.
