# 14. RMSNorm tiled (with residual)

Computes `out = rmsnorm(x) * weight + y` for arbitrarily large seqLen by tiling the sequence
axis. RMSNorm is per-token and the residual add is elementwise, so every L-tile is fully
independent.

## Tiling and double-buffering

The activation `x` and residual `y` (both seqLen × dModel, BF16) are split into `nb_tiles`
L-tiles (`L_tile` a multiple of `simdLanes_bf16 = 16`). Both are laid out group-major
`[group][D][lane]` (seqLen unrolled over 16 lanes), so an L-tile is a contiguous byte range.

Two x slots and two y slots are held in TCDM and ping-ponged. While the SIMD core processes tile
*n* in slot `n&1`, the DM core concurrently DMAs tile *n−1*'s result out and tile *n+1*'s x and y
inputs in (all via slot `(n+1)&1`), so all L3↔TCDM traffic hides behind compute. The result is
written in place into the x slot. The per-token `rms` scratch is a single compute-local buffer
(not double-buffered); `weight` and the `sqrt(D)` lane constant are resident. Output is staged to
an L3 buffer and verified against the golden at the end.


## Six SIMD passes per tile

The five RMSNorm passes (the `× 1/D` is folded into the reciprocal, see below) plus one big Add
for the residual. Each launch carries a large fixed overhead (streamer start + pipeline drain in
the busy-poll, dwarfing the CSR writes), so pass count is the main runtime lever after tile size.

| # | mode | size | operation |
|---|------|------|-----------|
| 1 | Rms  | LtD (big)  | `rms ← Σ x²` |
| 2 | Sqrt | Lt (small) | `rms ← √rms = √(Σ x²)` |
| 3 | Div  | Lt (small) | `rms ← √D / rms = 1/rms_true` |
| 4 | Mul  | LtD (big)  | `x ← x · rms` (1/rms_true broadcast over D) |
| 5 | Mul  | LtD (big)  | `x ← x · weight` (per-channel) |
| 6 | Add  | LtD (big)  | `x ← x + y` (elementwise residual) |

Net: `x · √D / √(Σ x²) · weight + y = x / √(mean x²) · weight + y = x / rms_true · weight + y`.



Two design points keep this fast and correct:

- The `× 1/D` pass is folded into the reciprocal by using `√D` (not `1.0`) as the Div
  numerator — a free 16-lane constant filled once. This removes one pass per tile at no per-tile
  cost.
- Sqrt and Div run only on the tiny per-token `rms` vector (Lt elems), never on the full x
  tile. They wrap an iterative unit (~10× slower per element and back-pressuring the streamer
  the whole time), so the full-tile work must stay cheap `Rms`/`Mul`.
