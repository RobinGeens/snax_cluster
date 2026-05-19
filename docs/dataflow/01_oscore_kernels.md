# 1. OS-core kernels: `osgemm` and `osgemm-tiled`

> Byte layouts of A, B, D: [memory_layouts/02](../../../chisel-ssm/docs/memory_layouts/02_gemm_layouts.md).

The OS-core computes `D = A · B` with no input bias. The output lands in
ConvFormat.

## `osgemm`

A single-shot OS-core launch.

| Tensor | Role            |
| ------ | --------------- |
| `A`    | activations (in)|
| `B`    | weights (in)    |
| `D`    | output          |

## `osgemm-tiled`

The same GeMM, but the **output (N) axis** is tiled.

| Tensor | Role                 | Lifecycle                                                          |
| ------ | -------------------- | ------------------------------------------------------------------ |
| `A`    | activations (in)     | **shared** — preloaded once, reused across all tiles               |
| `B`    | weights (in)         | **tiled** — per-tile slice, ping-pong'd                            |
| `D`    | output               | **tiled** — per-tile slice, ping-pong'd, reassembled into a FULL buffer by DMA-out |

**Why N, not K or M.** The OS-core has no psum read-back, so K-tiling would
need accumulation logic the hardware doesn't provide. M-tiling would force
per-tile A-tile loads and lose the "shared A" reuse. N-tiling fits cleanly
because each tile's output is independent.

**Pipeline.** Three stages overlap across tiles: DMA-in the next B-tile,
compute the current tile, DMA-out the previous tile's D-slice into the
reassembled buffer.
