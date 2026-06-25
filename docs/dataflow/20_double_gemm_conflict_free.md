# 20. Double GEMM, bank-conflict-free

`double-gemm-conflict-free` runs the same two independent GEMMs in parallel as
[`is-osgemm-tiled`](../../target/snitch_cluster/sw/apps/is-osgemm-tiled). The two GEMMs share no data; their only coupling is
TCDM bank bandwidth. This app removes that coupling with a bank-partitioned memory layout, so the
two cores can never contend for the same bank.


## Disjoint bank halves (skip-128 layout)

Because the two GEMMs share no data, each can own half the banks. OSGEMM buffers are placed in
banks 0–15 and ISGEMM buffers in banks 16–31. Then no osCore streamer can ever address a bank an isCore streamer uses, so all
cross-core conflicts vanish; only each GEMM's own three/four streamers contend, within its own 16
banks.

Confining a buffer to one bank half requires a sparse **skip-128** layout, because a contiguous
buffer sweeps all 32 banks. Each 128 B logical block (the low half of a 256 B region = banks 0–15,
or the high half = banks 16–31) is placed on a 256 B centre; the opposite half of every region
belongs to the other GEMM. So a logical offset `L` maps to physical
`2·(L & ~127) + (L & 127)` within an arena whose base fixes `addr[7]` (0 for OSGEMM, +128 for
ISGEMM).

### Overlapping the two heaps

Each buffer still costs 2× its logical size, but the wasted half is exactly the half the *other*
GEMM owns. So the two heaps share the **same** base (ISGEMM at base + 128), overlapping the same
address region: OSGEMM fills the low halves, ISGEMM the high halves of the same 256 B blocks. They
can never share a bank, yet no half is wasted. The resident set is therefore `2·max(OS, IS)`, not
`2·OS + 2·IS` — when the two GEMMs are balanced this equals the non-partitioned `is-osgemm-tiled`
footprint. At `seqLen=64, dModel=192, dInner=96` it is **96 KiB** (OS heap 72 KiB, IS heap 96 KiB),
versus 168 KiB for two end-to-end heaps and 84 KiB for the unpartitioned baseline.

### Streamer addressing

The skip-128 map is realised purely in the streamer AGUs: the innermost loop
dimension is split so it sweeps exactly one 128 B block, then the block stride and every original
outer stride are doubled. For the AGU's linear addressing to reproduce the
skip-128 layout, two constraints must hold, which the datagen asserts:

- each streamer's contiguous beat run must split cleanly into 128 B blocks (`dim0_bound` divisible
  by `128 / beat_stride`);
- every outer stride must already be a multiple of 128. With `dInnerUnroll = 24` and the 256-bit
  weight width, this reduces to **`dModel` being a multiple of 16** (and the usual seqLen/dInner
  unroll-divisibility), plus every DMA'd buffer length being a multiple of 128 B.

### DMA

The DM core scatters contiguous L3 inputs into the skip-128 TCDM buffers and gathers the skip-128
outputs back to contiguous L3 with one 2-D transfer each (128 B blocks, 256 B / 128 B strides).
The osCore output `D_os` is spilled per tile; the isCore accumulator `CD_is` is gathered to L3 once
at the end for verification.


## The same partition applied to EinFFT MLP

`einfft-double-conflictfree` applies the identical skip-128 idea to the dual-core EinFFT
MLP of [`einfft-tiled-is-osgemm`](../../target/snitch_cluster/sw/apps/einfft-tiled-is-osgemm) (see
[6](06_einfft_mlp.md#69-dual-core-variant-einfft-tiled-is-osgemm)). 

The new wrinkle is the SIMD fuse that runs after the GEMM. It is single-array (no cross-core
contention, so its scratch/bias/output buffers stay contiguous), but it reads back the skip-128
`rr/ii/P` the GEMM left behind. The two readers that touch those buffers — the widen reader over
`rr/ii` and the folded imag-narrow reader over `P` — therefore walk skip-128 too (`R7_widen` and a
dedicated `R7_bf16_skip`). Because the skip-128 map is a pure function of the logical offset, a
reader striding differently from the writer still lands on the same physical bytes, so the read-back
is correct regardless of the writer's beat stride.


