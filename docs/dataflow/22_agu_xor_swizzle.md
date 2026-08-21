# AGU XOR bank swizzle

Runtime-selectable address remap in every streamer AGU that decorrelates TCDM bank
conflicts between concurrently sweeping streamer ports, with zero memory footprint.

## Hardware

Each reader/writer AGU has a per-port `ADDR_REMAP_INDEX_<PORT>` CSR (generated
`streamer_csr_addr_map.h`):

- `0` (reset default): identity — all existing apps are unaffected.
- `1`: XOR bank swizzle, applied to every generated address:

```
addr[7:5] ^= addr[10:8]
```

- `2`: half-preserving variant `addr[6:5] ^= addr[9:8]` — bit 7 (the skip-128
  partition bit) is untouched, so it swizzles within a 16-bank half.
- `3`: deep variant — the key XOR-folds every address bit above the bank group, so
  strides that are multiples of the plain swizzle's 2 KiB key period (e.g. the fft
  region-split ±8 KiB) still de-alias. Currently no in-tree user (see below).

TCDM is word-interleaved with `bank = addr[7:3]` (32 banks x 8 B). The swizzle rotates
the 32 B four-bank group within each 256 B TCDM row by the row index (bits [10:8],
period 2 KiB). Bits [4:0] are untouched, so `bank % 4` is preserved and every sparse
interconnect access granularity (<= 4 in the simbacore config) stays legal.

Implementation: sentinel entries `0` / `-1` / `-2` in `tcdm_logic_word_size` of the
cluster hjson (`snax_simbacore_cluster.hjson`) select the mapping functions in
`hw/chisel/src/main/scala/snax/readerWriter/AddressGenUnit.scala`. All 14 readers and
4 writers have `[256, 0, -1, -2]` (identity, swizzle, half swizzle, deep swizzle).

## Why it kills bank conflicts

Without the swizzle, equal-alignment streams (e.g. all buffers 256 B-aligned, each
port sweeping +4 banks/cycle) hammer the same bank group every cycle: is-osgemm-tiled
fused runs both GEMM cores at ~74% because ~22% of streamer requests lose arbitration.
Bank-phase padding cannot fix this (aggregate demand is conserved); partitioning fixes
it at the cost of layout rigidity and footprint (skip-128, see
[20_double_gemm_conflict_free.md](20_double_gemm_conflict_free.md)).

With the swizzle, the instantaneous bank group of a port depends on which 256 B row it
is in. Streams based at different `bits[10:8]` phases sweep pairwise-disjoint bank
groups while in lockstep, and decorrelate to ~1/8 collision probability otherwise.
Up to 8 concurrent streams can be given distinct phases.

## Software recipe

The swizzle moves data only within each aligned 256 B row, and it is an involution
(applying it twice is the identity). The DMA does not go through the AGU, so DMA
images must be pre-swizzled host-side; goldens stay logical.

1. Give every concurrently-live stream a distinct phase: place its buffer at byte
   offset `phase * 256 (mod 2048)` from a 2 KiB-aligned TCDM base. Ping-pong halves
   share their stream's phase. Buffer sizes and bases must be multiples of 256 B.
2. Pre-swizzle each DMA image in datagen with the same map, per DMA destination:
   byte at buffer offset `o` goes to offset `swz(base + o) - base` (32 B chunks move
   as a unit). A plain 1-D DMA then lands every element at its swizzled address.
   Tiled tensors are swizzled tile-by-tile against the ping-pong destination bases.
3. Write `ADDR_REMAP_INDEX_<PORT> = 1` for every port that touches swizzled buffers
   (after `set_streamer_csr`; the value persists across launches).
4. Result checks: outputs sit at swizzled addresses. Sample checks read the result at
   the swizzled index and the golden at the logical index (datagen emits both arrays,
   `M<id>_test_samples_<T>` and `..._swz`).
5. Keep read-modify-write pairs on the same buffer (R13/W3 psum) identity-mapped: the
   write-back trails the read by a few 256 B rows, so a swizzled pair lands in
   different rows -> different keys -> the pair collides with itself ~1/8 of the time
   that the identity map never does. Measured: swizzling R13/W3 too costs back half
   the win (4142 vs 3879 below).

Reference implementation: `sw/apps/is-osgemm-tiled` (the swizzle is its default and
only layout; ~12 KiB extra TCDM for phase alignment vs a packed layout).

## Measured result (vsim, seqLen=64 dModel=192 dInner=96 nb_tiles=2)

| config | Simbacore cc | wall cc | check |
|---|---|---|---|
| identity, packed layout (old default) | 4,451 | 5,518 | PASS 0/50 |
| swizzle on all 7 ports (incl. R13/W3) | 4,142 | 5,137 | PASS 0/50 |
| swizzle, psum identity (the default) | 3,879 | 5,070 | PASS 0/50 |

At the production shapes the win holds: in-proj A (784/96, dInner=192, nb4)
54,168 -> 48,243 (-10.9%); out-proj C (64/768, dInner=384, nb8) 70,835 -> 62,444
(-11.8%). Note the batch runs these shapes on double-gemm-conflict-free, which stays
faster still (~42.0k / ~54.2k) where its constraints fit (2-way partition, dModel a
multiple of 16, 2-D DMAs, 2x-max footprint); the swizzle is the drop-in win for the
plain fused app.

-12.8% Simbacore vs the fused baseline, zero extra data movement, plain 1-D DMAs.
Reference points: hard skip-128 bank partitioning reaches ~3,440 on this app
([20_double_gemm_conflict_free.md](20_double_gemm_conflict_free.md)) at the cost of a
rigid two-way layout; single-core ideal is ~3,200. The residual above partitioning is
statistical: streams at different byte rates (R0 16 B/cyc, R11/W0 8 B/cyc vs the
32 B/cyc weight/psum streams) drift through key alignments and collide ~1/8 of the
time regardless of phase choice. Same-rate streams with distinct phases stay
conflict-free while in lockstep.

## Replacing bc_pad_banks (SUC BC read, `main` with `bc_swizzle: 1`)

The SU-core R7 reads BC as a `[2,2]` spatial spread with a 16-bank stride, so two of
the four per-cycle banks alias (see [the padding fix](../../chisel-ssm/docs/memory_layouts/05_xproj_format.md)).
Under the swizzle the four spatial addresses always land in four distinct bank groups
(the spread spans consecutive 256 B rows, and `g ^ r` never collides across
`r, r+1, r+2` for offsets of 4), so the conflict dies deterministically — no padding,
no footprint. dt_BC is produced through the swizzle by P1 (W3/R13 + pre-swizzled bias
init) and consumed through it by P2 (R7/R2); P2's own psum R13/W3 goes back to
identity. `bc_swizzle: 1` in `main`'s params_in enables it (mutually exclusive with
`bc_pad_banks`).

Measured (vsim, seqLen=64 dModel=48 dtRank=24):

| config | P1 cc | P2 cc | dt_BC footprint |
|---|---|---|---|
| bc_pad_banks=4 (old fix) | 2,895 | 7,746 | padded (+bc_pad per matrix) |
| unpadded, no swizzle | 2,759 | 12,022 | unpadded |
| bc_swizzle=1 (this fix) | 2,869 | 7,757 | unpadded |

The swizzle matches the padded P2 cycles at the unpadded footprint. P1 pays ~110 cc
over the pure-unpadded run (the swizzled W3/R13 psum RMW self-collides occasionally)
but stays below the padded P1.

`bc_swizzle: 1` is the default across the bc_pad family: `main`, `main-tiled`
(P1+P2 11,938 vs padded 11,856, PASS 0/125 at 64/48/nb4), and the golden-dt_BC apps
`suc-carry` / `p2-carry` (dt and BC live in separate slots there, so only R7 reads
through the swizzle; main.c puts the BC slots on 2 KiB boundaries and datagen
pre-swizzles the BC windows of the L3 `dt_BC` image against those phase-0 slots).
suc-carry at production scale (3136/96/nb_l 14): PASS, 618,382 cc vs padded 623,641.

Exception: `vmamba-tiled` carries the swizzle machinery (inherited from main-tiled)
but ships with `bc_pad_banks: 4` / `bc_swizzle: 0`. At H12/W16/192 its y_norm check
is race-marginal regardless of layout: the unpadded no-swizzle control fails 20/50
samples including a zero-valued tear (torn z through the R10 release), and the
swizzle's timing shift wobbles that to 24/50 at matching cycles (793,130 vs padded
794,749). Flip it to the swizzle after the underlying R10 race is fixed.

The biggest single-app win is `P2-async-OS-no-IS`, whose BC ring alias had never been
padded: 171,460 -> 104,989 SimbaCore cycles (-38.8%, PASS 0/50 at 448/96/nb_l 4). Its
BC ring is pre-swizzled per ring position (slot = lt % nb_slots, which the refill
keeps stable; the ring base is 2 KiB-aligned). `main-tiled-oscore` (same
previously-unpadded family) also runs the swizzle now.

A note on release margins: pad-era hand-tuned safe-to-start values were tuned with a
conflict-free SUC and stay valid under the swizzle (blanket-widening them cost
main-tiled +23% before being reverted). Only apps whose tuning baseline was
BC-conflicted (p2-carry, main-tiled-oscore, P2-async) need the widened swizzle-aware
formula.

Two pitfalls found while porting:

- Removing the BC bottleneck speeds up the SUC front-end, so `z` consumption catches
  the osCore sooner in apps that produce z in-launch behind the R10 delayed start.
  Symptom: scattered all-zero `y` samples (`y = SiLU(z) * ...` with torn z). The
  safe-to-start formulas in `main`/`main-tiled`/`p2-carry` widen their margin when
  bc_swizzle is set; pad-era hand-tuned `safe_to_start_*` overrides should be retired
  or re-tuned.
- `suc-carry`'s hand-emitted W3 state spatial-stride array predated the W3 `[2,2]`
  spatial split and carried one stride instead of two — an illegal-bank-access fatal
  on any post-split RTL, independent of the swizzle (fixed: `[s, 2s]`).

## Negative result: composing with skip-128

The half-preserving variant (remap `2`) inside skip-128 halves was measured on the
double-GEMM app: 3,672 cc vs plain skip-128's 3,440 (PASS 0/50 both, so the mode is
functionally validated). It loses because the partitioned baseline's intra-half
residual is small and partly consists of same-rate streams in naturally conflict-free
phase (R12 vs the R13/W3 psum walk); the swizzle replaces those patterns with ~1/4
statistical collisions. Use the swizzle against heavy correlated hotspots (the +55%
BC alias, the ~26% fused-GEMM stall), not to shave an already-small (<~10%) residual.
`double-gemm-conflict-free` therefore stays pure skip-128, and remap index 2
currently has no in-tree user.

## Negative result: fft reorder aliases are not rate-limiting

Every fft app's reorder streams carry deterministic same-bank aliases (R7 gathers and
the W3_2 re/im split write, offsets differing by multiples of 256 B). An fft-tiled
A/B (deep swizzle on the on-chip partition1_out lifecycle + a 32 B re/im pad for the
spilled reorder tile) measured 25,989 vs 25,460 baseline (+2.1%, both PASS 0/50): the
CMUL pass consumes 16 B/cycle through its serializer, so the aliased 4-channel ports
run at half their peak demand anyway — the alias never stalls anything, while the
pass-1 psum pair pays the swizzled-RMW straddle. The experiment was reverted. Rule of
thumb: an alias found by stride inspection only costs cycles if the port is
throughput-saturated; check the consumer's beat width first.
