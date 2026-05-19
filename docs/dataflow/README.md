# SNAX Program Dataflow Reference

> **All pages:**
> **README (this page)** ·
> [1. OS-core kernels](01_oscore_kernels.md) ·
> [2. IS-core kernels](02_iscore_kernels.md) ·
> [3. SIMD / RMSNorm kernels](03_simd_kernels.md) ·
> [4. Mamba main (Phase 1 + Phase 2)](04_mamba_main.md) ·
> [5. FFT family](05_fft.md)

This folder describes, for every program under
[target/snitch_cluster/sw/apps/](../../target/snitch_cluster/sw/apps/), the
**high-level dataflow**: the order of accelerator stages, the tiling axis (and
why), and which buffers are reused vs. tiled between stages.

These docs are intentionally code-free. For byte-level tensor layouts see
[../../../chisel-ssm/docs/memory_layouts/](../../../chisel-ssm/docs/memory_layouts/README.md);
for the exact streamer/CSR programming see the program sources.

## Programs at a glance

| Program            | Kernel                                  | Tiled? | Tile axis            |
| ------------------ | --------------------------------------- | ------ | -------------------- |
| `osgemm`           | OS-core GeMM                            | no     | —                    |
| `osgemm-tiled`     | OS-core GeMM                            | yes    | N (dInner)           |
| `isgemm`           | IS-core GeMM (with / without requant)   | no     | —                    |
| `isgemm-tiled`     | IS-core GeMM                            | yes    | K (dInner)           |
| `simd`             | Every SIMD op (BF16 + FP8 coverage)     | no     | —                    |
| `rmsnorm`          | RMSNorm via 6 chained SIMD ops          | no     | —                    |
| `main`             | Mamba block: Phase 1 → Phase 2          | no     | —                    |
| `main-tiled`       | Mamba block                             | yes    | dInner (both phases) |
| `main-full`        | Same as `main`, different parameters    | no     | —                    |
| `suc`              | SU-core stand-alone (debug)             | no     | —                    |
| `fft`              | 2-way partitioned EinFFT                | no     | —                    |
| `fft-tiled`        | 2-way partitioned EinFFT                | yes    | Phase A: dModel; Phase B: K |
| `fft-3way`         | 3-way partitioned EinFFT                | no     | —                    |
| `fft-3way-tiled`   | 3-way partitioned EinFFT                | yes    | Phase A: dModel; Phase B: —; Phase C: K |

## Cardinal rules

These constraints are baked into the hardware and shape every program.

1. **Loop order is fixed.** OS-core uses `N_M_K`, IS-core uses `K_M_N`,
   Switch-core uses `N_M_K`. You cannot pick another order.
2. **OS-core output and Switch-core input are always in ConvFormat** — the
   reshuffle happens inside the chip, not in software.
3. **IS-core output bytes are sized as the accumulator type (BF16).** When
   FP8 requant is on, the FP8 value lives in the low byte of each BF16 slot.
4. **The IS-core requantizes on its last K iteration.** This is the reason
   accumulating tiled IS GeMMs must tile K, not N: tiling N puts a fake
   last-iteration in every tile and corrupts the requant timing.
5. **Inter-stage byte reorders must be folded into streamer programming.**
   Scalar reorders over TCDM via Snitch are too slow and are not used in any
   program here.
6. **Padding is the programmer's responsibility** unless an explicit hardware
   helper does it (e.g., `ISGEMM_SQ` pads the input internally; the weight
   must still be pre-padded).

## Recurring patterns

**Tiled-and-accumulating** (used in every IS-core-accumulating tiled program).
The accumulator stays FULL in TCDM and is updated in place across tiles.
Non-final tiles skip the requant; the final tile applies it. This works on K
but not on N (see rule 4).

**Multi-stage pipeline.** A `nb_tiles + (NB_STAGES − 1)` iteration loop fires
DMA-in for tile `i`, compute for tile `i−1`, and (where relevant) DMA-out for
tile `i−2` in lockstep, with the DM core and compute core 0 working in
parallel. The compute streamer is configured once with per-tile bounds; only
the moving base pointers change per tile.

**On-chip forwarding.** In Phase 2 the OS-core writes its output and the
SU-core reads it on the next cycle via an on-chip wire, bypassing TCDM. The
same pattern carries SU-core → IS-core. This is what lets one Phase-2 launch
run the OS-core, Switch-core, SU-core, and IS-core concurrently.

**Buffer reuse across phases.** In Mamba main, Phase 1's IS-core output
(`xProj`) doubles as Phase 2's Switch-core input (`dt_in`), and Phase 1's
Switch-core output (`conv_out`) doubles as Phase 2's SU-core input (`x`). No
re-DMA between phases.
