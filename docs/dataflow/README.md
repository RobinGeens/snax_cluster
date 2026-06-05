# SNAX Program Dataflow Reference

> **All pages:**
> **README (this page)** ·
> [1. OS-core kernels](01_oscore_kernels.md) ·
> [2. IS-core kernels](02_iscore_kernels.md) ·
> [3. SIMD / RMSNorm kernels](03_simd_kernels.md) ·
> [4. Mamba main](04_mamba_main.md) ·
> [5. FFT family](05_fft.md) ·
> [6. EinFFT MLP](06_einfft_mlp.md) ·
> [7. VMamba SS2D](07_vmamba.md) ·
> [8. Performance optimization](08_performance_optimization.md) ·
> [9. Async tiling](09_async_tiling.md)

This folder describes, for every program under
[target/snitch_cluster/sw/apps/](../../target/snitch_cluster/sw/apps/), the
**high-level dataflow**: the order of accelerator stages, the tiling axis (and
why), and which buffers are reused vs. tiled between stages.

For byte-level tensor layouts see
[../../../chisel-ssm/docs/memory_layouts/](../../../chisel-ssm/docs/memory_layouts/README.md);
for the exact streamer/CSR programming see the program sources.

## App index

Every program under [`sw/apps/`](../../target/snitch_cluster/sw/apps/), grouped
by family. **Build** is whether the app is currently enabled in
[`sw/apps/Makefile`](../../target/snitch_cluster/sw/apps/Makefile) `SUBDIRS`
(**on**) or commented-out / unregistered (**off**); this changes often and says
nothing about correctness. **Detail** links to the page that describes the
dataflow. One-liners here are pointers only — the linked page is the single home
for each app's design.

### GEMM building blocks

| App | Build | What it does | Detail |
| --- | :---: | --- | :---: |
| `osgemm` | off | Single-shot OS-core `D = A·B` (ConvFormat output) | [1](01_oscore_kernels.md) |
| `osgemm-tiled` | off | OS GEMM, output **N-axis** tiled | [1](01_oscore_kernels.md) |
| `osgemm-tiled-async` | off | OS GEMM with the `A` input L-tiled into an async TCDM ring (input-side ring, paced by `R10`) | [9](09_async_tiling.md) |
| `isgemm` | off | Single-shot IS-core `D = C + A·B` (psum read-back) | [2](02_iscore_kernels.md) |
| `isgemm-tiled` | off | IS GEMM, **K-axis** accumulating tiles | [2](02_iscore_kernels.md) |
| `isgemm-tiled-async` | **on** | IS GEMM with the PSUM streamed through an async output-side ring (paced by `ISCORE_TILE_CNT`) | [9](09_async_tiling.md) |
| `is-osgemm` | off | OS + IS GEMM concurrently in one un-tiled `IS_OSGEMM` kernel | [9](09_async_tiling.md) |
| `is-osgemm-tiled` | off | Same, dInner-tiled and double-buffered | [9](09_async_tiling.md) |
| `is-osgemm-tiled-async` | **on** | Both async rings live at once (A-input refill + PSUM spill/reload), double-paced | [9](09_async_tiling.md) |

### SIMD / normalization

| App | Build | What it does | Detail |
| --- | :---: | --- | :---: |
| `simd` | off | Coverage test of every BF16 / FP8 SIMD op | [3](03_simd_kernels.md) |
| `rmsnorm` | off | RMSNorm fused into 6 SIMD launches | [3](03_simd_kernels.md) |
| `batchnorm` | **on** | Folded BatchNorm + ReLU (`ReLU(x·scale + shift)`) in 2 SIMD passes; SegFormer ConvModule tail | [3](03_simd_kernels.md) |

### Mamba main

| App | Build | What it does | Detail |
| --- | :---: | --- | :---: |
| `main` | **on** | Mamba block, Phase 1 → Phase 2, un-tiled | [4](04_mamba_main.md) |
| `main-full` | off | `main` with full-size params | [4](04_mamba_main.md) |
| `main-tiled` | **on** | Both phases dInner-tiled (`x`/`z`/`y` staged via L3) | [4](04_mamba_main.md) |
| `main-tiled-oscore` | **on** | `main-tiled` + async L-tiling of `oscore_in` (input ring, `R10`) | [4](04_mamba_main.md) · [9](09_async_tiling.md) |
| `main-tiled-iscore` | off (WIP) | `main-tiled` + async L-tiling of the IS-core output psum (output ring, `ISCORE_TILE_CNT`) | [9](09_async_tiling.md) |
| `suc-only` | off | Stand-alone SU-core probe / `BC` bank-conflict demo | [4](04_mamba_main.md) |

### FFT

| App | Build | What it does | Detail |
| --- | :---: | --- | :---: |
| `fft` | off | 2-way EinFFT partitioned DFT | [5](05_fft.md) |
| `fft-tiled` | **on** | 2-way DFT, Phase A dModel-tiled + Phase B K-tiled | [5](05_fft.md) |
| `fft-3way` | off | 3-way partitioned DFT | [5](05_fft.md) |
| `fft-3way-tiled` | **on** | 3-way DFT, per-phase tiling (dModel / un-tiled / K) | [5](05_fft.md) |

### EinFFT MLP

| App | Build | What it does | Detail |
| --- | :---: | --- | :---: |
| `einfft` | off | Un-tiled 2-layer EinFFT MLP (OS-core matmuls + SIMD fuse) | [6](06_einfft_mlp.md) |
| `einfft-tiled` | **on** | Same, OS-core **N-axis** tiled (per-tile SIMD fuse) | [6](06_einfft_mlp.md) |
| `einfft-tiled-is-osgemm` | **on** | Splits the 4 matmuls by complex side across OS + IS cores (`IS_OSGEMM`) | [6](06_einfft_mlp.md#69-dual-core-variant-einfft-tiled-is-osgemm) |

### VMamba SS2D

| App | Build | What it does | Detail |
| --- | :---: | --- | :---: |
| `vmamba` | off | Full SS2D forward (K=4 cross-scan → per-dir P1/P2 → cross-merge → RMSNorm) | [7](07_vmamba.md) |
| `vmamba-tiled` | off | dInner-tiled SS2D (single P1 → P2 pass) | [7](07_vmamba.md) |

### Microbenchmarks / diagnostics

| App | Build | What it does | Detail |
| --- | :---: | --- | :---: |
| `nop` | **on** | 1024 `nop`s — boot / baseline sanity test | — |
| `core-burner` | off | Runs a hardcoded mode that drives only the compute cores in isolation (activity probe) | — |
| `streamer-burner` | off | Runs a hardcoded mode that drives only the streamers in isolation (activity probe) | — |

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
