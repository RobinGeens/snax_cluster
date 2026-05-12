# SNAX Memory Layouts — Reference for the Programmer Agent

> **All pages:**
> **README (this page)** ·
> [1. Concepts](01_concepts.md) ·
> [2. GEMM layouts](02_gemm_layouts.md) ·
> [3. ConvFormat](03_conv_format.md) ·
> [4. SUCFormat](04_suc_format.md) ·
> [5. xProj layout](05_xproj_format.md) ·
> [6. Delta weight split](06_delta_weight.md) ·
> [7. Per-mode reference](07_mode_reference.md) ·
> [8. Helpers](08_helpers.md) ·
> [9. FFT](09_fft.md) ·
> [10. SIMD](10_simd.md)

This folder documents every memory layout that the SNAX programmer needs to
understand to feed and read data from the SimbaCore (which wraps the OS-core,
Switch-core, SU-core, IS-core and SIMD-core).

## Index

1. [Concepts and conventions](01_concepts.md) — banks, tiles, unroll factors, `MatmulDims`, FP types, requantization padding.
2. [GEMM layouts (OS-core and IS-core)](02_gemm_layouts.md) — `flattenA`, `flattenB`, `flattenCD`, and the role of loop order (`M_N_K`, `N_M_K`, `K_M_N`).
3. [ConvFormat](03_conv_format.md) — sub-tiled column-first layout used between OS-core ↔ Switch-core ↔ IS-core for time-domain data.
4. [SUCFormat](04_suc_format.md) — `delaySU`-strided layout used at the SU-core (state update) interface.
5. [xProj layout (`interleaveColsXProj`)](05_xproj_format.md) — how the `(dt, B, C)` projection output is interleaved, plus `splitXProj`.
6. [Delta weight split](06_delta_weight.md) — `splitDeltaWeight`: the `(dtRank, dInner)` matrix is split into a conv weight stream and a matmul weight stream.
7. [Per-mode layout reference](07_mode_reference.md) — quick lookup: for each `SimbaCoreMode` (`PHASE1`, `PHASE2`, `OSGEMM`, `ISGEMM`, `ISGEMM_SQ`, `SIMD_*`, …) which input/output buffer uses which layout.
8. [Helpers](08_helpers.md) — `bankTranspose`, padding (`padWithZeros`, `padMatrix*`), `simdInterleaveRealImag`, `duplicateEachElement`.
9. [FFT — partitioned DFT](09_fft.md) — how to run a length-`L` DFT as two `ISGEMM_SQ` matmuls plus one `SIMD_CMUL_FP8` Hadamard, with all buffer layouts.
10. [SIMD core layouts](10_simd.md) — lane ordering for `Add/Sub/Mul/Div/Sqrt/Noop`, the `[Re×N/2 | Im×N/2]` block layout for `CMul`, channel-wise reduction for `InProd/Rms`, `SoftShrink` and `Requant` post-ops.

## TL;DR — which format do I need?

| Buffer                                        | Format                                                                               | File                     |
| --------------------------------------------- | ------------------------------------------------------------------------------------ | ------------------------ |
| OS-core inputs A (activations, `dim0 × dim1`) | `flattenA` with `osCoreLoopOrder = N_M_K`                                            | [02](02_gemm_layouts.md) |
| OS-core weights B (`dim1 × dim2`)             | `flattenB` with `osCoreLoopOrder = N_M_K`                                            | [02](02_gemm_layouts.md) |
| OS-core output D (`dim0 × dim2`)              | **ConvFormat**                                                                       | [03](03_conv_format.md)  |
| OS-core bias C (only in OSGEMM mode)          | ConvFormat, in `accType` (BF16)                                                      | [03](03_conv_format.md)  |
| IS-core input A                               | ConvFormat (input comes from Switch-core or from a previous OS-core/SIMD output)     | [03](03_conv_format.md)  |
| IS-core weights B                             | `flattenB` with `isCoreLoopOrder = K_M_N`                                            | [02](02_gemm_layouts.md) |
| IS-core bias C and output D                   | `flattenCD` with `isCoreLoopOrder = K_M_N` (output in `accType`; FP8 cast is padded) | [02](02_gemm_layouts.md) |
| Phase-1 `(dt, B, C)` IS-core output (xProj)   | `flattenCD(...)` then **`interleaveColsXProj`** then `bankTranspose`                 | [05](05_xproj_format.md) |
| SU-core `x` input and SU-core output          | ConvFormat in memory (streamers convert ↔ SUCFormat at the SU-core port)             | [04](04_suc_format.md)   |
| SU-core `A`, `D`, `state`                     | row-major                                                                            | [04](04_suc_format.md)   |
| Switch-core delta-weight (`dtRank × dInner`)  | `splitDeltaWeight` → two flat buffers (`dt_weight_1`, `dt_weight_2`)                 | [06](06_delta_weight.md) |
| SIMD real-op inputs / outputs                 | flat tile-major, `N` elements per tile                                               | [10](10_simd.md)         |
| SIMD `CMul` inputs / outputs (FP8 only)       | per tile: `[Re×N/2 | Im×N/2]` block layout                                           | [10](10_simd.md)         |
| SIMD `InProd`/`Rms` inputs                    | `N` lanes = `N` independent channels reduced over `n_acc` consecutive tiles          | [10](10_simd.md)         |

## Cardinal rules

1. **The hardware loop order is fixed.** OS-core is `N_M_K`, IS-core is `K_M_N`,
   Switch-core is `N_M_K`. You cannot pick another loop order — picking the
   wrong one corrupts A or B silently. See [02](02_gemm_layouts.md).
2. **OS-core output and Switch-core input are _always_ in ConvFormat.** The
   reshuffle is done inside the chip
   ([MambaCore.scala:208](../../src/main/scala/mambacore/MambaCore.scala#L208)).
   When a buffer lives in TCDM and was produced by the OS-core or feeds the
   IS-core via the Switch-core, expect ConvFormat — see [03](03_conv_format.md).
3. **SU-core sees a `delaySU`-strided format at its port, but TCDM stores
   ConvFormat.** Streamers transpose on the fly; the binary file uses
   ConvFormat. See [04](04_suc_format.md).
4. **IS-core output bytes are always sized as `accType` (BF16).** If
   `en_isCoreRequant`, the FP8 values are written in the low half of each
   BF16 slot and the high half is zero-padded — use `padWithZeros` in
   the reference flow. See [02](02_gemm_layouts.md).
5. **`dim0` = sequence-length axis** (unrolled by `seqLenUnroll = Mu`).
   **`dim1` = reduction axis** (`Ku` for IS-core, `Nu` for OS-core wait — see
   [02](02_gemm_layouts.md) for the exact mapping of `dim*` ↔ `M/K/N`).
6. **Padding is the programmer's responsibility** unless an explicit hardware
   helper handles it (e.g., `ISGEMM_SQ` pads the input internally; the weight
   must still be pre-padded). See [08](08_helpers.md).
