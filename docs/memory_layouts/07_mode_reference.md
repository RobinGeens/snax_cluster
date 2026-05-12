# 7. Per-mode layout reference

> **All pages:**
> [README](README.md) ·
> [1. Concepts](01_concepts.md) ·
> [2. GEMM layouts](02_gemm_layouts.md) ·
> [3. ConvFormat](03_conv_format.md) ·
> [4. SUCFormat](04_suc_format.md) ·
> [5. xProj layout](05_xproj_format.md) ·
> [6. Delta weight split](06_delta_weight.md) ·
> **7. Per-mode reference (this page)** ·
> [8. Helpers](08_helpers.md) ·
> [9. FFT](09_fft.md) ·
> [10. SIMD](10_simd.md)

For each `SimbaCoreMode`, this page lists every TCDM buffer the SNAX program
must produce or consume, with a pointer to the layout doc and to the exact
data-generator line that emits it. When in doubt, read the cited
`DataGenerator*.scala` block — it is the executable spec.

`SimbaCoreMode` definitions are in
[src/main/scala/simbacore/SimbaCoreMode.scala](../../src/main/scala/simbacore/SimbaCoreMode.scala).
Each mode is a packed bitfield (see `SimbaCoreCtrlBundle`).

## 7.1 `OSGEMM` — standalone OS-core GEMM

Generator: [DataGeneratorOSGEMM.scala](../../src/test/scala/datagen/DataGeneratorOSGEMM.scala).
Shape: `D[M·Mu, N·Nu] = A[M·Mu, K·Ku] · B[K·Ku, N·Nu] + C[M·Mu, N·Nu]`.

| Buffer | Type | Layout | Function |
|---|---|---|---|
| `A` | FP8 | OS-core A | `flattenA(aMatrix, osCoreLoopOrder)` |
| `B` | FP8 | OS-core B | `flattenB(bMatrix, osCoreLoopOrder)` |
| `C` | BF16 | **ConvFormat** | `flattenConvFormat(cMatrix)` |
| `D` | FP8 | **ConvFormat** | `flattenConvFormat(dMatrix)` |

Loop order: `N_M_K`. `dim0 = seqLen`, `dim1` = reduction, `dim2 = dInner`.

## 7.2 `ISGEMM` / `ISGEMM_NO_REQUANT` — standalone IS-core GEMM

Generator: [DataGeneratorISGEMM.scala](../../src/test/scala/datagen/DataGeneratorISGEMM.scala).

| Buffer | Type | Layout | Function |
|---|---|---|---|
| `A` | FP8 | **ConvFormat** | `flattenConvFormat(aMatrix)` |
| `B` | FP8 | IS-core B | `flattenB(bMatrix, isCoreLoopOrder)` |
| `C` | BF16 | IS-core CD | `flattenCD(cMatrix, isCoreLoopOrder)` |
| `D` | FP8 (requant) | IS-core CD + `padWithZeros` | `padWithZeros(flattenCD(...), accType, inType)` |
| `D_no_requant` | BF16 | IS-core CD | `flattenCD(...)` |

Loop order: `K_M_N`. `dim0 = seqLen`, `dim1 = dInner`, `dim2 = output channels`.

For `ISGEMM_NO_REQUANT`, only the BF16 view (`D_no_requant`) is meaningful;
the streamer keeps the wider stride.

## 7.3 `ISGEMM_SQ` / `ISGEMM_SQ_TRANSPOSE` — IS-core square mode

Used by the EinFFT path
([DataGeneratorFFT.scala](../../src/test/scala/datagen/DataGeneratorFFT.scala)).
Key feature: input axis is allowed to extend beyond `dInner`, and the
hardware handles input padding internally. Weight padding is still on you.

| Buffer | Layout | Function |
|---|---|---|
| `dft_weight1` | ConvFormat over the padded `(2·L1, L1_padded)` weight | `flattenDftWeight1` → `padMatrixColumnsPerBlock` + `flattenConvFormat` |
| `dft_weight2` | ConvFormat over padded `(2·L2, 2·L2_padded)` | `flattenDftWeight2` |
| `dft_in` | "L1-leading and strided", col-major flatten of `stackLeading_L1(x, L1, L2)` | `flattenDftIn` |
| `partition1_expected` | IS-core CD + bank-transposed + `padWithZeros` | `flattenPartition1` |
| `partition2_expected` | IS-core CD + `padWithZeros` (no bank-transpose) | `flattenPartition2` |
| `twiddles` | reals/imags interleaved per 16 lanes | `simdInterleaveRealImag` |
| `hadamard_expected` | col-major reals/imags interleaved | `flattenHadamardOut` |
| `hadamard_reordered` | reals then imags, col-major | `flattenHadamardReordered` |

Padding helper: `padMatrixColumnsPerBlock(x, rowDivisor=seqLenUnroll,
blockSize=seqLenUnroll, paddedBlockSize=dInnerUnroll)` — see
[08](08_helpers.md).

## 7.4 `PHASE1` — OS-core → Switch-core (conv + SiLU) → IS-core (xProj)

Generator: [DataGeneratorMain.scala::genData_PHASE1](../../src/test/scala/datagen/DataGeneratorMain.scala#L48).

Flow (semantically):
`oscore_in × oscore_weight → conv_out (ConvFormat) → conv1d + SiLU → iscore_in → iscore_weight → iscore_out (dt|B|C)`.

| Buffer | Layout |
|---|---|
| `oscore_in` (FP8) | `flattenA(osCoreIn, osCoreLoopOrder)` over `(seqLen, dModel)` |
| `oscore_weight` (FP8) | `flattenB(osCoreWeight, osCoreLoopOrder)` over `(dModel, dInner)` |
| `conv_weight` (FP8) | `convWeight.flatten` — row-major, `(dInner, dConv)` |
| `conv_bias` (FP8) | plain vector of length `dInner` |
| `conv_out` (FP8) | **ConvFormat** over `(seqLen, dInner)` |
| `iscore_weight` (FP8) | `flattenB(interleaveColsXProj(isCoreWeight), isCoreLoopOrder)` over `(dInner, xProjDim)` |
| `iscore_bias` (BF16) | `flattenCD(zeros, isCoreLoopOrder)` |
| `iscore_out` (FP8) | `flattenXProjOut(interleaveColsXProj(isCoreExpected))` (= **interleaved + flatten + bankTranspose**) |

`PHASE1_NO_REQUANT` is the same flow, but the IS-core's `D` is BF16 in TCDM
(no requant, no transpose).

⚠️ The IS-core's A input in Phase-1 comes from the conv output through the
Switch-core — it is in ConvFormat. You do not write A explicitly.

## 7.5 `PHASE2` — OS-core, Switch-core (matmul + bias), SU-core, IS-core

Generator: [DataGeneratorMain.scala::genData_PHASE2](../../src/test/scala/datagen/DataGeneratorMain.scala#L97).

| Buffer | Layout |
|---|---|
| `dt_BC` (FP8) | `interleaveAndFlattenDtBC(dt, B, C)` — see [05](05_xproj_format.md) |
| `oscore_in` (FP8) | `flattenA(osCoreIn, osCoreLoopOrder)` over `(seqLen, dModel)` |
| `oscore_weight` (FP8) | `flattenB(osCoreWeight, osCoreLoopOrder)` |
| `oscore_expected` (FP8) | **ConvFormat** over `(seqLen, dInner)` |
| `dt_weight_1` (FP8) | `splitDeltaWeight` stream 1 — see [06](06_delta_weight.md) |
| `dt_weight_2` (FP8) | `splitDeltaWeight` stream 2 |
| `dt_bias` (BF16) | vector of length `dInner` |
| `suc_state` (BF16) | `(dInner, dState)` zeros, `state.flatten` |
| `suc_A` (BF16) | `(dInner, dState)`, `A.flatten` |
| `suc_D` (BF16) | `(dInner)` |
| `suc_x` (FP8) | **ConvFormat** — comes from P1 OS-core; streamer converts to SUCFormat at the port |
| `suc_expected` (FP8) | **ConvFormat** — SU-core writes via streamer back to ConvFormat |
| `iscore_weight` (FP8) | `flattenB(isCoreWeight, isCoreLoopOrder)` over `(dInner, dModel)` |
| `iscore_bias` (BF16) | `flattenCD(zeros, isCoreLoopOrder)` |
| `iscore_expected` (FP8) | `flattenCD(isCoreExpected, isCoreLoopOrder)` (BF16-padded) |

`PHASE2_NO_REQUANT` differs only by `en_isCoreRequant = 0`.

## 7.6 `SUC_ONLY`

Same SU-core / Switch-core buffers as Phase-2 (`dt_BC`, `dt_weight_*`,
`dt_bias`, `suc_state`, `suc_A`, `suc_D`, `suc_x`, `suc_expected`). No OS-core
or IS-core involvement. Use this for SU-core isolation tests.

## 7.7 `SIMD_*` — SIMD core modes

Generator: [DataGeneratorSIMD.scala](../../src/test/scala/datagen/DataGeneratorSIMD.scala).

Buffers are flat vectors in the natural FP type of the mode (`accType` for
BF16 modes, `inType` for FP8 modes):

- `in_a_bf16` / `in_a_fp8` — input A
- `in_b_bf16` / `in_b_fp8` — input B
- `add_out_*`, `sub_out_*`, `mul_out_*`, `cmul_out_*`, `inprod_out_*`,
  `rms_out_*`, `div_out_*`, `sqrt_out_*`, `noop_out_*`, `softshrink_out_*`
- requantized variants suffixed with `_requant` (FP8↔BF16)

Special points:

- **`cmul`** (FP8 only): each `simdLanes`-element tile is laid out as
  `[Re_0, Re_1, ..., Re_{lanes/2-1}, Im_0, ..., Im_{lanes/2-1}]`. The output
  preserves this convention.
- **`InProd` / `Rms`**: input vectors are read in groups of `simdLanes` and
  accumulated over `n_acc` groups. The output length is `numElem / n_acc`
  (one element per accumulator span).
- **`softshrink_fp8`**: half-lanes carry the comparison signs; see
  the data generator's inline code for the exact element layout.
- `simdInterleaveRealImag(reals, imags)` packs reals and imags into the
  16-real-then-16-imag pattern used by the FFT path — see [08](08_helpers.md).

## 7.8 `SIMD_INPROD_BF16` driven by RMSNorm

Generator: [DataGeneratorRmsNorm.scala](../../src/test/scala/datagen/DataGeneratorRmsNorm.scala).

| Buffer | Layout |
|---|---|
| `x` (BF16) | IS-core CD over `(seqLen, dModel)` (`flattenIsCoreOut`) |
| `weight` (BF16) | per-channel weight duplicated `simdLanes_bf16` times (`duplicateEachElement`) |
| `out` (BF16) | IS-core CD over `(seqLen, dModel)` |

Even though the SIMD core consumes the data, the **memory format is still
the IS-core's output format** — the SIMD core sits downstream of the IS-core
in this dataflow, and the producer is what dictates the layout.

## 7.9 Complex MLP (`ISGEMM` reused)

Generator: [DataGeneratorComplexMlp.scala](../../src/test/scala/datagen/DataGeneratorComplexMlp.scala).

The complex MLP needs 4 independent `(L, D/4) × (D/4, D/4)` matmuls per
`multiply(...)`. Each branch is laid out **independently** in IS-core A/B/CD
layout, then the 4 buffers are **concatenated** into a single binary so the
SNAX program can step through them with branch-sized strides:

```scala
def flattenPerBranchA(x): Seq[Float] =
  x.map(branch => flattenA(branch, isCoreLoopOrder)).flatten
```

Bias is `(4, D/4)` laid out as 4 back-to-back D/4 vectors (the streamer
broadcasts over `L`, so no need to materialize the L dimension):

```scala
def flattenBiasPerBranch(bias: Matrix) = bias.flatten
```

Layer-1 outputs are written twice (BF16 view and FP8 view) — the FP8 view is
what feeds layer 2.

## 7.10 At-a-glance: which TCDM buffer uses which format

| Producer → Consumer | TCDM layout |
|---|---|
| Host → OS-core A | OS-core A (`flattenA`, `N_M_K`) |
| Host → OS-core B | OS-core B (`flattenB`, `N_M_K`) |
| OS-core D → host / Switch-core / IS-core | **ConvFormat** |
| Host → Switch-core conv weight | row-major flat |
| Host → Switch-core delta weight | **two streams** (`dt_weight_1/_2`) via `splitDeltaWeight` |
| Switch-core (conv mode) → IS-core | ConvFormat |
| Host → IS-core A (standalone ISGEMM) | ConvFormat |
| Host → IS-core B | IS-core B (`flattenB`, `K_M_N`) |
| Host → IS-core C (bias) | IS-core CD (`flattenCD`, BF16) |
| IS-core D → host (requant) | IS-core CD + `padWithZeros` (FP8 in BF16 slots) |
| IS-core D (xProj, Phase-1) → host | `interleaveColsXProj` + `flattenCD` + `bankTranspose` |
| OS-core / SU-core → host (Phase-2) | **ConvFormat** |
| SIMD core in/out | row-major flat, per-mode special packing for cmul/inprod/rms |
