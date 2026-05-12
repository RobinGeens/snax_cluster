# 9. FFT — partitioned DFT on SimbaCore

> **All pages:**
> [README](README.md) ·
> [1. Concepts](01_concepts.md) ·
> [2. GEMM layouts](02_gemm_layouts.md) ·
> [3. ConvFormat](03_conv_format.md) ·
> [4. SUCFormat](04_suc_format.md) ·
> [5. xProj layout](05_xproj_format.md) ·
> [6. Delta weight split](06_delta_weight.md) ·
> [7. Per-mode reference](07_mode_reference.md) ·
> [8. Helpers](08_helpers.md) ·
> **9. FFT (this page)** ·
> [10. SIMD](10_simd.md)

This page is the SNAX programmer's guide to running an FFT on SimbaCore.
It documents the **partitioned DFT** (EinFFT) algorithm, which is the only
FFT flow the chip supports, and explains how the IS-core square mode
(`ISGEMM_SQ`), the SIMD core (`SIMD_CMUL_FP8`) and the helper memory
re-shuffles cooperate to compute a length-`L` complex DFT.

Source-of-truth files:

- Algorithm reference: [src/test/scala/simbalib/EinfftLib.scala](../../src/test/scala/simbalib/EinfftLib.scala)
- Python golden model: [src/test/reference/einfft_test_reference.py](../../src/test/reference/einfft_test_reference.py)
- Data generator (canonical layouts): [src/test/scala/datagen/DataGeneratorFFT.scala](../../src/test/scala/datagen/DataGeneratorFFT.scala)
- Test suite: [src/test/scala/simbalib/EinfftLibTest.scala](../../src/test/scala/simbalib/EinfftLibTest.scala)

## 9.1 Why partition the DFT

A length-`L` complex DFT computed naïvely is one `(2L × 2L)` real matmul.
On SimbaCore the IS-core square mode (`ISGEMM_SQ`) tops out at the
on-chip square tile (16×16-ish), so anything beyond a tiny DFT must be
factorised. EinFFT uses the **two-step Cooley–Tukey factorisation**

```
L = L1 · L2
```

and computes

```
X[k1 + L1·k2]  =  Σ_{m=0..L1-1} w(m,k2)        // size-L1 sub-DFT, "partition 1"
                  Σ_{l=0..L2-1} x[m·L2 + l]    //   inner loop over the L2 leading axis
                    · exp(-j 2π · m·k2 / L)    // Hadamard twiddle
                    · exp(-j 2π · k1·l / L2)   // size-L2 sub-DFT, "partition 2"
```

Each sub-DFT becomes a *real* `(2·Li, 2·Li)` matmul (the 2× factor packs Re
and Im as stacked rows). With reasonable `L1, L2`, both fit in `ISGEMM_SQ`.

The data axis `D` (e.g., `dModel`) is just batched along the matmul's
output dimension — every DFT sample becomes one column of the right-hand
matrix, so a single GEMM transforms the full `(L, D)` tensor.

## 9.2 The three steps

For a real input `x : (L, D)` (the SNAX flow does not currently handle
purely-complex inputs — pass an imag matrix of zeros if you need them):

### Step 1 — Partition 1 (size-L1 sub-DFTs)

```
weight   W1 = real DFT matrix of size L1, real-input variant
              shape (2·L1, L1)  =  [Re; Im] stacked vertically
input    rearrange x into "L1-leading, strided" layout
              shape (2·L1, L2·D)
output   X1 = W1 · input
              shape (2·L1, L2·D)  =  [Re; Im] for L1·L2 = L points across D channels
```

The `Re`/`Im` block layout means: rows `0..L1-1` carry the real parts of the
sub-DFT outputs, rows `L1..2·L1-1` carry the imaginary parts.

Computed on: **IS-core, `ISGEMM_SQ` mode**.

### Step 2 — Hadamard twiddle

```
input        X1 unstacked back into (L, D) complex   (= reals + imags)
twiddles     phase factors exp(-j 2π · m·l / L)   shape (L1, L2) -> flattened (L)
output       X2 = X1 ⊙ twiddles                    elementwise complex multiplication
                  shape (L, D)
```

Computed on: **SIMD core, `SIMD_CMUL_FP8` mode**, which uses the per-tile
"first 16 lanes are reals, next 16 are imags" packing — see [08](08_helpers.md).

### Step 3 — Partition 2 (size-L2 sub-DFTs)

```
weight   W2 = real DFT matrix of size L2, full complex variant
              shape (2·L2, 2·L2)  =  block matrix [[Re, -Im]; [Im, Re]]
input    rearrange X2 into "L2-leading" layout
              shape (2·L2, L1·D)
output   Xf = W2 · input
              shape (2·L2, L1·D)
```

Reconstruct the final complex result by interpreting `Xf[0..L2-1]` as the
reals and `Xf[L2..2·L2-1]` as the imags, then flatten using
`k = k1 + L1·k2`. (For the host: this is `recombine_L2` in `EinfftLib`.)

Computed on: **IS-core, `ISGEMM_SQ` mode** again.

## 9.3 Why the weight matrices differ

The two DFT kernels are not the same shape:

- **`W1`** has shape `(2·L1, L1)`. The input to step 1 is *real* — the
  full complex matrix is `[Re; Im]` stacked **after** the matmul. So step
  1's matmul is real, and `W1` is just `[Re(W); Im(W)]`. This is the
  `realInput=true` branch of `getDftMatrix` in `EinfftLib`.
- **`W2`** has shape `(2·L2, 2·L2)`. The input to step 2 is *complex*
  (post-twiddle), so the matmul takes `[Re(x); Im(x)]` and produces
  `[Re(X); Im(X)]`. This needs the full 2L × 2L block matrix
  `[[Re, -Im]; [Im, Re]]` — the `realInput=false` branch.

This asymmetry is invisible from the SNAX program's point of view (both are
just FP8 weights laid out in `ISGEMM_SQ` weight format), but it means the
two weight files have different sizes:

```
dft_weight1   :  (2·L1) × L1_padded             = 2·L1 × ceil_to(dInnerUnroll, L1)
dft_weight2   :  (2·L2) × (2·L2)_padded         = 2·L2 × ceil_to(dInnerUnroll, 2·L2)
```

## 9.4 ISGEMM_SQ matmul shape

In `ISGEMM_SQ` mode the IS-core treats:

- `A` as the **weight** (left-hand matrix of shape `(M·Mu, K·Ku)`)
- `B` as the **data**   (right-hand matrix of shape `(K·Ku, N)`)
- `D` as the **output** (shape `(M·Mu, N)`)

The hardware pads `B` along the `K`-tile axis at runtime (each
`seqLenUnroll`-sized tile is padded to `dInnerUnroll` per K-tile). **The
programmer must pre-pad `A` the same way** — that is why the data generator
calls `padMatrixColumnsPerBlock(..., seqLenUnroll, dInnerUnroll)` on each
DFT weight before laying it out. If you forget, the zero columns in B
won't line up with zero columns in A and you'll get garbage. See
`matmulKernel` in `EinfftLib` for the reference padding rule.

## 9.5 TCDM buffer layouts

From `DataGeneratorFFT.genData()` — these are what the streamer reads/writes.

| Buffer | Shape (logical) | Layout function | Notes |
|---|---|---|---|
| `dft_weight1` | `(2·L1, L1_padded)` FP8 | `flattenDftWeight1` → `padMatrixColumnsPerBlock` + `flattenConvFormat` | ConvFormat over the padded weight |
| `dft_weight2` | `(2·L2, 2·L2_padded)` FP8 | `flattenDftWeight2` → same pattern as `dft_weight1` | ConvFormat |
| `dft_in`      | `(L, dModel)` FP8 reals  | `flattenDftIn` → `stackLeading_L1(x, L1, L2)` then col-major | Input rearranged to "L1-leading, strided" before storing |
| `partition1_expected` | `(2·L1, L2·dModel)` FP8 | `flattenPartition1` → IS-core CD + `bankTranspose` + `padWithZeros` | Step-1 reference output. Bank-transposed because step 2 is the SIMD core consumer |
| `twiddles`    | `(L1, L2)` × `(cos, sin)` FP8 | `simdInterleaveRealImag(re, im)` | 16-real-then-16-imag tile packing — see [08.5](08_helpers.md#85-simd-realsimags-interleaving) |
| `hadamard_expected` | `(L, dModel)` complex FP8 | `flattenHadamardOut` → col-major + `simdInterleaveRealImag` | Direct SIMD output — reals/imags interleaved per 16-lane tile |
| `hadamard_reordered` | `(L, dModel)` complex FP8 | `flattenHadamardReordered` → col-major reals **then** col-major imags | "Re-block" of hadamard_expected — the form step 3 expects to read |
| `partition2_expected` | `(2·L2, L1·dModel)` FP8 | `flattenPartition2` → IS-core CD + `padWithZeros` (**no** `bankTranspose`) | Step-3 final output. No bank transpose because nothing consumes it on-chip |

Why `partition1` is bank-transposed but `partition2` is not:

- After step 1, the data is consumed *on-chip* by the SIMD core (step 2),
  which reads the IS-core output port. The streamer dispatches via banks,
  so the reference must apply `bankTranspose` to match what the SIMD core
  sees.
- After step 3, the data is the **final output** and is just read by the
  host. No bank dispatch happens.

## 9.6 Input layout — what `dft_in` actually contains

`flattenDftIn` does:

```scala
val transformed = einfft.stackLeading_L1(dftIn, L1, L2)
flattenMatrix(transformed, nRow=1, nCol=L2*dModel,
              nRowUnrolled=L1, nColUnrolled=1, ...)
```

`stackLeading_L1(x, L1, L2)` takes the `(L, D)` real input and arranges it
into an `(L1, L2·D)` matrix where each column of length `L1` holds elements
strided by `L2`:

```
column f·L2 + l  →   x[0·L2+l, f], x[1·L2+l, f], ..., x[(L1-1)·L2+l, f]
```

This is the "L1-leading, strided" layout that the size-L1 sub-DFT needs.
After `flattenMatrix(..., rowMajor=true, ...)` the buffer is just
column-major over `L2·dModel`, with each column being a contiguous run of
`L1` reals. **The DRAM layout for `dft_in` is purely `(L1·L2·dModel)` FP8
values in this stride-permuted order.**

⚠️ **The host must apply `stackLeading_L1` before writing `dft_in` to
TCDM.** The hardware does not do this re-shuffle. If you DMA the natural
row-major `(L, D)` tensor, partition 1 will compute the DFT of a permuted
sequence.

## 9.7 Step 1 → step 2 reordering — the bank-transpose trick

The output of `partition1` is laid out by the IS-core's bank dispatcher:
elements within one bank come from multiple temporal tiles. The SIMD core
expects to read its inputs in a different (stride-1) order.

Two operations bridge the gap:

1. **`bankTranspose`** in the reference flow (and the hardware bank
   transposer module on-chip) re-groups the banked output so consecutive
   elements within a bank come from consecutive tile rows.
2. **`unstackComplex_L1`** in the reference flow turns the `[Re; Im]`
   block layout back into "stride-1" complex `(L, D)` rows (i.e. samples
   appear in order `0, 1, 2, ..., L-1`).

In hardware these are the same physical re-shuffle, implemented by the
streamer between IS-core and SIMD core. The programmer just needs to
ensure the SIMD core's read pattern matches the bank-transposed pattern.
The data generator's `flattenPartition1` is the canonical byte order in
TCDM after step 1.

## 9.8 Step 2 → step 3 reordering

After the SIMD core multiplies by the twiddles, the data is in stride-1
`(L, D)` complex layout (`hadamard_expected` in the generator). Step 3
needs `(2·L2, L1·D)` `[Re; Im]` stacked rows — equivalent to a
permutation+stack:

```scala
buildStackedInput_L2(x2Re, x2Im, L1, L2, D)   // (2·L2, L1·D)
```

For the SNAX program, the practical recipe is the
`hadamard_reordered` buffer: take the complex output of step 2, write
**all reals as col-major** then **all imags as col-major**. That is
exactly the `[Re; Im]` stacked layout that step 3 reads via its B port.

This re-blocking is implemented by configuring the streamer's stride
pattern when reading from `hadamard_*` and writing to the IS-core's B
input — no explicit shuffle pass is required on the data, only address
generation.

## 9.9 Padding cheat sheet for FFT

`L1_padded = extendDim(L1, dInnerUnroll)` — the IS-core's `Nu` (output
channel) dimension demands a multiple of `dInnerUnroll`. For each weight
file, the second dimension is padded with zeros up to that multiple:

```scala
padMatrixColumnsPerBlock(weight, seqLenUnroll,
                         blockSize=seqLenUnroll, paddedBlockSize=dInnerUnroll)
```

This pads each `seqLenUnroll`-wide column block to `dInnerUnroll`. With the
default config (`seqLenUnroll=16`, `dInnerUnroll=24`), an `L1=16` weight
column becomes `24` columns (8 zeros tacked on).

Validity checks the data generator enforces (`validateParams`):

- `L1 > 0`, `L2 > 0`, `L1 * L2 == seqLen`
- `L1 % seqLenUnroll == 0`   (so the weight tile aligns)
- `seqLen % L1 == 0`, `seqLen % L2 == 0`
- `seqLen * inType.width % gemmWeightWidth == 0` — DFT input must fill the
  IS-core's wide input port cleanly

If you pick `L1, L2` outside these constraints the generator will exit;
the SNAX program would otherwise read the wrong addresses.

## 9.10 End-to-end SNAX program sketch

What the host program does, in order, for one DFT of `(L, D)`:

```
1. Pre-arrange input
   - On the host, apply stackLeading_L1(x, L1, L2) and DMA the result
     to dft_in.

2. Step 1: ISGEMM_SQ, size-L1 sub-DFT
   - A port: stream dft_weight1   (already padded, ConvFormat).
   - B port: stream dft_in        (the L1-leading layout).
   - D port: write partition1_buf in TCDM, in bank-transposed order
     (set the IS-core output streamer to bank-transpose mode).

3. Step 2: SIMD_CMUL_FP8, Hadamard twiddle
   - Inputs A and B: partition1_buf and twiddles (read in stride-1).
   - Output: hadamard_buf, in col-major real-then-imag form
     (alternatively: write to TCDM in the "Re-then-Im stacked" form
     by configuring the streamer's stride to mimic
     buildStackedInput_L2 directly).

4. Step 3: ISGEMM_SQ, size-L2 sub-DFT
   - A port: stream dft_weight2.
   - B port: stream the L2-leading form of hadamard_buf
     (= hadamard_reordered layout; configure streamer strides).
   - D port: write final output_buf.

5. (Optional) ortho normalisation by 1/sqrt(L)
   - Run SIMD_MUL_BF16_REQUANT (or any scalar multiply mode) with the
     constant 1/sqrt(L) broadcast across all lanes.
```

The data generator generates `expected` files for each intermediate
(`partition1_expected`, `hadamard_expected`, `partition2_expected`) so the
SNAX program can checkpoint after every stage and compare byte-for-byte
against the reference.

## 9.11 Choosing `L1, L2`

- Make both factors **as close to √L as possible**, so neither sub-DFT
  dominates the cost.
- `L1` must be a multiple of `seqLenUnroll` (= 16 in `MySystem`).
- Both `L1` and `L2` are eventually multiplied by 2 (the `[Re; Im]`
  stacking) and that 2× dimension must fit one IS-core square tile.
  Practically: `2·L1 ≤ seqLen` and `2·L2 ≤ seqLen`.
- `dft_weight1` size is `2·L1 × L1_padded`. `dft_weight2` size is
  `2·L2 × 2·L2_padded`. Pick the factors so neither weight dominates
  TCDM.

A worked example (`MySystem`, `seqLen=256, dModel=36`):

```
L1 = 16, L2 = 16   →  L = 256
L1_padded    = ceil(16, 24)  = 24
L2_padded    = ceil(16, 24)  = 24
dft_weight1  : 32 × 24  FP8
dft_weight2  : 32 × 48  FP8
```

This matches the configuration emitted by `DataGeneratorDefault`
(see [DataGeneratorDefault.scala](../../src/test/scala/datagen/DataGeneratorDefault.scala)).

## 9.12 Inverse DFT

Every helper in `EinfftLib` takes an `inverse: Boolean` flag. Forward and
inverse share the same byte layouts — the only difference is the sign of
the angle in the DFT matrices and twiddles. Generate fresh
`dft_weight1`/`dft_weight2`/`twiddles` buffers with `inverse = true` if
you want the host program to switch direction. The host code path,
streamers and core configurations do not change.

The Python reference `dft_partitioned_ref(...)` also takes `inverse`; use
it as the golden reference when validating an inverse-FFT flow.

## 9.13 Common pitfalls

- **Forgetting the input pre-shuffle.** `dft_in` is not `x.flatten`. Apply
  `stackLeading_L1` first. The hardware never does this for you.
- **Skipping the weight padding.** `ISGEMM_SQ` pads B at runtime but
  **not** A. Use `padMatrixColumnsPerBlock` on every DFT weight before
  flattening.
- **Writing partition1 without bank-transpose.** Step 2 reads via the
  banked port — if you skip the bank-transpose configuration on the
  output streamer, the SIMD core sees scrambled rows.
- **Writing hadamard buffer in interleaved form and then trying to read it
  as `[Re; Im]` stacked.** The two layouts are different. Pick one
  consistent convention (`hadamard_reordered` is the safer choice — reals
  fully come first, then imags) and configure the SIMD output streamer
  to write that pattern directly.
- **Mismatched `inverse` flag between weights and twiddles.** Both must
  use the same direction. The data generator handles this automatically;
  hand-rolled programs sometimes don't.
- **Asymmetric padding between weight and data.** `matmulKernel` in
  `EinfftLib` documents that "when A is per-K-tile column-padded, B must
  also be per-K-tile row-padded so that zero columns in A align with
  zero rows in B." The hardware already does the B padding for you in
  `ISGEMM_SQ`; just don't try to pre-pad B yourself — let `ISGEMM_SQ` do
  it and only pre-pad A.
