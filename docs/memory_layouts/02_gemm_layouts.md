# 2. GEMM layouts — OS-core and IS-core

> **All pages:**
> [README](README.md) ·
> [1. Concepts](01_concepts.md) ·
> **2. GEMM layouts (this page)** ·
> [3. ConvFormat](03_conv_format.md) ·
> [4. SUCFormat](04_suc_format.md) ·
> [5. xProj layout](05_xproj_format.md) ·
> [6. Delta weight split](06_delta_weight.md) ·
> [7. Per-mode reference](07_mode_reference.md) ·
> [8. Helpers](08_helpers.md) ·
> [9. FFT](09_fft.md) ·
> [10. SIMD](10_simd.md)

This is the layout used by every "plain GEMM" buffer in memory:
`A` (input activations), `B` (weights), `C` (bias / partial sum input),
`D` (output). The functions live in
[src/test/scala/utils/TensorUtils.scala](../../src/test/scala/utils/TensorUtils.scala)
(`flattenA`, `flattenB`, `flattenCD`, `flattenMatrix`).

A matmul is `D = A·B + C`, with shapes:

- `A`: `(M·Mu) × (K·Ku)`
- `B`: `(K·Ku) × (N·Nu)`
- `C`, `D`: `(M·Mu) × (N·Nu)`

The flattening always proceeds in **two nested levels**:

1. **Tile traversal** — iterating over the temporal indices (`M`, `K`, `N`)
   that *the hardware* uses.
2. **Within-tile traversal** — iterating over `(Mu, Ku)` or `(Ku, Nu)` or
   `(Mu, Nu)` depending on which matrix.

`flattenMatrix` is the generic helper:

```scala
def flattenMatrix(x, nRow, nCol, nRowUnrolled, nColUnrolled, rowMajor, rowMajorTile)
```

- `rowMajor = true`  → outer loop visits tiles in row-major order over the
  temporal grid.
- `rowMajor = false` → col-major over the temporal grid.
- `rowMajorTile`     → same choice *inside* a tile.

## 2.1 `flattenA(a, loopOrder)` — input activations

```text
M_N_K → flattenMatrix(a, M, K, Mu, Ku, rowMajor=true,  rowMajorTile=true)
N_M_K → flattenMatrix(a, M, K, Mu, Ku, rowMajor=true,  rowMajorTile=true)
K_M_N → flattenMatrix(a, M, K, Mu, Ku, rowMajor=false, rowMajorTile=true)
```

- For OS-core (`N_M_K`): tiles are row-major; A is read row by row, with
  the K-tile changing fastest along that row.
- For IS-core (`K_M_N`): tiles are col-major across the temporal grid; A stays
  stationary for `N` iterations.

## 2.2 `flattenB(b, loopOrder)` — weights

```text
M_N_K → flattenMatrix(b, K, N, Ku, Nu, rowMajor=false, rowMajorTile=false)
N_M_K → flattenMatrix(b, K, N, Ku, Nu, rowMajor=false, rowMajorTile=false)
K_M_N → flattenMatrix(b, K, N, Ku, Nu, rowMajor=true,  rowMajorTile=false)
```

Note `rowMajorTile = false` for B in every case — within a tile, B is
**column-major** (Nu is the inner index, Ku is the outer).

## 2.3 `flattenCD(d, loopOrder)` — bias C and output D

```text
M_N_K → flattenMatrix(d, M, N, Mu, Nu, rowMajor=true,  rowMajorTile=true)
N_M_K → flattenMatrix(d, M, N, Mu, Nu, rowMajor=false, rowMajorTile=true)
K_M_N → flattenMatrix(d, M, N, Mu, Nu, rowMajor=true,  rowMajorTile=true)
```

The `M_N_K`/`K_M_N` cases are identical here: D is produced row by row in both,
because the output is M·N tiles independent of K. The `N_M_K` case lays D out
col-major (since D is produced column by column).

## 2.4 OS-core GEMM (`SimbaCoreMode.OSGEMM`)

OS-core uses `osCoreLoopOrder = N_M_K`. Mapping:

- `dim0 = M·Mu` = sequence length
- `dim1 = K·Ku` = OS-core reduction axis (K_u = 1, so K is the full inner dim)
- `dim2 = N·Nu` = `dInner` axis

The data generator
([DataGeneratorOSGEMM.scala](../../src/test/scala/datagen/DataGeneratorOSGEMM.scala))
writes:

```scala
writeFloatToFile("A", mode, flattenA(aMatrix, params.osCoreLoopOrder))   // FP8
writeFloatToFile("B", mode, flattenB(bMatrix, params.osCoreLoopOrder))   // FP8
writeFloatToFile("C", mode, flattenConvFormat(cMatrix), accType)         // BF16, ConvFormat
writeFloatToFile("D", mode, flattenConvFormat(dMatrix))                  // FP8, ConvFormat
```

⚠️ **The OS-core output D is in ConvFormat, not in `flattenCD` format.**
This is because the OS-core's output goes through a reshuffle
(`rowMajor2ConvFormat` in
[MambaCore.scala:208](../../src/main/scala/mambacore/MambaCore.scala#L208))
before it reaches TCDM. The bias C, which is loaded into the accumulators,
is also expected in ConvFormat — see [03](03_conv_format.md).

## 2.5 IS-core GEMM (`SimbaCoreMode.ISGEMM`)

IS-core uses `isCoreLoopOrder = K_M_N`. Mapping:

- `dim0 = M·Mu` = sequence length
- `dim1 = K·Ku` = `dInner`  (the IS-core unrolls `dInner` along Ku)
- `dim2 = N·Nu` = output channels

From [DataGeneratorISGEMM.scala](../../src/test/scala/datagen/DataGeneratorISGEMM.scala):

```scala
writeFloatToFile("A", mode, flattenConvFormat(aMatrix))                       // FP8
writeFloatToFile("B", mode, flattenB(bMatrix, params.isCoreLoopOrder))        // FP8
writeFloatToFile("C", mode, flattenCD(cMatrix, params.isCoreLoopOrder),       // BF16 bias
                            accType)
writeFloatToFile("D", mode, padWithZeros(flattenCD(dMatrix, params.isCoreLoopOrder),
                                         accType, inType))                    // FP8 with requant pad
writeFloatToFile("D_no_requant", mode, flattenCD(dMatrix, params.isCoreLoopOrder),
                                       accType)                               // BF16 raw
```

Three things to highlight:

1. **A is `flattenConvFormat`, not `flattenA`.** In a standalone ISGEMM the
   A input would normally come from a previous OS-core / Switch-core stage,
   which already deposits its output in ConvFormat. So the test generator
   produces A directly in ConvFormat. See [03](03_conv_format.md).
2. **C is in `accType` (BF16).** The IS-core's bias path is BF16.
3. **D byte sizing.** With `en_isCoreRequant=1`, the IS-core requantizes its
   BF16 psum to FP8, **but** the streamer still writes one BF16-sized slot
   per element. The FP8 value sits in the low bits and the high bits are
   zeroed. `padWithZeros(x, psumType, finalType)` mimics this — appends
   zeros so that `totalLength = (psumType.width / finalType.width) * x.length`.
   The `D_no_requant` file shows the un-padded BF16 reference.

## 2.6 IS-core square mode (`ISGEMM_SQ`, `ISGEMM_SQ_TRANSPOSE`)

Used by the EinFFT path
([DataGeneratorFFT.scala](../../src/test/scala/datagen/DataGeneratorFFT.scala)).
The IS-core runs with `sw_isCoreSquareMode = 1`, which lets the input axis
extend beyond `dInner`. The weight is still in IS-core B layout, but it must
be **pre-padded** so its second dimension is a multiple of `dInnerUnroll`. The
generator's `flattenDftWeight*` apply `padMatrixColumnsPerBlock` first and
then call `flattenConvFormat` on the padded weight.

## 2.7 IS-core output → bank-transposed

In Phase-1, after the IS-core requantizes to FP8, the streamer dispatches the
output through banks. The data generator emulates this with
`bankTranspose(...)` over `seqLenUnroll`, with `elemPerBank = BANKWIDTH /
inType.width` (= 8 for FP8). See
[DataGeneratorMain.scala:27](../../src/test/scala/datagen/DataGeneratorMain.scala#L27).

When writing IS-core output buffers that are read by downstream cores via
the same banked port (e.g., the SIMD core in Phase-1), apply `bankTranspose`
after `flattenCD` / `interleaveColsXProj`. See [08](08_helpers.md) for details.
