# 5. xProj layout — `interleaveColsXProj`

> **All pages:**
> [README](README.md) ·
> [1. Concepts](01_concepts.md) ·
> [2. GEMM layouts](02_gemm_layouts.md) ·
> [3. ConvFormat](03_conv_format.md) ·
> [4. SUCFormat](04_suc_format.md) ·
> **5. xProj layout (this page)** ·
> [6. Delta weight split](06_delta_weight.md) ·
> [7. Per-mode reference](07_mode_reference.md) ·
> [8. Helpers](08_helpers.md) ·
> [9. FFT](09_fft.md) ·
> [10. SIMD](10_simd.md)

In Mamba's Phase-1 the IS-core computes the **x-projection**, which produces a
concatenated `(dt, B, C)` triple along the channel axis:

```
xProj : (seqLen,  dtRank + dState + dState)
                  |--dt--| |--B--| |--C--|
```

Width = `dtRank + 2 * dState = xProjDim`
(see [SsmParams.scala:23](../../src/main/scala/mambacore/SsmParams.scala#L23)).

The downstream SU-core needs `B` and `C` in **interleaved chunks**, because it
processes `dState` along the `delaySU`-wide port and consumes one chunk of B
then one chunk of C per cycle. This interleaving is performed at the layout
level — both for the IS-core's *weight* (so the projection it computes
already produces the interleaved order) and for the IS-core's *output*.

The function lives in
[src/main/scala/simbacore/SimbaCoreUtil.scala](../../src/main/scala/simbacore/SimbaCoreUtil.scala).

## 5.1 The transform

```scala
def interleaveColsXProj(
  xProjMatrix:  Seq[Seq[Float]],  // (seqLen, dtRank + 2*dState)
  dtRank:       Int,
  dState:       Int,
  elemPerChunk: Int
): Seq[Seq[Float]]
```

For each row:

```
[ dt | b0 b1 ... b_{N-1} | c0 c1 ... c_{N-1} ]
                                                              ↓
[ dt | b0 ... b_{c-1} | c0 ... c_{c-1} |
       b_c ... b_{2c-1} | c_c ... c_{2c-1} |
       ...                                              ]
```

where `c = elemPerChunk`. The `dt` slice (first `dtRank` columns) is left
untouched; the `B` and `C` slices each get cut into `dState / elemPerChunk`
chunks and the chunks are zipped pairwise.

There is also a 3-arg overload that builds `xProjMatrix` from separate
`(dt, B, C)` matrices first.

## 5.2 Choosing `elemPerChunk`

In `DataGeneratorMain.scala`:

```scala
def interleaveColsXProj(xProjOut) =
  SimbaCoreUtil.interleaveColsXProj(
    xProjOut, dtRank, params.dState,
    elemPerChunk = params.suCoreSerialWidthBC / inType.width
  )
```

`suCoreSerialWidthBC = suCoreParallelWidthVec / delaySU` — i.e., the serial
width of the SU-core's B/C port. Dividing by `inType.width` (FP8 = 8) gives
the number of FP8 elements that flow over the port per cycle. This is the
SU-core's natural chunk size, so the interleaving is driven by *consumer*
constraints, not by the bank width.

For BF16 you would substitute `accType.width`.

## 5.3 Inverse — `splitXProj`

```scala
def splitXProj(xProjOut, dtRank, dState): (dt, B, C)
```

Re-slices the *non-interleaved* `xProjMatrix` into three matrices. Use this
when you need to feed a reference Selective-Scan model with reconstructed
`dt`, `B`, `C` matrices (e.g., the data generator after computing the
expected IS-core output, before quantizing to FP8).

`splitXProj` does **not** undo the chunk interleaving; call it on the matrix
*before* you interleave.

## 5.4 Layout pipeline in Phase-1

Putting it together — what the data generator emits for the IS-core output
buffer (which is the actual `(dt, B, C)` activation tensor in TCDM):

```scala
def flattenXProjOut(xProjOut) =
  bankTranspose(
    flattenCD(xProjOut, isCoreLoopOrder)(xProjDims),
    seqLenUnroll,
    elemPerBank = BANKWIDTH / inType.width
  )

def interleaveAndFlattenDtBC(dt, B, C) = {
  val xProjMatrix = dt.zip(B).zip(C).map { ((d, b), c) => d ++ b ++ c }
  val xProjOut    = interleaveColsXProj(xProjMatrix)
  flattenXProjOut(xProjOut)
}
```

The order matters:

1. `dt ++ B ++ C` row-wise to build the wide matrix.
2. `interleaveColsXProj` to re-order the columns.
3. `flattenCD` with `K_M_N` to flatten through IS-core tiling.
4. `bankTranspose` to mimic the streamer's bank dispatch.

## 5.5 IS-core weight layout in Phase-1

The IS-core weight that *produces* this layout must itself be column-permuted
the same way (since the IS-core has output channels along Nu):

```scala
val isCoreWeightFlat = flattenB(
  interleaveColsXProj(isCoreWeight),
  params.isCoreLoopOrder
)(xProjDims)
```

If you forget to interleave the weight, the hardware will still produce some
matrix, but the chunks of B and C will be in the wrong positions and the
SU-core will read mis-ordered data — without any error.

## 5.6 In Phase-2

In Phase-2 the dt/B/C are *inputs* (they were produced and stored at the end
of Phase-1). The data generator still uses `interleaveAndFlattenDtBC` to put
them into the **Phase-1 output layout**, so that the Phase-2 program can
issue the same address pattern when reading them back:

```scala
val xProjFlat = interleaveAndFlattenDtBC(dt, matrixB, matrixC)
writeFloatToFile("dt_BC", mode, xProjFlat)
```

This is the buffer `dt_BC` you'll see in the `M<id>_dt_BC.bin` file.

## 5.7 Summary

- **`interleaveColsXProj`** = "after the `dt` slice, interleave B-chunks and
  C-chunks of size `elemPerChunk` so that the SU-core's port sees alternating
  chunks of B and C".
- `elemPerChunk` is determined by the SU-core's B/C port serial width.
- Apply to both the **IS-core weight** (so the matmul output is already
  interleaved) and to the **reference output** (so it matches the byte order
  the streamer will produce).
- It must precede `flattenCD` and `bankTranspose` in the layout pipeline.
- `splitXProj` is the inverse — but only of the (dt | B | C) concatenation,
  not of the column interleaving.
