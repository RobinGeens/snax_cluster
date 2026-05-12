# 6. Delta-weight split — `splitDeltaWeight`

> **All pages:**
> [README](README.md) ·
> [1. Concepts](01_concepts.md) ·
> [2. GEMM layouts](02_gemm_layouts.md) ·
> [3. ConvFormat](03_conv_format.md) ·
> [4. SUCFormat](04_suc_format.md) ·
> [5. xProj layout](05_xproj_format.md) ·
> **6. Delta weight split (this page)** ·
> [7. Per-mode reference](07_mode_reference.md) ·
> [8. Helpers](08_helpers.md) ·
> [9. FFT](09_fft.md) ·
> [10. SIMD](10_simd.md)

The Switch-core in **Matmul mode** uses the `dtRank → dInner` delta weight
(used in Phase-2 to project `dt` onto `dInner`). This matrix has shape
`(dtRank, dInner)` and is loaded through **two different ports** of the
Switch-core:

- The **shared (conv) weight port** carries the first `dConv` rows of each
  weight tile.
- The **matmul weight port** carries the remaining `(Ku - dConv)` rows.

This split exists because in Conv mode the same physical hardware lane is
used as the dConv-deep filter; in Matmul mode the lane is reused as part of
the (Ku-dConv)-wide matmul reduction. Mixing both into one wide matrix and
splitting it at layout time keeps the dataflow consistent.

`splitDeltaWeight` lives in
[src/main/scala/mambacore/MambaCoreUtil.scala](../../src/main/scala/mambacore/MambaCoreUtil.scala).

## 6.1 Signature

```scala
def splitDeltaWeight(
  deltaWeight: Seq[Seq[Float]],  // (dtRank, dInner)
  switchCoreParams: SwitchCoreParams
): (Seq[Float], Seq[Float])      // (conv-weight stream, matmul-weight stream)
```

## 6.2 Tiling

The matrix is tiled into `(Ku, convUnroll)` blocks:

- `nTilesRow = dtRank / Ku`
- `nTilesCol = dInner / convUnroll`

For each tile, the first `dConv` rows go into stream 1 (conv port) and rows
`dConv..Ku` go into stream 2 (matmul port).

Within a tile, both streams are flattened **col-major** (transpose then
flatten), so the column index along `convUnroll` is the *outer* dimension
inside the tile. This matches `DecoupledVec.unFlatten`, which reads
`convUnroll` parallel lanes from the wide port.

Across tiles, the flattening is **col-major** as well — the temporal index
walks the K direction first (i.e., `i` is the inner of the outer `(i, j)`
loop).

## 6.3 Where it's used

[DataGeneratorMain.scala:114](../../src/test/scala/datagen/DataGeneratorMain.scala#L114):

```scala
val (switchCoreWeight_1, switchCoreWeight_2) =
  splitDeltaWeight(switchCoreWeight, params.switchCoreParams)

writeFloatToFile("dt_weight_1", mode, switchCoreWeight_1)
writeFloatToFile("dt_weight_2", mode, switchCoreWeight_2)
```

Both files are simple flat FP8 streams. The streamer points the conv-port
DMA at `dt_weight_1` and the matmul-port DMA at `dt_weight_2`. Each stream
is consumed in order — its `temporalToSpatialIdxDeltaWeight(t) = t` mapping
(see [MambaCoreUtil.scala:265](../../src/main/scala/mambacore/MambaCoreUtil.scala#L265)).

## 6.4 Caveats

- **`Ku > dConv` is required.** If `Ku == dConv` the second stream would be
  empty — the code does not handle this case (`require(Ku > dConv, ...)` in
  [SwitchCoreParams.scala:26](../../src/main/scala/mambacore/SwitchCoreParams.scala#L26)).
- **`Ku - dConv` must be a power of two** (so the adder-tree-2 reduction
  works). The matmul-tree split is done with
  `adderTreeSplit = Some(params.dConv)` in
  [DataGeneratorMain.scala:125](../../src/test/scala/datagen/DataGeneratorMain.scala#L125)
  — when generating the reference output, pass this so the partial sums
  reflect the two independent adder trees in hardware.
- **`dtRank % Ku == 0`** and **`dInner % convUnroll == 0`**, otherwise the
  tiling is undefined.
- **Bias** for the delta projection (`dt_bias`) is a single vector of length
  `dInner`. The reference flow broadcasts it over `seqLen` rows before doing
  the matmul.

## 6.5 Conv weight in Phase-1 (separate matrix)

Do not confuse the *delta* weight (used in Phase-2 matmul mode) with the
*conv* weight (used in Phase-1 conv mode). The conv weight is a
`(dInner, dConv)` matrix; it is written **row-major flat** (just
`convWeight.flatten`) — no tile split, no interleaving:

```scala
writeFloatToFile("conv_weight", mode, convWeight.flatten)
```

The naming in the test data is intentional:

- `conv_weight` — Phase-1 conv kernel, `(dInner, dConv)`.
- `conv_bias`   — Phase-1 conv bias, length `dInner`.
- `dt_weight_1` + `dt_weight_2` — Phase-2 delta-projection split.
- `dt_bias` — Phase-2 delta-projection bias, length `dInner`.
