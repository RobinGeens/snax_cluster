# 3. ConvFormat

> **All pages:**
> [README](README.md) ·
> [1. Concepts](01_concepts.md) ·
> [2. GEMM layouts](02_gemm_layouts.md) ·
> **3. ConvFormat (this page)** ·
> [4. SUCFormat](04_suc_format.md) ·
> [5. xProj layout](05_xproj_format.md) ·
> [6. Delta weight split](06_delta_weight.md) ·
> [7. Per-mode reference](07_mode_reference.md) ·
> [8. Helpers](08_helpers.md) ·
> [9. FFT](09_fft.md) ·
> [10. SIMD](10_simd.md)

ConvFormat is the **column-sub-tiled, sequence-first** layout used everywhere
the OS-core, Switch-core (in conv mode) or IS-core (square mode) touch a
sequence-time tensor in TCDM. It exists because the Switch-core processes
`convUnroll` time-direction samples in parallel and wants the `L` axis to be
the *inner* iteration index.

Defined in
[src/main/scala/mambacore/MambaCoreUtil.scala](../../src/main/scala/mambacore/MambaCoreUtil.scala):

- `flattenConvFormat(a, seqLen, dInner, convUnroll, rowsPerTile, colsPerTile)`
- `temporalToSpatialIdxConvFormat`, `spatialToTemporalIdxConvFormat`
- `rowMajor2ConvFormat`, `convFormat2RowMajor` (Chisel modules that re-shuffle
  the data on-chip — you don't need to call these from software)

## 3.1 Shape

The matrix `(seqLen, dInner)` is divided into **tiles** of size
`(rowsPerTile, colsPerTile)`, and each tile is divided into
**sub-tiles** of size `(rowsPerTile, convUnroll)`.

For the MambaCore:

- `rowsPerTile = convTilesizeL = seqLenUnroll`  (OS-core's `Mu`)
- `colsPerTile = convUnroll * convRingDepth`    (= `dInnerUnroll`)
- `convUnroll = delaySU = 4`

So a tile is `(seqLenUnroll, dInnerUnroll)` = `(16, 24)` in the production
system, and a sub-tile is `(16, 4)`.

## 3.2 Order

Quoting the canonical comment in `rowMajor2ConvFormat`:

```
e.g. the output stationary core's output tile
  - 0 1 2 3
  - 4 5 6 7
  - 8 9 10 11
is outputted as 0 1 2 3 4 5 6 7 8 9 10 11 (row-major),
and needs to be transformed to 0 1 4 5 8 9 2 3 6 7 10 11 (convUnroll = 2)
```

In general the iteration order at element granularity is

```
for d3 in 0 .. dInner / colsPerTile           // tile column
  for l2 in 0 .. seqLen  / rowsPerTile        // tile row
    for d2 in 0 .. colsPerTile / convUnroll   // sub-tile column
      for l1 in 0 .. rowsPerTile              // row within sub-tile
        for c  in 0 .. convUnroll             // column within sub-tile
          write matrix[l2*rowsPerTile + l1][d3*colsPerTile + d2*convUnroll + c]
```

`temporalToSpatialIdxConvFormat(temporalIdx, ...)` returns, for each cycle's
slot `temporalIdx`, the row-major spatial index of the element it carries.
The diagrammatic comment in
[MambaCoreUtil.scala:66](../../src/main/scala/mambacore/MambaCoreUtil.scala#L66):

```
- 00 01 | 06 07 | 24 25 ..
- 02 03 | 08 09 | 26 27 ..
- 04 05 | 10 11 | ...
- ------------- |
- 12 13 | 18 19 | /\
- 14 15 | 20 21 | | rowsPerTile
- 16 17 | 22 23 | \/
- <----> convUnroll
- <-----------> colsPerTile
```

Each two-digit number is the temporal index. Tile (0,0) holds temporal
indices 0..11; tile (1,0) holds 12..23; tile (0,1) holds 24..35.

## 3.3 When to use it

ConvFormat is what TCDM holds whenever any of these is true:

| Buffer | Mode |
|---|---|
| OS-core output D (raw or requantized FP8) | OSGEMM, PHASE1, PHASE2 |
| OS-core bias C input | OSGEMM (BF16 in ConvFormat) |
| IS-core input A — when it comes from OS-core/Switch-core or is the test stand-in | ISGEMM, ISGEMM_SQ, PHASE1 |
| Switch-core conv weight (`conv_weight`) | PHASE1 |
| Switch-core output that feeds IS-core | PHASE1 |
| FFT DFT weights after column-block padding | ISGEMM_SQ |

See the per-mode reference in [07](07_mode_reference.md).

## 3.4 Convenience wrappers

In `DataGeneratorTrait`:

```scala
def flattenConvFormat(x: Matrix): Seq[Float] =
  flattenConvFormat(x, dim0 = seqLen, dim1 = dInner)
```

This always uses `convUnroll`, `rowsPerTile = seqLenUnroll`,
`colsPerTile = dInnerUnroll`. For non-standard tile shapes (FFT, ISGEMM_SQ),
call the 6-arg version in `MambaCoreUtil` directly.

There is also an overload in `MambaCoreUtil`:

```scala
flattenConvFormat(a: Seq[Seq[Float]], switchCoreParams: SwitchCoreParams)
```

which pulls `seqLen`, `dInner`, `convUnroll`, `convTilesizeL`, `convRingDepth`
straight from `SwitchCoreParams`.

## 3.5 Round-trip / debug

`flattenConvFormat` and `flattenCD(..., M_N_K)` are *not* the same. Sanity
test in `worksheet/fputils.worksheet.sc`:

```scala
val conv     = (0 until seqLen*dInner).map(temporalToSpatialIdxConvFormat(_, params))
val convBack = conv.map(spatialToTemporalIdxConvFormat(_, params))
// convBack == identity
```

If you flatten with ConvFormat and read back with row-major, you will get
the elements in the order shown in the diagram above — that is normal.

## 3.6 Relationship to the SU-core flow

In Phase-2, the OS-core output is routed *out* of TCDM in ConvFormat, and the
SU-core's `x` input is *read from* TCDM in ConvFormat — but the SU-core port
expects SUCFormat. The streamer transposes between the two on the fly. The
SNAX programmer just needs to make sure the TCDM contents are in ConvFormat.
See [04](04_suc_format.md).
