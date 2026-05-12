# 4. SUCFormat — State Update Core layout

> **All pages:**
> [README](README.md) ·
> [1. Concepts](01_concepts.md) ·
> [2. GEMM layouts](02_gemm_layouts.md) ·
> [3. ConvFormat](03_conv_format.md) ·
> **4. SUCFormat (this page)** ·
> [5. xProj layout](05_xproj_format.md) ·
> [6. Delta weight split](06_delta_weight.md) ·
> [7. Per-mode reference](07_mode_reference.md) ·
> [8. Helpers](08_helpers.md) ·
> [9. FFT](09_fft.md) ·
> [10. SIMD](10_simd.md)

SUCFormat is the layout the **SU-core's data port** uses. It groups
`delaySU` consecutive columns together, then steps through all rows, then
moves to the next group of `delaySU` columns.

`delaySU` is the SU-core's pipeline depth for a single FMA on an SSM
vec. In `MySystem`: `delaySU = 4`, hard-coded
([MambaCoreParams.scala:134](../../src/main/scala/mambacore/MambaCoreParams.scala#L134)).
By construction `convUnroll = delaySU`, so a SUCFormat slice of width
`delaySU` matches one ConvFormat sub-tile's width.

Defined in
[src/main/scala/mambacore/MambaCoreUtil.scala](../../src/main/scala/mambacore/MambaCoreUtil.scala):

- `flattenSUCFormat(a, delaySU)`
- `temporalToSpatialIdxSUC`, `spatialToTemporalIdxSUC`

## 4.1 Shape and order

For a matrix `(seqLen, dInner)`, SUCFormat emits values in this order:

```scala
for (i <- 0 until (dInner / delaySU))
  for (j <- 0 until seqLen)
    for (k <- 0 until delaySU)
      matrix(j)(i * delaySU + k)
```

There is **no row tiling** in SUCFormat — it walks the *full* `seqLen` rows
within each column-stripe of width `delaySU`. Compare with ConvFormat, which
also tiles along the `L` axis (`rowsPerTile`).

This is the natural format for the SU-core: each time-step gets one
`delaySU`-element vector slice from the matmul output's column-stripe, which
matches the way the SU-core writes its accumulators.

## 4.2 SUCFormat in TCDM vs at the port

The hardware policy is:

- **TCDM stores ConvFormat.** This is true for both:
  - `suc_x` — the OS-core output that feeds the SU-core's `x` input (it was
    produced by the OS-core in ConvFormat in Phase 1).
  - The SU-core's output `y`, which is written back to TCDM.
- **Streamers convert** between ConvFormat (memory side) and SUCFormat (port
  side). The programmer never writes SUCFormat to memory directly.

See
[DataGeneratorMain.scala:153](../../src/test/scala/datagen/DataGeneratorMain.scala#L153)
for the canonical comments:

```scala
// suc_x comes from P1 osCore output. Streamers must convert to SUC format
writeFloatToFile("suc_x", mode, flattenConvFormat(x))
// Produced in SUC format, but streamers must convert to conv format before writing to file
writeFloatToFile("suc_expected", mode, flattenConvFormat(stateUpdateExpected))
```

In short: anything `x`-like or `y`-like in the SU-core dataflow is written in
**ConvFormat** in the .bin file. SUCFormat is only relevant when describing
what the SU-core sees at its data port.

## 4.3 SU-core scalar inputs

These do not use SUCFormat:

- `suc_state` — `(dInner, dState)`, written as `state.flatten` (row-major).
  In Phase-2 testing the state is initialized to zero; the hardware reuses
  the on-chip state between invocations.
- `suc_A` — `(dInner, dState)`, row-major flatten.
- `suc_D` — `(dInner)`, plain vector.

These are loaded once per layer, not streamed at sequence-rate.

## 4.4 Round-trip / debug

In `worksheet/fputils.worksheet.sc`:

```scala
val suc      = (0 until seqLen*dInner).map(temporalToSpatialIdxSUC(_, seqLen, delaySU, dInner))
val suckBack = suc.map(spatialToTemporalIdxSUC(_, seqLen, delaySU, dInner))
// identity check
```

If you really need a SUCFormat-flattened sequence (e.g., to drive a unit test
of the SU-core directly), use `flattenSUCFormat(matrix, delaySU)`. See its use
in [MambaCoreBehaviour.scala:303](../../src/test/scala/mambacore/MambaCoreBehaviour.scala#L303).

## 4.5 Why `convUnroll == delaySU`

Stamped into MambaCoreParams
([MambaCoreParams.scala:164](../../src/main/scala/mambacore/MambaCoreParams.scala#L164)):

```scala
val switchCore = SwitchCoreParams(
  ...
  convUnroll    = delaySU, // Must match delaySU to allow proper dataflow
  convRingDepth = osCore.Nu / delaySU,
  ...
)
```

This guarantees that one ConvFormat sub-tile (`rowsPerTile, convUnroll`) and
one SUCFormat column-stripe of width `delaySU` carry the same number of
columns. The streamer can therefore re-tile from ConvFormat to SUCFormat
without buffering more than a single sub-tile.
