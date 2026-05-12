# 10. SIMD core layouts

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
> [9. FFT](09_fft.md) ·
> **10. SIMD (this page)**

This page documents how SNAX buffers must be laid out for the SIMD core,
with special attention to **lane ordering for real and complex operations**
(`Add`, `Sub`, `Mul`, `CMul`, `InProd`, `Rms`, `Div`, `Sqrt`, `Noop`,
`SoftShrink`, and any of those followed by `Requant`).

Source-of-truth files:

- Core RTL: [src/main/scala/simbacore/SimdCore.scala](../../src/main/scala/simbacore/SimdCore.scala)
- Modes enum: [SimdMode object](../../src/main/scala/simbacore/SimdCore.scala#L31)
- Reference behavior: [src/test/scala/simbacore/SimdCoreBehavior.scala](../../src/test/scala/simbacore/SimdCoreBehavior.scala)
- Data generator: [src/test/scala/datagen/DataGeneratorSIMD.scala](../../src/test/scala/datagen/DataGeneratorSIMD.scala)
- In the SimbaCore wrapper: [SimbaCore.scala::SIMD Core](../../src/main/scala/simbacore/SimbaCore.scala#L141)

## 10.1 Two physical SIMD cores, one mode bitfield

The chip instantiates **two** SIMD cores
([SimbaCore.scala:144](../../src/main/scala/simbacore/SimbaCore.scala#L144)):

| Core    | `fpType` | `requantType` | Lanes (`N`)        | Has `CMul` | Has `DivSqrt` | Has `Relu` |
|---------|----------|---------------|--------------------|------------|---------------|------------|
| BF16    | `accType` (BF16) | `inType` (FP8) | `simdLanes_bf16` | ✗ | ✓ | ✓ |
| FP8     | `inType` (FP8)   | `accType` (BF16) | `simdLanes_fp8`  | ✓ | ✗ | ✓ |

The selector bit is `sw_simdInType` in the `SimbaCoreCtrlBundle`. Every
`SimbaCoreMode` with a `_BF16` suffix sets it to 0; `_FP8` modes set it to 1.

`simdLanes_*` come from
[MambaCoreParams.scala:102](../../src/main/scala/mambacore/MambaCoreParams.scala#L102):

```
simdLanes_bf16 = (2 * suCoreSerialWidthBC) / accType.width
simdLanes_fp8  = (2 * suCoreSerialWidthBC) / inType.width
```

In `MySystem` (`suCoreSerialWidthBC = 128`):

- `simdLanes_bf16 = 16`
- `simdLanes_fp8  = 32`

A **SIMD tile** = `N` elements. The streamer hands the SIMD core one tile of
`A` and one tile of `B` per cycle (under back-pressure). Memory buffers are
therefore always sized in tiles of `N` elements of the matching type.

## 10.2 Common lane layout (real-valued ops)

For `Add`, `Sub`, `Mul`, `Div`, `Sqrt`, `Noop`:

```
A buffer: [a_0, a_1, ..., a_{N-1},   a_N, ..., a_{2N-1}, ...]   (FP8 or BF16, flat)
B buffer: same                                                   (only A used for Sqrt, Noop)
Output  : same shape, element-wise op applied per lane
```

- Each element in A pairs with the **element at the same lane index** in B.
- Tile boundaries do not matter for math — every lane is independent.
- Buffer length must be a multiple of `N` (the streamer reads tile-by-tile).

This is the layout produced by `genRandomTensor(numElem)` then
`writeFloatToFile`, see
[DataGeneratorSIMD.scala::genData](../../src/test/scala/datagen/DataGeneratorSIMD.scala#L24).

## 10.3 CMul lane layout (FP8 only)

`SimdMode.CMul` is **only available on the FP8 SIMD core** (the BF16 core
is built with `hasCMul = false`). It computes one complex multiplication
per **pair of lanes**:

```
For each SIMD tile of N lanes (N = simdLanes_fp8 = 32):

A: [ a_re_0  a_re_1  ...  a_re_{N/2-1}  |  a_im_0  a_im_1  ...  a_im_{N/2-1} ]
        first N/2 lanes                       second N/2 lanes
B: [ b_re_0  b_re_1  ...  b_re_{N/2-1}  |  b_im_0  b_im_1  ...  b_im_{N/2-1} ]

Output (same shape):
   [ result_re_0  ...  result_re_{N/2-1}  |  result_im_0  ...  result_im_{N/2-1} ]

where (result_re_i, result_im_i) = cmul(a_re_i, a_im_i, b_re_i, b_im_i)
                                 = ( a_re*b_re - a_im*b_im ,
                                     a_re*b_im + a_im*b_re )
```

⚠️ **The reals/imags split is at the half-tile boundary, not "every other
lane".** With `N = 32`, this means **lanes 0..15 carry reals, lanes 16..31
carry imags**, for both A and B.

The reference implementation in the data generator
([DataGeneratorSIMD.scala::cmulOut](../../src/test/scala/datagen/DataGeneratorSIMD.scala#L42)):

```scala
val cmulOut = (0 until aVec.length by simdLanes).flatMap { tileStart =>
  val results = (0 until simdLanes / 2).map { i =>
    cmul(
      aVec(tileStart + i),
      aVec(tileStart + simdLanes / 2 + i),    // a_im is at + N/2
      bVec(tileStart + i),
      bVec(tileStart + simdLanes / 2 + i),    // b_im is at + N/2
      fpType
    )
  }
  results.map { case (re, _) => re } ++ results.map { case (_, im) => im }
}
```

### How to produce a CMul input buffer from `(reals, imags)`

`simdInterleaveRealImag` is the canonical packer (used by the FFT path —
see [09](09_fft.md)):

```scala
def simdInterleaveRealImag(reals: Seq[Float], imags: Seq[Float]): Seq[Float] =
  reals.grouped(16).zip(imags.grouped(16))
       .flatMap { case (r, i) => r ++ i }.toSeq
```

The hardcoded `16` is `simdLanes_fp8 / 2`. For each tile, it packs 16
consecutive reals then 16 consecutive imags. Inputs `reals` and `imags`
must be the same length and a multiple of 16.

### Important: this is NOT "every other lane" interleaving

Some FFT references store complex data as `[re_0, im_0, re_1, im_1, ...]`
("alternating" or "stride-2") layout. **The SimbaCore SIMD core does not
support that layout** — it strictly uses the "block" layout above. If your
upstream producer emits alternating Re/Im, you must repack (e.g., via
streamer stride configuration, or a host-side rearrangement, or the
`bankTranspose` machinery the FFT path uses).

## 10.4 InProd / Rms — reduction along the tile axis

These modes reduce **across tiles**, not within a tile. The `N` lanes are
treated as `N` independent **channels**, each of which gets its own
dot-product over `n_acc` consecutive input tiles.

```
A buffer (one accumulation window):
   tile 0: [a_0_0,  a_0_1,  ..., a_0_{N-1}]
   tile 1: [a_1_0,  a_1_1,  ..., a_1_{N-1}]
   ...
   tile n_acc-1: [a_{n_acc-1}_0, ..., a_{n_acc-1}_{N-1}]

For channel c: output_c = Σ_{t=0..n_acc-1}  a_t_c · b_t_c
                                          (or a_t_c · a_t_c for Rms)
```

Output for one accumulation window is **one** tile of `N` elements
(channel-major). After `nTiles / n_acc` windows, the output is a flat
buffer of `(nTiles / n_acc) · N` elements.

`n_acc` is set via `dut.io.n_acc` at the IO level
([SimdCore.scala::n_acc](../../src/main/scala/simbacore/SimdCore.scala#L80)).
In SimbaCore it is driven by `mambaCore.io.loadedConfig.dModel`
([SimbaCore.scala:177](../../src/main/scala/simbacore/SimbaCore.scala#L177)).

For `Rms`, only `A` is consumed (B is ignored), and each lane gets `a*a`
multiplied before accumulating.

The reference flattening is `SimdCoreBehaviorUtil.inProdFromFlattened`
([SimdCoreBehavior.scala:191](../../src/test/scala/simbacore/SimdCoreBehavior.scala#L191)):

```scala
def inProdFromFlattened(a, b, N, n_acc, fpType): Seq[Float] = {
  val vecsA = a.grouped(N).toSeq      // tiles of length N
  val vecsB = b.grouped(N).toSeq
  val nOutTiles = vecsA.length / n_acc
  (0 until nOutTiles).flatMap { tileIdx =>
    (0 until N).map { channelIdx =>
      val span = (tileIdx*n_acc until (tileIdx+1)*n_acc)
      val aGroup = span.map(vecsA(_)(channelIdx))
      val bGroup = span.map(vecsB(_)(channelIdx))
      inProd(aGroup, bGroup, fpType)
    }
  }
}
```

Note the per-step `quantize(fpType, ...)` in `inProd` — each partial sum is
re-quantized to BF16/FP8 to match the hardware accumulator's register
precision. Don't compute the inner product in FP32 and expect a bit-exact
match.

### RMSNorm via SIMD

`DataGeneratorRmsNorm` uses `SIMD_INPROD_BF16`. The buffers are:

- `x` — BF16 in IS-core CD layout (the producer is the IS-core).
- `weight` — BF16, per-channel weight **duplicated across SIMD lanes**
  via `duplicateEachElement(weight, simdLanes_bf16)`. The same channel
  weight appears on every lane in a tile.
- `out` — BF16 in IS-core CD layout.

The duplicate is necessary because the SIMD core sees lanes as parallel
channels; broadcasting a per-channel scale means each tile's lanes carry
the *same* weight for that tile's channel. See [07.8](07_mode_reference.md#78-simd_inprod_bf16-driven-by-rmsnorm).

## 10.5 SoftShrink — N → N/2 with sign-driven selection

`SoftShrink` is a post-op gate triggered by `simdCtrl.doSoftShrink`. It
runs **after** the main SIMD op (typically `Add`). Per input tile of `N`
lanes, it produces `N/2` output lanes:

```
Input tile: [v_0, v_1, ..., v_{N/2-1},   w_0, w_1, ..., w_{N/2-1}]
             first half = "primary"        second half = "fallback"

For each i in 0..N/2-1:
   out_i = v_i        if v_i >= 0           (primary non-negative)
         = w_i        else if w_i  < 0      (primary neg, fallback also neg)
         = 0          otherwise
```

The hardware bases the sign check on the **MSB of the quantized FP byte**
([SimdCore.scala:393](../../src/main/scala/simbacore/SimdCore.scala#L393)) —
positive zero counts as non-negative, the IEEE-style negative zero
counts as negative. The reference in `DataGeneratorSIMD` mirrors this
exactly:

```scala
val softShrinkOut = addOut.grouped(simdLanes).toSeq.flatMap { tile =>
  (0 until simdLanes / 2).map { i =>
    val first    = quantize(fpType, tile(i))
    val second   = quantize(fpType, tile(i + simdLanes / 2))
    val firstMsb  = floatToUInt(fpType, first)  >> (fpType.width - 1)
    val secondMsb = floatToUInt(fpType, second) >> (fpType.width - 1)
    if (firstMsb == 0) first
    else if (secondMsb != 0) second
    else 0.0f
  }
}
```

To compute `softshrink(x, λ)` with the chip, drive `Add` mode with:

- `A` = `x - λ`  (primary; soft-shrunk if `x > λ`)
- `B` = `x + λ`  (fallback; soft-shrunk if `x < -λ`)
- `λ` is a per-tile constant; `A` and `B` are pre-computed by the host or
  by an earlier SIMD pass.

Pack `A` into the first `N/2` lanes of each tile and `B` into the second
half. After SIMD with `doSoftShrink = 1`, you get `N/2` output lanes per
tile (i.e., the output buffer is half the size).

⚠️ **`Requant` and `SoftShrink` are mutually exclusive** — the RTL asserts
this ([SimdCore.scala:397](../../src/main/scala/simbacore/SimdCore.scala#L397)).
Run them as two separate SIMD passes if you need both.

## 10.6 Requant — N → N elements but different bytes per element

`Requant` is a post-op triggered by `simdCtrl.doRequant`. It re-quantizes
the SIMD output to the **other** FP type (`requantType`):

| SIMD core | `fpType` → `requantType` | Byte change |
|---|---|---|
| BF16 | BF16 → FP8 | each element shrinks 2× |
| FP8  | FP8 → BF16 | each element grows 2× |

The number of *elements* in the output is the same as in the SIMD result.
The number of *bytes* changes. The output port serializes (BF16→FP8) or
parallelizes (FP8→BF16) accordingly.

For an `InProd_*_requant` mode, the rule of thumb is:

```
nOutTiles_input_to_requant = nInTiles / n_acc      // after accumulation
nOutTiles_in_bytes         = nOutTiles_input_to_requant * (fpType.width / requantType.width)
                              when narrowing (BF16→FP8)
                            = nOutTiles_input_to_requant * (requantType.width / fpType.width)
                              when widening  (FP8→BF16)
```

The data generator enforces `(nTiles / n_acc) % requantPackFactor == 0`
([SimdCoreBehavior.scala:71](../../src/test/scala/simbacore/SimdCoreBehavior.scala#L71)) so
the output stream fills full output tiles; respect this when sizing your
buffer.

## 10.7 Per-mode quick reference (lane semantics)

| Mode                  | A semantics | B semantics | Output / lane    | Output / tile |
|-----------------------|-------------|-------------|------------------|---------------|
| `Add`, `Sub`, `Mul`   | element     | element     | `op(a, b)` per lane | N |
| `Div`                 | numerator   | denominator | `a / b` per lane | N (BF16 only) |
| `Sqrt`                | input       | unused      | `sqrt(a)` per lane | N (BF16 only) |
| `Noop`                | input       | unused      | `quantize(a)` per lane | N |
| `CMul`                | `[Re×N/2 | Im×N/2]` | `[Re×N/2 | Im×N/2]` | `[Re×N/2 | Im×N/2]` | N (FP8 only) |
| `InProd`              | tile-stream | tile-stream | per-channel Σ over n_acc tiles | N (per accumulation window) |
| `Rms`                 | tile-stream | (ignored)   | per-channel Σ of a² | N (per accumulation window) |
| `SoftShrink` post     | `[primary×N/2 | fallback×N/2]` per tile | (used) | gated select | N/2 |
| `Requant` post        | (transparent) | (transparent) | re-quantize fpType↔requantType | N (element count unchanged; bytes change) |
| `Relu` post           | (transparent) | (transparent) | max(0, x) per lane | N |

`post` ops apply after the main SIMD op. They are bit-flags in
`SimbaCoreCtrlBundle.m_simd`: `doRelu`, `doRequant`, `doSoftShrink`.

## 10.8 Where SIMD reads from

In standalone SIMD test modes (`SIMD_*_BF16`, `SIMD_*_FP8`) the streamer
feeds the core directly from TCDM. The buffer layout is **flat
tile-major** — `N` elements per tile, tiles concatenated.

In Phase-1/Phase-2 flows, the SIMD core is wired inside SimbaCore and can
take its inputs from either the IS-core's output or directly from TCDM.
When the upstream is the IS-core, the buffer layout follows IS-core CD,
not "plain flat" — see [02](02_gemm_layouts.md) and [07.8](07_mode_reference.md#78-simd_inprod_bf16-driven-by-rmsnorm).

## 10.9 Common pitfalls

- **Mixing up "block" and "stride-2" Re/Im layouts for CMul.** The chip
  uses block layout `[Re×N/2 | Im×N/2]`. Any reference that produces
  `[Re,Im,Re,Im,...]` must be repacked.
- **Forgetting that BF16 has no CMul.** The BF16 SIMD core is wired with
  `hasCMul = false` ([SimbaCore.scala:152](../../src/main/scala/simbacore/SimbaCore.scala#L152)).
  Any complex multiplication must run on the FP8 core; pre-quantize first
  if your data is BF16.
- **Forgetting that FP8 has no Div/Sqrt.** Conversely, the FP8 core is
  wired with `hasDivSqrt = false` ([SimbaCore.scala:165](../../src/main/scala/simbacore/SimbaCore.scala#L165)).
  RMSNorm therefore stays on the BF16 core all the way through the
  `sqrt` step.
- **Wrong `n_acc`.** `n_acc` must divide the input tile count. The
  hardware reads `n_acc` from `mambaCore.io.loadedConfig.dModel` in
  Phase mode — make sure your program sets `dModel` consistently with the
  reduction window you want.
- **Lane-mismatched weight in RMSNorm.** Each lane in a tile must carry
  the *same* per-channel weight for that tile's channel. Use
  `duplicateEachElement(weight, simdLanes_bf16)`, not a single copy of
  `weight`.
- **`Requant` + `SoftShrink` in the same pass.** Disallowed by RTL
  assertion. Split into two SIMD invocations.
- **Output buffer size for SoftShrink/Requant.** SoftShrink halves the
  element count per tile; Requant changes the bytes per element (not the
  count). Size the output buffer accordingly or the streamer will
  truncate.
- **Sign of zero in SoftShrink.** The MSB-based sign check treats `+0`
  as non-negative and `-0` as negative. If you're producing `A = x - λ`
  with `λ = 0`, you may hit IEEE negative zero on one path and positive
  zero on the other — match the reference's `quantize → MSB` rule rather
  than `< 0` in FP arithmetic.
