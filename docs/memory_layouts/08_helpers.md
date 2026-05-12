# 8. Helpers and edge-case layouts

> **All pages:**
> [README](README.md) ·
> [1. Concepts](01_concepts.md) ·
> [2. GEMM layouts](02_gemm_layouts.md) ·
> [3. ConvFormat](03_conv_format.md) ·
> [4. SUCFormat](04_suc_format.md) ·
> [5. xProj layout](05_xproj_format.md) ·
> [6. Delta weight split](06_delta_weight.md) ·
> [7. Per-mode reference](07_mode_reference.md) ·
> **8. Helpers (this page)** ·
> [9. FFT](09_fft.md) ·
> [10. SIMD](10_simd.md)

The pieces that don't belong to a single core but show up across multiple
flows: bank transpose, padding, requant-padding, SIMD reals/imags
interleaving, and a couple of small but important utilities.

## 8.1 BankTransposer / `bankTranspose`

The bank transposer exists because there is a mismatch between
*producer* and *consumer* of an IS-core output buffer:

- The IS-core delivers one **tile** of `elementsPerCycle` elements per
  cycle, with all elements coming from **the same tile row** (one matrix
  row, sliced into output channels). This is the natural form for a GEMM
  output port that fires once per output row.
- TCDM is organised as `N_BANKS = 32` banks of `BANKWIDTH = 64` bits
  each. The downstream consumer (SIMD core, another IS-core stage, host
  read) addresses one or more **banks** at a time, where each bank holds
  `elemPerBank = BANKWIDTH / elementWidth` consecutive elements of the
  matrix. A bank-aligned read packs *multiple* output rows together.

Without bank transpose, the streamer writes each tile straight into one
slot per bank. With bank transpose, `elemPerBank` consecutive tiles get
collected into a small buffer, and the buffer is read out column-first so
that each emitted bank carries `elemPerBank` consecutive elements from
one column of that buffer — i.e. consecutive matrix rows on the same
channel. This is exactly the access pattern a banked TCDM reader expects.

The hardware module lives at
[src/main/scala/utilmodules/BankTransposer.scala](../../src/main/scala/utilmodules/BankTransposer.scala).
The byte-level reference lives at
[src/test/scala/utils/TensorUtils.scala::bankTranspose](../../src/test/scala/utils/TensorUtils.scala#L368).
The two are kept in lockstep by
[BankTransposerTest.scala](../../src/test/scala/utilmodules/BankTransposerTest.scala) —
the RTL output is compared bit-for-bit against the reference for several
shapes.

### 8.1.1 Parameters and shape

```scala
class BankTransposer(
  val elementsPerCycle: Int,        // tile width (= number of input elems per cycle)
  val elementWidth:     Int,        // bits per element
  val bankWidth:        Int,        // BANKWIDTH = 64
  forceNumBanks:        Option[Int] // override the default numBanks
)
```

Derived quantities ([BankTransposer.scala:33](../../src/main/scala/utilmodules/BankTransposer.scala#L33)):

```text
elemPerBank      = bankWidth / elementWidth
numBanksDefault  = elementsPerCycle / elemPerBank
numBanks         = forceNumBanks.getOrElse(numBanksDefault)
colGroups        = elementsPerCycle / numBanks      // output cycles per buffered "matrix"
```

Constraints:

- `elementsPerCycle * elementWidth % bankWidth == 0` — every input tile
  must occupy an integer number of banks.
- `numBanks >= numBanksDefault` and `elementsPerCycle % numBanks == 0`.
  Increasing `numBanks` widens the output port (more banks emitted per
  cycle) at the cost of more parallel TCDM ports; the default is the
  *minimum* number of banks that still allows continuous draining.

IO shapes:

```
in  : DecoupledIO[Vec(elementsPerCycle, UInt(elementWidth.W))]
out : DecoupledIO[Vec(numBanks, Vec(elemPerBank, UInt(elementWidth.W)))]
```

The output is "`numBanks` parallel banks, each carrying `elemPerBank`
consecutive elements" — exactly what a sparse TCDM connection consumes.

### 8.1.2 Byte-level transformation

Treat each `elemPerBank` consecutive input tiles as a small matrix:

```
matrix shape  : (elemPerBank rows) × (elementsPerCycle cols)
              =  one column per output channel,
                 one row    per input cycle
```

The transposer streams that matrix out column-wise, packing
`elemPerBank` rows into one bank and emitting `numBanks` banks per
cycle. For `colGroup` in `0 .. elemPerBank-1` and `bank` in
`0 .. numBanks-1`:

```
out_bank[colGroup, bank][row]  =  matrix[row][colGroup * numBanks + bank]
                                  for row in 0 .. elemPerBank-1
```

`colGroup` is the output cycle index; `bank` is the output lane index;
`row` is the slot within a bank.

The reference function makes this explicit:

```scala
def bankTranspose[T](x: Seq[T], tileSize: Int, elemPerBank: Int): Seq[T] = {
  require(tileSize % elemPerBank == 0, "tileSize must be divisible by elemPerBank")
  val numBanks = tileSize / elemPerBank
  val tiles    = x.grouped(tileSize).toSeq
  val matrices = tiles.grouped(elemPerBank).toSeq   // group elemPerBank cycles
  matrices.flatMap { matrix =>
    (0 until elemPerBank).flatMap { colGroup =>     // output cycle
      (0 until numBanks).flatMap { bank =>          // output lane
        (0 until elemPerBank).map { row =>          // slot within bank
          val colIdx = colGroup * numBanks + bank
          matrix(row)(colIdx)
        }
      }
    }
  }
}
```

### 8.1.3 Worked example — FP8 IS-core output

Take the production config: `elementsPerCycle = 16` (one IS-core tile
row of FP8 elements), `elementWidth = 8`, `bankWidth = 64`. Then
`elemPerBank = 8`, `numBanksDefault = 2`.

The transposer groups `elemPerBank = 8` consecutive input tiles into one
`8 × 16` matrix. Let the inputs be labelled `(row, col)`:

```
input cycle 0:   (0,0)  (0,1)  (0,2)  ...  (0,15)
input cycle 1:   (1,0)  (1,1)  (1,2)  ...  (1,15)
...
input cycle 7:   (7,0)  (7,1)  (7,2)  ...  (7,15)
```

Output cycles (`numBanks = 2` banks of `elemPerBank = 8` elements):

```
output cycle 0:
  bank 0 = [ (0,0)  (1,0)  (2,0)  (3,0)  (4,0)  (5,0)  (6,0)  (7,0) ]   ← col 0
  bank 1 = [ (0,1)  (1,1)  (2,1)  (3,1)  (4,1)  (5,1)  (6,1)  (7,1) ]   ← col 1
output cycle 1:
  bank 0 = column 2
  bank 1 = column 3
...
output cycle 7:
  bank 0 = column 14
  bank 1 = column 15
```

Each output bank therefore carries `elemPerBank` **consecutive rows of
one channel** — packed FP8 elements ready to be written into a single
TCDM bank.

### 8.1.4 Ping-pong double buffering

The hardware uses a ping-pong register
([BankTransposer.scala:45-61](../../src/main/scala/utilmodules/BankTransposer.scala#L45)):

```text
buffer : Reg(Vec(2 * elemPerBank, Vec(elementsPerCycle, ...)))
```

Capacity is **two** `elemPerBank × elementsPerCycle` matrices. The write
side accumulates `elemPerBank` input tiles into one half; once that half
is full, it flips and starts filling the other half while the read side
drains the just-completed half.

State machine in words:

1. **Write side**: input fires (`io.in.fire`) on each accepted tile. The
   write pointer `rowWritePtr` advances from `0` to `elemPerBank-1`.
   When it wraps, the half is marked valid (`pingPongValid`), and
   `pingOrPongWrite` flips.
2. **Read side**: output fires on each consumed cycle. The read pointer
   `colReadPtr` advances from `0` to `colGroups-1 = elemPerBank-1`.
   When it wraps, the half is marked invalid and `pingOrPongRead`
   flips. If the read side becomes empty while the other side is full,
   the read pointer can switch sides immediately
   ([BankTransposer.scala:94](../../src/main/scala/utilmodules/BankTransposer.scala#L94)).

Back-pressure:

- `io.in.ready = !pingPongValid(pingOrPongWrite)` — accept new input
  only when the current write half has free rows.
- `io.out.valid = pingPongValid(pingOrPongRead)` — produce output only
  when the current read half holds a complete matrix.

Throughput: in steady state with both sides used, the transposer
consumes one tile and produces one bank-vector per cycle. There is no
fundamental latency penalty other than the initial `elemPerBank`-cycle
fill before the first output appears.

### 8.1.5 The factory `BankTransposer(in, elementWidth, outputWidth, en)`

Wrapper at the bottom of the file
([BankTransposer.scala:113](../../src/main/scala/utilmodules/BankTransposer.scala#L113)):

```scala
def apply(in: DecoupledIO[UInt], elementWidth: Int, outputWidth: Int, en: Bool) = {
  require(outputWidth % BANKWIDTH == 0)
  val elementsPerCycle = in.bits.getWidth / elementWidth
  val numParallelBanks = outputWidth / BANKWIDTH
  val transposer = Module(new BankTransposer(
    elementsPerCycle, elementWidth, BANKWIDTH,
    forceNumBanks = Some(numParallelBanks)
  ))
  // ...
}
```

This is the form used inside the IS-core output path. It pins
`bankWidth = BANKWIDTH` and forces `numBanks` so the output port matches
the IS-core's wide output bus. Hand-rolled instantiations are rarely
needed; use the factory.

### 8.1.6 Where it fires in MambaCore

The IS-core output path conditionally routes through the transposer
([MambaCore.scala:370-381](../../src/main/scala/mambacore/MambaCore.scala#L370)):

```scala
val doTranspose    = isCoreOutIsFinal && ctrl.en_isCoreTranspose && ctrl.en_isCoreRequant
val (toTranspose, toPacker) = DecoupledMux(doTranspose, requant, en = true.B)
val transposed  = BankTransposer(toTranspose, inType.width, isCoreOutWidth,
                                 en = ctrl.en_isCoreTranspose)
val finalPacked = SerialToParallel(toPacker, isCoreOutWidth, en = true.B)
val toPadding   = DecoupledMux(doTranspose, transposed, finalPacked, en = true.B)
```

So the transposer is **enabled when all three are true**:

1. `isCoreOutIsFinal` — the IS-core output is heading to TCDM (not to
   another on-chip consumer that bypasses banking).
2. `en_isCoreTranspose` — the SimbaCore mode requests transpose.
3. `en_isCoreRequant` — the output has been requantized to `inType`
   (FP8). Without requant, the output bytes are BF16 and the natural
   per-tile alignment already fills a bank; transpose is unnecessary.

From [SimbaCoreMode.scala](../../src/main/scala/simbacore/SimbaCoreMode.scala),
the only mode that sets `en_isCoreTranspose = 1` is **`PHASE1`** — the
xProj output of a Mamba block.

Other modes' relation to the transposer:

| Mode | `en_isCoreTranspose` | Notes |
|---|---|---|
| `PHASE1` | 1 | IS-core output is consumed by SU-core via banks → needs transpose |
| `PHASE1_NO_REQUANT` | 0 | BF16 output, no transpose needed |
| `PHASE2`, `PHASE2_NO_REQUANT` | 0 | IS-core is the final stage, host-read does not need transposed banks |
| `ISGEMM`, `ISGEMM_NO_REQUANT` | 0 | Same — final output |
| `ISGEMM_SQ` | 0 | Square mode without hardware transpose. The data generator still applies `bankTranspose` in software (see `flattenPartition1` for FFT) — that models the streamer's stride configuration, not a hardware transposer pass. |
| `ISGEMM_SQ_TRANSPOSE` | 1 | Square mode with the hardware transposer enabled. Pick this mode whenever a downstream core (SIMD, another IS-core stage) reads the IS-core's FP8 square-mode output via banks. |

### 8.1.7 Software-side helper

Use the `bankTranspose` reference whenever you need to predict what the
hardware will write into TCDM after the transposer fires. Two canonical
call sites:

- [DataGeneratorMain.scala::flattenXProjOut](../../src/test/scala/datagen/DataGeneratorMain.scala#L27)
  — Phase-1 xProj output (`elemPerBank = BANKWIDTH / inType.width = 8`
  for FP8).
- [DataGeneratorFFT.scala::flattenPartition1](../../src/test/scala/datagen/DataGeneratorFFT.scala#L146)
  — FFT partition-1 output, where step 2 (SIMD CMul) reads the banked
  layout.

Skeleton:

```scala
bankTranspose(
  flattenCD(...),                          // IS-core CD layout
  tileSize    = seqLenUnroll,              // = elementsPerCycle (16 in MySystem)
  elemPerBank = BANKWIDTH / inType.width   // = 8 for FP8, 4 for BF16
)
```

If your call site doesn't apply `padWithZeros` afterwards, double-check
that the output type is `accType`; otherwise the byte sizes won't match
the streamer's expected stride.

### 8.1.8 When you can skip it

Skip `bankTranspose` (and the corresponding `en_isCoreTranspose`) when:

- The IS-core output is in `accType` (BF16) and not requantized — one
  output tile already fills one slot per bank cleanly.
- The IS-core output is the **final** output of the block and the host
  reads it back as a flat blob without striding across banks. See
  `DataGeneratorFFT.flattenPartition2` for a non-transposed final output.
- The output is consumed by an on-chip core that does not go through
  TCDM banks (e.g. directly piped into the Switch-core in Phase-2).

In all other cases — particularly Phase-1 xProj output, and any FP8
IS-core output that another core reads back from TCDM — the bank
transpose is required and `en_isCoreTranspose` must be set.

## 8.2 `padWithZeros(x, psumType, finalType)`

The IS-core's output port is always sized for `psumType` (BF16). When the
final cast is to a narrower type (e.g., FP8), the streamer keeps the BF16
slot and zero-pads the unused high bits. The reference flow mimics that:

```scala
def padWithZeros(x: Vec, psumType: FpType, finalType: FpType): Vec = {
  val totalLength = (psumType.width / finalType.width) * x.length
  x ++ Seq.fill(totalLength - x.length)(0f)
}
```

For BF16→FP8 this appends `x.length` zeros (one per real element).

Apply only to the FP8 *output* — bias C is read in BF16, so it does **not**
need padding.

## 8.3 Matrix padding

Defined in `TensorUtils`:

- `padMatrix(x, dim0, dim1)` — pads with zeros to `(dim0, dim1)` (rows
  bottom, cols right).
- `extendDim(dim, divisor)` — `ceil(dim / divisor) * divisor`.
- `padMatrixToMultiple(x, divisor_0, divisor_1)` — combine the two:
  pad each dim to a multiple of the matching divisor.
- `padMatrixColumnsPerBlock(x, rowDivisor, blockSize, paddedBlockSize)` —
  splits each row into `blockSize`-wide column blocks, pads each block
  individually to `paddedBlockSize`, then pads rows to a multiple of
  `rowDivisor`.

`padMatrixColumnsPerBlock` is used by the FFT path to align each L1- or
L2-sized weight column-block to `dInnerUnroll` (the OS-/IS-core's column
tile width).

## 8.4 `duplicateEachElement(x, duplicates)`

```scala
def duplicateEachElement(x: Vec, duplicates: Int): Vec =
  x.map(xi => Seq.fill(duplicates)(xi)).flatten
```

Used in `DataGeneratorRmsNorm` to broadcast a per-channel weight across the
SIMD lanes:

```scala
val weightFlat = duplicateEachElement(weight, params.simdLanes_bf16)
```

The SIMD core sees the same weight on every lane within a SIMD tile.

## 8.5 SIMD reals/imags interleaving

Defined in `DataGeneratorFFT.scala`:

```scala
def simdInterleaveRealImag(reals: Seq[Float], imags: Seq[Float]): Seq[Float] =
  reals.grouped(16).zip(imags.grouped(16))
       .flatMap { case (r, i) => r ++ i }.toSeq
```

The FP8 SIMD core has 32 lanes. In `cmul` mode it expects the first 16 lanes
to be reals and the next 16 to be imags — for every tile of 32 lanes. The
helper packs vectors of reals/imags into this `[Re×16][Im×16]` pattern.

The BF16 SIMD core uses a different lane count (`simdLanes_bf16`), and most
BF16 modes are real-only — so this interleaving is FP8 / FFT specific.

## 8.6 Identity flattens (good fallbacks)

Use these when a buffer doesn't fit any named layout:

- `matrix.flatten` — row-major flatten. Used for `conv_weight`, `dt_bias`,
  scalar SU-core inputs, SIMD inputs.
- `vec` (Seq[Float]) — used directly when the buffer is 1-D.

## 8.7 `MatmulDims` (re-)construction tricks

- `MatmulDims.fromTensorSizes(dim0, dim1, dim2, arrayDims)` — preferred.
- `MatmulDims.fromTensorAB(a, b, arrayDims)` — when only the matrices are
  in hand.
- For the IS-core final out in RMSNorm, the data generator uses a
  *workaround*: a fake matmul reduction dim of `dInnerUnroll` just to
  reuse `flattenCD`. See
  [DataGeneratorTrait.scala::flattenIsCoreOut](../../src/test/scala/datagen/DataGeneratorTrait.scala#L59):

  ```scala
  // NOTE this is just a workaround to use the tensorUtils flatten function,
  // we don't care about the GEMM reduction dim
  val dims = MatmulDims.fromTensorSizes(L, dInnerUnroll, D, params.isCoreParams.dimensions)
  flattenCD(x, params.isCoreLoopOrder)(dims)
  ```

  Translation: when you just want IS-core CD layout for an `(L, D)` matrix
  without an actual matmul to plan, pick any dummy `K·Ku ≥ Ku` value that
  makes `MatmulDims` happy.

## 8.8 Validity checks the generator performs

These rules apply to *every* generator. Failing them is a hard programmer
error — the streamer cannot recover:

From `DataGeneratorTrait.validateParams`:

- `seqLen % seqLenUnroll == 0`
- `dInner * expand % dInnerUnroll == 0`
  (i.e. inner dim must be a multiple of `dInnerUnroll / expand`)
- If a `dtRank` is involved:
  - `dtRank % dtRankUnroll == 0`
  - `dtRank * inType.width % BANKWIDTH == 0` (must fill a bank cleanly)

From Switch-core:

- `dInner % (convUnroll * convRingDepth) == 0`
- `seqLen % convTilesizeL == 0`
- `dtRank % Ku == 0`, `Ku > dConv`, `(Ku - dConv)` is power of two

From SU-core:

- `dInner % delaySU == 0`
- `dState` is a power of two
- `dInnerUnroll % delaySU == 0`

If you generate buffers programmatically, run the same checks before laying
them out — silent corruption otherwise.
