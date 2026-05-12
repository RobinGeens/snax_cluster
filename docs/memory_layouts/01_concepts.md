# 1. Concepts and conventions

> **All pages:**
> [README](README.md) ·
> **1. Concepts (this page)** ·
> [2. GEMM layouts](02_gemm_layouts.md) ·
> [3. ConvFormat](03_conv_format.md) ·
> [4. SUCFormat](04_suc_format.md) ·
> [5. xProj layout](05_xproj_format.md) ·
> [6. Delta weight split](06_delta_weight.md) ·
> [7. Per-mode reference](07_mode_reference.md) ·
> [8. Helpers](08_helpers.md) ·
> [9. FFT](09_fft.md) ·
> [10. SIMD](10_simd.md)

This page defines the vocabulary used by every other layout doc. Read it once
and refer back when the names "tile", "Mu/Ku/Nu", "bank", "convUnroll" appear.

## 1.1 Tensor model

The SNAX accelerator works on 2-D matrices. The reference Python/Scala model
sees a matrix as `Seq[Seq[Float]]` (rows of values). In hardware, every matrix
is sliced into **tiles**:

- A tile is the unit that the spatial array processes in one step.
- The tile shape comes from the array shape parameters `Mu`, `Ku`, `Nu`.
- Outside the tile, there is a **temporal** loop iterating over `M`, `K`, `N`.
  The temporal loop order is fixed per core (see [02](02_gemm_layouts.md)).

`utils.MatmulDims(M, K, N, Mu, Ku, Nu)` encapsulates these dimensions. Helpers:

- `MatmulDims.fromTensorSizes(dim0, dim1, dim2, Seq(Mu, Ku, Nu))` builds it from
  full tensor sizes.
- `dim0 = M*Mu`, `dim1 = K*Ku`, `dim2 = N*Nu`.

## 1.2 Cores and their array dimensions

From [src/main/scala/mambacore/MambaCoreParams.scala](../../src/main/scala/mambacore/MambaCoreParams.scala):

| Core         | `Mu` | `Ku`           | `Nu`           | Loop order | Stationarity |
|--------------|------|----------------|----------------|------------|--------------|
| OS-core      | `seqLenUnroll` | 1              | `dInnerUnroll` | `N_M_K`    | Output-stationary |
| IS-core      | `seqLenUnroll` | `dInnerUnroll` | 1              | `K_M_N`    | Input-stationary |
| Switch-core  | 1    | `Ku` (= `dtRankUnroll`) | `convUnroll` | `N_M_K` | Output-stationary |
| SU-core      | (vector) `delaySU` lanes; reduction tree over `N = dState` | — | — | — | — |

Hardwired constants for the production system
([SystemParams.scala](../../src/main/scala/simbacore/SystemParams.scala)):

- `BANKWIDTH = 64` bits (= 8 FP8 elements, or 4 BF16 elements).
- `N_BANKS = 32`.
- `seqLenUnroll = 16`, `dInnerUnroll = 24`, `dtRankUnroll = 6`, `dConv = 4`,
  `dState = 64`, `delaySU = 4`, `convUnroll = delaySU = 4`.

## 1.3 The fixed loop orders (and why)

`LoopOrder` lives in
[src/main/scala/utils/LoopOrder.scala](../../src/main/scala/utils/LoopOrder.scala).
For a matmul `D = A · B + C`:

| Loop order | Outer → inner | Used for |
|---|---|---|
| `M_N_K` | M → N → K | (generic / VersaCore tests) |
| `N_M_K` | N → M → K | **OS-core**, **Switch-core** |
| `K_M_N` | K → M → N | **IS-core** |

The loop order determines:

- Which axis is traversed fastest in the flattened input buffer.
- Whether tiles inside a buffer are row-major or col-major across the temporal
  outer loops.
- The stationarity setting (`OutputStationary` for `M_N_K`/`N_M_K`,
  `InputStationary` for `K_M_N`).

The exact mapping is in `flattenA`/`flattenB`/`flattenCD` (see [02](02_gemm_layouts.md)).

## 1.4 FP types

The MambaCore is parametrized with two FP types
([SsmParams.scala](../../src/main/scala/mambacore/SsmParams.scala)):

- `inType` — activation/weight type for matmul inputs. In `MySystem`: `FP8_ALT`
  (8 bits, ~exp4/mant3 with alt bias).
- `accType` — accumulator / psum / bias type. In `MySystem`: `BF16` (16 bits).

Implications for memory:

- Every matmul **bias C** is stored in `accType`. The IS-core takes its `C`
  input straight in BF16; the OSGEMM-mode `C` is also BF16.
- Every matmul **output D** is produced in `accType`. If a subsequent step
  needs FP8 (e.g., to feed another matmul), the OS-core or IS-core
  requantizes inline; the *byte size* of the buffer in TCDM, however, stays at
  `accType.width` — see `padWithZeros` rule in [02](02_gemm_layouts.md).
- The SIMD core has two flavours: BF16 lanes (`simdLanes_bf16 = (2 *
  suCoreSerialWidthBC) / accType.width`) and FP8 lanes (`simdLanes_fp8`).

## 1.5 Banks and bank width

TCDM is structured as `N_BANKS` banks of `BANKWIDTH` bits each. Streamers
read/write parallel-by-bank.

Two things to remember:

1. Many serial widths are *aligned to* `BANKWIDTH` even when the natural
   width would be smaller. `extendToBankWidth` does this rounding
   ([SimbaCoreUtil.scala:15](../../src/main/scala/simbacore/SimbaCoreUtil.scala#L15)).
2. The IS-core's output port is sized to `BANKWIDTH` along the seqLen direction,
   so its outputs are emitted in "banked" groups. The reference flow uses
   `bankTranspose` to mimic this — see [08](08_helpers.md).

## 1.6 Generated data layout on disk

Every data generator writes:

- `M<id>_<name>.bin` — one line per element, decimal UInt of the raw FP bits.
  `<id>` is the integer assigned to the `SimbaCoreMode`
  (`SimbaCoreMode.getId`).
- `params.hjson` — algorithmic + hardware parameters consumed by the SNAX app.

Files are written into `generated/data/<dirName>/`.
See [DataGeneratorTrait.scala](../../src/test/scala/datagen/DataGeneratorTrait.scala).

## 1.7 Common shorthand

- `seqLen` (or `L`) — sequence-length axis. Maps to `dim0` of a matmul.
- `dInner` (or `D`) — inner channel axis. Maps to `dim2` for OS-core, to
  `dim1` for IS-core.
- `dtRank` — delta-projection rank (Mamba-specific).
- `dState` — SSM hidden state size (`N`).
- `dConv` — 1-D convolution depth.
- `expand` — `dInner = expand * dModel`.
