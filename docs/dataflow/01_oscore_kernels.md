# 1. OS-core kernels: `osgemm` and `osgemm-tiled`

> See also: [byte layouts for OS-core A, B, D](../../../chisel-ssm/docs/memory_layouts/02_gemm_layouts.md).
> This page covers **program structure**, not byte layout.

The OS-core computes `D[dim0, dim2] = A[dim0, dim1] · B[dim1, dim2]` in
`N_M_K` loop order. Both A and B are read in `flattenA` / `flattenB`. The
output D is written in **ConvFormat** (see
[memory_layouts/03](../../../chisel-ssm/docs/memory_layouts/03_conv_format.md)).

## 1.1 `osgemm` — single-shot

**Source**: [osgemm/src/osgemm.c](../../target/snitch_cluster/sw/apps/osgemm/src/osgemm.c).

**TCDM allocation** (offsets from `snrt_l1_next()`):

```
[ A (M3_addr_a) | B (M3_addr_b) | D (M3_addr_d) ]
```

**DMA-in**: A and B are DMA'ed once from L3. There is no input C / bias —
the OS-core does not consume a psum.

**Compute** (single stage):

| Step | Streamer ports        | MODE         | `set_simbacore_csr` args                |
| ---- | --------------------- | ------------ | --------------------------------------- |
| 1    | `R0=A`, `R1=B`, `W0=D`| `M3_OSGEMM`  | `dim0, dim1, dim2, 1, 1`                |

`start_simbacore_and_streamers(M3_R10_en, 0, M3_R11_en, 0)` is called; only
`R0/R1/W0` are active in this program. R10/R11 enables are passed as data.h
constants — in OSGEMM mode both are 0 (no Phase-2 forwarding).

## 1.2 `osgemm-tiled` — dInner-tiled with a 3-stage pipeline

**Source**: [osgemm-tiled/src/main.c](../../target/snitch_cluster/sw/apps/osgemm-tiled/src/main.c).

**What is tiled**: the **dInner axis** (`dim2`, the OS-core's N axis). A is the
non-reduction-non-tiled axis input and is loaded **once** at startup; B is the
tiled input; D is the tiled output.

**TCDM allocation**:

```
[ A (full) | B ping | B pong | D ping | D pong | D_full ]
```

`D_full` is a contiguous TCDM buffer sized at `M3_length_d` that emulates the
off-chip destination for `transfer_out`. (The L3 `M3_D` slot is the golden
reference and is read-only.)

**3-stage pipeline** (per iteration `i ∈ [0, nb_tiles + 2)`):

| Iteration window     | Stage                                              | Core        |
| -------------------- | -------------------------------------------------- | ----------- |
| `i ∈ [0, nb_tiles)`  | DMA-in B-tile `i` into the ping/pong buffer        | DM core     |
| `i ∈ [1, nb_tiles+1)`| Compute tile `i−1`: `D_tile = A · B_tile`          | core 0      |
| `i ∈ [2, nb_tiles+2)`| DMA-out D-tile `i−2` from D ping/pong → `D_full`   | DM core     |

The ping/pong index is `buf = (i-1) % 2` for the compute stage; the same `buf`
is used for the matching transfer-out two iterations later. Each iteration
ends with `snrt_dma_wait_all()` (DM core) and a cluster barrier.

**Streamer / MODE setup**: configured **once** with per-tile bounds:

```c
set_osgemm_streamer_csr(ptr_a, R0_ss/tb/ts,
                        ptr_b[0], R1_ss/tb/ts,
                        ptr_d[0], W0_ss/tb/ts);
set_simbacore_csr(M3_OSGEMM, dim0, dim1, M3_dim2_tile, 1, 1);
```

Note `dim2` becomes `M3_dim2_tile` — the streamer sees a smaller N per tile;
the *full* output is assembled in `D_full` by the transfer-out stage.

**Per-tile CSR updates** (compute stage):

```c
write_csr(BASE_PTR_READER_1_LOW, ptr_b[buf]);
write_csr(BASE_PTR_WRITER_0_LOW, ptr_d[buf]);
start_simbacore_and_streamers(...);
wait_simbacore_and_streamer();
```

Only the **two** moving base pointers are rewritten. `A`'s base pointer,
bounds, strides, and the simbacore CSRs are constants set once before the loop.

**Why dInner (not dim0 or dim1)?** dim1 is the OS-core *reduction* axis — it
would require accumulation logic that the OS-core does not have (no psum
read-back port). dim0 would tile the sequence-length axis, requiring per-tile
A-tile loads and per-tile B-tile reuse — strictly worse than reusing the
single shared A. dInner tiles the output's N axis, which is independent and
maps cleanly to ping/pong B-tiles.

## 1.3 Recreate the program

To recreate `osgemm-tiled` from scratch:

1. Allocate `A | B0 | B1 | D0 | D1 | D_full` in TCDM.
2. Preload A once, barrier.
3. Configure the OS-core streamer **once** with per-tile bounds (only
   `M3_dim2_tile` differs from the untiled version).
4. Loop `nb_tiles + 2` iterations. Inside the loop, fire DMA-in / compute /
   DMA-out based on the iteration index, rewrite the two moving base pointers
   for the compute stage, and end with `dma_wait_all + cluster_hw_barrier`.
5. After the loop, check `D_full` against the golden reference.
