# Performance optimization techniques for tiled Snitch+SimbaCore programs

> **All pages:**
> [README](README.md) ·
> [1. OS-core kernels](01_oscore_kernels.md) ·
> [2. IS-core kernels](02_iscore_kernels.md) ·
> [3. SIMD / RMSNorm kernels](03_simd_kernels.md) ·
> [4. Mamba main](04_mamba_main.md) ·
> [5. FFT family](05_fft.md) ·
> [6. EinFFT MLP](06_einfft_mlp.md) ·
> [7. VMamba SS2D](07_vmamba.md) ·
> **8. Performance optimization (this page)** ·
> [9. Async tiling](09_async_tiling.md)

SW-level techniques for reducing Snitch elapsed time (wall-clock cycles) in
tiled programs that pipeline DMA transfers with SimbaCore kernel invocations.

---

## 1. CSR preloading during busy-wait

**Problem.**  Between two back-to-back kernel invocations the CPU must write new
base-pointer CSRs, then assert START.  Each `write_csr` takes ~2-5 cycles
through the SNAX interface.  With 5-12 base pointers per tile, this adds 10-60
dead cycles where the accelerator sits idle.

**Technique.**  After asserting START, immediately de-assert the start signals
and write the *next* tile's CSRs while the current tile is still computing.  The
streamer AGU latches its configuration at the rising edge of START; subsequent
CSR writes only update the register file and take effect on the next START.

```c
_set_streamer_start();              // latch current CSRs, begin compute
_set_simbacore_start();
write_csr(STREAMER_START_CSR, 0);   // de-assert (HW finishes current config)
write_csr(SIMBACORE_START, 0);

// --- overlapped with compute ---
write_csr(BASE_PTR_READER_1_LOW, next_tile_ptr);   // updates register file only
write_csr(BASE_PTR_READER_3_LOW, next_tile_wgt);
...
// --- end overlap ---

while (read_csr(SIMBACORE_BUSY));      // poll for completion
while (read_csr(STREAMER_BUSY_CSR));
```

On the **next** tile the CPU just asserts START (2 writes) — zero base-pointer
setup on the critical path.

**Rules.**
- The first tile gets its CSRs from the initial `set_streamer_*()` call — no
  preload from a previous iteration.
- The tile before a configuration change (e.g. the last bulk tile before a
  finalStep kernel) must NOT preload, because the next kernel rewrites all CSRs.
- `MODE` (SimbaCore config): do NOT preload during compute.  The HW reads MODE
  in real-time (e.g. to gate requant on the final K-step).  Always write MODE
  before START.
- Streamer base-pointer CSRs and temporal bounds/strides ARE safe to preload
  (latched at START, not read continuously).

---

## 2. Printf removal from hot loops

`printf` inside the tile loop costs ~2000+ hardware cycles per call (integer
formatting + putc_buffer writes to L3).  This is often the single largest source
of overhead.

Remove per-tile printf from the hot loop.  Print only phase-level messages and
the final timing summary.  Keep per-tile printf behind `#ifdef VERBOSE` for
debugging.

---

## 3. Inline start/wait when delayed streamers are unused

The library functions `start_simbacore_and_streamers()` and
`wait_simbacore_and_streamer()` handle all streamers generically, including
R10/R11 delayed start.  When delayed streamers are disabled (e.g. P1 kernels),
the delay-gauge polling and DELAYED_START clear writes are wasted.

Inline the start+wait for these kernels:

```c
_set_streamer_start();
_set_simbacore_start();
write_csr(STREAMER_START_CSR, 0);
write_csr(SIMBACORE_START, 0);
// skip DELAYED_START_READER_10/11 clear — they're disabled
while (read_csr(SIMBACORE_BUSY));
while (read_csr(STREAMER_BUSY_CSR));
```

Saves 4 CSR writes + 2 poll loops per tile vs the library function.  Combine
with CSR preloading (section 1) by inserting the preload writes between the
de-assert and the BUSY poll.

---

## 4. DMA / compute overlap

### 4.1 Three-stage pipelined loop

The standard tiled loop overlaps DMA with compute using a 3-stage pipeline
and double-buffered TCDM slots:

```
iter i:  DMA LOAD(i)  ||  COMPUTE(i-1)  ||  DMA SPILL(i-2)
         ─────────────────────────────────────────────────────
                          hw_barrier
```

The DMA core and compute core run their branches in parallel within each
iteration; the hardware barrier at the end synchronizes them.  The pipeline
needs `nb_tiles + 2` iterations (2 extra for fill and drain).

### 4.2 Ping-pong buffer sizing

Each tiled tensor needs two TCDM slots (buf[0], buf[1]).  The DMA loads into
`buf[i % 2]` while the accelerator reads from `buf[(i-1) % 2]`.  Slots must
be 64-byte aligned to match AXI burst width and sparse-interconnect bank
granularity.

### 4.3 Flag-based sync (experimental)

Replacing the hw_barrier with TCDM-based volatile flags (`dma_done`,
`compute_done`) would let the faster core run ahead without waiting.  The flags
must be in TCDM (not `static` / L3) for cross-core visibility.  This was
attempted but requires careful handling of the DMA engine's transaction
ordering.

---

## 5. Data-array alignment for DMA

The DMA engine requires source and destination addresses to be aligned to the
AXI data width (8 bytes).  The Python datagen emits test-data arrays without
alignment attributes by default.  Depending on the binary layout, arrays may
land at 2- or 4-byte aligned addresses, causing `Misaligned Load/Store` faults
on the `dmsrc` instruction.

**Fix** (in `datagen_base.py`):

```python
def format_vector(self, type, var_name, value):
    self.lines_data.append(format_vector_definition(type, var_name, value, alignment=8))
```

This emits `__attribute__((aligned(8)))` on every data array in `data.h`.
The `alignment` parameter is already supported by `data_utils.py`'s
`format_vector_definition()` — it was just never passed.

---
