// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// Tiled 3-way partitioned EinFFT (L = L1 * L2 * L3).
// See docs/dataflow/05_fft.md §5.4 for the full dataflow, tile-regime
// rationale, TCDM overlay map, L3 layout, and verification caveats.

#include "../data/data.h"
#include "snax-simbacore-lib.h"

static inline uint32_t align64(uint32_t x) { return (x + 63u) & ~63u; }

int test() {
    int err = 0;

    // L3 alloc on core 0; static so the pointer is visible to all cores.
    // First reserve 16 KiB at the L3 base to skip past the snRuntime's
    // `putc_buffer` (sized N_CORES * 1024 B at `_edram`). Without this,
    // Phase A's first DMA-out clobbers `putc_buffer.hdr.size`, and the next
    // printf's `sys_write` syscall reads a corrupted length — flooding stdout
    // with megabytes of garbage. fft-tiled (2-way) has the same overlap but
    // happens to write FP8 values at offset 0..3 that, by luck, decode to a
    // small `size`; this test data hit huge values.
    if (snrt_global_core_idx() == 0) {
        (void)snrt_l3alloc(16 * 1024);
    }
    snrt_cluster_hw_barrier();

    static uint8_t* ptr_hadamard1_packed_l3 = NULL;
    static uint8_t* ptr_hadamard2_packed_l3 = NULL;
    if (snrt_global_core_idx() == 0) {
        ptr_hadamard1_packed_l3 = (uint8_t*)snrt_l3alloc(M6_length_hadamard1_packed);
        ptr_hadamard2_packed_l3 = (uint8_t*)snrt_l3alloc(M6_length_hadamard2_packed);
    }
    snrt_cluster_hw_barrier();

    // TCDM allocation. Phase A/B/C working regions all overlay at ptr_work_base.
    void* tcdm_base_ptr    = snrt_l1_next();
    uint8_t* ptr_weight1   = (uint8_t*)tcdm_base_ptr;
    uint8_t* ptr_weight2   = ptr_weight1 + align64(M6_length_weight1);
    uint8_t* ptr_weight3   = ptr_weight2 + align64(M6_length_weight2);
    uint8_t* ptr_twiddles1 = ptr_weight3 + align64(M6_length_weight3);
    uint8_t* ptr_twiddles2 = ptr_twiddles1 + align64(M6_length_twiddles1);
    uint8_t* ptr_work_base = ptr_twiddles2 + align64(M6_length_twiddles2);

    // Phase A slots — double-buffered ping-pong so DMA(load/spill) overlaps compute.
    uint8_t* ptr_in_tile[2];
    uint8_t* ptr_partition1_out_tile[2];
    uint8_t* ptr_hadamard1_out_tile[2];
    uint8_t* ptr_hadamard1_packed_a[2];
    ptr_in_tile[0]             = ptr_work_base;
    ptr_in_tile[1]             = ptr_in_tile[0] + align64(M6_length_in_tile);
    ptr_partition1_out_tile[0] = ptr_in_tile[1] + align64(M6_length_in_tile);
    ptr_partition1_out_tile[1] = ptr_partition1_out_tile[0] + align64(M6_length_partition1_out_tile);
    ptr_hadamard1_out_tile[0]  = ptr_partition1_out_tile[1] + align64(M6_length_partition1_out_tile);
    ptr_hadamard1_out_tile[1]  = ptr_hadamard1_out_tile[0] + align64(M6_length_hadamard1_out_tile);
    ptr_hadamard1_packed_a[0]  = ptr_hadamard1_out_tile[1] + align64(M6_length_hadamard1_out_tile);
    ptr_hadamard1_packed_a[1]  = ptr_hadamard1_packed_a[0] + align64(M6_length_hadamard1_packed_tile);

    // Phase B slots (hadamard2_out overlays hadamard1_packed_full after partition2 consumes it)
    uint8_t* ptr_hadamard1_packed_full = ptr_work_base;
    uint8_t* ptr_partition2_out        = ptr_hadamard1_packed_full + align64(M6_length_hadamard1_packed);
    uint8_t* ptr_hadamard2_out         = ptr_hadamard1_packed_full;
    uint8_t* ptr_hadamard2_packed      = ptr_partition2_out + align64(M6_length_partition2_out);

    // Phase C slots
    uint8_t* ptr_hadamard2_packed_b_ktile = ptr_work_base;
    uint8_t* ptr_partition3_out           = ptr_hadamard2_packed_b_ktile + align64(M6_length_hadamard2_packed_ktile);

    if (snrt_global_core_idx() == 0) init_cycle_counter();
    snrt_cluster_hw_barrier();

    // Weights and twiddles depend only on (l1, l2, l3), not d — loaded once.
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_weight1, M6_dft_weight1, M6_length_weight1);
        snrt_dma_start_1d(ptr_weight2, M6_dft_weight2, M6_length_weight2);
        snrt_dma_start_1d(ptr_weight3, M6_dft_weight3, M6_length_weight3);
        snrt_dma_start_1d(ptr_twiddles1, M6_twiddles1, M6_length_twiddles1);
        snrt_dma_start_1d(ptr_twiddles2, M6_twiddles2, M6_length_twiddles2);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    uint32_t start_cycles            = 0;
    uint32_t simbacore_cycles_phaseA = 0;
    uint32_t simbacore_cycles_phaseB = 0;
    uint32_t simbacore_cycles_phaseC = 0;
    if (snrt_global_core_idx() == 0) {
        printf(
            "\nStarting program: tiled 3-way FFT (nb_tiles_A=%d, nb_tiles_C=%d, L=%d, dModel=%d, "
            "L1=%d, L2=%d, L3=%d)\n\n",
            nb_tiles_A, nb_tiles_C, seqLen, dModel, L1, L2, L3);
        start_cycles = snrt_mcycle();
    }

    // Zero both ping-pong buffers ONCE up front. Each tile's compute fully overwrites
    // the data region; the streamers leave the fixed padding cells (read downstream)
    // untouched, so a single zero-fill keeps them valid for every tile.
    if (snrt_is_dm_core()) {
        for (int b = 0; b < 2; b++) {
            snrt_dma_start_1d(ptr_partition1_out_tile[b], (void*)snrt_zero_memory_ptr(),
                              M6_length_partition1_out_tile);
            snrt_dma_start_1d(ptr_hadamard1_out_tile[b], (void*)snrt_zero_memory_ptr(), M6_length_hadamard1_out_tile);
            snrt_dma_start_1d(ptr_hadamard1_packed_a[b], (void*)snrt_zero_memory_ptr(),
                              M6_length_hadamard1_packed_tile);
        }
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    // ========================================================================
    // Phase A tile loop: partition1 + hadamard1 + reorder1, dModel-tiled.
    // 3-stage pipeline: DMA load(i) || compute(i-1) || DMA spill(i-2), with
    // inlined start/wait + CSR preloading (docs/dataflow/08_performance_optimization.md).
    // ========================================================================
    for (uint32_t i = 0; i < nb_tiles_A + 2; i++) {
        int buf = i & 1;

        if (snrt_is_dm_core()) {
            // Spill tile i-2 first (FIFO order: spill reads complete before the load
            // below overwrites the same slot).
            if (i >= 2) {
                uint32_t spill_tile = i - 2;
                int sbuf            = spill_tile & 1;
                uint8_t* reals_dst  = ptr_hadamard1_packed_l3 + spill_tile * M6_phaseA_dma_per_tile_per_part;
                uint8_t* imags_dst  = reals_dst + (M6_length_hadamard1_packed / 2);
                snrt_dma_start_1d(reals_dst, ptr_hadamard1_packed_a[sbuf], M6_phaseA_dma_per_tile_per_part);
                snrt_dma_start_1d(imags_dst, ptr_hadamard1_packed_a[sbuf] + M6_length_hadamard1_packed_tile_re,
                                  M6_phaseA_dma_per_tile_per_part);
            }
            if (i < nb_tiles_A) {
                snrt_dma_start_1d(ptr_in_tile[buf], M6_dft_in + i * M6_length_in_tile, M6_length_in_tile);
            }
            snrt_dma_wait_all();
        }

        if (i >= 1 && i <= nb_tiles_A && snrt_global_core_idx() == 0) {
            uint32_t tile = i - 1;
            int cbuf      = tile & 1;

            // ---- Step 1: partition1 (IS-core). Tile 0: full setup; later tiles get
            //      their Step 1 CSRs preloaded by the previous tile's Step 2B. ----
            if (tile == 0) {
                set_isgemm_streamer_csr((uint32_t)ptr_weight1, M6_R11_1_ss, M6_R11_1_tb, M6_R11_1_ts,
                                        (uint32_t)ptr_in_tile[0], M6_R12_1_ss, M6_R12_1_tb, M6_R12_1_ts,
                                        (uint32_t)ptr_partition1_out_tile[0], M6_W3_1_ss, M6_W3_1_tb, M6_W3_1_ts);
                write_csr(SEQ_LEN, 2 * L1);
                write_csr(D_MODEL, 1);
                write_csr(D_INNER, L1_padded);
                write_csr(DT_RANK, 1);
                write_csr(D_FINAL, M6_N_1_tile);
            }
            write_csr(MODE, M7_ISGEMM_SQ_TRANSPOSE);
            _set_streamer_start();
            _set_simbacore_start();
            if (M6_R10_en) write_csr(DELAYED_START_READER_10, 1);
            write_csr(DELAYED_START_READER_11, 1);
            write_csr(STREAMER_START_CSR, 0);
            write_csr(SIMBACORE_START, 0);
            if (M6_R10_en) write_csr(DELAYED_START_READER_10, 0);
            write_csr(DELAYED_START_READER_11, 0);
            // Preload Step 2 SIMD streamers during IS-core compute.
            set_simd_streamer_csr((uint32_t)ptr_partition1_out_tile[cbuf], M6_R7_2_ss, M6_R7_2_tb, M6_R7_2_ts,
                                  (uint32_t)ptr_twiddles1, M6_R13_2_ss, M6_R13_2_tb, M6_R13_2_ts,
                                  (uint32_t)ptr_hadamard1_out_tile[cbuf], M6_W3_2_ss, M6_W3_2_tb, M6_W3_2_ts);
            while (read_csr(SIMBACORE_BUSY));
            while (read_csr(STREAMER_BUSY_CSR));
            simbacore_cycles_phaseA += read_simbacore_perf_counter();

            asm volatile("fence" ::: "memory");

            // ---- Step 2: hadamard1 CMUL (streamers preloaded, just MODE + START). ----
            write_csr(MODE, M20_SIMD_CMUL_FP8);
            _set_streamer_start();
            _set_simbacore_start();
            write_csr(STREAMER_START_CSR, 0);
            write_csr(SIMBACORE_START, 0);
            // Preload Step 2B SIMD streamers during CMUL compute.
            set_simd_streamer_no_b((uint32_t)ptr_hadamard1_out_tile[cbuf], M6_R7_2B_ss, M6_R7_2B_tb, M6_R7_2B_ts,
                                   (uint32_t)ptr_hadamard1_packed_a[cbuf], M6_W3_2B_ss, M6_W3_2B_tb, M6_W3_2B_ts);
            while (read_csr(SIMBACORE_BUSY));
            while (read_csr(STREAMER_BUSY_CSR));
            simbacore_cycles_phaseA += read_simbacore_perf_counter();

            // ---- Step 2B: reorder1 NOOP (streamers preloaded, just MODE + START). ----
            write_csr(MODE, M23_SIMD_NOOP_FP8);
            _set_streamer_start();
            _set_simbacore_start();
            write_csr(STREAMER_START_CSR, 0);
            write_csr(SIMBACORE_START, 0);
            // Preload next tile's Step 1 IS-core streamers during NOOP compute. MODE is
            // NOT preloaded (HW reads it in real time) — only the streamers + dims, which
            // are identical every tile.
            if (tile < nb_tiles_A - 1) {
                int nbuf = (tile + 1) & 1;
                set_isgemm_streamer_csr((uint32_t)ptr_weight1, M6_R11_1_ss, M6_R11_1_tb, M6_R11_1_ts,
                                        (uint32_t)ptr_in_tile[nbuf], M6_R12_1_ss, M6_R12_1_tb, M6_R12_1_ts,
                                        (uint32_t)ptr_partition1_out_tile[nbuf], M6_W3_1_ss, M6_W3_1_tb, M6_W3_1_ts);
                write_csr(SEQ_LEN, 2 * L1);
                write_csr(D_MODEL, 1);
                write_csr(D_INNER, L1_padded);
                write_csr(DT_RANK, 1);
                write_csr(D_FINAL, M6_N_1_tile);
            }
            while (read_csr(SIMBACORE_BUSY));
            while (read_csr(STREAMER_BUSY_CSR));
            simbacore_cycles_phaseA += read_simbacore_perf_counter();
            // Clear delayed-start state so Phase B starts clean.
            write_csr(DELAYED_START_READER_10, 0);
            write_csr(DELAYED_START_READER_11, 0);
        }

        snrt_cluster_hw_barrier();
    }

    // Phase B: partition2 + hadamard2 + reorder2, un-tiled.
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_hadamard1_packed_full, ptr_hadamard1_packed_l3, M6_length_hadamard1_packed);
        snrt_dma_start_1d(ptr_partition2_out, (void*)snrt_zero_memory_ptr(), M6_length_partition2_out);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    if (snrt_global_core_idx() == 0) {
        // Step 3: partition2 (TRANSPOSE for hadamard2's banked port). IS-core.
        // Three steps run back-to-back on core 0 with inlined start/wait + CSR
        // preloading (docs/dataflow/08_performance_optimization.md §1, §3).
        set_isgemm_streamer_csr((uint32_t)ptr_weight2, M6_R11_3_ss, M6_R11_3_tb, M6_R11_3_ts,
                                (uint32_t)ptr_hadamard1_packed_full, M6_R12_3_ss, M6_R12_3_tb, M6_R12_3_ts,
                                (uint32_t)ptr_partition2_out, M6_W3_3_ss, M6_W3_3_tb, M6_W3_3_ts);
        write_csr(SEQ_LEN, 2 * L2);
        write_csr(D_MODEL, 1);
        write_csr(D_INNER, 2 * L2_padded);
        write_csr(DT_RANK, 1);
        write_csr(D_FINAL, dModel * L1 * L3);
        write_csr(MODE, M7_ISGEMM_SQ_TRANSPOSE);
        _set_streamer_start();
        _set_simbacore_start();
        if (M6_R10_en) write_csr(DELAYED_START_READER_10, 1);
        write_csr(DELAYED_START_READER_11, 1);
        write_csr(STREAMER_START_CSR, 0);
        write_csr(SIMBACORE_START, 0);
        if (M6_R10_en) write_csr(DELAYED_START_READER_10, 0);
        write_csr(DELAYED_START_READER_11, 0);
        // Preload Step 4 SIMD streamers during partition2 compute.
        set_simd_streamer_csr((uint32_t)ptr_partition2_out, M6_R7_4_ss, M6_R7_4_tb, M6_R7_4_ts, (uint32_t)ptr_twiddles2,
                              M6_R13_4_ss, M6_R13_4_tb, M6_R13_4_ts, (uint32_t)ptr_hadamard2_out, M6_W3_4_ss,
                              M6_W3_4_tb, M6_W3_4_ts);
        while (read_csr(SIMBACORE_BUSY));
        while (read_csr(STREAMER_BUSY_CSR));
        simbacore_cycles_phaseB += read_simbacore_perf_counter();

        asm volatile("fence" ::: "memory");

        // Step 4: hadamard2 CMUL (streamers preloaded). hadamard2_out overlays hadamard1_packed_full.
        write_csr(MODE, M20_SIMD_CMUL_FP8);
        _set_streamer_start();
        _set_simbacore_start();
        write_csr(STREAMER_START_CSR, 0);
        write_csr(SIMBACORE_START, 0);
        // Preload Step 4B SIMD streamers during CMUL compute.
        set_simd_streamer_no_b((uint32_t)ptr_hadamard2_out, M6_R7_4B_ss, M6_R7_4B_tb, M6_R7_4B_ts,
                               (uint32_t)ptr_hadamard2_packed, M6_W3_4B_ss, M6_W3_4B_tb, M6_W3_4B_ts);
        while (read_csr(SIMBACORE_BUSY));
        while (read_csr(STREAMER_BUSY_CSR));
        simbacore_cycles_phaseB += read_simbacore_perf_counter();

        // Step 4B: reorder2 NOOP (streamers preloaded, just MODE + START).
        write_csr(MODE, M23_SIMD_NOOP_FP8);
        _set_streamer_start();
        _set_simbacore_start();
        write_csr(STREAMER_START_CSR, 0);
        write_csr(SIMBACORE_START, 0);
        while (read_csr(SIMBACORE_BUSY));
        while (read_csr(STREAMER_BUSY_CSR));
        simbacore_cycles_phaseB += read_simbacore_perf_counter();
        write_csr(DELAYED_START_READER_10, 0);
        write_csr(DELAYED_START_READER_11, 0);
    }
    snrt_cluster_hw_barrier();

    // Spill hadamard2_packed to L3 so Phase C can overlay the TCDM region.
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_hadamard2_packed_l3, ptr_hadamard2_packed, M6_length_hadamard2_packed);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    // Phase C: partition3, K-axis tiled. R13/W3 accumulate in place on partition3_out.
    // Overlap the partition3_out zero-init (DMA core) with the weight3/simbacore CSR
    // setup (core 0) and the tile-0 ktile prefetch (perf-opt §4).
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_partition3_out, (void*)snrt_zero_memory_ptr(), M6_length_partition3_out);
        snrt_dma_start_1d(ptr_hadamard2_packed_b_ktile, ptr_hadamard2_packed_l3, M6_length_hadamard2_packed_ktile);
    }
    if (snrt_global_core_idx() == 0) {
        set_isgemm_streamer_csr((uint32_t)ptr_weight3, M6_R11_5_ss, M6_R11_5_tb, M6_R11_5_ts,
                                (uint32_t)ptr_hadamard2_packed_b_ktile, M6_R12_5_ss, M6_R12_5_tb, M6_R12_5_ts,
                                (uint32_t)ptr_partition3_out, M6_W3_5_ss, M6_W3_5_tb, M6_W3_5_ts);
        set_simbacore_csr(M6_ISGEMM_SQ, 2 * L3, 1, M6_dInner_3_tile, 1, dModel * L1 * L2);
    }
    if (snrt_is_dm_core()) snrt_dma_wait_all();
    snrt_cluster_hw_barrier();

    for (uint32_t tile = 0; tile < nb_tiles_C; tile++) {
        // tile 0 prefetched above; load tile>0 here (single-buffered).
        if (snrt_is_dm_core() && tile > 0) {
            snrt_dma_start_1d(ptr_hadamard2_packed_b_ktile,
                              ptr_hadamard2_packed_l3 + tile * M6_length_hadamard2_packed_ktile,
                              M6_length_hadamard2_packed_ktile);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();

        if (snrt_global_core_idx() == 0) {
            // BASE_PTR_READER_12_LOW (= ptr_hadamard2_packed_b_ktile) and the
            // streamer bounds/strides + SimbaCore CSR were set ONCE above the
            // loop — they don't change per tile. Only weight3's K-slice base
            // and the MODE (NO_REQUANT vs ISGEMM_SQ on final) change here.
            write_csr(BASE_PTR_READER_11_LOW, (uint32_t)(ptr_weight3 + tile * M6_length_weight3_ktile));
            write_csr(MODE, (tile == nb_tiles_C - 1) ? M6_ISGEMM_SQ : M30_ISGEMM_SQ_NO_REQUANT);
            start_simbacore_and_streamers(M6_R10_en, 0, 1, 0);
            wait_simbacore_and_streamer();
            simbacore_cycles_phaseC += read_simbacore_perf_counter();
        }
        snrt_cluster_hw_barrier();
    }

    // --- Verification ---
    if (snrt_global_core_idx() == 0) {
        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Simbacore Phase A (sum over tiles, 3 steps each): %u cycles\n", end_cycles,
               simbacore_cycles_phaseA);
        printf("[%d cc] Simbacore Phase B (partition2 + hadamard2 + reorder2): %u cycles\n", end_cycles,
               simbacore_cycles_phaseB);
        printf("[%d cc] Simbacore Phase C (partition3 K-tiles): %u cycles\n", end_cycles, simbacore_cycles_phaseC);
        printf("[%d cc] Simbacore elapsed time: %u cycles\n", end_cycles,
               simbacore_cycles_phaseA + simbacore_cycles_phaseB + simbacore_cycles_phaseC);
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        // Only the final output is verified — intermediate buffers diverge at byte
        // positions the streamer does not read. See docs/dataflow/05_fft.md §5.4.
        err += check_result_sample(ptr_partition3_out, M6_partition3_expected, M6_test_samples_expected,
                                   nb_test_samples, "partition3_out (TCDM)");

        printf(
            "Test FFT 3-way tiled (Phase A dModel-tiled, Phase C K-tiled, partition2 un-tiled): "
            "(%d x %d), L1=%d L2=%d L3=%d, nb_tiles_A=%d, nb_tiles_C=%d\n",
            seqLen, dModel, L1, L2, L3, nb_tiles_A, nb_tiles_C);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test(); }
