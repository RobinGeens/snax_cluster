// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// Tiled 2-way partitioned EinFFT. Phase A is L2-tiled (full dModel per tile); Phase B is
// K-tiled. See docs/dataflow/05_fft.md §5.2 for dataflow, tile-axis
// rationale, TCDM overlay map, and L3 layout.
//
// Phase A uses nb_tiles (can be large) with ping-pong DMA/compute overlap.
// Phase B uses nb_tiles_B (limited by K_2) with serial DMA/compute.

#include "../data/data.h"
#include "snax-simbacore-lib.h"

static inline uint32_t align64(uint32_t x) { return (x + 63u) & ~63u; }

int test() {
    int err = 0;

    // ---------- L3 buffers --------------------------------------------------
    static uint8_t* ptr_hadamard_reordered_l3 = NULL;
    if (snrt_global_core_idx() == 0) {
        (void)snrt_l3alloc(16 * 1024);
        ptr_hadamard_reordered_l3 = (uint8_t*)snrt_l3alloc(M6_length_hadamard_reordered);
    }
    snrt_cluster_hw_barrier();

    // ---------- TCDM allocation --------------------------------------------
    void* tcdm_base_ptr         = snrt_l1_next();
    uint8_t* ptr_weight1        = (uint8_t*)tcdm_base_ptr;
    uint8_t* ptr_weight2        = ptr_weight1 + align64(M6_length_weight1);
    uint8_t* ptr_twiddles_tiled = ptr_weight2 + align64(M6_length_weight2);

    uint8_t* ptr_work_base = ptr_twiddles_tiled + align64(M6_length_twiddles_tiled_total);

    // ---- Phase A: double-buffered working slots (ping-pong) ----
    uint8_t* ptr_in_tile[2];
    uint8_t* ptr_partition1_out_tile[2];
    uint8_t* ptr_hadamard_out_tile[2];
    uint8_t* ptr_had_reord_a_tile[2];

    ptr_in_tile[0]             = ptr_work_base;
    ptr_in_tile[1]             = ptr_in_tile[0] + align64(M6_length_in_tile);
    ptr_partition1_out_tile[0] = ptr_in_tile[1] + align64(M6_length_in_tile);
    ptr_partition1_out_tile[1] = ptr_partition1_out_tile[0] + align64(M6_length_partition1_out_tile);
    ptr_hadamard_out_tile[0]   = ptr_partition1_out_tile[1] + align64(M6_length_partition1_out_tile);
    ptr_hadamard_out_tile[1]   = ptr_hadamard_out_tile[0] + align64(M6_length_hadamard_out_tile);
    ptr_had_reord_a_tile[0]    = ptr_hadamard_out_tile[1] + align64(M6_length_hadamard_out_tile);
    ptr_had_reord_a_tile[1]    = ptr_had_reord_a_tile[0] + align64(M6_length_hadamard_reordered_tile);

    // Phase B: overlays Phase A region after barrier. Single-buffered.
    uint8_t* ptr_had_reord_b_ktile = ptr_work_base;
    uint8_t* ptr_partition2_out    = ptr_had_reord_b_ktile + align64(M6_length_hadamard_reordered_ktile);

    if (snrt_global_core_idx() == 0) init_cycle_counter();
    snrt_cluster_hw_barrier();

    // Preload always-live small inputs: weights + twiddles
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_weight1, M6_dft_weight1, M6_length_weight1);
        snrt_dma_start_1d(ptr_weight2, M6_dft_weight2, M6_length_weight2);
        snrt_dma_start_1d(ptr_twiddles_tiled, M6_twiddles, M6_length_twiddles);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    uint32_t start_cycles            = 0;
    uint32_t phaseA_end_cycles       = 0;
    uint32_t simbacore_cycles_phaseA = 0;
    uint32_t simbacore_cycles_phaseB = 0;
    static uint32_t _dma_done = 0, _compute_done = 0;

    if (snrt_global_core_idx() == 0) {
        printf("\nFFT tiled: L=%d D=%d L1=%d L2=%d nb_tiles=%d nb_tiles_B=%d\n", seqLen, dModel, L1, L2, nb_tiles,
               M6_nb_tiles_B);
        printf("Expected L1 TCDM usage: %u B (%u KiB)\n", (uint32_t)L1_TCDM_PEAK_BYTES,
               (uint32_t)(L1_TCDM_PEAK_BYTES / 1024));
        printf("Phase A tile: in=%d p1out=%d hadout=%d reord=%d\n", M6_length_in_tile, M6_length_partition1_out_tile,
               M6_length_hadamard_out_tile, M6_length_hadamard_reordered_tile);
        start_cycles = snrt_mcycle();
    }

    // Zero both ping-pong buffers ONCE up front. Streamers leave the fixed padding cells
    // untouched (downstream reads them), but each tile's compute fully overwrites the data
    // region. This way, we don't need to zero this region for each tile
    if (snrt_is_dm_core()) {
        for (int b = 0; b < 2; b++) {
            snrt_dma_start_1d(ptr_partition1_out_tile[b], (void*)snrt_zero_memory_ptr(), M6_length_partition1_out_tile);
            snrt_dma_start_1d(ptr_hadamard_out_tile[b], (void*)snrt_zero_memory_ptr(), M6_length_hadamard_out_tile);
            snrt_dma_start_1d(ptr_had_reord_a_tile[b], (void*)snrt_zero_memory_ptr(),
                              M6_length_hadamard_reordered_tile);
        }
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    // ========================================================================
    // Phase A tile loop: partition1 + hadamard + reorder, tile along L2.
    // 3-stage pipeline: DMA load(i) || compute(i-1) || DMA spill(i-2).
    // ========================================================================

    for (uint32_t i = 0; i < nb_tiles + 2; i++) {
        int buf = i & 1;

        if (snrt_is_dm_core()) {
            // Start spill BEFORE load (FIFO ordering ensures spill reads complete before load's zero-fill overwrites
            // the same slot).
            if (i >= 2) {
                uint32_t spill_tile = i - 2;
                int sbuf            = spill_tile & 1;
                uint8_t* reals_dst  = ptr_hadamard_reordered_l3 + spill_tile * M6_phaseA_dma_per_tile_per_part;
                uint8_t* imags_dst  = reals_dst + (M6_length_hadamard_reordered / 2);
                snrt_dma_start_1d(reals_dst, ptr_had_reord_a_tile[sbuf], M6_phaseA_dma_per_tile_per_part);
                snrt_dma_start_1d(imags_dst, ptr_had_reord_a_tile[sbuf] + M6_length_hadamard_reordered_tile_re,
                                  M6_phaseA_dma_per_tile_per_part);
            }

            if (i < nb_tiles) {
                // dft_in is global K-tile-major ([K-tile0: all cols | K-tile1: ...]). Each
                // Phase A tile is a dModel-block needing ALL K-tiles, so gather them with a
                // 2D DMA (one chunk per K-tile) into a per-tile K-tile-major slot. For
                // L1 == seqLenUnroll there is a single K-tile and this is a plain 1D copy.
                uint32_t n_ktile   = L1 / 16;
                uint32_t per_ktile = M6_length_in_tile / n_ktile;
                snrt_dma_start_2d(ptr_in_tile[buf], M6_dft_in + i * per_ktile, per_ktile, per_ktile,
                                  M6_length_in / n_ktile, n_ktile);
            }

            snrt_dma_wait_all();
            // First iteration: check if time(DMA) < time(compute) for latency hiding
            if (i == 1) _dma_done = snrt_mcycle();
        }

        if (i >= 1 && i <= nb_tiles && snrt_global_core_idx() == 0) {
            uint32_t tile = i - 1;
            int cbuf      = tile & 1;

            // ---- Step 1: partition1 (IS-core) ----
            // Tile 0: full setup. Tile > 0: CSRs preloaded from prev Step 2B.
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

            // Preload Step 2 SIMD streamers during IS-core compute
            set_simd_streamer_csr((uint32_t)ptr_partition1_out_tile[cbuf], M6_R7_2_ss, M6_R7_2_tb, M6_R7_2_ts,
                                  (uint32_t)ptr_twiddles_tiled, M6_R13_2_ss, M6_R13_2_tb, M6_R13_2_ts,
                                  (uint32_t)ptr_hadamard_out_tile[cbuf], M6_W3_2_ss, M6_W3_2_tb, M6_W3_2_ts);
            while (read_csr(SIMBACORE_BUSY));
            while (read_csr(STREAMER_BUSY_CSR));
            simbacore_cycles_phaseA += read_simbacore_perf_counter();

            asm volatile("fence" ::: "memory");

            // ---- Step 2: hadamard CMUL (streamers preloaded, just MODE+START) ----
            write_csr(MODE, M20_SIMD_CMUL_FP8);
            _set_streamer_start();
            _set_simbacore_start();
            write_csr(STREAMER_START_CSR, 0);
            write_csr(SIMBACORE_START, 0);
            // Preload Step 2B SIMD streamers during CMUL compute
            set_simd_streamer_no_b((uint32_t)ptr_hadamard_out_tile[cbuf], M6_R7_2B_ss, M6_R7_2B_tb, M6_R7_2B_ts,
                                   (uint32_t)ptr_had_reord_a_tile[cbuf], M6_W3_2B_ss, M6_W3_2B_tb, M6_W3_2B_ts);
            while (read_csr(SIMBACORE_BUSY));
            while (read_csr(STREAMER_BUSY_CSR));
            simbacore_cycles_phaseA += read_simbacore_perf_counter();

            // ---- Step 2B: reorder NOOP (streamers preloaded, just MODE+START) ----
            write_csr(MODE, M23_SIMD_NOOP_FP8);
            _set_streamer_start();
            _set_simbacore_start();
            write_csr(STREAMER_START_CSR, 0);
            write_csr(SIMBACORE_START, 0);
            // Preload next tile's Step 1 IS-core CSRs during NOOP compute
            if (tile < nb_tiles - 1) {
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
            // Clean delayed-start state so Phase B starts clean
            write_csr(DELAYED_START_READER_10, 0);
            write_csr(DELAYED_START_READER_11, 0);
        }

        snrt_cluster_hw_barrier();
    }

    if (snrt_global_core_idx() == 0) phaseA_end_cycles = snrt_mcycle();

    // ========================================================================
    // Phase B: partition 2, K-axis tiled. The partition2_out zero-init (DMA core) overlaps
    // the weight2/simbacore CSR setup (core 0) and the tile-0 had_reord prefetch instead of
    // running serially before them.
    // ========================================================================
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_partition2_out, (void*)snrt_zero_memory_ptr(), M6_length_partition2_out);
        snrt_dma_start_1d(ptr_had_reord_b_ktile, ptr_hadamard_reordered_l3, M6_length_hadamard_reordered_ktile);
    }
    if (snrt_global_core_idx() == 0) {
        set_isgemm_streamer_csr((uint32_t)ptr_weight2, M6_R11_3_ss, M6_R11_3_tb, M6_R11_3_ts,
                                (uint32_t)ptr_had_reord_b_ktile, M6_R12_3_ss, M6_R12_3_tb, M6_R12_3_ts,
                                (uint32_t)ptr_partition2_out, M6_W3_3_ss, M6_W3_3_tb, M6_W3_3_ts);
        set_simbacore_csr(M6_ISGEMM_SQ, 2 * L2, 1, M6_dInner_2_tile, 1, dModel * L1);
    }
    if (snrt_is_dm_core()) snrt_dma_wait_all();
    snrt_cluster_hw_barrier();

    for (uint32_t tile = 0; tile < M6_nb_tiles_B; tile++) {
        // tile 0 is already prefetched above; load tile>0 here (single-buffered).
        if (snrt_is_dm_core() && tile > 0) {
            snrt_dma_start_1d(ptr_had_reord_b_ktile,
                              ptr_hadamard_reordered_l3 + tile * M6_length_hadamard_reordered_ktile,
                              M6_length_hadamard_reordered_ktile);
            snrt_dma_wait_all();
        }

        snrt_cluster_hw_barrier();

        if (snrt_global_core_idx() == 0) {
            write_csr(BASE_PTR_READER_11_LOW, (uint32_t)(ptr_weight2 + tile * M6_length_weight2_ktile));
            write_csr(BASE_PTR_READER_12_LOW, (uint32_t)ptr_had_reord_b_ktile);
            write_csr(MODE, (tile == M6_nb_tiles_B - 1) ? M6_ISGEMM_SQ : M30_ISGEMM_SQ_NO_REQUANT);
            start_simbacore_and_streamers(M6_R10_en, 0, 1, 0);
            wait_simbacore_and_streamer();
            simbacore_cycles_phaseB += read_simbacore_perf_counter();
        }

        snrt_cluster_hw_barrier();
    }

    // --- Verification ---
    if (snrt_global_core_idx() == 0) {
        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Simbacore elapsed time: %u cycles\n", end_cycles,
               simbacore_cycles_phaseA + simbacore_cycles_phaseB);
        printf("[%d cc] Simbacore Phase A (sum over tiles, 3 steps each): %u cycles\n", end_cycles,
               simbacore_cycles_phaseA);
        printf("[%d cc] Simbacore Phase B (sum over tiles): %u cycles\n", end_cycles, simbacore_cycles_phaseB);
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);
        printf("  Phase A wall: %u cc (compute %u)  Phase B wall: %u cc (compute %u)\n",
               phaseA_end_cycles - start_cycles, simbacore_cycles_phaseA, end_cycles - phaseA_end_cycles,
               simbacore_cycles_phaseB);
        printf("DMA latency hiding: phaseA=%s\n", _dma_done < _compute_done ? "ok" : "STALL");

        err += check_result_sample(ptr_hadamard_reordered_l3, M6_hadamard_reordered, M6_test_samples_expected,
                                   nb_test_samples, "hadamard_reordered (L3)");
        err += check_result_sample(ptr_partition2_out, M6_partition2_expected, M6_test_samples_expected,
                                   nb_test_samples, "partition2_out (TCDM)");

        printf("Test FFT tiled (Phase A L2-tiled, Phase B K-tiled): (%d x %d), nb_tiles=%d/%d\n", seqLen, dModel,
               nb_tiles, M6_nb_tiles_B);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, 2 * nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test(); }
