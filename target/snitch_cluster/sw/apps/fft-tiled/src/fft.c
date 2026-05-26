// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// Tiled 2-way partitioned EinFFT. Phase A is dModel-tiled; Phase B is
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

    // ---------- Preload always-live small inputs: weights + twiddles ----
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_weight1, M6_dft_weight1, M6_length_weight1);
        snrt_dma_start_1d(ptr_weight2, M6_dft_weight2, M6_length_weight2);
        snrt_dma_start_1d(ptr_twiddles_tiled, M6_twiddles, M6_length_twiddles);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    uint32_t start_cycles            = 0;
    uint32_t simbacore_cycles_phaseA = 0;
    uint32_t simbacore_cycles_phaseB = 0;
    static uint32_t _dma_done = 0, _compute_done = 0;

    if (snrt_global_core_idx() == 0) {
        printf("\nFFT tiled: L=%d D=%d L1=%d L2=%d nb_tiles=%d nb_tiles_B=%d\n", seqLen, dModel, L1, L2, nb_tiles,
               M6_nb_tiles_B);
        printf("Phase A tile: in=%d p1out=%d hadout=%d reord=%d\n", M6_length_in_tile, M6_length_partition1_out_tile,
               M6_length_hadamard_out_tile, M6_length_hadamard_reordered_tile);
        start_cycles = snrt_mcycle();
    }

    // ========================================================================
    // Phase A tile loop: partition1 + hadamard + reorder, tile along L2.
    // 3-stage pipeline: DMA load(i) || compute(i-1) || DMA spill(i-2).
    // ========================================================================

    for (uint32_t i = 0; i < nb_tiles + 2; i++) {
        int buf = i & 1;

        if (snrt_is_dm_core()) {
            // Start spill BEFORE load (FIFO ordering ensures spill reads
            // complete before load's zero-fill overwrites the same slot).
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
                snrt_dma_start_1d(ptr_in_tile[buf], M6_dft_in + i * M6_length_in_tile, M6_length_in_tile);
                snrt_dma_start_1d(ptr_partition1_out_tile[buf], (void*)snrt_zero_memory_ptr(),
                                  M6_length_partition1_out_tile);
                snrt_dma_start_1d(ptr_hadamard_out_tile[buf], (void*)snrt_zero_memory_ptr(),
                                  M6_length_hadamard_out_tile);
                snrt_dma_start_1d(ptr_had_reord_a_tile[buf], (void*)snrt_zero_memory_ptr(),
                                  M6_length_hadamard_reordered_tile);
            }

            snrt_dma_wait_all();
            // First iteration: check if time(DMA) < time(compute) for latency hiding
            if (i == 1) _dma_done = snrt_mcycle();
        }

        if (i >= 1 && i <= nb_tiles && snrt_global_core_idx() == 0) {
            uint32_t tile = i - 1;
            int cbuf      = tile & 1;

            // ---- Step 1: partition1 (IS-core) ----
            // Full streamer setup every tile: Steps 2/2B clobber IS-core bounds.
            set_isgemm_streamer_csr((uint32_t)ptr_weight1, M6_R11_1_ss, M6_R11_1_tb, M6_R11_1_ts,
                                    (uint32_t)ptr_in_tile[cbuf], M6_R12_1_ss, M6_R12_1_tb, M6_R12_1_ts,
                                    (uint32_t)ptr_partition1_out_tile[cbuf], M6_W3_1_ss, M6_W3_1_tb, M6_W3_1_ts);
            set_simbacore_csr(M7_ISGEMM_SQ_TRANSPOSE, 2 * L1, 1, L1_padded, 1, M6_N_1_tile);
            start_simbacore_and_streamers(M6_R10_en, 0, 1, 0);
            wait_simbacore_and_streamer();
            simbacore_cycles_phaseA += read_simbacore_perf_counter();

            asm volatile("fence" ::: "memory");

            // ---- Step 2: hadamard CMUL (SIMD) ----
            set_simd_streamer_csr((uint32_t)ptr_partition1_out_tile[cbuf], M6_R7_2_ss, M6_R7_2_tb, M6_R7_2_ts,
                                  (uint32_t)ptr_twiddles_tiled, M6_R13_2_ss, M6_R13_2_tb, M6_R13_2_ts,
                                  (uint32_t)ptr_hadamard_out_tile[cbuf], M6_W3_2_ss, M6_W3_2_tb, M6_W3_2_ts);
            set_simbacore_csr(M20_SIMD_CMUL_FP8, 0, 0, 0, 0, 0);
            start_simbacore_and_streamers(0, 0, 0, 0);
            wait_simbacore_and_streamer();
            simbacore_cycles_phaseA += read_simbacore_perf_counter();

            // ---- Step 2B: reorder NOOP (SIMD) ----
            set_simd_streamer_no_b((uint32_t)ptr_hadamard_out_tile[cbuf], M6_R7_2B_ss, M6_R7_2B_tb, M6_R7_2B_ts,
                                   (uint32_t)ptr_had_reord_a_tile[cbuf], M6_W3_2B_ss, M6_W3_2B_tb, M6_W3_2B_ts);
            set_simbacore_csr(M23_SIMD_NOOP_FP8, 0, 0, 0, 0, 0);
            start_simbacore_and_streamers(0, 0, 0, 0);
            wait_simbacore_and_streamer();
            simbacore_cycles_phaseA += read_simbacore_perf_counter();
            if (tile == 0) _compute_done = snrt_mcycle();

            printf("[%u cc] Phase A tile %u/%d done\n", snrt_mcycle(), tile + 1, nb_tiles);
        }

        snrt_cluster_hw_barrier();
    }

    // ========================================================================
    // Phase B: partition 2, K-axis tiled. Serial DMA/compute.
    // ========================================================================
    if (snrt_global_core_idx() == 0)
        printf("[%u cc] Phase B: zeroing p2out (%d B)\n", snrt_mcycle(), M6_length_partition2_out);
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_partition2_out, (void*)snrt_zero_memory_ptr(), M6_length_partition2_out);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    if (snrt_global_core_idx() == 0) {
        printf("[%u cc] Phase B: setting CSRs\n", snrt_mcycle());
        set_isgemm_streamer_csr((uint32_t)ptr_weight2, M6_R11_3_ss, M6_R11_3_tb, M6_R11_3_ts,
                                (uint32_t)ptr_had_reord_b_ktile, M6_R12_3_ss, M6_R12_3_tb, M6_R12_3_ts,
                                (uint32_t)ptr_partition2_out, M6_W3_3_ss, M6_W3_3_tb, M6_W3_3_ts);
        set_simbacore_csr(M6_ISGEMM_SQ, 2 * L2, 1, M6_dInner_2_tile, 1, dModel * L1);
    }
    snrt_cluster_hw_barrier();

    for (uint32_t tile = 0; tile < M6_nb_tiles_B; tile++) {
        if (snrt_global_core_idx() == 0)
            printf("[%u cc] Phase B tile %d/%d: DMA load\n", snrt_mcycle(), tile, M6_nb_tiles_B);
        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(ptr_had_reord_b_ktile,
                              ptr_hadamard_reordered_l3 + tile * M6_length_hadamard_reordered_ktile,
                              M6_length_hadamard_reordered_ktile);
            snrt_dma_wait_all();
        }

        snrt_cluster_hw_barrier();

        if (snrt_global_core_idx() == 0) {
            printf("[%u cc] Phase B tile %d/%d: starting compute\n", snrt_mcycle(), tile, M6_nb_tiles_B);
            write_csr(BASE_PTR_READER_11_LOW, (uint32_t)(ptr_weight2 + tile * M6_length_weight2_ktile));
            write_csr(BASE_PTR_READER_12_LOW, (uint32_t)ptr_had_reord_b_ktile);
            write_csr(MODE, (tile == M6_nb_tiles_B - 1) ? M6_ISGEMM_SQ : M30_ISGEMM_SQ_NO_REQUANT);
            start_simbacore_and_streamers(M6_R10_en, 0, 1, 0);
            wait_simbacore_and_streamer();
            simbacore_cycles_phaseB += read_simbacore_perf_counter();
            printf("[%u cc] Phase B tile %d/%d done\n", snrt_mcycle(), tile + 1, M6_nb_tiles_B);
        }

        snrt_cluster_hw_barrier();
    }

    // --- Verification ---
    if (snrt_global_core_idx() == 0) {
        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Simbacore Phase A (sum over tiles, 3 steps each): %u cycles\n", end_cycles,
               simbacore_cycles_phaseA);
        printf("[%d cc] Simbacore Phase B (sum over tiles): %u cycles\n", end_cycles, simbacore_cycles_phaseB);
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);
        printf("DMA latency hiding: phaseA=%s\n", _dma_done < _compute_done ? "hidden" : "STALL");

        err += check_result_sample(ptr_hadamard_reordered_l3, M6_hadamard_reordered, M6_test_samples_expected,
                                   nb_test_samples, "hadamard_reordered (L3)");
        err += check_result_sample(ptr_partition2_out, M6_partition2_expected, M6_test_samples_expected,
                                   nb_test_samples, "partition2_out (TCDM)");

        printf("Test FFT tiled (Phase A dModel-tiled, Phase B K-tiled): (%d x %d), nb_tiles=%d/%d\n", seqLen, dModel,
               nb_tiles, M6_nb_tiles_B);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, 2 * nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test(); }
