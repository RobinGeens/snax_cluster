// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// Tiled 2-way partitioned EinFFT. Phase A is dModel-tiled with ping-pong DMA/compute overlap
// (the fused CMul write emits each tile in flattenB order, so the L3 spill is one chunk per
// k2-group); Phase B is K-tiled into the full psum, opened by a K=1 head launch that reads its
// psum from a 64 B zero block. See docs/dataflow/05_fft.md "fft-tiled".

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
    uint8_t* ptr_zero           = ptr_twiddles_tiled + align64(M6_length_twiddles_tiled_total);  // 64 B

    uint8_t* ptr_work_base = ptr_zero + 64;

    // ---- Phase A: double-buffered working slots (ping-pong) ----
    uint8_t* ptr_in_tile[2];
    uint8_t* ptr_partition1_out_tile[2];
    uint8_t* ptr_had_reord_a_tile[2];

    ptr_in_tile[0]             = ptr_work_base;
    ptr_in_tile[1]             = ptr_in_tile[0] + align64(M6_length_in_tile);
    ptr_partition1_out_tile[0] = ptr_in_tile[1] + align64(M6_length_in_tile);
    ptr_partition1_out_tile[1] = ptr_partition1_out_tile[0] + align64(M6_length_partition1_out_tile);
    ptr_had_reord_a_tile[0]    = ptr_partition1_out_tile[1] + align64(M6_length_partition1_out_tile);
    ptr_had_reord_a_tile[1]    = ptr_had_reord_a_tile[0] + align64(M6_length_hadamard_reordered_tile);

    // Phase B: overlays Phase A region after barrier. Single-buffered.
    uint8_t* ptr_had_reord_b_ktile = ptr_work_base;
    uint8_t* ptr_partition2_out    = ptr_had_reord_b_ktile + align64(M6_length_hadamard_reordered_ktile);
    // K-tile 0 can load behind Phase A's last spill when its slot stays below the had_reord tiles.
    const bool ktile0_early = ptr_had_reord_b_ktile + M6_length_hadamard_reordered_ktile <= ptr_had_reord_a_tile[0];
    uint32_t start_cycles   = 0;

    if (snrt_global_core_idx() == 0) {
        printf("\nFFT tiled: L=%d D=%d L1=%d L2=%d nb_tiles=%d nb_tiles_B=%d\n", seqLen, dModel, L1, L2, nb_tiles,
               M6_nb_tiles_B);
        printf("Expected L1 TCDM usage: %u B (%u KiB)\n", (uint32_t)L1_TCDM_PEAK_BYTES,
               (uint32_t)(L1_TCDM_PEAK_BYTES / 1024));
        printf("Phase A tile: in=%d p1out=%d reord=%d\n", M6_length_in_tile, M6_length_partition1_out_tile,
               M6_length_hadamard_reordered_tile);
        init_cycle_counter();
        start_cycles = snrt_mcycle();
    }

    // Preload always-live small inputs: weights + twiddles
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_weight1, M6_dft_weight1, M6_length_weight1);
        snrt_dma_start_1d(ptr_weight2, M6_dft_weight2, M6_length_weight2);
        snrt_dma_start_1d(ptr_twiddles_tiled, M6_twiddles, M6_length_twiddles);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    uint32_t phaseA_end_cycles       = 0;
    uint32_t simbacore_cycles_phaseA = 0;
    uint32_t simbacore_cycles_phaseB = 0;
    static uint32_t _dma_done = 0, _compute_done = 0;

    // Zero the partition1 psum ping-pong slots ONCE up front (streamers leave the fixed padding
    // cells untouched, and each tile's compute fully overwrites the data region) and the zero
    // block Phase B's head launch reads its psum from.
    if (snrt_is_dm_core()) {
        for (int b = 0; b < 2; b++) simba_dma_fill_zero(ptr_partition1_out_tile[b], M6_length_partition1_out_tile);
        simba_dma_fill_zero(ptr_zero, 64);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    // Phase B launch programs: K-tile 0's head (K=1, psum read from the zero block: R13 base
    // and stride 0), its tail (the remaining K steps) and full K-tiles. The streamer latches
    // its program at start, so each is written during the previous launch.
#define CFG_P2(N, W, B, FROM_ZERO)                                                                                \
    do {                                                                                                         \
        set_isgemm_streamer_csr((uint32_t)(W), M6_R11_##N##_ss, M6_R11_##N##_tb, M6_R11_##N##_ts, (uint32_t)(B),   \
                                M6_R12_##N##_ss, M6_R12_##N##_tb, M6_R12_##N##_ts, (uint32_t)ptr_partition2_out, \
                                M6_W3_##N##_ss, M6_W3_##N##_tb, M6_W3_##N##_ts);                                 \
        if (FROM_ZERO) {                                                                                         \
            write_csr(BASE_PTR_READER_13_LOW, (uint32_t)ptr_zero);                                               \
            write_csr(T_STRIDE_BASE_READER_13 + 0, 0);                                                           \
        }                                                                                                        \
    } while (0)
#define CFG_P2_FULL(T) CFG_P2(3, ptr_weight2 + (T) * M6_length_weight2_ktile, ptr_had_reord_b_ktile, false)

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
                // Spill the tile into the flattenB (K-tile-major) L3 layout the partition-2 IS-core
                // reads: the tile already holds each k2-group's [d][m] blocks contiguously, so one
                // chunk per k2-group and re/im lands at (k2g, d0).
                uint8_t* tile_re = ptr_had_reord_a_tile[sbuf];
                uint8_t* tile_im = tile_re + M6_length_hadamard_reordered_tile_re;
                uint8_t* l3_re   = ptr_hadamard_reordered_l3;
                uint8_t* l3_im   = l3_re + (M6_length_hadamard_reordered / 2);
                uint32_t d0      = spill_tile * M6_dModel_tile;
                for (uint32_t k2g = 0; k2g < M6_nk_2; k2g++) {
                    uint32_t dst = k2g * M6_spill_l3_kstride + d0 * L1 * seqLenUnroll;
                    snrt_dma_start_1d(l3_re + dst, tile_re + k2g * M6_spill_chunk, M6_spill_chunk);
                    snrt_dma_start_1d(l3_im + dst, tile_im + k2g * M6_spill_chunk, M6_spill_chunk);
                }
            }
            // Phase B's K-tile 0 rides behind the last spill.
            if (i == nb_tiles + 1 && ktile0_early)
                snrt_dma_start_1d(ptr_had_reord_b_ktile, ptr_hadamard_reordered_l3, M6_length_hadamard_reordered_ktile);

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
            // Tile 0: full setup. Tile > 0: CSRs preloaded during prev tile's Step 2.
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

            // Preload Step 2 SIMD streamers during IS-core compute. The fused CMUL writer
            // scatters re/im halves straight into the reorder tile slot (W3 2-D spatial).
            set_simd_streamer_csr((uint32_t)ptr_partition1_out_tile[cbuf], M6_R7_2_ss, M6_R7_2_tb, M6_R7_2_ts,
                                  (uint32_t)ptr_twiddles_tiled, M6_R13_2_ss, M6_R13_2_tb, M6_R13_2_ts,
                                  (uint32_t)ptr_had_reord_a_tile[cbuf], M6_W3_2_ss, M6_W3_2_tb, M6_W3_2_ts);
            while (read_csr(SIMBACORE_BUSY));
            while (read_csr(STREAMER_BUSY_CSR));
            simbacore_cycles_phaseA += read_simbacore_perf_counter();

            asm volatile("fence" ::: "memory");

            // ---- Step 2: hadamard CMUL + fused re/im reorder (streamers preloaded) ----
            write_csr(MODE, M20_SIMD_CMUL_FP8);
            _set_streamer_start();
            _set_simbacore_start();
            write_csr(STREAMER_START_CSR, 0);
            write_csr(SIMBACORE_START, 0);
            // Preload next tile's Step 1 IS-core CSRs (or Phase B's head launch) during CMUL compute
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
            } else {
                CFG_P2(3H, ptr_weight2, ptr_had_reord_b_ktile, true);
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

    if (snrt_global_core_idx() == 0) {
        phaseA_end_cycles = snrt_mcycle();
    }

    // ========================================================================
    // Phase B: partition 2, K-axis tiled into the FULL partition2_out. K-tile 0 opens the
    // accumulation with a K=1 head launch reading its psum from the zero block, then its tail;
    // later K-tiles accumulate (single-buffered slot); the last launch requantizes.
    // ========================================================================
#define LAUNCH_P2(FINAL, K_ELEMS)                                                                             \
    do {                                                                                                     \
        set_simbacore_csr((FINAL) ? M6_ISGEMM_SQ : M30_ISGEMM_SQ_NO_REQUANT, 2 * L2, 1, (K_ELEMS), 1, dModel * L1); \
        start_simbacore_and_streamers(M6_R10_en, 0, 1, 0);                                                   \
    } while (0)
#define FINISH_P2()                                                   \
    do {                                                              \
        wait_simbacore_and_streamer();                                \
        simbacore_cycles_phaseB += read_simbacore_perf_counter();     \
    } while (0)

    for (uint32_t tile = 0; tile < M6_nb_tiles_B; tile++) {
        if (snrt_is_dm_core() && (tile > 0 || !ktile0_early)) {
            snrt_dma_start_1d(ptr_had_reord_b_ktile,
                              ptr_hadamard_reordered_l3 + tile * M6_length_hadamard_reordered_ktile,
                              M6_length_hadamard_reordered_ktile);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();  // K-tile slot loaded

        if (snrt_global_core_idx() == 0) {
            bool last = tile + 1 == M6_nb_tiles_B;
            if (tile == 0) {
                // head (CSRs preloaded during Phase A's last CMUL), then the tail if K_2_t > 1
                LAUNCH_P2(last && M6_K_2_t == 1, dInnerUnroll);
                if (M6_K_2_t > 1)
                    CFG_P2(3T, ptr_weight2 + M6_weight2_kmacro_bytes, ptr_had_reord_b_ktile + M6_had_reord_kmacro_bytes,
                           false);
                else if (!last)
                    CFG_P2_FULL(1);
                FINISH_P2();
                if (M6_K_2_t > 1) {
                    LAUNCH_P2(last, M6_dInner_2_tail);
                    if (!last) CFG_P2_FULL(1);
                    FINISH_P2();
                }
            } else {
                LAUNCH_P2(last, M6_dInner_2_tile);
                if (!last) CFG_P2_FULL(tile + 1);
                FINISH_P2();
            }
        }
        snrt_cluster_hw_barrier();  // K-tile slot consumed
    }
#undef CFG_P2
#undef CFG_P2_FULL
#undef LAUNCH_P2
#undef FINISH_P2

    // --- Verification ---
    if (snrt_global_core_idx() == 0) {
        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Simbacore elapsed time: %u cycles\n", end_cycles,
               simbacore_cycles_phaseA + simbacore_cycles_phaseB);
        printf("[%d cc] Simbacore Phase A (sum over tiles, 2 steps each): %u cycles\n", end_cycles,
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
