// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// Tiled 3-way partitioned EinFFT (L = L1*L2*L3) with an extra l3-tile loop.
// See docs/dataflow/05_fft.md.

#include "../data/data.h"
#include "snax-simbacore-lib.h"

static inline uint32_t align64(uint32_t x) { return (x + 63u) & ~63u; }

// --- l3-tile gathers from DRAM into contiguous tile-local buffers ---
// in_tile: per k1-tile, copy slice s / l3-tile lt's m3-block (repeat over d).
static inline void gather_in_tile(uint8_t* dst, uint32_t s, uint32_t lt) {
    for (uint32_t kt = 0; kt < M6_in_ktile_count; kt++) {
        uint8_t* src = M6_dft_in + kt * M6_in_ktile_stride + s * M6_in_slice_chunk + lt * M6_in_gather_chunk;
        snrt_dma_start_2d(dst + kt * M6_in_gather_dst_ktile, src, M6_in_gather_chunk,
                          /*dst_stride=*/M6_in_gather_chunk, /*src_stride=*/M6_in_gather_d_stride,
                          /*repeat=*/M6_dModel_slice);
    }
}
// twiddles1 (k1, j=m3*L2+m2): gather the tile's m3 j-block per k1 row.
static inline void gather_tw1(uint8_t* dst, uint32_t lt) {
    snrt_dma_start_2d(dst, M6_twiddles1 + lt * M6_tw1_gather_chunk, M6_tw1_gather_chunk,
                      /*dst_stride=*/M6_tw1_gather_chunk, /*src_stride=*/M6_tw1_gather_src_stride, /*repeat=*/L1);
}
// twiddles2 (l2, l3): gather the tile's L3t columns per l2 row.
static inline void gather_tw2(uint8_t* dst, uint32_t lt) {
    snrt_dma_start_2d(dst, M6_twiddles2 + lt * M6_tw2_gather_chunk, M6_tw2_gather_chunk,
                      /*dst_stride=*/M6_tw2_gather_chunk, /*src_stride=*/M6_tw2_gather_src_stride, /*repeat=*/L2);
}

int test() {
    int err = 0;

    // L3: assembled output + the full H2 (partition-3 input) staged out of TCDM.
    static uint8_t* ptr_output_l3 = NULL;
    static uint8_t* ptr_h2_l3     = NULL;
    if (snrt_global_core_idx() == 0) {
        (void)snrt_l3alloc(16 * 1024);
        ptr_output_l3 = (uint8_t*)snrt_l3alloc(M6_length_partition3_out);
        ptr_h2_l3     = (uint8_t*)snrt_l3alloc(M6_full_h2_bytes);
    }
    snrt_cluster_hw_barrier();

    // TCDM: full weights, tile-local gather + scratch, one gathered N-tile of H2, the N-tile psum.
    void* base           = snrt_l1_next();
    uint8_t* ptr_weight1 = (uint8_t*)base;
    uint8_t* ptr_weight2 = ptr_weight1 + align64(M6_length_weight1);
    uint8_t* ptr_weight3 = ptr_weight2 + align64(M6_length_weight2);
    uint8_t* ptr_in      = ptr_weight3 + align64(M6_length_weight3);
    uint8_t* ptr_tw1     = ptr_in + align64(M6_in_tile_bytes);
    uint8_t* ptr_tw2     = ptr_tw1 + align64(M6_tw1_tile_bytes);
    uint8_t* ptr_P       = ptr_tw2 + align64(M6_tw2_tile_bytes);
    // H2 ping-pong: the cmuls write one slot while the previous tile's stage-out DMA
    // drains the other.
    uint8_t* ptr_H2s[2];
    ptr_H2s[0] = ptr_P + M6_slot_size_tile;
    ptr_H2s[1] = ptr_H2s[0] + align64(M6_hsize_tile);
    // Partition 3 runs after stages 1-4, so overlay its buffers on the dead scratch.
    uint8_t* ptr_h2ntile = ptr_in;
    uint8_t* ptr_P3      = ptr_h2ntile + align64(M6_h2_ntile_bytes);

    uint32_t start_cycles     = 0;
    uint32_t simbacore_cycles = 0;

    if (snrt_global_core_idx() == 0) {
        printf(
            "\nStarting program: tiled 3-way FFT (l3-stream, nb_d=%d, nb_l3=%d, seqLen=%d, dModel=%d, "
            "L1=%d, L2=%d, L3=%d)\n\n",
            M6_nb_d, M6_nb_l3, seqLen, dModel, L1, L2, L3);
        printf("Expected L1 TCDM usage: %u B (%u KiB)\n", (uint32_t)L1_TCDM_PEAK_BYTES,
               (uint32_t)(L1_TCDM_PEAK_BYTES / 1024));
        init_cycle_counter();
        start_cycles = snrt_mcycle();
    }

    // Weights: loaded once, broadcast across all slices and l3-tiles.
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_weight1, M6_dft_weight1, M6_length_weight1);
        snrt_dma_start_1d(ptr_weight2, M6_dft_weight2, M6_length_weight2);
        snrt_dma_start_1d(ptr_weight3, M6_dft_weight3, M6_length_weight3);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

#define CFG_GEMM_P(W, IN_, N, PSUM)                                                                              \
    set_isgemm_streamer_csr((uint32_t)(W), M6_R11_##N##_ss, M6_R11_##N##_tb, M6_R11_##N##_ts, (uint32_t)(IN_),   \
                            M6_R12_##N##_ss, M6_R12_##N##_tb, M6_R12_##N##_ts, (uint32_t)(PSUM), M6_W3_##N##_ss, \
                            M6_W3_##N##_tb, M6_W3_##N##_ts)

    for (uint32_t s = 0; s < M6_nb_d; s++) {
        // --- stages 1-4 per l3-tile: assemble the full H2 (partition-3 input) ---
        // Prologue: gather tile 0's inputs + zero its gemm1 psum. Later tiles' gathers +
        // P-zeros run in the serial DMA windows below (they are only free once the cmul
        // that reads them finishes, and the fused cmuls leave no compute to hide behind).
        if (snrt_is_dm_core()) {
            gather_in_tile(ptr_in, s, 0);
            gather_tw1(ptr_tw1, 0);
            gather_tw2(ptr_tw2, 0);
            snrt_dma_start_1d(ptr_P, (void*)snrt_zero_memory_ptr(), M6_slot_size_tile);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();

        for (uint32_t lt = 0; lt < M6_nb_l3; lt++) {
            uint32_t lt_next = lt + 1;  // tile prefetched in this tile's DMA windows
            uint8_t* H2c     = ptr_H2s[lt & 1];

            if (snrt_global_core_idx() == 0) {
                // gemm1 (in_tile -> P): start, then program cmul1's streamer behind the run.
                CFG_GEMM_P(ptr_weight1, ptr_in, 1, ptr_P);
                set_simbacore_csr(M7_ISGEMM_SQ_TRANSPOSE, 2 * L1, 1, L1_padded, 1, M6_dModel_slice * L2 * M6_l3_tile);
                start_simbacore_and_streamers(M6_R10_en, 0, 1, 0);
                set_simd_streamer_csr((uint32_t)ptr_P, M6_R7_2_ss, M6_R7_2_tb, M6_R7_2_ts, (uint32_t)ptr_tw1,
                                      M6_R13_2_ss, M6_R13_2_tb, M6_R13_2_ts, (uint32_t)H2c, M6_W3_2_ss, M6_W3_2_tb,
                                      M6_W3_2_ts);
                wait_simbacore_and_streamer();
                simbacore_cycles += read_simbacore_perf_counter();

                // cmul1 (P, tw1 -> H2): the fused W3 split write deinterleaves re/im.
                set_simbacore_csr(M20_SIMD_CMUL_FP8, 0, 0, 0, 0, 0);
                start_simbacore_and_streamers(0, 0, 0, 0);
                wait_simbacore_and_streamer();
                simbacore_cycles += read_simbacore_perf_counter();
            }
            // Hidden behind gemm1+cmul1: stage the previous tile's H2 [re|im] halves out to
            // the L3 re/im regions (cmul1 writes the other slot) + gather this tile's tw2
            // (its buffer is free since the previous cmul2).
            if (snrt_is_dm_core() && lt > 0) {
                uint8_t* H2p = ptr_H2s[(lt - 1) & 1];
                snrt_dma_start_1d(ptr_h2_l3 + (lt - 1) * M6_h2_half_bytes, H2p, M6_h2_half_bytes);
                snrt_dma_start_1d(ptr_h2_l3 + M6_h2_im_region + (lt - 1) * M6_h2_half_bytes,
                                  H2p + M6_h2_half_bytes, M6_h2_half_bytes);
                gather_tw2(ptr_tw2, lt);
                snrt_dma_wait_all();
            }
            snrt_cluster_hw_barrier();  // A: H2 ready, P free (cmul1 read it); in & tw1 consumed

            // Serial DMA window: zero P for gemm2 (only free once cmul1 read it; nothing to
            // hide behind). The next tile's in & tw1 gathers ride behind gemm2+cmul2 instead.
            if (snrt_is_dm_core()) {
                snrt_dma_start_1d(ptr_P, (void*)snrt_zero_memory_ptr(), M6_slot_size_tile);
                snrt_dma_wait_all();
            }
            snrt_cluster_hw_barrier();  // B: P zeroed

            // Hidden behind gemm2+cmul2: prefetch next tile's in & tw1 (dead since gemm1/cmul1).
            if (snrt_is_dm_core() && lt_next < M6_nb_l3) {
                gather_in_tile(ptr_in, s, lt_next);
                gather_tw1(ptr_tw1, lt_next);
                snrt_dma_wait_all();
            }
            if (snrt_global_core_idx() == 0) {
                // gemm2 (H2 -> P): start, then program cmul2's streamer behind the run.
                CFG_GEMM_P(ptr_weight2, H2c, 3, ptr_P);
                set_simbacore_csr(M7_ISGEMM_SQ_TRANSPOSE, 2 * L2, 1, 2 * L2_padded, 1,
                                  M6_dModel_slice * L1 * M6_l3_tile);
                start_simbacore_and_streamers(M6_R10_en, 0, 1, 0);
                set_simd_streamer_csr((uint32_t)ptr_P, M6_R7_4_ss, M6_R7_4_tb, M6_R7_4_ts, (uint32_t)ptr_tw2,
                                      M6_R13_4_ss, M6_R13_4_tb, M6_R13_4_ts, (uint32_t)H2c, M6_W3_4_ss, M6_W3_4_tb,
                                      M6_W3_4_ts);
                wait_simbacore_and_streamer();
                simbacore_cycles += read_simbacore_perf_counter();

                // cmul2 (P, tw2 -> H2): fused re/im split write (gemm2 has consumed H2).
                set_simbacore_csr(M20_SIMD_CMUL_FP8, 0, 0, 0, 0, 0);
                start_simbacore_and_streamers(0, 0, 0, 0);
                wait_simbacore_and_streamer();
                simbacore_cycles += read_simbacore_perf_counter();
                asm volatile("fence" ::: "memory");
            }
            snrt_cluster_hw_barrier();  // C: H2 ready (deinterleaved), P free; tw2 consumed

            // Serial DMA window: zero the psum for the next tile's gemm1 (only free once
            // cmul2 is done). This tile's H2 stage-out rides behind the next tile's
            // gemm1+cmul1 (or ahead of the first N-tile gather for the last tile).
            if (snrt_is_dm_core() && lt_next < M6_nb_l3) {
                snrt_dma_start_1d(ptr_P, (void*)snrt_zero_memory_ptr(), M6_slot_size_tile);
                snrt_dma_wait_all();
            }
            snrt_cluster_hw_barrier();  // D: next psum ready
        }

        // --- partition 3: one full-K GEMM per N-tile, then scatter. ---
        for (uint32_t nt = 0; nt < M6_nb_ntile; nt++) {
            // Gather this N-tile's full-K H2 (every K-step's N-run) from L3 in one 2-D transfer.
            if (snrt_is_dm_core()) {
                // First N-tile: stage the last l3-tile's H2 out first (no compute window was
                // left to hide it behind). The DMA engine runs transfers in issue order, so
                // the gather below reads a complete L3 image.
                if (nt == 0) {
                    uint8_t* H2p = ptr_H2s[(M6_nb_l3 - 1) & 1];
                    snrt_dma_start_1d(ptr_h2_l3 + (M6_nb_l3 - 1) * M6_h2_half_bytes, H2p, M6_h2_half_bytes);
                    snrt_dma_start_1d(ptr_h2_l3 + M6_h2_im_region + (M6_nb_l3 - 1) * M6_h2_half_bytes,
                                      H2p + M6_h2_half_bytes, M6_h2_half_bytes);
                }
                snrt_dma_start_2d(ptr_h2ntile, ptr_h2_l3 + nt * M6_ntile_n_off, M6_h2_gather_chunk,
                                  /*dst_stride=*/M6_h2_gather_chunk, /*src_stride=*/M6_h2_gather_src_stride,
                                  /*repeat=*/M6_h2_gather_kreps);
                snrt_dma_start_1d(ptr_P3, (void*)snrt_zero_memory_ptr(), M6_p3_ntile_bytes);
                snrt_dma_wait_all();
            }
            snrt_cluster_hw_barrier();

            if (snrt_global_core_idx() == 0) {
                // K-tile the contraction into nb_l3 chunks; accumulate, requant on the last chunk.
                for (uint32_t kc = 0; kc < M6_nb_l3; kc++) {
                    CFG_GEMM_P(ptr_weight3 + kc * M6_weight3_kchunk_bytes, ptr_h2ntile + kc * M6_h2_kchunk_bytes, 5,
                               ptr_P3);
                    uint32_t mode = (kc + 1 == M6_nb_l3) ? M6_ISGEMM_SQ : M30_ISGEMM_SQ_NO_REQUANT;
                    set_simbacore_csr(mode, 2 * L3, 1, 2 * L3_padded / M6_nb_l3, 1,
                                      M6_dModel_slice * L1 * L2 / M6_nb_ntile);
                    start_simbacore_and_streamers(M6_R10_en, 0, 1, 0);
                    wait_simbacore_and_streamer();
                    simbacore_cycles += read_simbacore_perf_counter();
                }
                asm volatile("fence" ::: "memory");
            }
            snrt_cluster_hw_barrier();

            // Scatter this (slice, N-tile)'s partition-3 output into the L3 output.
            if (snrt_is_dm_core()) {
                snrt_dma_start_2d(ptr_output_l3 + s * M6_out_block_slice + nt * M6_out_ntile_chunk, ptr_P3,
                                  M6_out_ntile_chunk, /*dst_stride=*/M6_out_block_full,
                                  /*src_stride=*/M6_out_ntile_chunk, /*repeat=*/M6_out_nblk);
                snrt_dma_wait_all();
            }
            snrt_cluster_hw_barrier();
        }
    }
#undef CFG_GEMM_P

    if (snrt_global_core_idx() == 0) {
        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Simbacore elapsed time: %u cycles\n", end_cycles, simbacore_cycles);
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        err += check_result_sample(ptr_output_l3, M6_partition3_expected, M6_test_samples_expected, nb_test_samples,
                                   "partition3_out (L3)");
        printf("Test FFT 3-way tiled (l3-stream): (%d x %d), L1=%d L2=%d L3=%d, nb_d=%d nb_l3=%d\n", seqLen, dModel, L1,
               L2, L3, M6_nb_d, M6_nb_l3);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, nb_test_samples);
    }
    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test(); }
