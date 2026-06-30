// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// l3-streamed 4-way partitioned EinFFT (L = L1*L2*L3*L4): stages 1-2 run per l3-tile on gathered
// tile-local buffers, each tile's hadamard2 reorder2-transposed into the resident partition3
// input; then partition3 -> cmul3 -> reorder3 -> partition4 finish off-tile.
// See docs/dataflow/05_fft.md "fft-4way-tiled-async".

#include "../data/data.h"
#include "snax-simbacore-lib.h"

static inline uint32_t align64(uint32_t x) { return (x + 63u) & ~63u; }

// --- l3-tile gathers from DRAM into contiguous tile-local buffers (L3 -> L3t). ---
static inline void gather_in_tile(uint8_t* dst, uint32_t lt) {
    for (uint32_t d = 0; d < M6_dModel_slice; d++) {
        uint8_t* src = M6_dft_in + d * M6_in_gather_d_stride + lt * M6_in_gather_chunk;
        snrt_dma_start_2d(dst + d * L4 * M6_in_gather_chunk, src, M6_in_gather_chunk,
                          /*dst_stride=*/M6_in_gather_chunk, /*src_stride=*/M6_in_gather_m4_stride, /*repeat=*/L4);
    }
}
// twiddles1 [k1][jc][m2] (re,im adjacent): gather the tile's m3-block per (k1, m4).
static inline void gather_tw1(uint8_t* dst, uint32_t lt) {
    for (uint32_t k1 = 0; k1 < L1; k1++) {
        uint8_t* src = M6_twiddles1 + k1 * M6_tw1_k1_stride + lt * M6_tw1_gather_chunk;
        snrt_dma_start_2d(dst + k1 * L4 * M6_tw1_gather_chunk, src, M6_tw1_gather_chunk,
                          /*dst_stride=*/M6_tw1_gather_chunk, /*src_stride=*/M6_tw1_gather_m4_stride, /*repeat=*/L4);
    }
}
// twiddles2 [k2][jc] (re,im adjacent): gather the tile's m3-block per (k2, m4).
static inline void gather_tw2(uint8_t* dst, uint32_t lt) {
    for (uint32_t k2 = 0; k2 < L2; k2++) {
        uint8_t* src = M6_twiddles2 + k2 * M6_tw2_k2_stride + lt * M6_tw2_gather_chunk;
        snrt_dma_start_2d(dst + k2 * L4 * M6_tw2_gather_chunk, src, M6_tw2_gather_chunk,
                          /*dst_stride=*/M6_tw2_gather_chunk, /*src_stride=*/M6_tw2_gather_m4_stride, /*repeat=*/L4);
    }
}

int test() {
    int err = 0;

    static uint8_t* ptr_output_l3 = NULL;
    if (snrt_global_core_idx() == 0) {
        (void)snrt_l3alloc(16 * 1024);
        ptr_output_l3 = (uint8_t*)snrt_l3alloc(M6_length_partition4_out);
    }
    snrt_cluster_hw_barrier();

    void* base           = snrt_l1_next();
    uint8_t* ptr_weight1 = (uint8_t*)base;
    uint8_t* ptr_weight2 = ptr_weight1 + align64(M6_length_weight1);
    uint8_t* ptr_weight3 = ptr_weight2 + align64(M6_length_weight2);
    uint8_t* ptr_weight4 = ptr_weight3 + align64(M6_length_weight3);
    uint8_t* ptr_tw3     = ptr_weight4 + align64(M6_length_weight4);
    uint8_t* ptr_packed3 = ptr_tw3 + align64(M6_length_twiddles3);  // persists across l3-tiles
    // Stage-1-2 tile-local buffers.
    uint8_t* ptr_in  = ptr_packed3 + align64(M6_full_packed3_bytes);
    uint8_t* ptr_tw1 = ptr_in + align64(M6_in_tile_bytes);
    uint8_t* ptr_tw2 = ptr_tw1 + align64(M6_tw1_tile_bytes);
    uint8_t* ptr_P   = ptr_tw2 + align64(M6_tw2_tile_bytes);
    uint8_t* ptr_H1  = ptr_P + M6_slot_size_tile;
    uint8_t* ptr_H2  = ptr_H1 + M6_hsize_tile;
    // Stage-3-4 full buffers overlay the dead stage-1-2 scratch (from ptr_in).
    uint8_t* ptr_P3      = ptr_in;
    uint8_t* ptr_H3      = ptr_P3 + M6_slot_size_full;
    uint8_t* ptr_packed4 = ptr_H3 + align64(M6_hsize_full);
    uint8_t* ptr_P4      = ptr_packed4 + align64(M6_full_packed3_bytes);

    uint32_t start_cycles     = 0;
    uint32_t simbacore_cycles = 0;

    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: 4-way FFT (l3-stream, nb_l3=%d, L=%d, dModel=%d, L1=%d L2=%d L3=%d L4=%d)\n\n",
               M6_nb_l3, seqLen, dModel, L1, L2, L3, L4);
        init_cycle_counter();
        start_cycles = snrt_mcycle();
    }

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_weight1, M6_dft_weight1, M6_length_weight1);
        snrt_dma_start_1d(ptr_weight2, M6_dft_weight2, M6_length_weight2);
        snrt_dma_start_1d(ptr_weight3, M6_dft_weight3, M6_length_weight3);
        snrt_dma_start_1d(ptr_weight4, M6_dft_weight4, M6_length_weight4);
        snrt_dma_start_1d(ptr_tw3, M6_twiddles3, M6_length_twiddles3);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

#define CFG_GEMM(W, IN_, N, PSUM)                                                                                \
    set_isgemm_streamer_csr((uint32_t)(W), M6_R11_##N##_ss, M6_R11_##N##_tb, M6_R11_##N##_ts, (uint32_t)(IN_),   \
                            M6_R12_##N##_ss, M6_R12_##N##_tb, M6_R12_##N##_ts, (uint32_t)(PSUM), M6_W3_##N##_ss, \
                            M6_W3_##N##_tb, M6_W3_##N##_ts)

    // ===== Stages 1-2 per l3-tile: assemble the full partition3 input (packed3) =====
    for (uint32_t lt = 0; lt < M6_nb_l3; lt++) {
        if (snrt_is_dm_core()) {
            gather_in_tile(ptr_in, lt);
            gather_tw1(ptr_tw1, lt);
            gather_tw2(ptr_tw2, lt);
            snrt_dma_start_1d(ptr_P, (void*)snrt_zero_memory_ptr(), M6_slot_size_tile);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();

        if (snrt_global_core_idx() == 0) {
            // p1 (in_tile, weight1 -> P)
            CFG_GEMM(ptr_weight1, ptr_in, 1, ptr_P);
            set_simbacore_csr(M7_ISGEMM_SQ_TRANSPOSE, 2 * L1, 1, L1_padded, 1, M6_dModel_slice * L2 * M6_l3_tile * L4);
            start_simbacore_and_streamers(M6_R10_en, 0, 1, 0);
            wait_simbacore_and_streamer();
            simbacore_cycles += read_simbacore_perf_counter();

            // cmul1 (P, tw1 -> H1)
            set_simd_streamer_csr((uint32_t)ptr_P, M6_R7_2_ss, M6_R7_2_tb, M6_R7_2_ts, (uint32_t)ptr_tw1, M6_R13_2_ss,
                                  M6_R13_2_tb, M6_R13_2_ts, (uint32_t)ptr_H1, M6_W3_2_ss, M6_W3_2_tb, M6_W3_2_ts);
            set_simbacore_csr(M20_SIMD_CMUL_FP8, 0, 0, 0, 0, 0);
            start_simbacore_and_streamers(0, 0, 0, 0);
            wait_simbacore_and_streamer();
            simbacore_cycles += read_simbacore_perf_counter();

            // reorder1 (H1 -> H2)
            set_simd_streamer_no_b((uint32_t)ptr_H1, M6_R7_2B_ss, M6_R7_2B_tb, M6_R7_2B_ts, (uint32_t)ptr_H2,
                                   M6_W3_2B_ss, M6_W3_2B_tb, M6_W3_2B_ts);
            set_simbacore_csr(M23_SIMD_NOOP_FP8, 0, 0, 0, 0, 0);
            start_simbacore_and_streamers(0, 0, 0, 0);
            wait_simbacore_and_streamer();
            simbacore_cycles += read_simbacore_perf_counter();
            asm volatile("fence" ::: "memory");
        }
        snrt_cluster_hw_barrier();

        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(ptr_P, (void*)snrt_zero_memory_ptr(), M6_slot_size_tile);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();

        if (snrt_global_core_idx() == 0) {
            // p2 (H2, weight2 -> P)
            CFG_GEMM(ptr_weight2, ptr_H2, 3, ptr_P);
            set_simbacore_csr(M7_ISGEMM_SQ_TRANSPOSE, 2 * L2, 1, 2 * L2_padded, 1,
                              M6_dModel_slice * L1 * M6_l3_tile * L4);
            start_simbacore_and_streamers(M6_R10_en, 0, 1, 0);
            wait_simbacore_and_streamer();
            simbacore_cycles += read_simbacore_perf_counter();

            // cmul2 (P, tw2 -> H1)
            set_simd_streamer_csr((uint32_t)ptr_P, M6_R7_4_ss, M6_R7_4_tb, M6_R7_4_ts, (uint32_t)ptr_tw2, M6_R13_4_ss,
                                  M6_R13_4_tb, M6_R13_4_ts, (uint32_t)ptr_H1, M6_W3_4_ss, M6_W3_4_tb, M6_W3_4_ts);
            set_simbacore_csr(M20_SIMD_CMUL_FP8, 0, 0, 0, 0, 0);
            start_simbacore_and_streamers(0, 0, 0, 0);
            wait_simbacore_and_streamer();
            simbacore_cycles += read_simbacore_perf_counter();
            asm volatile("fence" ::: "memory");
        }
        snrt_cluster_hw_barrier();

        // reorder2: deinterleave cmul2's re/im (in H1) into the partition3 input packed3.
        // Two regimes: L3 == 16 DMA gathers vs 2*L3 <= 16 block-swap NOOP.
        if ((int)L3 == 16) {
            if (snrt_is_dm_core()) {
                snrt_dma_start_2d(ptr_packed3, ptr_H1, L3, L3, 2 * L3, M6_n_full);
                snrt_dma_start_2d(ptr_packed3 + M6_pk_kstride, ptr_H1 + L3, L3, L3, 2 * L3, M6_n_full);
                snrt_dma_wait_all();
            }
        } else if (snrt_global_core_idx() == 0) {
            set_simd_streamer_no_b((uint32_t)ptr_H1, M6_R7_2B_ss, M6_R7_2B_tb, M6_R7_2B_ts, (uint32_t)ptr_packed3,
                                   M6_W3_2B_ss, M6_W3_2B_tb, M6_W3_2B_ts);
            set_simbacore_csr(M23_SIMD_NOOP_FP8, 0, 0, 0, 0, 0);
            start_simbacore_and_streamers(0, 0, 0, 0);
            wait_simbacore_and_streamer();
            simbacore_cycles += read_simbacore_perf_counter();
            asm volatile("fence" ::: "memory");
        }
        snrt_cluster_hw_barrier();
    }

    // ===== Stage 3: partition 3 (full, single K=2*L3_padded invocation, bank-transposed) =====
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_P3, (void*)snrt_zero_memory_ptr(), M6_slot_size_full);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    if (snrt_global_core_idx() == 0) {
        CFG_GEMM(ptr_weight3, ptr_packed3, 5, ptr_P3);
        set_simbacore_csr(M7_ISGEMM_SQ_TRANSPOSE, 2 * L3, 1, 2 * L3_padded, 1, M6_dModel_slice * L1 * L2 * L4);
        start_simbacore_and_streamers(M6_R10_en, 0, 1, 0);
        wait_simbacore_and_streamer();
        simbacore_cycles += read_simbacore_perf_counter();
        asm volatile("fence" ::: "memory");
    }
    snrt_cluster_hw_barrier();

    // cmul3 (P3, tw3 -> H3) on core 0 while the DM core zeroes P4 concurrently: P4 is idle until
    // partition4, so its 128 KiB init hides behind cmul3 instead of sitting on the critical path.
    if (snrt_global_core_idx() == 0) {
        set_simd_streamer_csr((uint32_t)ptr_P3, M6_R7_6_ss, M6_R7_6_tb, M6_R7_6_ts, (uint32_t)ptr_tw3, M6_R13_6_ss,
                              M6_R13_6_tb, M6_R13_6_ts, (uint32_t)ptr_H3, M6_W3_6_ss, M6_W3_6_tb, M6_W3_6_ts);
        set_simbacore_csr(M20_SIMD_CMUL_FP8, 0, 0, 0, 0, 0);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();
        simbacore_cycles += read_simbacore_perf_counter();
        asm volatile("fence" ::: "memory");
    }
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_P4, (void*)snrt_zero_memory_ptr(), M6_slot_size_full);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    // reorder3: same deinterleave as reorder2, for partition4's m4 contraction (no transpose --
    // m4 already innermost). Two regimes as reorder2.
    if ((int)L4 == 16) {
        if (snrt_is_dm_core()) {
            snrt_dma_start_2d(ptr_packed4, ptr_H3, L4, L4, 2 * L4, M6_n4_full);
            snrt_dma_start_2d(ptr_packed4 + M6_pk_kstride4, ptr_H3 + L4, L4, L4, 2 * L4, M6_n4_full);
            snrt_dma_wait_all();
        }
    } else if (snrt_global_core_idx() == 0) {
        set_simd_streamer_no_b((uint32_t)ptr_H3, M6_R7_6B_ss, M6_R7_6B_tb, M6_R7_6B_ts, (uint32_t)ptr_packed4,
                               M6_W3_6B_ss, M6_W3_6B_tb, M6_W3_6B_ts);
        set_simbacore_csr(M23_SIMD_NOOP_FP8, 0, 0, 0, 0, 0);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();
        simbacore_cycles += read_simbacore_perf_counter();
        asm volatile("fence" ::: "memory");
    }
    snrt_cluster_hw_barrier();

    // ===== Stage 4: partition 4 (plain, final; P4 was zeroed during cmul3) =====
    if (snrt_global_core_idx() == 0) {
        CFG_GEMM(ptr_weight4, ptr_packed4, 7, ptr_P4);
        set_simbacore_csr(M6_ISGEMM_SQ, 2 * L4, 1, 2 * L4_padded, 1, M6_dModel_slice * L1 * L2 * L3);
        start_simbacore_and_streamers(M6_R10_en, 0, 1, 0);
        wait_simbacore_and_streamer();
        simbacore_cycles += read_simbacore_perf_counter();
        asm volatile("fence" ::: "memory");
    }
    snrt_cluster_hw_barrier();

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_output_l3, ptr_P4, M6_length_partition4_out);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();
#undef CFG_GEMM

    if (snrt_global_core_idx() == 0) {
        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Simbacore elapsed time: %u cycles\n", end_cycles, simbacore_cycles);
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        err += check_result_sample(ptr_output_l3, M6_partition4_expected, M6_test_samples_expected, nb_test_samples,
                                   "partition4_out (L3)");
        printf("Test FFT 4-way tiled (l3-stream): (%d x %d), L1=%d L2=%d L3=%d L4=%d, nb_l3=%d\n", seqLen, dModel, L1,
               L2, L3, L4, M6_nb_l3);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, nb_test_samples);
    }
    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test(); }
