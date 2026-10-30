// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// l3-streamed 4-way partitioned EinFFT (L = L1*L2*L3*L4): stages 1-2 run per tile on tile-local
// buffers, each hadamard's re/im split write producing the next partition's input (cmul2
// assembles the resident partition3 input directly); then partition3 -> cmul3 -> partition4 run
// per N-chunk. See docs/dataflow/05_fft.md "fft-4way-tiled-async".

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
// Tile inputs: m4-tile slices are contiguous in the chip layouts (1-D input, one chunk per
// k1/k2 twiddle row); m3-tile slices are gathered.
static inline void load_in_tw1(uint8_t* in, uint8_t* tw1, uint32_t lt) {
    if (M6_nb_m4 > 1) {
        snrt_dma_start_1d(in, M6_dft_in + lt * M6_in_tile_bytes, M6_in_tile_bytes);
        snrt_dma_start_2d(tw1, M6_twiddles1 + lt * (M6_tw1_tile_bytes / L1), M6_tw1_tile_bytes / L1,
                          M6_tw1_tile_bytes / L1, M6_tw1_k1_stride, L1);
    } else {
        gather_in_tile(in, lt);
        gather_tw1(tw1, lt);
    }
}
static inline void load_tw2(uint8_t* tw2, uint32_t lt) {
    if (M6_nb_m4 > 1)
        snrt_dma_start_2d(tw2, M6_twiddles2 + lt * (M6_tw2_tile_bytes / L2), M6_tw2_tile_bytes / L2,
                          M6_tw2_tile_bytes / L2, M6_tw2_k2_stride, L2);
    else
        gather_tw2(tw2, lt);
}

// Every GEMM opens its own accumulation: R13 reads the K-step-0 psum from the zero buffer and
// K-step 1 from the psum itself (one R13 K-stride); W3 keeps writing the psum.
static inline void psum_from_zero(uint8_t* zero, uint8_t* psum) {
    write_csr(BASE_PTR_READER_13_LOW, (uint32_t)zero);
    write_csr(T_STRIDE_BASE_READER_13 + 1, (uint32_t)(psum - zero));
}

static inline void launch_gemm(uint32_t mode, uint32_t m, uint32_t k, uint32_t n) {
    set_simbacore_csr(mode, m, 1, k, 1, n);
    start_simbacore_and_streamers(M6_R10_en, 0, 1, 0);
}
static inline void launch_cmul(void) {
    set_simbacore_csr(M20_SIMD_CMUL_FP8, 0, 0, 0, 0, 0);
    start_simbacore_and_streamers(0, 0, 0, 0);
}
static inline void launch_noop(void) {
    set_simbacore_csr(M23_SIMD_NOOP_FP8, 0, 0, 0, 0, 0);
    start_simbacore_and_streamers(0, 0, 0, 0);
}

int test() {
    int err = 0;

    static uint8_t* ptr_output_l3 = NULL;
    if (snrt_global_core_idx() == 0) {
        (void)snrt_l3alloc(16 * 1024);
        ptr_output_l3 = (uint8_t*)snrt_l3alloc(M6_length_partition4_out);
    }
    snrt_cluster_hw_barrier();

    // TCDM layout: weights, the resident partition3 input and zero buffer, then the stage-1-2
    // tile scratch; the stage-3-4 chunk buffers overlay the dead tile scratch (from ptr_in).
    void* base           = snrt_l1_next();
    uint8_t* ptr_weight1 = (uint8_t*)base;
    uint8_t* ptr_weight2 = ptr_weight1 + align64(M6_length_weight1);
    uint8_t* ptr_weight3 = ptr_weight2 + align64(M6_length_weight2);
    uint8_t* ptr_weight4 = ptr_weight3 + align64(M6_length_weight3);
    uint8_t* ptr_tw3     = ptr_weight4 + align64(M6_length_weight4);
    uint8_t* ptr_packed3 = ptr_tw3 + align64(M6_length_twiddles3);
    uint8_t* ptr_zero    = ptr_packed3 + align64(M6_full_packed3_bytes);
    uint8_t* ptr_in      = ptr_zero + M6_zero_bytes;
    uint8_t* ptr_tw1     = ptr_in + align64(M6_in_tile_bytes);
    uint8_t* ptr_tw2     = ptr_tw1 + align64(M6_tw1_tile_bytes);
    uint8_t* ptr_P       = ptr_tw2 + align64(M6_tw2_tile_bytes);
    uint8_t* ptr_H2      = ptr_P + M6_slot_size_tile;
    uint8_t* ptr_H1      = ptr_H2 + M6_hsize_tile;  // block-swap NOOP scratch (K = 1 regimes only)
    uint8_t* ptr_P3      = ptr_in;
    uint8_t* ptr_packed4 = ptr_P3 + M6_slot_size_chunk;
    uint8_t* ptr_H3      = ptr_packed4 + M6_hsize_chunk;  // idem, for reorder3
    // A hadamard's re/im split is fused into its write when the halves are separate K-blocks
    // (axis == 16); at K = 1 the cmul writes H1/H3 and a block-swap NOOP finishes the deinterleave.
    const bool fused1 = 2 * (int)L2 > (int)seqLenUnroll;
    const bool fused2 = (int)L3 == (int)seqLenUnroll;
    const bool fused3 = (int)L4 == (int)seqLenUnroll;
    // P4 ping-pongs against the output scatter when there is more than one chunk.
    uint8_t* ptr_P4[2] = {ptr_H3 + (fused3 ? 0 : M6_hsize_chunk),
                          ptr_H3 + (fused3 ? 0 : M6_hsize_chunk) + (M6_nb_ntile > 1 ? M6_slot_size_chunk : 0)};

    uint32_t start_cycles     = 0;
    uint32_t simbacore_cycles = 0;

    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: 4-way FFT (l3-stream, nb_12=%d nb_ntile=%d, L=%d, dModel=%d, L1=%d L2=%d L3=%d "
               "L4=%d)\n\n",
               M6_nb_12, M6_nb_ntile, seqLen, dModel, L1, L2, L3, L4);
        init_cycle_counter();
        start_cycles = snrt_mcycle();
    }

#define CFG_GEMM(W, IN_, N, PSUM)                                                                                \
    do {                                                                                                         \
        set_isgemm_streamer_csr((uint32_t)(W), M6_R11_##N##_ss, M6_R11_##N##_tb, M6_R11_##N##_ts, (uint32_t)(IN_), \
                                M6_R12_##N##_ss, M6_R12_##N##_tb, M6_R12_##N##_ts, (uint32_t)(PSUM),              \
                                M6_W3_##N##_ss, M6_W3_##N##_tb, M6_W3_##N##_ts);                                 \
        psum_from_zero(ptr_zero, (PSUM));                                                                        \
    } while (0)
#define CFG_CMUL(IN_, TW, N, OUT)                                                                                 \
    set_simd_streamer_csr((uint32_t)(IN_), M6_R7_##N##_ss, M6_R7_##N##_tb, M6_R7_##N##_ts, (uint32_t)(TW),        \
                          M6_R13_##N##_ss, M6_R13_##N##_tb, M6_R13_##N##_ts, (uint32_t)(OUT), M6_W3_##N##_ss, \
                          M6_W3_##N##_tb, M6_W3_##N##_ts)
#define CFG_NOOP(IN_, N, OUT)                                                                               \
    set_simd_streamer_no_b((uint32_t)(IN_), M6_R7_##N##_ss, M6_R7_##N##_tb, M6_R7_##N##_ts, (uint32_t)(OUT), \
                           M6_W3_##N##_ss, M6_W3_##N##_tb, M6_W3_##N##_ts)
#define FINISH()                                            \
    do {                                                    \
        wait_simbacore_and_streamer();                      \
        simbacore_cycles += read_simbacore_perf_counter();  \
    } while (0)

    // Weights, twiddles3, the zero buffer and tile 0's inputs once. Every later streamer program
    // is written while the previous launch runs (the streamer latches its config at start).
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_weight1, M6_dft_weight1, M6_length_weight1);
        snrt_dma_start_1d(ptr_weight2, M6_dft_weight2, M6_length_weight2);
        snrt_dma_start_1d(ptr_weight3, M6_dft_weight3, M6_length_weight3);
        snrt_dma_start_1d(ptr_weight4, M6_dft_weight4, M6_length_weight4);
        snrt_dma_start_1d(ptr_tw3, M6_twiddles3, M6_length_twiddles3);
        simba_dma_fill_zero(ptr_zero, M6_zero_bytes);
        load_in_tw1(ptr_in, ptr_tw1, 0);
        load_tw2(ptr_tw2, 0);
        snrt_dma_wait_all();
    }
    if (snrt_global_core_idx() == 0) CFG_GEMM(ptr_weight1, ptr_in, 1, ptr_P);
    snrt_cluster_hw_barrier();

    // ===== Stages 1-2 per tile (m3-blocks via l3_tile, or m4-blocks via nb_m4): p1 -> cmul1 ->
    // p2 -> cmul2, the fused split writes assembling the full partition3 input (packed3) =====
    for (uint32_t lt = 0; lt < M6_nb_12; lt++) {
        if (snrt_global_core_idx() == 0) {
            // p1 (in, weight1 -> P)
            launch_gemm(M7_ISGEMM_SQ_TRANSPOSE, 2 * L1, L1_padded, M6_gemm_N1);
            CFG_CMUL(ptr_P, ptr_tw1, 2, fused1 ? ptr_H2 : ptr_H1);
            FINISH();
            // cmul1 (P, tw1 -> H2): re/im split into partition2's K-blocks (reorder1 NOOP at K = 1)
            launch_cmul();
            if (!fused1) {
                CFG_NOOP(ptr_H1, 2B, ptr_H2);
                FINISH();
                launch_noop();
            }
            CFG_GEMM(ptr_weight2, ptr_H2, 3, ptr_P);
            FINISH();
        }
        // Behind p1+cmul1: this tile's tw2 (free since the previous cmul2).
        if (snrt_is_dm_core() && lt > 0) {
            load_tw2(ptr_tw2, lt);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();  // A: in & tw1 consumed; tw2 ready

        if (snrt_global_core_idx() == 0) {
            // p2 (H2, weight2 -> P)
            launch_gemm(M7_ISGEMM_SQ_TRANSPOSE, 2 * L2, 2 * L2_padded, M6_gemm_N2);
            CFG_CMUL(ptr_P, ptr_tw2, 4, fused2 ? ptr_packed3 + lt * M6_pk_tile_off : ptr_H1);
            FINISH();
            // cmul2 (P, tw2 -> packed3): re/im split into partition3's K-blocks at the tile's m4
            // slots (reorder2 NOOP at K = 1)
            launch_cmul();
            if (!fused2) {
                CFG_NOOP(ptr_H1, 2B, ptr_packed3);
                FINISH();
                launch_noop();
            }
            if (lt + 1 < M6_nb_12)
                CFG_GEMM(ptr_weight1, ptr_in, 1, ptr_P);
            else
                CFG_GEMM(ptr_weight3, ptr_packed3, 5, ptr_P3);
            FINISH();
            asm volatile("fence" ::: "memory");
        }
        // Behind p2+cmul2: the next tile's in & tw1 (dead since p1/cmul1).
        if (snrt_is_dm_core() && lt + 1 < M6_nb_12) {
            load_in_tw1(ptr_in, ptr_tw1, lt + 1);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();  // C: packed3 tile written; tw2 consumed
    }

    // ===== Stages 3-4 per (k1,k2) N-chunk over the resident packed3: p3 -> cmul3 -> p4, the
    // previous chunk's output draining to L3 meanwhile =====
    for (uint32_t c = 0; c < M6_nb_ntile; c++) {
        if (snrt_global_core_idx() == 0) {
            // partition 3 (K=2*L3_padded, bank-transposed)
            launch_gemm(M7_ISGEMM_SQ_TRANSPOSE, 2 * L3, 2 * L3_padded, M6_gemm_N3);
            CFG_CMUL(ptr_P3, ptr_tw3, 6, fused3 ? ptr_packed4 : ptr_H3);
            FINISH();
            // cmul3 (P3, tw3 -> packed4): re/im split into partition4's K-blocks (reorder3 NOOP at K = 1)
            launch_cmul();
            if (!fused3) {
                CFG_NOOP(ptr_H3, 6B, ptr_packed4);
                FINISH();
                launch_noop();
            }
            CFG_GEMM(ptr_weight4, ptr_packed4, 7, ptr_P4[c & 1]);
            FINISH();
            // partition 4 (plain, final)
            launch_gemm(M6_ISGEMM_SQ, 2 * L4, 2 * L4_padded, M6_gemm_N4);
            if (c + 1 < M6_nb_ntile) CFG_GEMM(ptr_weight3, ptr_packed3 + (c + 1) * M6_pk3_chunk_off, 5, ptr_P3);
            FINISH();
            asm volatile("fence" ::: "memory");
        }
        // Behind the chunk: drain the previous chunk's slice of each M_4 row-block to the L3 output.
        if (snrt_is_dm_core() && c > 0) {
            snrt_dma_start_2d(ptr_output_l3 + (c - 1) * M6_p4c_blk_bytes, ptr_P4[(c - 1) & 1], M6_p4c_blk_bytes,
                              M6_p4_blk_bytes, M6_p4c_blk_bytes, M6_out_nblk);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();
    }
    if (snrt_is_dm_core()) {
        uint32_t c = M6_nb_ntile - 1;
        if (M6_nb_ntile == 1)
            snrt_dma_start_1d(ptr_output_l3, ptr_P4[0], M6_length_partition4_out);
        else
            snrt_dma_start_2d(ptr_output_l3 + c * M6_p4c_blk_bytes, ptr_P4[c & 1], M6_p4c_blk_bytes, M6_p4_blk_bytes,
                              M6_p4c_blk_bytes, M6_out_nblk);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();
#undef CFG_GEMM
#undef CFG_CMUL
#undef CFG_NOOP
#undef FINISH

    if (snrt_global_core_idx() == 0) {
        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Simbacore elapsed time: %u cycles\n", end_cycles, simbacore_cycles);
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        err += check_result_sample(ptr_output_l3, M6_partition4_expected, M6_test_samples_expected, nb_test_samples,
                                   "partition4_out (L3)");
        printf("Test FFT 4-way tiled (l3-stream): (%d x %d), L1=%d L2=%d L3=%d L4=%d, nb_12=%d nb_ntile=%d\n", seqLen,
               dModel, L1, L2, L3, L4, M6_nb_12, M6_nb_ntile);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, nb_test_samples);
    }
    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test(); }
