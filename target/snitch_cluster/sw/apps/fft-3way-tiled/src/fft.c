// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// Tiled 3-way partitioned EinFFT (L = L1*L2*L3), outer dModel-tile + inner l3-tile loop:
// stages 1-4 run per l3-tile on DMA-gathered buffers and write the full TCDM H2 directly, then
// partition3 K-accumulates the l3-tiles from TCDM. See docs/dataflow/05_fft.md "fft-3way-tiled".

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

// A launch that opens an accumulation reads its K-step-0 psum from the zero buffer and
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

int test() {
    int err = 0;

    // L3: only the assembled full-dModel output (the H2 stays on-chip).
    static uint8_t* ptr_output_l3 = NULL;
    if (snrt_global_core_idx() == 0) {
        (void)snrt_l3alloc(16 * 1024);
        ptr_output_l3 = (uint8_t*)snrt_l3alloc(M6_length_partition3_out);
    }
    snrt_cluster_hw_barrier();

    // TCDM layout: H2 (partition-3 input) and the zero buffer sit below the stage-1-4 scratch,
    // so P3 overlays the dead scratch and may spill upward into free TCDM without touching them.
    void* base           = snrt_l1_next();
    uint8_t* ptr_weight1 = (uint8_t*)base;
    uint8_t* ptr_weight2 = ptr_weight1 + align64(M6_length_weight1);
    uint8_t* ptr_weight3 = ptr_weight2 + align64(M6_length_weight2);
    uint8_t* ptr_H2_full = ptr_weight3 + align64(M6_length_weight3);
    uint8_t* ptr_zero    = ptr_H2_full + align64(M6_h2_full_bytes);
    uint8_t* ptr_in      = ptr_zero + M6_zero_bytes;
    uint8_t* ptr_tw1     = ptr_in + align64(M6_in_tile_bytes);
    uint8_t* ptr_tw2     = ptr_tw1 + align64(M6_tw1_tile_bytes);
    uint8_t* ptr_P       = ptr_tw2 + align64(M6_tw2_tile_bytes);
    uint8_t* ptr_H2c     = ptr_P + M6_slot_size_tile;
    uint8_t* ptr_P3      = ptr_in;
    // The zero buffer covers partition3's first psum read only when it is slot-sized.
    const bool p3_from_zero = M6_zero_bytes >= M6_slot_size;

    uint32_t start_cycles     = 0;
    uint32_t simbacore_cycles = 0;

    if (snrt_global_core_idx() == 0) {
        printf(
            "\nStarting program: tiled 3-way FFT (outer dModel-tile, nb_d=%d, nb_l3=%d, seqLen=%d, dModel=%d, "
            "L1=%d, L2=%d, L3=%d)\n\n",
            M6_nb_d, M6_nb_l3, seqLen, dModel, L1, L2, L3);
        printf("Expected L1 TCDM usage: %u B (%u KiB)\n", (uint32_t)L1_TCDM_PEAK_BYTES,
               (uint32_t)(L1_TCDM_PEAK_BYTES / 1024));
        init_cycle_counter();
        start_cycles = snrt_mcycle();
    }

#define CFG_GEMM(W, IN_, N, PSUM, FIRST)                                                                         \
    do {                                                                                                         \
        set_isgemm_streamer_csr((uint32_t)(W), M6_R11_##N##_ss, M6_R11_##N##_tb, M6_R11_##N##_ts, (uint32_t)(IN_), \
                                M6_R12_##N##_ss, M6_R12_##N##_tb, M6_R12_##N##_ts, (uint32_t)(PSUM),              \
                                M6_W3_##N##_ss, M6_W3_##N##_tb, M6_W3_##N##_ts);                                 \
        if (FIRST) psum_from_zero(ptr_zero, (PSUM));                                                             \
    } while (0)
#define CFG_GEMM1(C) CFG_GEMM(ptr_weight1 + (C) * M6_weight1_kchunk_bytes, ptr_in + (C) * M6_in_kchunk_bytes, 1, ptr_P, (C) == 0)
#define CFG_P3(KC) \
    CFG_GEMM(ptr_weight3 + (KC) * M6_weight3_kchunk_bytes, ptr_H2_full + (KC) * M6_h2_kchunk_bytes, 5, ptr_P3, \
             (KC) == 0 && p3_from_zero)
#define FINISH()                                            \
    do {                                                    \
        wait_simbacore_and_streamer();                      \
        simbacore_cycles += read_simbacore_perf_counter();  \
    } while (0)

    // Weights + zero buffer once; slice 0 / tile 0 inputs. Every later streamer program is
    // written while the previous launch runs (the streamer latches its config at start).
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_weight1, M6_dft_weight1, M6_length_weight1);
        snrt_dma_start_1d(ptr_weight2, M6_dft_weight2, M6_length_weight2);
        snrt_dma_start_1d(ptr_weight3, M6_dft_weight3, M6_length_weight3);
        simba_dma_fill_zero(ptr_zero, M6_zero_bytes);
        gather_in_tile(ptr_in, 0, 0);
        gather_tw1(ptr_tw1, 0);
        gather_tw2(ptr_tw2, 0);
        snrt_dma_wait_all();
    }
    if (snrt_global_core_idx() == 0) CFG_GEMM1(0);
    snrt_cluster_hw_barrier();

    for (uint32_t s = 0; s < M6_nb_d; s++) {
        // --- stages 1-4 per l3-tile, writing the tile's [re|im] K-blocks of the full H2 ---
        for (uint32_t lt = 0; lt < M6_nb_l3; lt++) {
            if (snrt_global_core_idx() == 0) {
                // gemm1 (in_tile -> P), K-chunked; requant + transpose on the last chunk.
                for (uint32_t c = 0; c < M6_nb_k1; c++) {
                    bool last = c + 1 == M6_nb_k1;
                    launch_gemm(last ? M7_ISGEMM_SQ_TRANSPOSE : M30_ISGEMM_SQ_NO_REQUANT, 2 * L1,
                                L1_padded / M6_nb_k1, M6_dModel_slice * L2 * M6_l3_tile);
                    if (!last)
                        CFG_GEMM1(c + 1);
                    else
                        set_simd_streamer_csr((uint32_t)ptr_P, M6_R7_2_ss, M6_R7_2_tb, M6_R7_2_ts, (uint32_t)ptr_tw1,
                                              M6_R13_2_ss, M6_R13_2_tb, M6_R13_2_ts, (uint32_t)ptr_H2c, M6_W3_2_ss,
                                              M6_W3_2_tb, M6_W3_2_ts);
                    FINISH();
                }
                // cmul1 (P, tw1 -> H2c): the fused W3 split write deinterleaves re/im.
                launch_cmul();
                CFG_GEMM(ptr_weight2, ptr_H2c, 3, ptr_P, true);
                FINISH();
            }
            // Behind gemm1+cmul1: gather this tile's tw2 (free since the previous cmul2).
            if (snrt_is_dm_core() && lt > 0) {
                gather_tw2(ptr_tw2, lt);
                snrt_dma_wait_all();
            }
            snrt_cluster_hw_barrier();  // A: in & tw1 consumed

            if (snrt_global_core_idx() == 0) {
                // gemm2 (H2c -> P).
                launch_gemm(M7_ISGEMM_SQ_TRANSPOSE, 2 * L2, 2 * L2_padded, M6_dModel_slice * L1 * M6_l3_tile);
                set_simd_streamer_csr((uint32_t)ptr_P, M6_R7_4_ss, M6_R7_4_tb, M6_R7_4_ts, (uint32_t)ptr_tw2,
                                      M6_R13_4_ss, M6_R13_4_tb, M6_R13_4_ts,
                                      (uint32_t)(ptr_H2_full + lt * M6_h2_half_bytes), M6_W3_4_ss, M6_W3_4_tb,
                                      M6_W3_4_ts);
                FINISH();
                // cmul2 (P, tw2 -> H2_full): fused re/im split write into the tile's K-blocks.
                launch_cmul();
                if (lt + 1 < M6_nb_l3)
                    CFG_GEMM1(0);
                else
                    CFG_P3(0);
                FINISH();
                asm volatile("fence" ::: "memory");
            }
            // Behind gemm2+cmul2: prefetch the next tile's in & tw1 (dead since gemm1/cmul1).
            if (snrt_is_dm_core() && lt + 1 < M6_nb_l3) {
                gather_in_tile(ptr_in, s, lt + 1);
                gather_tw1(ptr_tw1, lt + 1);
                snrt_dma_wait_all();
            }
            snrt_cluster_hw_barrier();  // C: H2_full ready; tw2 consumed
        }

        // --- partition 3: K-tile the 2*L3 contraction over nb_l3 chunks from the TCDM H2,
        //     accumulating NO_REQUANT and requantising on the last. ---
        if (!p3_from_zero) {
            if (snrt_is_dm_core()) {
                simba_dma_fill_zero(ptr_P3, M6_slot_size);
                snrt_dma_wait_all();
            }
            snrt_cluster_hw_barrier();
        }
        if (snrt_global_core_idx() == 0) {
            for (uint32_t kc = 0; kc < M6_nb_l3; kc++) {
                bool last = kc + 1 == M6_nb_l3;
                launch_gemm(last ? M6_ISGEMM_SQ : M30_ISGEMM_SQ_NO_REQUANT, 2 * L3, 2 * L3_padded / M6_nb_l3,
                            M6_dModel_slice * L1 * L2);
                if (!last)
                    CFG_P3(kc + 1);
                else if (s + 1 < M6_nb_d)
                    CFG_GEMM1(0);
                FINISH();
            }
            asm volatile("fence" ::: "memory");
        }
        snrt_cluster_hw_barrier();

        // Scatter this slice's partition-3 output into its d-slice of the full L3 output, then
        // gather the next slice's tile 0 (its in/tw buffers are under the P3 overlay).
        if (snrt_is_dm_core()) {
            snrt_dma_start_2d(ptr_output_l3 + s * M6_out_block_slice, ptr_P3, M6_out_block_slice,
                              /*dst_stride=*/M6_out_block_full, /*src_stride=*/M6_out_block_slice,
                              /*repeat=*/M6_out_nblk);
            snrt_dma_wait_all();
            if (s + 1 < M6_nb_d) {
                gather_in_tile(ptr_in, s + 1, 0);
                gather_tw1(ptr_tw1, 0);
                gather_tw2(ptr_tw2, 0);
                snrt_dma_wait_all();
            }
        }
        snrt_cluster_hw_barrier();
    }
#undef CFG_GEMM
#undef CFG_GEMM1
#undef CFG_P3
#undef FINISH

    if (snrt_global_core_idx() == 0) {
        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Simbacore elapsed time: %u cycles\n", end_cycles, simbacore_cycles);
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        err += check_result_sample(ptr_output_l3, M6_partition3_expected, M6_test_samples_expected, nb_test_samples,
                                   "partition3_out (L3)");
        printf("Test FFT 3-way tiled (outer dModel-tile): (%d x %d), L1=%d L2=%d L3=%d, nb_d=%d nb_l3=%d\n", seqLen,
               dModel, L1, L2, L3, M6_nb_d, M6_nb_l3);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, nb_test_samples);
    }
    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test(); }
