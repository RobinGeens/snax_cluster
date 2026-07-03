// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// 4-way partitioned EinFFT (L = L1 * L2 * L3 * L4), un-tiled (all buffers resident).
// Bring-up / validation harness for the 4-partition kernel: checks every stage's golden.
// See docs/dataflow/05_fft.md.

#include "../data/data.h"
#include "snax-simbacore-lib.h"

int test() {
    int err = 0;

    void* base                     = snrt_l1_next();
    uint8_t* ptr_weight1           = (uint8_t*)(base + M6_addr_weight1);
    uint8_t* ptr_weight2           = (uint8_t*)(base + M6_addr_weight2);
    uint8_t* ptr_weight3           = (uint8_t*)(base + M6_addr_weight3);
    uint8_t* ptr_weight4           = (uint8_t*)(base + M6_addr_weight4);
    uint8_t* ptr_in                = (uint8_t*)(base + M6_addr_in);
    uint8_t* ptr_partition1_out    = (uint8_t*)(base + M6_addr_partition1_out);
    uint8_t* ptr_twiddles1         = (uint8_t*)(base + M6_addr_twiddles1);
    uint8_t* ptr_hadamard1_out     = (uint8_t*)(base + M6_addr_hadamard1_out);
    uint8_t* ptr_hadamard1_packed  = (uint8_t*)(base + M6_addr_hadamard1_packed);
    uint8_t* ptr_partition2_out    = (uint8_t*)(base + M6_addr_partition2_out);
    uint8_t* ptr_twiddles2         = (uint8_t*)(base + M6_addr_twiddles2);
    uint8_t* ptr_hadamard2_out     = (uint8_t*)(base + M6_addr_hadamard2_out);
    uint8_t* ptr_hadamard2_packed  = (uint8_t*)(base + M6_addr_hadamard2_packed);
    uint8_t* ptr_partition3_out    = (uint8_t*)(base + M6_addr_partition3_out);
    uint8_t* ptr_twiddles3         = (uint8_t*)(base + M6_addr_twiddles3);
    uint8_t* ptr_hadamard3_out     = (uint8_t*)(base + M6_addr_hadamard3_out);
    uint8_t* ptr_hadamard3_packed  = (uint8_t*)(base + M6_addr_hadamard3_packed);
    uint8_t* ptr_partition4_out    = (uint8_t*)(base + M6_addr_partition4_out);

    if (snrt_global_core_idx() == 0) init_cycle_counter();
    snrt_cluster_hw_barrier();

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_weight1, M6_dft_weight1, M6_length_weight1);
        snrt_dma_start_1d(ptr_weight2, M6_dft_weight2, M6_length_weight2);
        snrt_dma_start_1d(ptr_weight3, M6_dft_weight3, M6_length_weight3);
        snrt_dma_start_1d(ptr_weight4, M6_dft_weight4, M6_length_weight4);
        snrt_dma_start_1d(ptr_in, M6_dft_in, M6_length_in);
        snrt_dma_start_1d(ptr_twiddles1, M6_twiddles1, M6_length_twiddles1);
        snrt_dma_start_1d(ptr_twiddles2, M6_twiddles2, M6_length_twiddles2);
        snrt_dma_start_1d(ptr_twiddles3, M6_twiddles3, M6_length_twiddles3);
        snrt_dma_start_1d(ptr_partition1_out, (void*)snrt_zero_memory_ptr(), M6_length_partition1_out);
        snrt_dma_start_1d(ptr_partition2_out, (void*)snrt_zero_memory_ptr(), M6_length_partition2_out);
        snrt_dma_start_1d(ptr_partition3_out, (void*)snrt_zero_memory_ptr(), M6_length_partition3_out);
        snrt_dma_start_1d(ptr_partition4_out, (void*)snrt_zero_memory_ptr(), M6_length_partition4_out);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    uint32_t start_cycles = 0;
    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: 4-way partitioned FFT (L=%d, dModel=%d, L1=%d, L2=%d, L3=%d, L4=%d)\n\n",
               seqLen, dModel, L1, L2, L3, L4);
        start_cycles = snrt_mcycle();

        // ===== Step 1: partition 1 (bank-transposed) =====
        set_isgemm_streamer_csr((uint32_t)ptr_weight1, M6_R11_1_ss, M6_R11_1_tb, M6_R11_1_ts,
                                (uint32_t)ptr_in, M6_R12_1_ss, M6_R12_1_tb, M6_R12_1_ts,
                                (uint32_t)ptr_partition1_out, M6_W3_1_ss, M6_W3_1_tb, M6_W3_1_ts);
        set_simbacore_csr(M7_ISGEMM_SQ_TRANSPOSE, 2 * L1, 1, L1_padded, 1, (dModel * L2 * L3 * L4));
        start_simbacore_and_streamers(M6_R10_en, 0, 1, 0);
        wait_simbacore_and_streamer();
        printf("  step done: p1\n");

        // ===== Step 2: hadamard 1 (CMUL) =====
        set_simd_streamer_csr((uint32_t)ptr_partition1_out, M6_R7_2_ss, M6_R7_2_tb, M6_R7_2_ts,
                              (uint32_t)ptr_twiddles1, M6_R13_2_ss, M6_R13_2_tb, M6_R13_2_ts,
                              (uint32_t)ptr_hadamard1_out, M6_W3_2_ss, M6_W3_2_tb, M6_W3_2_ts);
        set_simbacore_csr(M20_SIMD_CMUL_FP8, 0, 0, 0, 0, 0);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();
        printf("  step done: tw1\n");

        // ===== Step 2B: reorder 1 (NOOP deinterleave) =====
        set_simd_streamer_no_b((uint32_t)ptr_hadamard1_out, M6_R7_2B_ss, M6_R7_2B_tb, M6_R7_2B_ts,
                               (uint32_t)ptr_hadamard1_packed, M6_W3_2B_ss, M6_W3_2B_tb, M6_W3_2B_ts);
        set_simbacore_csr(M23_SIMD_NOOP_FP8, 0, 0, 0, 0, 0);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();
        printf("  step done: reorder1\n");
        asm volatile("fence" ::: "memory");

        // ===== Step 3: partition 2 (bank-transposed) =====
        set_isgemm_streamer_csr((uint32_t)ptr_weight2, M6_R11_3_ss, M6_R11_3_tb, M6_R11_3_ts,
                                (uint32_t)ptr_hadamard1_packed, M6_R12_3_ss, M6_R12_3_tb, M6_R12_3_ts,
                                (uint32_t)ptr_partition2_out, M6_W3_3_ss, M6_W3_3_tb, M6_W3_3_ts);
        set_simbacore_csr(M7_ISGEMM_SQ_TRANSPOSE, 2 * L2, 1, 2 * L2_padded, 1, (dModel * L1 * L3 * L4));
        start_simbacore_and_streamers(M6_R10_en, 0, 1, 0);
        wait_simbacore_and_streamer();
        printf("  step done: p2\n");

        // ===== Step 4: hadamard 2 (CMUL with twiddles2) =====
        set_simd_streamer_csr((uint32_t)ptr_partition2_out, M6_R7_4_ss, M6_R7_4_tb, M6_R7_4_ts,
                              (uint32_t)ptr_twiddles2, M6_R13_4_ss, M6_R13_4_tb, M6_R13_4_ts,
                              (uint32_t)ptr_hadamard2_out, M6_W3_4_ss, M6_W3_4_tb, M6_W3_4_ts);
        set_simbacore_csr(M20_SIMD_CMUL_FP8, 0, 0, 0, 0, 0);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();
        printf("  step done: tw2\n");

        // ===== Step 4B: reorder 2 = m3/m4 transpose (scalar, un-tiled validation only).
        // Gathers m3 to partition3's K-axis. =====
        {
            const int L34v = L3 * L4;
            for (int a = 0; a < L1 * L2; a++)
                for (int m4 = 0; m4 < L4; m4++)
                    for (int m3 = 0; m3 < L3; m3++)
                        for (int reim = 0; reim < 2; reim++) {
                            int l   = a * L34v + m4 * L3 + m3;
                            int src = (l / 16) * 32 + reim * 16 + (l % 16);
                            int dst = (a * L4 + m4) * (2 * L3) + reim * L3 + m3;
                            ptr_hadamard2_packed[dst] = ptr_hadamard2_out[src];
                        }
        }
        printf("  step done: reorder2\n");
        asm volatile("fence" ::: "memory");

        // ===== Step 5: partition 3 (bank-transposed) =====
        set_isgemm_streamer_csr((uint32_t)ptr_weight3, M6_R11_5_ss, M6_R11_5_tb, M6_R11_5_ts,
                                (uint32_t)ptr_hadamard2_packed, M6_R12_5_ss, M6_R12_5_tb, M6_R12_5_ts,
                                (uint32_t)ptr_partition3_out, M6_W3_5_ss, M6_W3_5_tb, M6_W3_5_ts);
        set_simbacore_csr(M7_ISGEMM_SQ_TRANSPOSE, 2 * L3, 1, 2 * L3_padded, 1, (dModel * L1 * L2 * L4));
        start_simbacore_and_streamers(M6_R10_en, 0, 1, 0);
        wait_simbacore_and_streamer();
        printf("  step done: p3\n");

        // ===== Step 6: hadamard 3 (CMUL with twiddles3) =====
        set_simd_streamer_csr((uint32_t)ptr_partition3_out, M6_R7_6_ss, M6_R7_6_tb, M6_R7_6_ts,
                              (uint32_t)ptr_twiddles3, M6_R13_6_ss, M6_R13_6_tb, M6_R13_6_ts,
                              (uint32_t)ptr_hadamard3_out, M6_W3_6_ss, M6_W3_6_tb, M6_W3_6_ts);
        set_simbacore_csr(M20_SIMD_CMUL_FP8, 0, 0, 0, 0, 0);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();
        printf("  step done: tw3\n");

        // ===== Step 6B: reorder 3 (NOOP deinterleave) =====
        set_simd_streamer_no_b((uint32_t)ptr_hadamard3_out, M6_R7_6B_ss, M6_R7_6B_tb, M6_R7_6B_ts,
                               (uint32_t)ptr_hadamard3_packed, M6_W3_6B_ss, M6_W3_6B_tb, M6_W3_6B_ts);
        set_simbacore_csr(M23_SIMD_NOOP_FP8, 0, 0, 0, 0, 0);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();
        printf("  step done: reorder3\n");
        asm volatile("fence" ::: "memory");

        // ===== Step 7: partition 4 (plain, final output) =====
        set_isgemm_streamer_csr((uint32_t)ptr_weight4, M6_R11_7_ss, M6_R11_7_tb, M6_R11_7_ts,
                                (uint32_t)ptr_hadamard3_packed, M6_R12_7_ss, M6_R12_7_tb, M6_R12_7_ts,
                                (uint32_t)ptr_partition4_out, M6_W3_7_ss, M6_W3_7_tb, M6_W3_7_ts);
        set_simbacore_csr(M6_ISGEMM_SQ, 2 * L4, 1, 2 * L4_padded, 1, (dModel * L1 * L2 * L3));
        start_simbacore_and_streamers(M6_R10_en, 0, 1, 0);
        wait_simbacore_and_streamer();
        printf("  step done: p4\n");

        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        // ----- Per-stage verification -----
        err += check_result_sample(ptr_partition1_out, M6_partition1_expected, M6_test_samples_expected,
                                   nb_test_samples, "partition 1");
        err += check_result_sample((uint8_t*)ptr_hadamard1_out, M6_hadamard1_expected, M6_test_samples_expected,
                                   nb_test_samples, "hadamard 1");
        err += check_result_sample(ptr_partition2_out, M6_partition2_expected, M6_test_samples_expected,
                                   nb_test_samples, "partition 2");
        err += check_result_sample((uint8_t*)ptr_hadamard2_out, M6_hadamard2_expected, M6_test_samples_expected,
                                   nb_test_samples, "hadamard 2");
        err += check_result_sample(ptr_partition3_out, M6_partition3_expected, M6_test_samples_expected,
                                   nb_test_samples, "partition 3");
        err += check_result_sample((uint8_t*)ptr_hadamard3_out, M6_hadamard3_expected, M6_test_samples_expected,
                                   nb_test_samples, "hadamard 3");
        err += check_result_sample(ptr_partition4_out, M6_partition4_expected, M6_test_samples_expected,
                                   nb_test_samples, "partition 4");

        printf("Test FFT 4-way: (%d x %d), L1=%d L2=%d L3=%d L4=%d\n", seqLen, dModel, L1, L2, L3, L4);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, 7 * nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test(); }
