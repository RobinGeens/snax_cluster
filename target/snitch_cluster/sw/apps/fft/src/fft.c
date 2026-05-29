// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>

#include "../data/data.h"
#include "snax-simbacore-lib.h"

int test() {
    int err = 0;

    // Define TCDM addresses
    void* tcdm_base_ptr           = snrt_l1_next();
    uint8_t* ptr_weight1          = (uint8_t*)(tcdm_base_ptr + M6_addr_weight1);
    uint8_t* ptr_weight2          = (uint8_t*)(tcdm_base_ptr + M6_addr_weight2);
    uint8_t* ptr_in               = (uint8_t*)(tcdm_base_ptr + M6_addr_in);
    uint8_t* ptr_partition1_out   = (uint8_t*)(tcdm_base_ptr + M6_addr_partition1_out);
    uint8_t* ptr_twiddle_factors  = (uint8_t*)(tcdm_base_ptr + M6_addr_twiddles);
    uint8_t* ptr_hadamard_out     = (uint8_t*)(tcdm_base_ptr + M6_addr_hadamard_out);
    uint8_t* ptr_hadamard_reorder = (uint8_t*)(tcdm_base_ptr + M6_addr_hadamard_reordered);
    uint8_t* ptr_partition2_out   = (uint8_t*)(tcdm_base_ptr + M6_addr_partition2_out);

    // Initialize cycle counter for timing
    if (snrt_global_core_idx() == 0) init_cycle_counter();

    snrt_cluster_hw_barrier();

    // Transfer data from L3 to L1 using DMA only
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_weight1, M6_dft_weight1, M6_length_weight1);
        snrt_dma_start_1d(ptr_weight2, M6_dft_weight2, M6_length_weight2);
        snrt_dma_start_1d(ptr_in, M6_dft_in, M6_length_in);
        snrt_dma_start_1d(ptr_twiddle_factors, M6_twiddles, M6_length_twiddles);
        snrt_dma_start_1d(ptr_partition1_out, (void*)snrt_zero_memory_ptr(), M6_length_partition1_out);
        snrt_dma_start_1d(ptr_partition2_out, (void*)snrt_zero_memory_ptr(), M6_length_partition2_out);
        snrt_dma_wait_all();
    }

    // Wait for DMA to finish
    snrt_cluster_hw_barrier();

    // Call compute core
    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: DFT partition 1\n\n");
        printf("Expected L1 TCDM usage: %u B (%u KiB)\n", (uint32_t)L1_TCDM_PEAK_BYTES,
               (uint32_t)(L1_TCDM_PEAK_BYTES / 1024));
        uint32_t start_cycles = snrt_mcycle();
#ifdef VERBOSE
        printf("[%d cc] Setting up Streamer and SimbaCore CSRs\n", start_cycles);
#endif

        // Step 1: partition 1
        set_isgemm_streamer_csr((uint32_t)ptr_weight1, M6_R11_1_ss, M6_R11_1_tb, M6_R11_1_ts,       // A
                                (uint32_t)ptr_in, M6_R12_1_ss, M6_R12_1_tb, M6_R12_1_ts,            // B
                                (uint32_t)ptr_partition1_out, M6_W3_1_ss, M6_W3_1_tb, M6_W3_1_ts);  // C/D

        set_simbacore_csr(M7_ISGEMM_SQ_TRANSPOSE, 2 * L1, 1, L1_padded, 1, (dModel * L2));
        start_simbacore_and_streamers(M6_R10_en, 0, 1, 0);
        wait_simbacore_and_streamer();

        // Step 2: Hadamard
        set_simd_streamer_csr((uint32_t)ptr_partition1_out, M6_R7_2_ss, M6_R7_2_tb, M6_R7_2_ts,      // SUC BC
                              (uint32_t)ptr_twiddle_factors, M6_R13_2_ss, M6_R13_2_tb, M6_R13_2_ts,  // isCore psum
                              (uint32_t)ptr_hadamard_out, M6_W3_2_ss, M6_W3_2_tb, M6_W3_2_ts         // isCore out
        );
        set_simbacore_csr(M20_SIMD_CMUL_FP8, 0, 0, 0, 0, 0);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

        // Step 2B: retransform the CMUL output: packed real/imag (interleaved every 16 elements) -> separate real/imag
        set_simd_streamer_no_b((uint32_t)ptr_hadamard_out, M6_R7_2B_ss, M6_R7_2B_tb, M6_R7_2B_ts,     // SUC BC
                               (uint32_t)ptr_hadamard_reorder, M6_W3_2B_ss, M6_W3_2B_tb, M6_W3_2B_ts  // isCore out
        );
        set_simbacore_csr(M23_SIMD_NOOP_FP8, 0, 0, 0, 0, 0);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

        // Step 3: partition 2
        set_isgemm_streamer_csr((uint32_t)ptr_weight2, M6_R11_3_ss, M6_R11_3_tb, M6_R11_3_ts,           // A
                                (uint32_t)ptr_hadamard_reorder, M6_R12_3_ss, M6_R12_3_tb, M6_R12_3_ts,  // B
                                (uint32_t)ptr_partition2_out, M6_W3_3_ss, M6_W3_3_tb, M6_W3_3_ts);      // C/D

        set_simbacore_csr(M6_ISGEMM_SQ, 2 * L2, 1, 2 * L2_padded, 1, (dModel * L1));
        start_simbacore_and_streamers(M6_R10_en, 0, 1, 0);
        wait_simbacore_and_streamer();

        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Simbacore elapsed time: %u cycles\n", end_cycles, read_simbacore_perf_counter());
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        err += check_result_sample(ptr_partition1_out, M6_partition1_expected, M6_test_samples_expected,
                                   nb_test_samples, "partition 1");

        err += check_result_sample((uint8_t*)ptr_hadamard_out, M6_hadamard_expected, M6_test_samples_expected,
                                   nb_test_samples, "hadamard");
        // err += check_result_sample((uint8_t*)ptr_hadamard_reorder, M6_hadamard_reordered,
        // M6_test_samples_expected,
        //                            nb_test_samples, "hadamard reordered");
        err += check_result_sample((uint8_t*)ptr_partition2_out, M6_partition2_expected, M6_test_samples_expected,
                                   nb_test_samples, "partition 2");
        // check_result_all((uint8_t*)ptr_hadamard_out, M6_hadamard_expected, 2 *
        // M6_length_hadamard_out);

        printf("Test FFT: (%d x %d)\n", seqLen, dModel);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test(); }
