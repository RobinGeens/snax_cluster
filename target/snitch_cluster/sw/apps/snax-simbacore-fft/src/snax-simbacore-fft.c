// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>

#include "../data/data.h"
#include "snax-simbacore-lib.h"

int test() {
    int err = 0;

    // Define TCDM addresses
    void* tcdm_base_ptr          = snrt_l1_next();
    uint8_t* ptr_weight1         = (uint8_t*)(tcdm_base_ptr + M6_addr_weight1);
    uint8_t* ptr_in              = (uint8_t*)(tcdm_base_ptr + M6_addr_in);
    uint16_t* ptr_partition1_out = (uint16_t*)(tcdm_base_ptr + M6_addr_partition1_out);

    // Initialize cycle counter for timing
    if (snrt_global_core_idx() == 0) init_cycle_counter();

    snrt_cluster_hw_barrier();

    // Transfer data from L3 to L1 using DMA only
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_weight1, M6_dft_weight, M6_length_weight1);
        snrt_dma_start_1d(ptr_in, M6_dft_in, M6_length_in);
        // Initialize ptr_partition1_out with zeros from zero memory
        snrt_dma_start_1d(ptr_partition1_out, (void*)snrt_zero_memory_ptr(), M6_length_partition1_out);
        snrt_dma_wait_all();
    }

    // Wait for DMA to finish
    snrt_cluster_hw_barrier();

    // Call compute core
    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: DFT partition 1\n\n");
        uint32_t start_cycles = snrt_mcycle();
#ifdef VERBOSE
        printf("[%d cc] Setting up Streamer and SimbaCore CSRs\n", start_cycles);
#endif

        // Step 1: partition 1
        set_isgemm_streamer_csr((uint32_t)ptr_weight1, M6_R11_1_ss, M6_R11_1_tb, M6_R11_1_ts,       // A
                                (uint32_t)ptr_in, M6_R12_1_ss, M6_R12_1_tb, M6_R12_1_ts,            // B
                                (uint32_t)ptr_partition1_out, M6_W3_1_ss, M6_W3_1_tb, M6_W3_1_ts);  // C/D

        set_simbacore_csr(M7_ISGEMM_SQ_TRANSPOSE, L1_padded0, 1, L1_padded1, 1, (dModel * L2));
        start_simbacore_and_streamers(M6_R10_en, 0, 1, 0);
        wait_simbacore_and_streamer();
        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Simbacore elapsed time: %u cycles\n", end_cycles, read_simbacore_perf_counter());
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        err += check_result_sample((uint8_t*)ptr_partition1_out, M6_partition1_expected, M6_test_samples_expected,  //
                                   nb_test_samples, "partition 1");

        // Step 2: Hadamard
        // TODO

        printf("Test FFT: (%d x %d), channels=%d\n", L1_padded0, L1, dModel * L2);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test(); }
