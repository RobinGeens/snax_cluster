// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>

#include "../data/data.h"
#include "snax-simbacore-lib.h"

int test() {
    int err = 0;

    // Define TCDM addresses
    void* tcdm_base_ptr = snrt_l1_next();
    uint8_t* ptr_a      = (uint8_t*)(tcdm_base_ptr + M5_addr_a);
    uint8_t* ptr_b      = (uint8_t*)(tcdm_base_ptr + M5_addr_b);
    uint16_t* ptr_cd    = (uint16_t*)(tcdm_base_ptr + M5_addr_cd);

    // Initialize cycle counter for timing
    if (snrt_global_core_idx() == 0) init_cycle_counter();

    snrt_cluster_hw_barrier();

    // Transfer data from L3 to L1 using DMA only
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_a, M5_dft_weight, M5_length_a);
        snrt_dma_start_1d(ptr_b, M5_dft_in, M5_length_b);
        // Initialize ptr_cd with zeros from zero memory
        snrt_dma_start_1d(ptr_cd, (void*)snrt_zero_memory_ptr(), M5_length_cd);
        snrt_dma_wait_all();
    }

    // Wait for DMA to finish
    snrt_cluster_hw_barrier();

    // Call compute core
    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: DFT partition 1\n\n");
        uint32_t start_cycles = get_cycle_count();
#ifdef VERBOSE
        printf("[%d cc] Setting up Streamer and SimbaCore CSRs\n", start_cycles);
#endif

        set_isgemm_streamer_csr((uint32_t)ptr_a, M5_R11_ss, M5_R11_tb, M5_R11_ts,  // A
                                (uint32_t)ptr_b, M5_R12_ss, M5_R12_tb, M5_R12_ts,  // B
                                (uint32_t)ptr_cd, M5_W3_ss, M5_W3_tb, M5_W3_ts);   // C/D

        set_simbacore_csr(M5_ISGEMM_SQ, L1_padded0, 1, L1_padded1, 1, (dModel * L2));
        start_simbacore_and_streamers(M5_R10_en, 0, M5_R11_en, 0);
        wait_simbacore_and_streamer();
        uint32_t end_cycles = get_cycle_count();
        printf("[%d cc] Simbacore elapsed time: %u cycles\n", end_cycles, read_simbacore_perf_counter());
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        // check_result_all((uint8_t*)ptr_cd, M5_D, M5_length_cd);
        err += check_result_sample((uint8_t*)ptr_cd, M5_expected, M5_test_samples_expected,  //
                                   nb_test_samples, "out");

        printf("Test FFT: (%d x %d), channels=%d\n", L1_padded0, L1, dModel * L2);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test(); }
