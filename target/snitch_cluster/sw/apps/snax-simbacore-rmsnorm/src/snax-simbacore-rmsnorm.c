// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>

#include "../data/data.h"
#include "snax-simbacore-lib.h"

int test() {
    int err = 0;

    // Define TCDM addresses
    void* tcdm_base_ptr  = snrt_l1_next();
    uint16_t* ptr_x      = (uint16_t*)(tcdm_base_ptr + M11_addr_x);
    uint16_t* ptr_weight = (uint16_t*)(tcdm_base_ptr + M11_addr_weight);
    uint16_t* ptr_out    = (uint16_t*)(tcdm_base_ptr + M11_addr_xSqSum);

    // Initialize cycle counter for timing
    if (snrt_global_core_idx() == 0) init_cycle_counter();

    snrt_cluster_hw_barrier();

    // Transfer data from L3 to L1 using DMA only
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_x, M11_x, M11_length_x);
        snrt_dma_start_1d(ptr_weight, M11_weight, M11_length_weight);
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

        set_simd_streamer_no_b((uint32_t)ptr_x, M11_R7_ss, M11_R7_tb, M11_R7_ts,  //
                               (uint32_t)ptr_out, M11_W3_ss, M11_W3_tb, M11_W3_ts);

        set_simbacore_simd_mode(M11_SIMD_RMS);
        set_simbacore_simd_n_acc(dModel);
        start_simbacore_and_streamers(M11_R10_en, 0, M11_R11_en, 0);
        wait_simbacore_and_streamer();
        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Simbacore elapsed time: %u cycles\n", end_cycles, read_simbacore_perf_counter());
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        err += check_result_sample_u16(ptr_out, M11_xSqSum, M11_test_samples_expected,  //
                                       nb_test_samples, "out");

        printf("Test RMSNORM: (%d x %d)\n", seqLen, dModel);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test(); }
