// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>

#include <stdint.h>

#include "../data/data.h"
#include "snax-simbacore-lib.h"

/* BF16 = high 16 bits of IEEE 754 float. */
static inline uint16_t f32_to_bf16(float f) {
    union {
        float f;
        uint32_t u;
    } x = {.f = f};
    return (uint16_t)(x.u >> 16);
}

int test() {
    int err = 0;

    // Define TCDM addresses
    void* tcdm_base_ptr     = snrt_l1_next();
    uint16_t* ptr_x         = (uint16_t*)(tcdm_base_ptr + M11_addr_x);
    uint16_t* ptr_d_inverse = (uint16_t*)(tcdm_base_ptr + M11_addr_d_inverse);
    uint16_t* ptr_weight    = (uint16_t*)(tcdm_base_ptr + M11_addr_weight);
    uint16_t* ptr_out       = (uint16_t*)(tcdm_base_ptr + M11_addr_meanSq);

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
        printf("\nStarting program: RMSNorm\n\n");
        uint32_t start_cycles = snrt_mcycle();
#ifdef VERBOSE
        printf("[%d cc] Setting up Streamer and SimbaCore CSRs\n", start_cycles);
#endif

        // 1. Compute Σ(x^2)
        set_simd_streamer_no_b((uint32_t)ptr_x, M11_R7_LD_ss, M11_R7_LD_tb, M11_R7_LD_ts,  //
                               (uint32_t)ptr_out, M11_W3_L_ss, M11_W3_L_tb, M11_W3_L_ts);

        set_simbacore_simd_mode(M11_SIMD_RMS);
        set_simbacore_simd_n_acc(dModel);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

        // 2. Divide by D to get average
        uint16_t d_inverse = f32_to_bf16(1.0f / (float)dModel);  // BF16 encoding of 1/dModel
        printf("d_inverse (BF16 encoding of 1/dModel): 0x%04x\n", d_inverse);
        // Fill whole lane with d_inverse
        for (int i = 0; i < simdLanes; i++) {
            ptr_d_inverse[i] = d_inverse;
        }
        set_simd_streamer_csr((uint32_t)ptr_out, M11_R7_L_ss, M11_R7_L_tb, M11_R7_L_ts,  //
                              (uint32_t)ptr_d_inverse, M11_R7_L_ss, M11_R7_L_tb, 0,  // Same temporal bound, no stride
                              (uint32_t)ptr_out, M11_W3_L_ss, M11_W3_L_tb, M11_W3_L_ts);  // We write inplace
        set_simbacore_simd_mode(M8_SIMD_MUL);
        // set_simbacore_simd_n_acc(dModel);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

        // Final bookkeeping
        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Simbacore elapsed time: %u cycles\n", end_cycles, read_simbacore_perf_counter());
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        err += check_result_sample_u16(ptr_out, M11_meanSq, M11_test_samples_expected,  //
                                       nb_test_samples, "out");

        printf("Test RMSNORM: (%d x %d)\n", seqLen, dModel);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test(); }
