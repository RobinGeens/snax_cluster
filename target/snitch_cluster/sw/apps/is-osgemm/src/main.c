// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Un-tiled parallel OSGEMM + ISGEMM in a single kernel call.

#include "data.h"
#include "snax-simbacore-lib.h"

int main() {
    int err = 0;

    void* tcdm_base_ptr = snrt_l1_next();
    uint8_t* ptr_a_os   = (uint8_t*)tcdm_base_ptr;
    uint8_t* ptr_b_os   = ptr_a_os + M3_length_a;
    uint8_t* ptr_d_os   = ptr_b_os + M3_length_b;
    uint8_t* ptr_a_is   = ptr_d_os + M3_length_d;
    uint8_t* ptr_b_is   = ptr_a_is + M4_length_a;
    uint16_t* ptr_cd_is = (uint16_t*)(ptr_b_is + M4_length_b);

    if (snrt_global_core_idx() == 0) init_cycle_counter();
    snrt_cluster_hw_barrier();

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_a_os, M3_A, M3_length_a);
        snrt_dma_start_1d(ptr_b_os, M3_B, M3_length_b);
        snrt_dma_start_1d(ptr_a_is, M4_A, M4_length_a);
        snrt_dma_start_1d(ptr_b_is, M4_B, M4_length_b);
        snrt_dma_start_1d((uint8_t*)ptr_cd_is, M4_C, M4_length_cd);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: IS+OSGeMM (un-tiled)\n\n");
        uint32_t start_cycles = snrt_mcycle();

        set_streamer_csr(
            (uint32_t)ptr_a_os, M3_R0_ss, M3_R0_tb, M3_R0_ts, M3_R0_en,       // R0
            (uint32_t)ptr_b_os, M3_R1_ss, M3_R1_tb, M3_R1_ts, M3_R1_en,       // R1
            (uint32_t)0, 0, 0, 0, 0, (uint32_t)0, 0, 0, 0, 0,  // R2, R3
            (uint32_t)0, 0, 0, 0, 0, (uint32_t)0, 0, 0, 0, 0,  // R4, R5
            (uint32_t)0, 0, 0, 0, 0, (uint32_t)0, 0, 0, 0, 0,  // R6, R7
            (uint32_t)0, 0, 0, 0, 0, (uint32_t)0, 0, 0, 0, 0,  // R8, R9
            (uint32_t)0, 0, 0, 0, 0,                             // R10
            (uint32_t)ptr_a_is, M4_R11_ss, M4_R11_tb, M4_R11_ts, M4_R11_en,   // R11
            (uint32_t)ptr_b_is, M4_R12_ss, M4_R12_tb, M4_R12_ts, M4_R12_en,   // R12
            (uint32_t)ptr_cd_is, M4_R13_ss, M4_R13_tb, M4_R13_ts, M4_R13_en,  // R13
            (uint32_t)ptr_d_os, M3_W0_ss, M3_W0_tb, M3_W0_ts, M3_W0_en,       // W0
            (uint32_t)0, 0, 0, 0, 0, (uint32_t)0, 0, 0, 0, 0,  // W1, W2
            (uint32_t)ptr_cd_is, M4_W3_ss, M4_W3_tb, M4_W3_ts, M4_W3_en       // W3
        );

        set_simbacore_csr(M31_IS_OSGEMM, seqLen, dModel, dInner, 1, dModel);
        start_simbacore_and_streamers(0, 0, M4_R11_en, 0);
        wait_simbacore_and_streamer();

        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Simbacore elapsed time: %u cycles\n", end_cycles, read_simbacore_perf_counter());
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        err += check_result_sample(ptr_d_os, M3_D, M3_test_samples_D, nb_test_samples, "osgemm_out");
        err += check_result_sample((uint8_t*)ptr_cd_is, M4_D, M4_test_samples_D, nb_test_samples, "isgemm_out");

        printf("Test IS+OSGeMM: seqLen=%d, dModel=%d, dInner=%d\n", seqLen, dModel, dInner);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, 2 * nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}
