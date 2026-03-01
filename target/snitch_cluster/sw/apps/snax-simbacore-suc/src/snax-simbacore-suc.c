// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>

#include "snax-simbacore-helper.c"
#include "snax-simbacore-lib.h"

void set_streamer_suc_only(uint32_t ptr_z, uint32_t ptr_dt_in, uint32_t ptr_dt_weight_1, uint32_t ptr_dt_weight_2,
                           uint32_t ptr_dt_bias, uint32_t ptr_x, uint32_t ptr_A, uint32_t ptr_BC, uint32_t ptr_D,
                           uint32_t ptr_y) {
#ifdef VERBOSE
    printf("[%d cc] Setting up Streamer and SimbaCore for SUC only...\n", snrt_mcycle());
#endif
    // The regular (working) memory layout gives bank conflicts because the spatial stride of BC is 16 banks, so
    // every two elements come from the same bank. We overwrite this here with an incorrect stride, just to verify
    // utilization.
    uint32_t ss_BC_test[] = {16};  // 2 banks (wrong, correct one is M2_R7_ss)

    set_streamer_csr(

        (uint32_t)0, 0, 0, 0, 0,                                            // osCore in
        (uint32_t)0, 0, 0, 0, 0,                                            // oscore weight
        (uint32_t)ptr_dt_in, M2_R2_ss, M2_R2_tb, M2_R2_ts, M2_R2_en,        // switchCore in
        (uint32_t)ptr_dt_weight_1, M2_R3_ss, M2_R3_tb, M2_R3_ts, M2_R3_en,  // switchCore weight
        (uint32_t)ptr_dt_bias, M2_R4_ss, M2_R4_tb, M2_R4_ts, M2_R4_en,      // switchCore bias
        (uint32_t)ptr_dt_weight_2, M2_R5_ss, M2_R5_tb, M2_R5_ts, M2_R5_en,  // switchCore  matmul weight
        (uint32_t)ptr_A, M2_R6_ss, M2_R6_tb, M2_R6_ts, M2_R6_en,            //  SUC A
        (uint32_t)ptr_BC, ss_BC_test, M2_R7_tb, M2_R7_ts, M2_R7_en,         // SUC BC
        (uint32_t)ptr_D, M2_R8_ss, M2_R8_tb, M2_R8_ts, M2_R8_en,            // SUC  D
        (uint32_t)ptr_x, M2_R9_ss, M2_R9_tb, M2_R9_ts, M2_R9_en,            // SUC x
        (uint32_t)ptr_z, M2_R10_ss, M2_R10_tb, M2_R10_ts, M2_R10_en,        // SUC z = osCore out
        (uint32_t)ptr_y, M2_R11_ss, M2_R11_tb, M2_R11_ts, M2_R11_en,        // iscore in = SUC y
        (uint32_t)0, 0, 0, 0, 0,                                            // isCore weight
        (uint32_t)0, 0, 0, 0, 0,                                            // isCore psum

        (uint32_t)0, 0, 0, 0, 0,                                  // osCore out = z
        (uint32_t)0, 0, 0, 0, M2_W1_en,                           // disable
        (uint32_t)ptr_y, M2_W2_ss, M2_W2_tb, M2_W2_ts, M2_W2_en,  // SUC y
        (uint32_t)0, 0, 0, 0, 0                                   // isCore out

    );
}

int main() {
    int err = 0;

    // Define TCDM addresses
    void* tcdm_base_ptr        = snrt_l1_next();
    uint8_t* ptr_oscore_in     = (uint8_t*)(tcdm_base_ptr + M2_addr_oscore_in);
    uint8_t* ptr_oscore_weight = (uint8_t*)(tcdm_base_ptr + M2_addr_oscore_weight);
    uint8_t* ptr_z             = (uint8_t*)(tcdm_base_ptr + M2_addr_z);  // osCore out
    uint8_t* ptr_dt_in         = (uint8_t*)(tcdm_base_ptr + M2_addr_dt_BC);
    uint8_t* ptr_BC            = (uint8_t*)(tcdm_base_ptr + M2_addr_dt_BC + M2_dt_to_BC_offset);  //
    uint8_t* ptr_dt_weight_1   = (uint8_t*)(tcdm_base_ptr + M2_addr_dt_weight_1);
    uint8_t* ptr_dt_weight_2   = (uint8_t*)(tcdm_base_ptr + M2_addr_dt_weight_2);
    uint8_t* ptr_dt_bias       = (uint8_t*)(tcdm_base_ptr + M2_addr_dt_bias);
    uint8_t* ptr_x             = (uint8_t*)(tcdm_base_ptr + M2_addr_x);  // from Phase1
    uint8_t* ptr_A             = (uint8_t*)(tcdm_base_ptr + M2_addr_A);
    uint8_t* ptr_D             = (uint8_t*)(tcdm_base_ptr + M2_addr_D);
    uint8_t* ptr_y             = (uint8_t*)(tcdm_base_ptr + M2_addr_y);  // SUC out
    uint8_t* ptr_iscore_weight = (uint8_t*)(tcdm_base_ptr + M2_addr_iscore_weight);
    uint16_t* ptr_iscore_out   = (uint16_t*)(tcdm_base_ptr + M2_addr_iscore_out);

    // Initialize cycle counter for timing
    if (snrt_global_core_idx() == 0) init_cycle_counter();

    snrt_cluster_hw_barrier();

    // Transfer data from L3 to L1 using DMA only
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_z, M2_oscore_expected, M2_length_z);  // Directly load expected z
        snrt_dma_start_1d(ptr_dt_in, M2_dt_BC, M2_length_dt_BC);
        snrt_dma_start_1d(ptr_dt_weight_1, M2_dt_weight_1, M2_length_dt_weight_1);
        snrt_dma_start_1d(ptr_dt_weight_2, M2_dt_weight_2, M2_length_dt_weight_2);
        snrt_dma_start_1d(ptr_dt_bias, M2_dt_bias, M2_length_dt_bias);
        snrt_dma_start_1d(ptr_x, M2_suc_x, M2_length_x);
        snrt_dma_start_1d(ptr_A, M2_suc_A, M2_length_A);
        snrt_dma_start_1d(ptr_D, M2_suc_D, M2_length_D);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    // Call compute core
    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: SUC only\n\n");
        uint32_t start_cycles = snrt_mcycle();
#ifdef VERBOSE
        printf("[%d cc] Setting up Streamer and SimbaCore CSRs\n", start_cycles);
#endif

        set_streamer_suc_only((uint32_t)ptr_z, (uint32_t)ptr_dt_in, (uint32_t)ptr_dt_weight_1,
                              (uint32_t)ptr_dt_weight_2, (uint32_t)ptr_dt_bias, (uint32_t)ptr_x, (uint32_t)ptr_A,
                              (uint32_t)ptr_BC, (uint32_t)ptr_D, (uint32_t)ptr_y);

        set_simbacore_csr(M14_SUC_ONLY, seqLen, dModel, dInner, dtRank, dModel);
        start_simbacore_and_streamers(M2_R10_en, 0, 0, 0);
        wait_simbacore_and_streamer();
        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Simbacore elapsed time: %u cycles\n", end_cycles, read_simbacore_perf_counter());
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        err += check_result_sample(ptr_y, M2_suc_expected, M2_test_samples_y,  //
                                   nb_test_samples, "SUC y");

        printf("Test SUC only: seqLen=%d, dModel=%d\n", seqLen, dModel);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}
