// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// VMamba program — mirrors the main Mamba program.
//
// Phase 1: osCore (in_proj) → conv1d + SiLU → IsCore (x_proj)
//   produces conv_out (= Phase 2 SUC x) and iscore_out (= dt+B+C)
//
// Phase 2: osCore → z, SwitchCore (dt_proj) → delta, SUC → y, IsCore (out_proj) → output

#include "helper.c"
#include "snax-simbacore-lib.h"

int test_phase1_and_2() {
    int err = 0;

    printf("This Snitch is awake and ready to roll!\r\n");

    // ----------------------------------------------------------------
    // Allocation — Phase 1
    // ----------------------------------------------------------------
    void* tcdm_base_ptr = snrt_l1_next();

    uint8_t* ptr_oscore_in        = (uint8_t*)(tcdm_base_ptr + M1_addr_oscore_in);
    uint8_t* ptr_oscore_weight_P1 = (uint8_t*)(tcdm_base_ptr + M1_addr_oscore_weight);
    uint8_t* ptr_conv_weight      = (uint8_t*)(tcdm_base_ptr + M1_addr_conv_weight);
    uint8_t* ptr_conv_bias        = (uint8_t*)(tcdm_base_ptr + M1_addr_conv_bias);
    uint8_t* ptr_conv_out         = (uint8_t*)(tcdm_base_ptr + M1_addr_conv_out);
    uint8_t* ptr_iscore_weight_P1 = (uint8_t*)(tcdm_base_ptr + M1_addr_iscore_weight);
    uint16_t* ptr_iscore_out_P1   = (uint16_t*)(tcdm_base_ptr + M1_addr_iscore_out);

    // ----------------------------------------------------------------
    // Allocation — Phase 2 (starts after Phase 1 iscore_out)
    // ----------------------------------------------------------------
    void* phase2_base_ptr         = ((void*)ptr_iscore_out_P1 + M1_length_iscore_out);
    uint8_t* ptr_oscore_weight_P2 = (uint8_t*)(phase2_base_ptr + M2_addr_oscore_weight);
    uint8_t* ptr_z                = (uint8_t*)(phase2_base_ptr + M2_addr_z);
    uint8_t* ptr_dt_in            = (uint8_t*)ptr_iscore_out_P1;
    uint8_t* ptr_BC               = (void*)ptr_dt_in + M2_dt_to_BC_offset;
    uint8_t* ptr_dt_weight_1      = (uint8_t*)(phase2_base_ptr + M2_addr_dt_weight_1);
    uint8_t* ptr_dt_weight_2      = (uint8_t*)(phase2_base_ptr + M2_addr_dt_weight_2);
    uint8_t* ptr_dt_bias          = (uint8_t*)(phase2_base_ptr + M2_addr_dt_bias);
    uint8_t* ptr_x                = ptr_conv_out;
    uint8_t* ptr_A                = (uint8_t*)(phase2_base_ptr + M2_addr_A);
    uint8_t* ptr_D                = (uint8_t*)(phase2_base_ptr + M2_addr_D);
    uint8_t* ptr_y                = (uint8_t*)(phase2_base_ptr + M2_addr_y);
    uint8_t* ptr_iscore_weight_P2 = (uint8_t*)(phase2_base_ptr + M2_addr_iscore_weight);
    uint16_t* ptr_iscore_out_P2   = (uint16_t*)(phase2_base_ptr + M2_addr_iscore_out);

    if (snrt_global_core_idx() == 0) init_cycle_counter();

    snrt_cluster_hw_barrier();

    // ================================================================
    // DMA: load Phase 1 data
    // ================================================================
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_oscore_in, M1_oscore_in, M1_length_oscore_in);
        snrt_dma_start_1d(ptr_oscore_weight_P1, M1_oscore_weight, M1_length_oscore_weight);
        snrt_dma_start_1d(ptr_conv_weight, M1_conv_weight, M1_length_conv_weight);
        snrt_dma_start_1d(ptr_conv_bias, M1_conv_bias, M1_length_conv_bias);
        snrt_dma_start_1d(ptr_iscore_weight_P1, M1_iscore_weight, M1_length_iscore_weight);
        snrt_dma_start_1d(ptr_iscore_out_P1, M1_iscore_bias, M1_length_iscore_out);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    // ================================================================
    // Phase 1: in_proj → conv1d + SiLU → x_proj
    // ================================================================
    uint32_t start_cycles            = 0;
    uint32_t simbacore_cycles_phase1 = 0;
    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: VMamba (Phase1 and Phase2)\n\n");
        start_cycles = snrt_mcycle();

        set_streamer_phase1((uint32_t)ptr_oscore_in, (uint32_t)ptr_oscore_weight_P1,
                            (uint32_t)ptr_conv_weight, (uint32_t)ptr_conv_bias,
                            (uint32_t)ptr_iscore_weight_P1, (uint32_t)ptr_iscore_out_P1,
                            (uint32_t)ptr_conv_out);

        set_simbacore_csr(M1_PHASE1, seqLen, dModel, dInner, dtRank, xProjDim);
        start_simbacore_and_streamers(M1_R10_en, 0, M1_R11_en, 0);
        wait_simbacore_and_streamer();
        simbacore_cycles_phase1 = read_simbacore_perf_counter();

#ifdef VERBOSE
        uint32_t end_cycles_phase1 = snrt_mcycle();
        printf("[%d cc] SimbaCore Phase1 took %u cycles\n", end_cycles_phase1, simbacore_cycles_phase1);
#endif
    }

    // ================================================================
    // DMA: load Phase 2 data (can overlap with Phase 1 compute)
    // ================================================================
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_oscore_weight_P2, M2_oscore_weight, M2_length_oscore_weight);
        snrt_dma_start_1d(ptr_dt_weight_1, M2_dt_weight_1, M2_length_dt_weight_1);
        snrt_dma_start_1d(ptr_dt_weight_2, M2_dt_weight_2, M2_length_dt_weight_2);
        snrt_dma_start_1d(ptr_dt_bias, M2_dt_bias, M2_length_dt_bias);
        snrt_dma_start_1d(ptr_A, M2_suc_A, M2_length_A);
        snrt_dma_start_1d(ptr_D, M2_suc_D, M2_length_D);
        snrt_dma_start_1d(ptr_iscore_weight_P2, M2_iscore_weight, M2_length_iscore_weight);
        snrt_dma_start_1d(ptr_iscore_out_P2, M2_iscore_bias, M2_length_iscore_out);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    // ================================================================
    // Phase 2: osCore → z, SwitchCore → delta, SUC → y, IsCore → output
    // ================================================================
    if (snrt_global_core_idx() == 0) {
        set_streamer_phase2((uint32_t)ptr_oscore_in, (uint32_t)ptr_oscore_weight_P2,
                            (uint32_t)ptr_z, (uint32_t)ptr_dt_in,
                            (uint32_t)ptr_dt_weight_1, (uint32_t)ptr_dt_weight_2, (uint32_t)ptr_dt_bias,
                            (uint32_t)ptr_x, (uint32_t)ptr_A, (uint32_t)ptr_BC, (uint32_t)ptr_D,
                            (uint32_t)ptr_y, (uint32_t)ptr_iscore_weight_P2, (uint32_t)ptr_iscore_out_P2);

        set_simbacore_csr(M2_PHASE2, seqLen, dModel, dInner, dtRank, dModel);
        start_simbacore_and_streamers(M2_R10_en, M2_R10_start_cnt, M2_R11_en, M2_R11_start_cnt);
        wait_simbacore_and_streamer();

        uint32_t simbacore_cycles_phase2 = read_simbacore_perf_counter();
        uint32_t end_cycles              = snrt_mcycle();
        printf("[%d cc] Simbacore Phase1 took %u cycles\n", end_cycles, simbacore_cycles_phase1);
        printf("[%d cc] Simbacore Phase2 took %u cycles\n", end_cycles, simbacore_cycles_phase2);
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        err += check_result_sample(ptr_z, M2_oscore_expected, M2_test_samples_z,
                                   nb_test_samples, "z (osCore out)");
        err += check_result_sample(ptr_y, M2_suc_expected, M2_test_samples_y,
                                   nb_test_samples, "SUC y");
        err += check_result_sample((uint8_t*)ptr_iscore_out_P2, M2_iscore_expected,
                                   M2_test_samples_iscore_out, nb_test_samples, "iscore_out");

        printf("Test VMamba Phase1+2: seqLen=%d, dModel=%d\n", seqLen, dModel);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, 5 * nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() {
    int err = 0;
    err += test_phase1_and_2();
    return err;
}
