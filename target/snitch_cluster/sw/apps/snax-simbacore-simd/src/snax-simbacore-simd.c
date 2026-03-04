// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>

#include "../data/data.h"
#include "snax-simbacore-lib.h"
#include "streamer_csr_addr_map.h"

int test_simd() {
    int err = 0;

    // Define TCDM addresses
    void* tcdm_base_ptr      = snrt_l1_next();
    uint16_t* ptr_a          = (uint16_t*)(tcdm_base_ptr + M8_addr_in_a);
    uint16_t* ptr_b          = (uint16_t*)(tcdm_base_ptr + M8_addr_in_b);
    uint16_t* ptr_out_add    = (uint16_t*)(tcdm_base_ptr + M8_addr_add_out);
    uint16_t* ptr_out_sub    = (uint16_t*)(tcdm_base_ptr + M8_addr_sub_out);
    uint16_t* ptr_out_mul    = (uint16_t*)(tcdm_base_ptr + M8_addr_mul_out);
    uint16_t* ptr_out_cmul   = (uint16_t*)(tcdm_base_ptr + M8_addr_cmul_out);
    uint16_t* ptr_out_inprod = (uint16_t*)(tcdm_base_ptr + M8_addr_inprod_out);
    uint16_t* ptr_out_rms    = (uint16_t*)(tcdm_base_ptr + M8_addr_rms_out);
    uint16_t* ptr_out_div    = (uint16_t*)(tcdm_base_ptr + M8_addr_div_out);
    uint16_t* ptr_out_sqrt   = (uint16_t*)(tcdm_base_ptr + M8_addr_sqrt_out);

    // Initialize cycle counter for timing
    if (snrt_global_core_idx() == 0) init_cycle_counter();

    snrt_cluster_hw_barrier();

    // Transfer data from L3 to L1 using DMA only
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_a, M8_simd_a_bf16, M8_length_in_a);
        snrt_dma_start_1d(ptr_b, M8_simd_b_bf16, M8_length_in_b);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    // Call compute core
    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: SIMD\n\n");
        uint32_t start_cycles = snrt_mcycle();

        // CMUL - BF16 core has no CMUL
        // #ifdef VERBOSE
        //         printf("[%d cc] Setting up Streamer and SimbaCore CSRs\n", start_cycles);
        //         printf("[%d cc] CMUL\n", snrt_mcycle());
        // #endif
        //         set_simd_streamer_csr((uint32_t)ptr_a, M8_R7_ss, M8_R7_tb, M8_R7_ts,        // SUC BC
        //                               (uint32_t)ptr_b, M8_R13_ss, M8_R13_tb, M8_R13_ts,     // isCore psum
        //                               (uint32_t)ptr_out_cmul, M8_W3_ss, M8_W3_tb, M8_W3_ts  // isCore out
        //         );

        //         set_simbacore_csr(M8_SIMD_ADD_BF16, 0, 0, 0, 0, 0);
        //         start_simbacore_and_streamers(0, 0, 0, 0);
        //         wait_simbacore_and_streamer();

        // ADD
#ifdef VERBOSE
        printf("[%d cc] ADD\n", snrt_mcycle());
#endif
        write_csr(BASE_PTR_WRITER_3_LOW, ptr_out_add);
        set_simbacore_simd_mode(M8_SIMD_ADD_BF16);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

        // SUB
#ifdef VERBOSE
        printf("[%d cc] SUB\n", snrt_mcycle());
#endif
        write_csr(BASE_PTR_WRITER_3_LOW, ptr_out_sub);
        set_simbacore_simd_mode(M9_SIMD_SUB_BF16);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

        // MUL
#ifdef VERBOSE
        printf("[%d cc] MUL\n", snrt_mcycle());
#endif
        write_csr(BASE_PTR_WRITER_3_LOW, ptr_out_mul);
        set_simbacore_simd_mode(M10_SIMD_MUL_BF16);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

        // DIV
#ifdef VERBOSE
        printf("[%d cc] DIV\n", snrt_mcycle());
#endif
        write_csr(BASE_PTR_WRITER_3_LOW, ptr_out_div);
        set_simbacore_simd_mode(M14_SIMD_DIV_BF16);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

        // INPROD
#ifdef VERBOSE
        printf("[%d cc] INPROD\n", snrt_mcycle());
#endif
        set_simd_streamer_csr((uint32_t)ptr_a, M8_R7_ss, M8_R7_tb, M8_R7_ts,     // SUC BC
                              (uint32_t)ptr_b, M8_R13_ss, M8_R13_tb, M8_R13_ts,  // isCore psum
                              (uint32_t)ptr_out_inprod, M8_W3_reduce_ss, M8_W3_reduce_tb, M8_W3_reduce_ts  // isCore out
        );
        set_simbacore_simd_mode(M12_SIMD_INPROD_BF16);
        set_simbacore_simd_n_acc(n_acc);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

        // RMS
#ifdef VERBOSE
        printf("[%d cc] RMS\n", snrt_mcycle());
#endif
        set_simd_streamer_no_b((uint32_t)ptr_a, M8_R7_ss, M8_R7_tb, M8_R7_ts,                            // SUC BC
                               (uint32_t)ptr_out_rms, M8_W3_reduce_ss, M8_W3_reduce_tb, M8_W3_reduce_ts  // isCore out
        );
        set_simbacore_simd_mode(M13_SIMD_RMS_BF16);
        set_simbacore_simd_n_acc(n_acc);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

        // SQRT
#ifdef VERBOSE
        printf("[%d cc] SQRT\n", snrt_mcycle());
#endif
        set_simd_streamer_no_b((uint32_t)ptr_a, M8_R7_ss, M8_R7_tb, M8_R7_ts,        // SUC BC
                               (uint32_t)ptr_out_sqrt, M8_W3_ss, M8_W3_tb, M8_W3_ts  // isCore out
        );
        set_simbacore_simd_mode(M15_SIMD_SQRT_BF16);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

        // Final
        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Simbacore elapsed time: %u cycles\n", end_cycles, read_simbacore_perf_counter());
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        err += check_result_sample_u16(ptr_out_cmul, M8_cmul_out_bf16, M8_test_samples_out,  //
                                       nb_test_samples, "CMUL");
        err += check_result_sample_u16(ptr_out_add, M8_add_out_bf16, M8_test_samples_out,  //
                                       nb_test_samples, "ADD");
        err += check_result_sample_u16(ptr_out_sub, M8_sub_out_bf16, M8_test_samples_out,  //
                                       nb_test_samples, "SUB");
        err += check_result_sample_u16(ptr_out_mul, M8_mul_out_bf16, M8_test_samples_out,  //
                                       nb_test_samples, "MUL");
        err += check_result_sample_u16(ptr_out_inprod, M8_inprod_out_bf16, M8_test_samples_out_reduce,  //
                                       nb_test_samples, "INPROD");
        err += check_result_sample_u16(ptr_out_rms, M8_rms_out_bf16, M8_test_samples_out_reduce,  //
                                       nb_test_samples, "RMS");
        err += check_result_sample_u16(ptr_out_div, M8_div_out_bf16, M8_test_samples_out,  //
                                       nb_test_samples, "DIV");
        err += check_result_sample_u16(ptr_out_sqrt, M8_sqrt_out_bf16, M8_test_samples_out,  //
                                       nb_test_samples, "SQRT");

        printf("Test SIMD: numElem=%d\n", numElem);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, 8 * nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test_simd(); }
