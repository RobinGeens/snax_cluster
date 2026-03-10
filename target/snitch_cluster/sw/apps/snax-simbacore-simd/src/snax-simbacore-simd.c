// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>

#include "../data/data.h"
#include "snax-simbacore-lib.h"
#include "streamer_csr_addr_map.h"

int test_simd_bf16() {
    int err = 0;

    // Define TCDM addresses
    void* tcdm_base_ptr      = snrt_l1_next();
    uint16_t* ptr_a          = (uint16_t*)(tcdm_base_ptr + M8_addr_in_a_bf16);
    uint16_t* ptr_b          = (uint16_t*)(tcdm_base_ptr + M8_addr_in_b_bf16);
    uint16_t* ptr_out_add    = (uint16_t*)(tcdm_base_ptr + M8_addr_add_out_bf16);
    uint16_t* ptr_out_sub    = (uint16_t*)(tcdm_base_ptr + M8_addr_sub_out_bf16);
    uint16_t* ptr_out_mul    = (uint16_t*)(tcdm_base_ptr + M8_addr_mul_out_bf16);
    uint16_t* ptr_out_cmul   = (uint16_t*)(tcdm_base_ptr + M8_addr_cmul_out_bf16);
    uint16_t* ptr_out_inprod = (uint16_t*)(tcdm_base_ptr + M8_addr_inprod_out_bf16);
    uint16_t* ptr_out_rms    = (uint16_t*)(tcdm_base_ptr + M8_addr_rms_out_bf16);
    uint16_t* ptr_out_div    = (uint16_t*)(tcdm_base_ptr + M8_addr_div_out_bf16);
    uint16_t* ptr_out_sqrt   = (uint16_t*)(tcdm_base_ptr + M8_addr_sqrt_out_bf16);

    uint8_t* ptr_out_mul_requant  = (uint8_t*)(tcdm_base_ptr + M8_addr_mul_out_bf16_requant);
    uint8_t* ptr_out_noop_requant = (uint8_t*)(tcdm_base_ptr + M8_addr_noop_out_bf16_requant);

    // Initialize cycle counter for timing
    if (snrt_global_core_idx() == 0) init_cycle_counter();

    snrt_cluster_hw_barrier();

    // Transfer data from L3 to L1 using DMA only
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_a, M8_in_a_bf16, M8_length_in_a_bf16);
        snrt_dma_start_1d(ptr_b, M8_in_b_bf16, M8_length_in_b_bf16);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    // Call compute core
    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: SIMD BF16\n\n");
        uint32_t start_cycles = snrt_mcycle();

        // CMUL - BF16 core has no CMUL
        // #ifdef VERBOSE
        //         printf("[%d cc] Setting up Streamer and SimbaCore CSRs\n", start_cycles);
        //         printf("[%d cc] CMUL\n", snrt_mcycle());
        // #endif
        set_simd_streamer_csr((uint32_t)ptr_a, M8_R7_bf16_ss, M8_R7_bf16_tb, M8_R7_bf16_ts,        // SUC BC
                              (uint32_t)ptr_b, M8_R13_bf16_ss, M8_R13_bf16_tb, M8_R13_bf16_ts,     // isCore psum
                              (uint32_t)ptr_out_cmul, M8_W3_bf16_ss, M8_W3_bf16_tb, M8_W3_bf16_ts  // isCore out
        );

        set_simbacore_csr(M8_SIMD_ADD_BF16, 0, 0, 0, 0, 0);
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

#ifdef VERBOSE
        printf("[%d cc] MUL_REQUANT\n", snrt_mcycle());
#endif
        set_simd_streamer_csr((uint32_t)ptr_a, M8_R7_bf16_ss, M8_R7_bf16_tb, M8_R7_bf16_ts,     // SUC BC
                              (uint32_t)ptr_b, M8_R13_bf16_ss, M8_R13_bf16_tb, M8_R13_bf16_ts,  // isCore psum
                              (uint32_t)ptr_out_mul_requant, M8_W3_fp8_ss, M8_W3_fp8_tb,
                              M8_W3_fp8_ts  // Output are FP8's (less loops)

        );
        set_simbacore_simd_mode(M11_SIMD_MUL_BF16_REQUANT);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

// INPROD
#ifdef VERBOSE
        printf("[%d cc] INPROD\n", snrt_mcycle());
#endif
        set_simd_streamer_csr((uint32_t)ptr_a, M8_R7_bf16_ss, M8_R7_bf16_tb, M8_R7_bf16_ts,     // SUC BC
                              (uint32_t)ptr_b, M8_R13_bf16_ss, M8_R13_bf16_tb, M8_R13_bf16_ts,  // isCore psum
                              (uint32_t)ptr_out_inprod, M8_W3_reduce_bf16_ss, M8_W3_reduce_bf16_tb,
                              M8_W3_reduce_bf16_ts  // isCore out
        );
        set_simbacore_simd_mode(M12_SIMD_INPROD_BF16);
        set_simbacore_simd_n_acc(n_acc);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

// RMS
#ifdef VERBOSE
        printf("[%d cc] RMS\n", snrt_mcycle());
#endif
        set_simd_streamer_no_b((uint32_t)ptr_a, M8_R7_bf16_ss, M8_R7_bf16_tb, M8_R7_bf16_ts,  // SUC BC
                               (uint32_t)ptr_out_rms, M8_W3_reduce_bf16_ss, M8_W3_reduce_bf16_tb,
                               M8_W3_reduce_bf16_ts  // isCore out
        );
        set_simbacore_simd_mode(M13_SIMD_RMS_BF16);
        set_simbacore_simd_n_acc(n_acc);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

// SQRT
#ifdef VERBOSE
        printf("[%d cc] SQRT\n", snrt_mcycle());
#endif
        set_simd_streamer_no_b((uint32_t)ptr_a, M8_R7_bf16_ss, M8_R7_bf16_tb, M8_R7_bf16_ts,        // SUC BC
                               (uint32_t)ptr_out_sqrt, M8_W3_bf16_ss, M8_W3_bf16_tb, M8_W3_bf16_ts  // isCore out
        );
        set_simbacore_simd_mode(M15_SIMD_SQRT_BF16);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

// NOOP_REQUANT (BF16 -> FP8)
#ifdef VERBOSE
        printf("[%d cc] NOOP_REQUANT\n", snrt_mcycle());
#endif
        set_simd_streamer_no_b((uint32_t)ptr_a, M8_R7_bf16_ss, M8_R7_bf16_tb, M8_R7_bf16_ts,
                               (uint32_t)ptr_out_noop_requant, M8_W3_fp8_ss, M8_W3_fp8_tb, M8_W3_fp8_ts);
        set_simbacore_simd_mode(M24_SIMD_NOOP_BF16_REQUANT);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

        // Final
        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Simbacore elapsed time: %u cycles\n", end_cycles, read_simbacore_perf_counter());
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        err += check_result_sample_u16(ptr_out_add, M8_add_out_bf16,
                                       M8_test_samples_out,  //
                                       nb_test_samples, "ADD");
        err += check_result_sample_u16(ptr_out_sub, M8_sub_out_bf16,
                                       M8_test_samples_out,  //
                                       nb_test_samples, "SUB");
        err += check_result_sample_u16(ptr_out_mul, M8_mul_out_bf16,
                                       M8_test_samples_out,  //
                                       nb_test_samples, "MUL");
        err += check_result_sample_u16(ptr_out_inprod, M8_inprod_out_bf16,
                                       M8_test_samples_out_reduce,  //
                                       nb_test_samples, "INPROD");
        err += check_result_sample_u16(ptr_out_rms, M8_rms_out_bf16,
                                       M8_test_samples_out_reduce,  //
                                       nb_test_samples, "RMS");
        err += check_result_sample_u16(ptr_out_div, M8_div_out_bf16,
                                       M8_test_samples_out,  //
                                       nb_test_samples, "DIV");
        err += check_result_sample_u16(ptr_out_sqrt, M8_sqrt_out_bf16,
                                       M8_test_samples_out,  //
                                       nb_test_samples, "SQRT");
        err += check_result_sample(ptr_out_mul_requant, M8_mul_out_bf16_requant,
                                   M8_test_samples_out,  //
                                   nb_test_samples, "MUL_REQUANT");
        err += check_result_sample(ptr_out_noop_requant, M8_noop_out_bf16_requant,
                                   M8_test_samples_out,  //
                                   nb_test_samples, "NOOP_BF16_REQUANT");

        printf("Test SIMD: numElem=%d\n", numElem);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, 9 * nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int test_simd_fp8() {
    int err = 0;

    // Define TCDM addresses
    void* tcdm_base_ptr     = snrt_l1_next();
    uint8_t* ptr_a          = (uint8_t*)(tcdm_base_ptr + M8_addr_in_a_fp8);
    uint8_t* ptr_b          = (uint8_t*)(tcdm_base_ptr + M8_addr_in_b_fp8);
    uint8_t* ptr_out_add    = (uint8_t*)(tcdm_base_ptr + M8_addr_add_out_fp8);
    uint8_t* ptr_out_sub    = (uint8_t*)(tcdm_base_ptr + M8_addr_sub_out_fp8);
    uint8_t* ptr_out_mul    = (uint8_t*)(tcdm_base_ptr + M8_addr_mul_out_fp8);
    uint8_t* ptr_out_cmul   = (uint8_t*)(tcdm_base_ptr + M8_addr_cmul_out_fp8);
    uint8_t* ptr_out_inprod = (uint8_t*)(tcdm_base_ptr + M8_addr_inprod_out_fp8);
    uint8_t* ptr_out_rms    = (uint8_t*)(tcdm_base_ptr + M8_addr_rms_out_fp8);

    uint16_t* ptr_out_mul_requant  = (uint16_t*)(tcdm_base_ptr + M8_addr_mul_out_fp8_requant);
    uint16_t* ptr_out_noop_requant = (uint16_t*)(tcdm_base_ptr + M8_addr_noop_out_fp8_requant);
    uint8_t* ptr_out_softshrink    = (uint8_t*)(tcdm_base_ptr + M8_addr_softshrink_out_fp8);

    if (snrt_global_core_idx() == 0) init_cycle_counter();

    snrt_cluster_hw_barrier();

    // Transfer data from L3 to L1 using DMA only
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_a, M8_in_a_fp8, M8_length_in_a_fp8);
        snrt_dma_start_1d(ptr_b, M8_in_b_fp8, M8_length_in_b_fp8);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    // Call compute core
    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: SIMD FP8\n\n");
        uint32_t start_cycles = snrt_mcycle();

        // CMUL
#ifdef VERBOSE
        printf("[%d cc] CMUL\n", snrt_mcycle());
#endif
        set_simd_streamer_csr((uint32_t)ptr_a, M8_R7_fp8_ss, M8_R7_fp8_tb, M8_R7_fp8_ts, (uint32_t)ptr_b, M8_R13_fp8_ss,
                              M8_R13_fp8_tb, M8_R13_fp8_ts, (uint32_t)ptr_out_cmul, M8_W3_fp8_ss, M8_W3_fp8_tb,
                              M8_W3_fp8_ts);
        set_simbacore_csr(M20_SIMD_CMUL_FP8, 0, 0, 0, 0, 0);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

        // ADD
#ifdef VERBOSE
        printf("[%d cc] ADD\n", snrt_mcycle());
#endif
        write_csr(BASE_PTR_WRITER_3_LOW, ptr_out_add);
        set_simbacore_simd_mode(M16_SIMD_ADD_FP8);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

        // SUB
#ifdef VERBOSE
        printf("[%d cc] SUB\n", snrt_mcycle());
#endif
        write_csr(BASE_PTR_WRITER_3_LOW, ptr_out_sub);
        set_simbacore_simd_mode(M17_SIMD_SUB_FP8);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

        // MUL
#ifdef VERBOSE
        printf("[%d cc] MUL\n", snrt_mcycle());
#endif
        write_csr(BASE_PTR_WRITER_3_LOW, ptr_out_mul);
        set_simbacore_simd_mode(M18_SIMD_MUL_FP8);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

        // MUL_REQUANT (FP8 -> BF16)
#ifdef VERBOSE
        printf("[%d cc] MUL_REQUANT\n", snrt_mcycle());
#endif
        set_simd_streamer_csr((uint32_t)ptr_a, M8_R7_fp8_ss, M8_R7_fp8_tb, M8_R7_fp8_ts, (uint32_t)ptr_b, M8_R13_fp8_ss,
                              M8_R13_fp8_tb, M8_R13_fp8_ts, (uint32_t)ptr_out_mul_requant, M8_W3_bf16_ss, M8_W3_bf16_tb,
                              M8_W3_bf16_ts);

        set_simbacore_simd_mode(M19_SIMD_MUL_FP8_REQUANT);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

// INPROD
#ifdef VERBOSE
        printf("[%d cc] INPROD\n", snrt_mcycle());
#endif
        set_simd_streamer_csr((uint32_t)ptr_a, M8_R7_fp8_ss, M8_R7_fp8_tb, M8_R7_fp8_ts, (uint32_t)ptr_b, M8_R13_fp8_ss,
                              M8_R13_fp8_tb, M8_R13_fp8_ts, (uint32_t)ptr_out_inprod, M8_W3_reduce_fp8_ss,
                              M8_W3_reduce_fp8_tb, M8_W3_reduce_fp8_ts);
        set_simbacore_simd_mode(M21_SIMD_INPROD_FP8);
        set_simbacore_simd_n_acc(n_acc);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

// RMS
#ifdef VERBOSE
        printf("[%d cc] RMS\n", snrt_mcycle());
#endif
        set_simd_streamer_no_b((uint32_t)ptr_a, M8_R7_fp8_ss, M8_R7_fp8_tb, M8_R7_fp8_ts, (uint32_t)ptr_out_rms,
                               M8_W3_reduce_fp8_ss, M8_W3_reduce_fp8_tb, M8_W3_reduce_fp8_ts);
        set_simbacore_simd_mode(M22_SIMD_RMS_FP8);
        set_simbacore_simd_n_acc(n_acc);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

// NOOP_REQUANT (FP8 -> BF16)
#ifdef VERBOSE
        printf("[%d cc] NOOP_REQUANT\n", snrt_mcycle());
#endif
        set_simd_streamer_no_b((uint32_t)ptr_a, M8_R7_fp8_ss, M8_R7_fp8_tb, M8_R7_fp8_ts,
                               (uint32_t)ptr_out_noop_requant, M8_W3_bf16_ss, M8_W3_bf16_tb, M8_W3_bf16_ts);
        set_simbacore_simd_mode(M25_SIMD_NOOP_FP8_REQUANT);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

// SOFTSHRINK (FP8)
#ifdef VERBOSE
        printf("[%d cc] SOFTSHRINK\n", snrt_mcycle());
#endif
        set_simd_streamer_csr((uint32_t)ptr_a, M8_R7_fp8_ss, M8_R7_fp8_tb, M8_R7_fp8_ts, (uint32_t)ptr_b, M8_R13_fp8_ss,
                              M8_R13_fp8_tb, M8_R13_fp8_ts, (uint32_t)ptr_out_softshrink, M8_W3_softshrink_fp8_ss,
                              M8_W3_softshrink_fp8_tb, M8_W3_softshrink_fp8_ts);
        set_simbacore_simd_mode(M26_SIMD_SOFTSHRINK_FP8);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

        // Final
        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Simbacore elapsed time: %u cycles\n", end_cycles, read_simbacore_perf_counter());
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        err += check_result_sample(ptr_out_cmul, M8_cmul_out_fp8, M8_test_samples_out, nb_test_samples, "CMUL_FP8");
        err += check_result_sample(ptr_out_add, M8_add_out_fp8, M8_test_samples_out, nb_test_samples, "ADD_FP8");
        err += check_result_sample(ptr_out_sub, M8_sub_out_fp8, M8_test_samples_out, nb_test_samples, "SUB_FP8");
        err += check_result_sample(ptr_out_mul, M8_mul_out_fp8, M8_test_samples_out, nb_test_samples, "MUL_FP8");
        err += check_result_sample(ptr_out_inprod, M8_inprod_out_fp8, M8_test_samples_out_reduce, nb_test_samples,
                                   "INPROD_FP8");
        err += check_result_sample(ptr_out_rms, M8_rms_out_fp8, M8_test_samples_out_reduce, nb_test_samples, "RMS_FP8");
        err += check_result_sample_u16(ptr_out_mul_requant, M8_mul_out_fp8_requant, M8_test_samples_out,
                                       nb_test_samples, "MUL_FP8_REQUANT");
        err += check_result_sample_u16(ptr_out_noop_requant, M8_noop_out_fp8_requant, M8_test_samples_out,
                                       nb_test_samples, "NOOP_FP8_REQUANT");
        err += check_result_sample(ptr_out_softshrink, M8_softshrink_out_fp8, M8_test_samples_out_softshrink,
                                   nb_test_samples, "SOFTSHRINK_FP8");

        printf("Test SIMD FP8: numElem=%d\n", numElem);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, 9 * nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() {
    int err = 0;
    err += test_simd_fp8();
    err += test_simd_bf16();
    return err;
}
