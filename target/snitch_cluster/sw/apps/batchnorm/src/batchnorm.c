// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// Standalone folded-BatchNorm + ReLU: out = ReLU(x * scale + shift), with per-channel
// scale/shift. This is the SegFormer ConvModule tail that follows the 1x1-conv (= OS-core
// GEMM, the existing osgemm app). Two SIMD passes: a per-channel multiply by scale, then a
// per-channel add of shift fused with ReLU.

#include <stdint.h>

#include "../data/data.h"
#include "snax-simbacore-lib.h"

int test() {
    int err = 0;

    void* tcdm_base_ptr = snrt_l1_next();
    uint16_t* ptr_x     = (uint16_t*)(tcdm_base_ptr + M10_addr_x);
    uint16_t* ptr_scale = (uint16_t*)(tcdm_base_ptr + M10_addr_scale);
    uint16_t* ptr_shift = (uint16_t*)(tcdm_base_ptr + M10_addr_shift);
    uint16_t* ptr_out   = (uint16_t*)(tcdm_base_ptr + M10_addr_out);

    if (snrt_global_core_idx() == 0) init_cycle_counter();
    snrt_cluster_hw_barrier();

    // Load activations + per-channel scale/shift from L3.
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_x, M10_x, M10_length_x);
        snrt_dma_start_1d(ptr_scale, M10_scale, M10_length_scale);
        snrt_dma_start_1d(ptr_shift, M10_shift, M10_length_shift);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: BatchNorm + ReLU (%u x %u)\n\n", seqLen, channels);
        uint32_t start_cycles = snrt_mcycle();

        // Pass 1: out = x * scale   (per-channel multiply)
        set_simd_streamer_csr((uint32_t)ptr_x, M10_R7_xw_ss, M10_R7_xw_tb, M10_R7_xw_ts,        //
                              (uint32_t)ptr_scale, M10_R13_vec_ss, M10_R13_vec_tb, M10_R13_vec_ts,  //
                              (uint32_t)ptr_out, M10_W3_xw_ss, M10_W3_xw_tb, M10_W3_xw_ts);
        set_simbacore_simd_mode(M10_SIMD_MUL_BF16);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

        // Pass 2: out = ReLU(out + shift)   (per-channel add, fused ReLU; in place)
        set_simd_streamer_csr((uint32_t)ptr_out, M10_R7_xw_ss, M10_R7_xw_tb, M10_R7_xw_ts,         //
                              (uint32_t)ptr_shift, M10_R13_vec_ss, M10_R13_vec_tb, M10_R13_vec_ts,  //
                              (uint32_t)ptr_out, M10_W3_xw_ss, M10_W3_xw_tb, M10_W3_xw_ts);
        set_simbacore_simd_mode(M10_SIMD_ADD_BF16_RELU);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

        uint32_t end_cycles = snrt_mcycle();
        printf("[%u cc] Simbacore elapsed time: %u cycles\n", end_cycles, read_simbacore_perf_counter());
        printf("[%u cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        err += check_result_sample_u16(ptr_out, M10_out, M10_test_samples_out, nb_test_samples, "out");

        printf("Test BatchNorm + ReLU: (%u x %u)\n", seqLen, channels);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test(); }
