// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>

#include <stdint.h>

#include "../data/data.h"
#include "snax-simbacore-lib.h"

/* BF16 = high 16 bits of IEEE 754 float. This doesn't use any rounding, only truncation. */
static inline uint16_t fp32_to_bf16(float f) {
    union {
        float f;
        uint32_t u;
    } x = {.f = f};
    return (uint16_t)(x.u >> 16);
}

/* Convert BF16 to IEEE 754 float (FP32). */
static inline float bf16_to_fp32(uint16_t bf16) {
    union {
        float f;
        uint32_t u;
    } x = {.u = ((uint32_t)bf16 << 16)};
    return x.f;
}

// Temporal strides for broadcast (constant not advanced): all zeros
static const int32_t zero_ts[4] = {0, 0, 0, 0};

// Runs all sqrt/div operations on the CPU
void batch_sqrt_div_cpu(uint16_t* ptr_rms, int32_t seqLen) {
    for (int i = 0; i < seqLen; i++) {
        float f = bf16_to_fp32(ptr_rms[i]);
        f       = sqrt(f);
        f       = bf16_to_fp32(fp32_to_bf16(f));  // Quantize
        f       = 1.0f / f;
        // TODO this truncrates instead of "quantizing", which is a mismatch with datagen
        ptr_rms[i] = fp32_to_bf16(f);
    }
}

int test() {
    int err = 0;

    // Define TCDM addresses
    void* tcdm_base_ptr    = snrt_l1_next();
    uint16_t* ptr_x        = (uint16_t*)(tcdm_base_ptr + M12_addr_x);
    uint16_t* ptr_constant = (uint16_t*)(tcdm_base_ptr + M12_addr_d_inverse);
    uint16_t* ptr_weight   = (uint16_t*)(tcdm_base_ptr + M12_addr_weight);
    uint16_t* ptr_rms      = (uint16_t*)(tcdm_base_ptr + M12_addr_rms);

    // Initialize cycle counter for timing
    if (snrt_global_core_idx() == 0) init_cycle_counter();

    snrt_cluster_hw_barrier();

    // Transfer data from L3 to L1 using DMA only
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_x, M12_x, M12_length_x);
        snrt_dma_start_1d(ptr_weight, M12_weight, M12_length_weight);
        snrt_dma_wait_all();
    }

    // Wait for DMA to finish
    snrt_cluster_hw_barrier();

    // Call compute core
    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: RMSNorm\n\n");
        printf("Expected L1 TCDM usage: %u B (%u KiB)\n", (uint32_t)L1_TCDM_PEAK_BYTES,
               (uint32_t)(L1_TCDM_PEAK_BYTES / 1024));
        uint32_t start_cycles = snrt_mcycle();
#ifdef VERBOSE
        printf("[%d cc] Setting up Streamer and SimbaCore CSRs\n", start_cycles);
#endif

        // 1. Compute Σ(x^2)
        set_simd_streamer_no_b((uint32_t)ptr_x, M12_R7_x_ss, M12_R7_x_tb, M12_R7_x_ts,  //
                               (uint32_t)ptr_rms, M12_W3_rms_ss, M12_W3_rms_tb, M12_W3_rms_ts);

        set_simbacore_simd_mode(M13_SIMD_RMS_BF16);
        set_simbacore_simd_n_acc(dModel);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

        // 2. Divide by D to get average (inplace using ptr_rms)
        uint16_t d_inverse = fp32_to_bf16(1.0f / (float)dModel);  // BF16 encoding of 1/dModel
        // Fill whole lane with d_inverse
        for (int i = 0; i < simdLanes_bf16; i++) ptr_constant[i] = d_inverse;

        set_simd_streamer_csr((uint32_t)ptr_rms, M12_R7_rms_ss, M12_R7_rms_tb, M12_R7_rms_ts,  //
                              (uint32_t)ptr_constant, M12_R7_rms_ss, M12_R7_rms_tb,
                              (int32_t*)zero_ts,  // Same temporal bound, no stride
                              (uint32_t)ptr_rms, M12_W3_rms_ss, M12_W3_rms_tb, M12_W3_rms_ts);  // We write inplace

        set_simbacore_simd_mode(M10_SIMD_MUL_BF16);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

        // 3. Compute rms = 1 / sqrt(Σ(x^2) / D)  (single RSQRT pass: sqrt then reciprocal)
        set_simd_streamer_no_b((uint32_t)ptr_rms, M12_R7_rms_ss, M12_R7_rms_tb, M12_R7_rms_ts,   //
                               (uint32_t)ptr_rms, M12_W3_rms_ss, M12_W3_rms_tb, M12_W3_rms_ts);  // We write inplace

        set_simbacore_simd_mode(M45_SIMD_RSQRT_BF16);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

        // err += check_result_all_u16(ptr_rms, M12_invRms, M12_length_rms);
        // err += check_result_sample_u16(ptr_rms, M12_invRms, M12_test_samples_rms, nb_test_samples, "invRms");

        // 4. Multiply x by rms (inplace)
        set_simd_streamer_csr((uint32_t)ptr_x, M12_R7_x_ss, M12_R7_x_tb, M12_R7_x_ts,
                              // Slide over D (one rms norm per token)
                              (uint32_t)ptr_rms, M12_R13_x_rms_ss, M12_R13_x_rms_tb, M12_R13_x_rms_ts,  //
                              (uint32_t)ptr_x, M12_W3_x_ss, M12_W3_x_tb, M12_W3_x_ts);  // We write inplace

        set_simbacore_simd_mode(M10_SIMD_MUL_BF16);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

        // printf("normalized (x * rms)\n");
        // for (int i = 0; i < 32; i++) {
        //     printf("normalized[%d] = %u,\n", i, ptr_x[i]);
        // }

        // err += check_result_all_u16(ptr_x, M12_normalized, M12_length_normalized);
        // err += check_result_sample_u16(ptr_x, M12_normalized, M12_test_samples_expected, nb_test_samples,
        // "normalized");

        // 5. Multiply x by weight (inplace)
        set_simd_streamer_csr((uint32_t)ptr_x, M12_R7_x_w_ss, M12_R7_x_w_tb, M12_R7_x_w_ts,
                              // Keep weight stationary for L
                              (uint32_t)ptr_weight, M12_R13_x_w_ss, M12_R13_x_w_tb, M12_R13_x_w_ts,  //
                              (uint32_t)ptr_x, M12_W3_x_w_ss, M12_W3_x_w_tb, M12_W3_x_w_ts);         // We write inplace
        // (We are still in MUL mode)
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

        // Final bookkeeping
        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Simbacore elapsed time: %u cycles\n", end_cycles, read_simbacore_perf_counter());
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        err += check_result_sample_u16(ptr_x, M12_norm, M12_test_samples_expected,  //
                                       nb_test_samples, "norm");

        printf("Test RMSNORM: (%d x %d)\n", seqLen, dModel);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test(); }
