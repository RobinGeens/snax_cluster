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

int test() {
    int err = 0;

    // Define TCDM addresses
    void* tcdm_base_ptr     = snrt_l1_next();
    uint16_t* ptr_x         = (uint16_t*)(tcdm_base_ptr + M11_addr_x);
    uint16_t* ptr_d_inverse = (uint16_t*)(tcdm_base_ptr + M11_addr_d_inverse);
    uint16_t* ptr_weight    = (uint16_t*)(tcdm_base_ptr + M11_addr_weight);
    uint16_t* ptr_rms       = (uint16_t*)(tcdm_base_ptr + M11_addr_rms);

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
        set_simd_streamer_no_b((uint32_t)ptr_x, M11_R7_x_ss, M11_R7_x_tb, M11_R7_x_ts,  //
                               (uint32_t)ptr_rms, M11_W3_rms_ss, M11_W3_rms_tb, M11_W3_rms_ts);

        set_simbacore_simd_mode(M11_SIMD_RMS);
        set_simbacore_simd_n_acc(dModel);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

        // 2. Divide by D to get average (inplace using ptr_rms)
        uint16_t d_inverse = fp32_to_bf16(1.0f / (float)dModel);  // BF16 encoding of 1/dModel
        // Fill whole lane with d_inverse
        for (int i = 0; i < simdLanes; i++) {
            ptr_d_inverse[i] = d_inverse;
        }
        set_simd_streamer_csr((uint32_t)ptr_rms, M11_R7_rms_ss, M11_R7_rms_tb, M11_R7_rms_ts,  //
                              (uint32_t)ptr_d_inverse, M11_R7_rms_ss, M11_R7_rms_tb,
                              0,  // Same temporal bound, no stride
                              (uint32_t)ptr_rms, M11_W3_rms_ss, M11_W3_rms_tb, M11_W3_rms_ts);  // We write inplace
        set_simbacore_simd_mode(M8_SIMD_MUL);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

        // 3. Compute sqrt(Σ(x^2)/D)
        for (int i = 0; i < seqLen; i++) {
            float f = bf16_to_fp32(ptr_rms[i]);
            f       = sqrt(f);
            f       = 1.0f / f;
            // TODO this truncrates instead of "quantizing", which is a mismatch with datagen
            ptr_rms[i] = fp32_to_bf16(f);
        }

        // 4. Multiply x by rms (inplace)
        set_simd_streamer_csr((uint32_t)ptr_x, M11_R7_x_ss, M11_R7_x_tb, M11_R7_x_ts,
                              // Slide over D (one rms norm per token)
                              (uint32_t)ptr_rms, M11_R13_x_rms_ss, M11_R13_x_rms_tb, M11_R13_x_rms_ts,  //
                              (uint32_t)ptr_x, M11_W3_x_ss, M11_W3_x_tb, M11_W3_x_ts);  // We write inplace
        // (We are still in MUL mode)
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

        // 5. Multiply x by weight (inplace)
        set_simd_streamer_csr((uint32_t)ptr_x, M11_R7_x_w_ss, M11_R7_x_w_tb, M11_R7_x_w_ts,
                              // Keep weight stationary for L
                              (uint32_t)ptr_weight, M11_R13_x_w_ss, M11_R13_x_w_tb, M11_R13_x_w_ts,  //
                              (uint32_t)ptr_x, M11_W3_x_w_ss, M11_W3_x_w_tb, M11_W3_x_w_ts);         // We write inplace
        // (We are still in MUL mode)
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();

        // Final bookkeeping
        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Simbacore elapsed time: %u cycles\n", end_cycles, read_simbacore_perf_counter());
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        err += check_result_sample_u16(ptr_x, M11_out, M11_test_samples_expected,  //
                                       nb_test_samples, "out");

        printf("Test RMSNORM: (%d x %d)\n", seqLen, dModel);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test(); }
