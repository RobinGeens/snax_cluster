// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// VMamba SS2D program.
//
// Per direction k:
//   Phase 1: osCore(in_proj) + SwitchCore(conv1d+SiLU) + IS-core(x_proj)
//     → conv_out (= Phase 2 SUC x) and iscore_out (= dt+B+C in xProj format)
//   Phase 2: osCore(z recompute) + SwitchCore(dt_proj) + SUC + IS-core(out_proj)
//     → z, y (SUC out), iscore_out (final output)
//
// Per-direction oscore_in (cross-scanned) and iscore_weight (x_proj) are
// pre-computed in the Scala data generator and loaded from L3 via DMA.

#include "helper.c"
#include "snax-simbacore-lib.h"

static inline uint16_t fp32_to_bf16(float f) {
    union {
        float f;
        uint32_t u;
    } x = {.f = f};
    return (uint16_t)(x.u >> 16);
}

int test_ss2d() {
    int err = 0;

    printf("VMamba SS2D: H=%d, W=%d, seqLen=%d, dModel=%d, dInner=%d, K=%d\n", H, W, seqLen, dModel, dInner, K);

    void* tcdm_base_ptr = snrt_l1_next();

    // Phase 1 buffers
    uint8_t* ptr_oscore_in        = (uint8_t*)(tcdm_base_ptr + M1_addr_oscore_in);
    uint8_t* ptr_oscore_weight_P1 = (uint8_t*)(tcdm_base_ptr + M1_addr_oscore_weight);
    uint8_t* ptr_conv_weight      = (uint8_t*)(tcdm_base_ptr + M1_addr_conv_weight);
    uint8_t* ptr_conv_bias        = (uint8_t*)(tcdm_base_ptr + M1_addr_conv_bias);
    uint8_t* ptr_conv_out         = (uint8_t*)(tcdm_base_ptr + M1_addr_conv_out);
    uint8_t* ptr_iscore_weight_P1 = (uint8_t*)(tcdm_base_ptr + M1_addr_iscore_weight);
    uint16_t* ptr_iscore_out_P1   = (uint16_t*)(tcdm_base_ptr + M1_addr_iscore_out);

    // Phase 2 buffers
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

    // ---- Load shared weights ----
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_oscore_weight_P1, M2_oscore_weight, M1_length_oscore_weight);
        snrt_dma_start_1d(ptr_conv_weight, M2_conv_weight, M1_length_conv_weight);
        snrt_dma_start_1d(ptr_conv_bias, M2_conv_bias, M1_length_conv_bias);
        snrt_dma_start_1d(ptr_oscore_weight_P2, M2_oscore_weight_P2, M2_length_oscore_weight);
        snrt_dma_start_1d(ptr_dt_weight_1, M2_dt_weight_1, M2_length_dt_weight_1);
        snrt_dma_start_1d(ptr_dt_weight_2, M2_dt_weight_2, M2_length_dt_weight_2);
        snrt_dma_start_1d(ptr_dt_bias, M2_dt_bias, M2_length_dt_bias);
        snrt_dma_start_1d(ptr_A, M2_suc_A, M2_length_A);
        snrt_dma_start_1d(ptr_D, M2_suc_D, M2_length_D);
        snrt_dma_start_1d(ptr_iscore_weight_P2, M2_iscore_weight_P2, M2_length_iscore_weight);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    uint32_t start_cycles = 0;
    uint32_t p1_cycles = 0, p2_cycles = 0;

    if (snrt_global_core_idx() == 0) {
        printf("\nStarting VMamba SS2D: K=%d dirs, Phase1 + Phase2 per dir\n\n", K);
        printf("Expected L1 TCDM usage: %u B (%u KiB)\n", (uint32_t)L1_TCDM_PEAK_BYTES,
               (uint32_t)(L1_TCDM_PEAK_BYTES / 1024));
        start_cycles = snrt_mcycle();
    }

    for (uint32_t k = 0; k < K; k++) {
        // ---- Load per-direction data (pre-flattened in Scala generator) ----
        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(ptr_oscore_in, M2_oscore_in_K + k * dir_size_oscore_in, M1_length_oscore_in);
            snrt_dma_start_1d(ptr_iscore_weight_P1, M2_iscore_weight_K + k * dir_size_iscore_weight,
                              M1_length_iscore_weight);
            snrt_dma_start_1d(ptr_iscore_out_P1, M2_iscore_bias, M1_length_iscore_out);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();

        // ---- Phase 1: in_proj + conv + x_proj ----
        if (snrt_global_core_idx() == 0) {
            set_streamer_phase1((uint32_t)ptr_oscore_in, (uint32_t)ptr_oscore_weight_P1, (uint32_t)ptr_conv_weight,
                                (uint32_t)ptr_conv_bias, (uint32_t)ptr_iscore_weight_P1, (uint32_t)ptr_iscore_out_P1,
                                (uint32_t)ptr_conv_out);
            set_simbacore_csr(M1_PHASE1, seqLen, dModel, dInner, dtRank, xProjDim);
            start_simbacore_and_streamers(M1_R10_en, 0, M1_R11_en, 0);
            wait_simbacore_and_streamer();
            p1_cycles += read_simbacore_perf_counter();
        }

        // Phase 2 bias preload
        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(ptr_iscore_out_P2, M2_iscore_bias_P2, M2_length_iscore_out);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();

        // ---- Phase 2: osCore(z) + SwitchCore(dt_proj) + SUC + IsCore(out_proj) ----
        if (snrt_global_core_idx() == 0) {
            set_streamer_phase2((uint32_t)ptr_oscore_in, (uint32_t)ptr_oscore_weight_P2, (uint32_t)ptr_z,
                                (uint32_t)ptr_dt_in, (uint32_t)ptr_dt_weight_1, (uint32_t)ptr_dt_weight_2,
                                (uint32_t)ptr_dt_bias, (uint32_t)ptr_x, (uint32_t)ptr_A, (uint32_t)ptr_BC,
                                (uint32_t)ptr_D, (uint32_t)ptr_y, (uint32_t)ptr_iscore_weight_P2,
                                (uint32_t)ptr_iscore_out_P2);
            set_simbacore_csr(M2_PHASE2, seqLen, dModel, dInner, dtRank, dModel);
            start_simbacore_and_streamers(M2_R10_en, M2_R10_start_cnt, M2_R11_en, M2_R11_start_cnt);
            wait_simbacore_and_streamer();
            p2_cycles += read_simbacore_perf_counter();

            printf("[%u cc] Dir %u done\n", snrt_mcycle(), k);
        }
        snrt_cluster_hw_barrier();
    }

    // ---- Cross-merge: sum 4 inverse-permuted y_k via SIMD ADD ----
    uint8_t* ptr_merge_a = ptr_conv_out;
    uint8_t* ptr_merge_b = ptr_y;
    uint32_t merge_len   = dir_size_y_invperm;

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_merge_a, M2_y_invperm_K, merge_len);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    uint32_t merge_cycles = 0;
    for (uint32_t k = 1; k < K; k++) {
        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(ptr_merge_b, M2_y_invperm_K + k * dir_size_y_invperm, merge_len);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();

        if (snrt_global_core_idx() == 0) {
            set_simd_streamer_csr((uint32_t)ptr_merge_a, M16_R7_ss, M16_R7_tb, M16_R7_ts, (uint32_t)ptr_merge_b,
                                  M16_R13_ss, M16_R13_tb, M16_R13_ts, (uint32_t)ptr_merge_a, M16_W3_ss, M16_W3_tb,
                                  M16_W3_ts);
            set_simbacore_simd_mode(M16_SIMD_ADD_FP8);
            start_simbacore_and_streamers(0, 0, 0, 0);
            wait_simbacore_and_streamer();
            merge_cycles += read_simbacore_perf_counter();
        }
        snrt_cluster_hw_barrier();
    }

    if (snrt_global_core_idx() == 0) {
        err += check_result_sample(ptr_merge_a, M2_y_merged_flat, M2_test_samples_y_merged_flat, nb_test_samples,
                                   "y_merged");
        printf("[%u cc] Cross-merge done (%u cc)\n", snrt_mcycle(), merge_cycles);
    }
    snrt_cluster_hw_barrier();

    // ---- RMSNorm: 8-step SIMD chain ----
    uint16_t* ptr_rms_x             = (uint16_t*)(tcdm_base_ptr + M12_addr_rms_x_bf16);
    uint16_t* ptr_rms_c             = (uint16_t*)(tcdm_base_ptr + M12_addr_rms_const);
    uint16_t* ptr_rms_w             = (uint16_t*)(tcdm_base_ptr + M12_addr_rms_weight);
    uint16_t* ptr_rms_vec           = (uint16_t*)(tcdm_base_ptr + M12_addr_rms_vec);
    static const int32_t zero_ts[4] = {0, 0, 0, 0};

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_rms_w, M2_norm_weight, M12_length_rms_weight);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    uint32_t rms_cycles = 0;
    if (snrt_global_core_idx() == 0) {
        // Widen FP8 → BF16
        set_simd_streamer_no_b((uint32_t)ptr_merge_a, M12_R7_widen_ss, M12_R7_widen_tb, M12_R7_widen_ts,
                               (uint32_t)ptr_rms_x, M12_W3_widen_ss, M12_W3_widen_tb, M12_W3_widen_ts);
        set_simbacore_csr(M25_SIMD_NOOP_FP8_REQUANT, seqLen, dInner, dInner, 1, 1);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();
        rms_cycles += read_simbacore_perf_counter();

        // RMS: Σ(x²) per token
        set_simd_streamer_no_b((uint32_t)ptr_rms_x, M12_R7_x_ss, M12_R7_x_tb, M12_R7_x_ts, (uint32_t)ptr_rms_vec,
                               M12_W3_rms_ss, M12_W3_rms_tb, M12_W3_rms_ts);
        set_simbacore_simd_mode(M13_SIMD_RMS_BF16);
        set_simbacore_simd_n_acc(dInner);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();
        rms_cycles += read_simbacore_perf_counter();

        // ÷D
        uint16_t d_inv = fp32_to_bf16(1.0f / (float)dInner);
        for (int i = 0; i < simdLanes_bf16; i++) ptr_rms_c[i] = d_inv;
        set_simd_streamer_csr((uint32_t)ptr_rms_vec, M12_R7_rms_ss, M12_R7_rms_tb, M12_R7_rms_ts, (uint32_t)ptr_rms_c,
                              M12_R7_rms_ss, M12_R7_rms_tb, (int32_t*)zero_ts, (uint32_t)ptr_rms_vec, M12_W3_rms_ss,
                              M12_W3_rms_tb, M12_W3_rms_ts);
        set_simbacore_simd_mode(M10_SIMD_MUL_BF16);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();
        rms_cycles += read_simbacore_perf_counter();

        // √
        set_simd_streamer_no_b((uint32_t)ptr_rms_vec, M12_R7_rms_ss, M12_R7_rms_tb, M12_R7_rms_ts,
                               (uint32_t)ptr_rms_vec, M12_W3_rms_ss, M12_W3_rms_tb, M12_W3_rms_ts);
        set_simbacore_simd_mode(M15_SIMD_SQRT_BF16);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();
        rms_cycles += read_simbacore_perf_counter();

        // 1/√
        uint16_t one = fp32_to_bf16(1.0f);
        for (int i = 0; i < simdLanes_bf16; i++) ptr_rms_c[i] = one;
        set_simd_streamer_csr((uint32_t)ptr_rms_c, M12_R7_rms_ss, M12_R7_rms_tb, (int32_t*)zero_ts,
                              (uint32_t)ptr_rms_vec, M12_R7_rms_ss, M12_R7_rms_tb, M12_R7_rms_ts, (uint32_t)ptr_rms_vec,
                              M12_W3_rms_ss, M12_W3_rms_tb, M12_W3_rms_ts);
        set_simbacore_simd_mode(M14_SIMD_DIV_BF16);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();
        rms_cycles += read_simbacore_perf_counter();

        // x × (1/√)
        set_simd_streamer_csr((uint32_t)ptr_rms_x, M12_R7_x_ss, M12_R7_x_tb, M12_R7_x_ts, (uint32_t)ptr_rms_vec,
                              M12_R13_x_rms_ss, M12_R13_x_rms_tb, M12_R13_x_rms_ts, (uint32_t)ptr_rms_x, M12_W3_x_ss,
                              M12_W3_x_tb, M12_W3_x_ts);
        set_simbacore_simd_mode(M10_SIMD_MUL_BF16);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();
        rms_cycles += read_simbacore_perf_counter();

        // × weight
        set_simd_streamer_csr((uint32_t)ptr_rms_x, M12_R7_x_w_ss, M12_R7_x_w_tb, M12_R7_x_w_ts, (uint32_t)ptr_rms_w,
                              M12_R13_x_w_ss, M12_R13_x_w_tb, M12_R13_x_w_ts, (uint32_t)ptr_rms_x, M12_W3_x_w_ss,
                              M12_W3_x_w_tb, M12_W3_x_w_ts);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();
        rms_cycles += read_simbacore_perf_counter();

        // Narrow BF16 → FP8
        set_simd_streamer_no_b((uint32_t)ptr_rms_x, M12_R7_x_ss, M12_R7_x_tb, M12_R7_x_ts, (uint32_t)ptr_merge_a,
                               M12_W3_narrow_ss, M12_W3_narrow_tb, M12_W3_narrow_ts);
        set_simbacore_csr(M24_SIMD_NOOP_BF16_REQUANT, seqLen, dInner, dInner, 1, 1);
        start_simbacore_and_streamers(0, 0, 0, 0);
        wait_simbacore_and_streamer();
        rms_cycles += read_simbacore_perf_counter();

        err += check_result_sample(ptr_merge_a, M2_y_norm_flat, M2_test_samples_y_norm_flat, nb_test_samples, "y_norm");
        printf("[%u cc] RMSNorm done (%u cc)\n", snrt_mcycle(), rms_cycles);
    }
    snrt_cluster_hw_barrier();

    // ---- Summary ----
    if (snrt_global_core_idx() == 0) {
        uint32_t end_cycles = snrt_mcycle();
        printf("[%u cc] P1: %u cc, P2: %u cc, merge: %u cc, RMSNorm: %u cc, Snitch: %u cc\n", end_cycles, p1_cycles,
               p2_cycles, merge_cycles, rms_cycles, end_cycles - start_cycles);
        printf("[%u cc] Simbacore elapsed time: %u cycles\n", end_cycles,
               p1_cycles + p2_cycles + merge_cycles + rms_cycles);
        printf("[%u cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);
        printf("Test VMamba SS2D: H=%d, W=%d, dModel=%d, K=%d\n", H, W, dModel, K);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, 2 * nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test_ss2d(); }
