// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// dInner-tiled VMamba SS2D: per-direction tiled Phase 1 + Phase 2,
// then cross-merge (SIMD ADD) and RMSNorm.
//
// Per direction k (K=4):
//   Phase 1 (tiled): osCore(in_proj) + SwitchCore(conv1d+SiLU) + IS-core(x_proj)
//   Phase 2 (tiled): osCore(z) + SwitchCore(dt_proj) + SUC + IS-core(out_proj)
// Cross-merge: sum 4 inverse-permuted y_k via SIMD ADD
// RMSNorm: 8-step SIMD chain (widen, RMS, div, sqrt, recip, mul_rms, mul_weight, narrow)

#include "helper.c"
#include "snax-simbacore-lib.h"

static inline uint16_t fp32_to_bf16(float f) {
    union {
        float f;
        uint32_t u;
    } x = {.f = f};
    return (uint16_t)(x.u >> 16);
}

int test_ss2d_tiled() {
    int err = 0;

    printf("VMamba SS2D tiled: H=%d, W=%d, seqLen=%d, dModel=%d, dInner=%d, K=%d, nb_tiles=%d\n", H, W, seqLen, dModel,
           dInner, K, nb_tiles);

    // ---- L3 staging buffers ----
    static uint8_t* ptr_conv_out_l3 = NULL;
    static uint8_t* ptr_z_l3        = NULL;
    static uint8_t* ptr_y_l3        = NULL;
    if (snrt_global_core_idx() == 0) {
        (void)snrt_l3alloc(16 * 1024);
        ptr_conv_out_l3 = (uint8_t*)snrt_l3alloc(M1_length_conv_out);
        ptr_z_l3        = (uint8_t*)snrt_l3alloc(M2_length_z);
        ptr_y_l3        = (uint8_t*)snrt_l3alloc(M2_length_y);
    }
    snrt_cluster_hw_barrier();

    // ---- FULL TCDM buffers ----
    void* tcdm_base_ptr = snrt_l1_next();

    uint8_t* ptr_oscore_in      = (uint8_t*)tcdm_base_ptr;
    uint8_t* ptr_iscore_out_P1  = ptr_oscore_in + M1_length_oscore_in;
    uint8_t* ptr_dt_in          = ptr_iscore_out_P1;
    uint8_t* ptr_BC             = ptr_dt_in + M2_dt_to_BC_offset;
    uint16_t* ptr_iscore_out_P2 = (uint16_t*)(ptr_iscore_out_P1 + M1_length_iscore_out);

#define _ALIGN64(p) ((uint8_t*)(((uintptr_t)(p) + 63u) & ~(uintptr_t)63u))

    uint8_t* pingpong_base_ptr = _ALIGN64((uint8_t*)ptr_iscore_out_P2 + M2_length_iscore_out);

    // ---- Phase 1 ping-pong slots ----
    uint8_t* ptr_oscore_weight_P1[2] = {
        pingpong_base_ptr,
        _ALIGN64(pingpong_base_ptr + M1_length_oscore_weight_tile),
    };
    uint8_t* ptr_conv_weight[2] = {
        _ALIGN64(ptr_oscore_weight_P1[1] + M1_length_oscore_weight_tile),
        _ALIGN64(_ALIGN64(ptr_oscore_weight_P1[1] + M1_length_oscore_weight_tile) + M1_length_conv_weight_tile),
    };
    uint8_t* ptr_conv_bias[2] = {
        _ALIGN64(ptr_conv_weight[1] + M1_length_conv_weight_tile),
        _ALIGN64(_ALIGN64(ptr_conv_weight[1] + M1_length_conv_weight_tile) + M1_length_conv_bias_tile),
    };
    uint8_t* ptr_iscore_weight_P1[2] = {
        _ALIGN64(ptr_conv_bias[1] + M1_length_conv_bias_tile),
        _ALIGN64(_ALIGN64(ptr_conv_bias[1] + M1_length_conv_bias_tile) + M1_length_iscore_weight_tile),
    };
    uint8_t* ptr_conv_out_tile[2] = {
        _ALIGN64(ptr_iscore_weight_P1[1] + M1_length_iscore_weight_tile),
        _ALIGN64(_ALIGN64(ptr_iscore_weight_P1[1] + M1_length_iscore_weight_tile) + M1_length_conv_out_tile),
    };

    // ---- Phase 2 ping-pong slots (overlay Phase 1's ping-pong region) ----
    uint8_t* ptr_oscore_weight_P2[2] = {
        pingpong_base_ptr,
        _ALIGN64(pingpong_base_ptr + M2_length_oscore_weight_tile),
    };
    uint8_t* ptr_dt_weight_1[2] = {
        _ALIGN64(ptr_oscore_weight_P2[1] + M2_length_oscore_weight_tile),
        _ALIGN64(_ALIGN64(ptr_oscore_weight_P2[1] + M2_length_oscore_weight_tile) + M2_length_dt_weight_1_tile),
    };
    uint8_t* ptr_dt_weight_2[2] = {
        _ALIGN64(ptr_dt_weight_1[1] + M2_length_dt_weight_1_tile),
        _ALIGN64(_ALIGN64(ptr_dt_weight_1[1] + M2_length_dt_weight_1_tile) + M2_length_dt_weight_2_tile),
    };
    uint8_t* ptr_dt_bias[2] = {
        _ALIGN64(ptr_dt_weight_2[1] + M2_length_dt_weight_2_tile),
        _ALIGN64(_ALIGN64(ptr_dt_weight_2[1] + M2_length_dt_weight_2_tile) + M2_length_dt_bias_tile),
    };
    uint8_t* ptr_A[2] = {
        _ALIGN64(ptr_dt_bias[1] + M2_length_dt_bias_tile),
        _ALIGN64(_ALIGN64(ptr_dt_bias[1] + M2_length_dt_bias_tile) + M2_length_A_tile),
    };
    uint8_t* ptr_D[2] = {
        _ALIGN64(ptr_A[1] + M2_length_A_tile),
        _ALIGN64(_ALIGN64(ptr_A[1] + M2_length_A_tile) + M2_length_D_tile),
    };
    uint8_t* ptr_iscore_weight_P2[2] = {
        _ALIGN64(ptr_D[1] + M2_length_D_tile),
        _ALIGN64(_ALIGN64(ptr_D[1] + M2_length_D_tile) + M2_length_iscore_weight_tile),
    };
    uint8_t* ptr_x_tile[2] = {
        _ALIGN64(ptr_iscore_weight_P2[1] + M2_length_iscore_weight_tile),
        _ALIGN64(_ALIGN64(ptr_iscore_weight_P2[1] + M2_length_iscore_weight_tile) + M2_length_x_tile),
    };
    uint8_t* ptr_z_tile[2] = {
        _ALIGN64(ptr_x_tile[1] + M2_length_x_tile),
        _ALIGN64(_ALIGN64(ptr_x_tile[1] + M2_length_x_tile) + M2_length_z_tile),
    };
    uint8_t* ptr_y_tile[2] = {
        _ALIGN64(ptr_z_tile[1] + M2_length_z_tile),
        _ALIGN64(_ALIGN64(ptr_z_tile[1] + M2_length_z_tile) + M2_length_y_tile),
    };
#undef _ALIGN64

    if (snrt_global_core_idx() == 0) init_cycle_counter();
    snrt_cluster_hw_barrier();

    uint32_t start_cycles = 0;
    uint32_t p1_cycles = 0, p2_cycles = 0;
    static uint32_t _p1_dma_done = 0, _p1_compute_done = 0;
    static uint32_t _p2_dma_done = 0, _p2_compute_done = 0;

    const uint32_t K_i = M1_dInner_tile / dInnerUnroll;

    if (snrt_global_core_idx() == 0) {
        printf("\nStarting VMamba SS2D tiled: K=%d dirs, nb_tiles=%d, K_i=%u\n\n", K, nb_tiles, K_i);
        start_cycles = snrt_mcycle();
    }

    // ==========================
    // Per-direction loop
    // ==========================
    for (uint32_t k = 0; k < K; k++) {
        // ---- Load per-direction FULL data ----
        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(ptr_oscore_in, M2_oscore_in_K + k * dir_size_oscore_in, M1_length_oscore_in);
            snrt_dma_start_1d(ptr_iscore_out_P1, M2_iscore_bias, M1_length_iscore_out);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();

        /////////////////////////////////
        //////// Phase 1 (tiled) ////////
        /////////////////////////////////

        if (snrt_global_core_idx() == 0) {
            set_streamer_phase1((uint32_t)ptr_oscore_in, (uint32_t)ptr_oscore_weight_P1[0],
                                (uint32_t)ptr_conv_weight[0], (uint32_t)ptr_conv_bias[0],
                                (uint32_t)ptr_iscore_weight_P1[0], (uint32_t)ptr_iscore_out_P1,
                                (uint32_t)ptr_conv_out_tile[0]);
            set_simbacore_csr(M28_PHASE1_NO_REQUANT, seqLen, dModel, M1_dInner_tile, dtRank, xProjDim);
        }
        snrt_cluster_hw_barrier();

        for (uint32_t i = 0; i < nb_tiles + 2; i++) {
            int buf = i % 2;

            if (i < nb_tiles && snrt_is_dm_core()) {
                snrt_dma_start_1d(ptr_oscore_weight_P1[buf], M2_oscore_weight + i * M1_length_oscore_weight_tile,
                                  M1_length_oscore_weight_tile);
                snrt_dma_start_1d(ptr_conv_weight[buf], M2_conv_weight + i * M1_length_conv_weight_tile,
                                  M1_length_conv_weight_tile);
                snrt_dma_start_1d(ptr_conv_bias[buf], M2_conv_bias + i * M1_length_conv_bias_tile,
                                  M1_length_conv_bias_tile);
                snrt_dma_start_1d(ptr_iscore_weight_P1[buf],
                                  M2_iscore_weight_K + k * dir_size_iscore_weight + i * M1_length_iscore_weight_tile,
                                  M1_length_iscore_weight_tile);
            }

            if (i >= 1 && i <= nb_tiles && snrt_global_core_idx() == 0) {
                uint32_t tile      = i - 1;
                bool is_final_tile = (tile == nb_tiles - 1);

                if (is_final_tile) write_csr(MODE, M1_PHASE1);
                _set_streamer_start();
                _set_simbacore_start();
                write_csr(STREAMER_START_CSR, 0);
                write_csr(SIMBACORE_START, 0);
                if (!is_final_tile) {
                    uint32_t next_tile = tile + 1;
                    int nbuf = next_tile % 2;
                    write_csr(BASE_PTR_READER_1_LOW, (uint32_t)ptr_oscore_weight_P1[nbuf]);
                    write_csr(BASE_PTR_READER_3_LOW, (uint32_t)ptr_conv_weight[nbuf]);
                    write_csr(BASE_PTR_READER_4_LOW, (uint32_t)ptr_conv_bias[nbuf]);
                    write_csr(BASE_PTR_READER_12_LOW, (uint32_t)ptr_iscore_weight_P1[nbuf]);
                    write_csr(BASE_PTR_WRITER_1_LOW, (uint32_t)ptr_conv_out_tile[nbuf]);
                }
                while (read_csr(SIMBACORE_BUSY));
                while (read_csr(STREAMER_BUSY_CSR));
                p1_cycles += read_simbacore_perf_counter();
                if (i == 1) _p1_compute_done = snrt_mcycle();
            }

            if (i >= 2 && snrt_is_dm_core()) {
                uint32_t spill_tile = i - 2;
                int sbuf            = spill_tile % 2;
                snrt_dma_start_1d(ptr_conv_out_l3 + spill_tile * M1_length_conv_out_tile, ptr_conv_out_tile[sbuf],
                                  M1_length_conv_out_tile);
            }

            if (snrt_is_dm_core()) {
                snrt_dma_wait_all();
                if (i == 1) _p1_dma_done = snrt_mcycle();
            }
            snrt_cluster_hw_barrier();
        }

        /////////////////////////////////
        //////// Phase 2 bias preload ///
        /////////////////////////////////

        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(ptr_iscore_out_P2, M2_iscore_bias_P2, M2_length_iscore_out);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();

        /////////////////////////////////
        //////// Phase 2 (tiled) ////////
        /////////////////////////////////

        if (snrt_global_core_idx() == 0) {
            set_streamer_phase2((uint32_t)ptr_oscore_in, (uint32_t)ptr_oscore_weight_P2[0], (uint32_t)ptr_z_tile[0],
                                (uint32_t)ptr_dt_in, (uint32_t)ptr_dt_weight_1[0], (uint32_t)ptr_dt_weight_2[0],
                                (uint32_t)ptr_dt_bias[0], (uint32_t)ptr_x_tile[0], (uint32_t)ptr_A[0], (uint32_t)ptr_BC,
                                (uint32_t)ptr_D[0], (uint32_t)ptr_y_tile[0], (uint32_t)ptr_iscore_weight_P2[0],
                                (uint32_t)ptr_iscore_out_P2);
            set_simbacore_csr(M29_PHASE2_NO_REQUANT, seqLen, dModel, M2_dInner_tile, dtRank, dModel);
        }
        snrt_cluster_hw_barrier();

        for (uint32_t i = 0; i < nb_tiles + 2; i++) {
            int buf = i % 2;

            if (i < nb_tiles && snrt_is_dm_core()) {
                snrt_dma_start_1d(ptr_oscore_weight_P2[buf], M2_oscore_weight_P2 + i * M2_length_oscore_weight_tile,
                                  M2_length_oscore_weight_tile);
                snrt_dma_start_1d(ptr_dt_weight_1[buf], M2_dt_weight_1 + i * M2_length_dt_weight_1_tile,
                                  M2_length_dt_weight_1_tile);
                snrt_dma_start_1d(ptr_dt_weight_2[buf], M2_dt_weight_2 + i * M2_length_dt_weight_2_tile,
                                  M2_length_dt_weight_2_tile);
                snrt_dma_start_1d(ptr_dt_bias[buf], M2_dt_bias + i * M2_length_dt_bias_tile, M2_length_dt_bias_tile);
                snrt_dma_start_1d(ptr_A[buf], M2_suc_A + i * M2_length_A_tile, M2_length_A_tile);
                snrt_dma_start_1d(ptr_D[buf], M2_suc_D + i * M2_length_D_tile, M2_length_D_tile);
                snrt_dma_start_1d(ptr_iscore_weight_P2[buf], M2_iscore_weight_P2 + i * M2_length_iscore_weight_tile,
                                  M2_length_iscore_weight_tile);
                snrt_dma_start_1d(ptr_x_tile[buf], ptr_conv_out_l3 + i * M2_length_x_tile, M2_length_x_tile);
            }

            if (i >= 1 && i <= nb_tiles && snrt_global_core_idx() == 0) {
                uint32_t tile      = i - 1;
                bool is_final_tile = (tile == nb_tiles - 1);

                if (is_final_tile) write_csr(MODE, M2_PHASE2);
                start_simbacore_and_streamers(M2_R10_en, M2_R10_start_cnt, M2_R11_en, M2_R11_start_cnt);
                write_csr(STREAMER_START_CSR, 0);
                write_csr(SIMBACORE_START, 0);
                write_csr(DELAYED_START_READER_10, 0);
                write_csr(DELAYED_START_READER_11, 0);

                if (!is_final_tile) {
                    uint32_t next_tile = tile + 1;
                    int nbuf           = next_tile % 2;
                    write_csr(BASE_PTR_READER_1_LOW, (uint32_t)ptr_oscore_weight_P2[nbuf]);
                    write_csr(BASE_PTR_READER_3_LOW, (uint32_t)ptr_dt_weight_1[nbuf]);
                    write_csr(BASE_PTR_READER_4_LOW, (uint32_t)ptr_dt_bias[nbuf]);
                    write_csr(BASE_PTR_READER_5_LOW, (uint32_t)ptr_dt_weight_2[nbuf]);
                    write_csr(BASE_PTR_READER_6_LOW, (uint32_t)ptr_A[nbuf]);
                    write_csr(BASE_PTR_READER_8_LOW, (uint32_t)ptr_D[nbuf]);
                    write_csr(BASE_PTR_READER_9_LOW, (uint32_t)ptr_x_tile[nbuf]);
                    write_csr(BASE_PTR_READER_10_LOW, (uint32_t)ptr_z_tile[nbuf]);
                    write_csr(BASE_PTR_READER_11_LOW, (uint32_t)ptr_y_tile[nbuf]);
                    write_csr(BASE_PTR_READER_12_LOW, (uint32_t)ptr_iscore_weight_P2[nbuf]);
                    write_csr(BASE_PTR_WRITER_0_LOW, (uint32_t)ptr_z_tile[nbuf]);
                    write_csr(BASE_PTR_WRITER_2_LOW, (uint32_t)ptr_y_tile[nbuf]);
                }
                while (read_csr(SIMBACORE_BUSY));
                while (read_csr(STREAMER_BUSY_CSR));
                p2_cycles += read_simbacore_perf_counter();
                if (i == 1) _p2_compute_done = snrt_mcycle();
            }

            if (i >= 2 && snrt_is_dm_core()) {
                uint32_t spill_tile = i - 2;
                int sbuf            = spill_tile % 2;
                snrt_dma_start_1d(ptr_z_l3 + spill_tile * M2_length_z_tile, ptr_z_tile[sbuf], M2_length_z_tile);
                snrt_dma_start_1d(ptr_y_l3 + spill_tile * M2_length_y_tile, ptr_y_tile[sbuf], M2_length_y_tile);
            }

            if (snrt_is_dm_core()) {
                snrt_dma_wait_all();
                if (i == 1) _p2_dma_done = snrt_mcycle();
            }
            snrt_cluster_hw_barrier();
        }

        // if (snrt_global_core_idx() == 0) printf("[%u cc] Dir %u done\n", snrt_mcycle(), k);
        snrt_cluster_hw_barrier();
    }

    // ---- Cross-merge: sum 4 inverse-permuted y_k via SIMD ADD ----
    // Reuse TCDM base for merge buffers (P1/P2 data no longer needed)
    uint8_t* ptr_merge_a = (uint8_t*)tcdm_base_ptr;
    uint8_t* ptr_merge_b = (uint8_t*)tcdm_base_ptr + dir_size_y_invperm;
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

    // if (snrt_global_core_idx() == 0) {
    //     err += check_result_sample(ptr_merge_a, M2_y_merged_flat, M2_test_samples_y_merged_flat, nb_test_samples,
    //                                "y_merged");
    //     printf("[%u cc] Cross-merge done (%u cc)\n", snrt_mcycle(), merge_cycles);
    // }
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
        // printf("[%u cc] RMSNorm done (%u cc)\n", snrt_mcycle(), rms_cycles);
    }
    snrt_cluster_hw_barrier();

    // --- Verification ---
    if (snrt_global_core_idx() == 0) {
        uint32_t end_cycles = snrt_mcycle();
        printf("[%u cc] P1: %u cc, P2: %u cc, merge: %u cc, RMSNorm: %u cc, Snitch: %u cc\n", end_cycles, p1_cycles,
               p2_cycles, merge_cycles, rms_cycles, end_cycles - start_cycles);
        printf("[%u cc] Simbacore elapsed time: %u cycles\n", end_cycles,
               p1_cycles + p2_cycles + merge_cycles + rms_cycles);
        printf("[%u cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);
        printf("DMA latency hiding: P1=%s, P2=%s\n", _p1_dma_done < _p1_compute_done ? "hidden" : "STALL",
               _p2_dma_done < _p2_compute_done ? "hidden" : "STALL");
        printf("Test VMamba SS2D tiled: H=%d, W=%d, dModel=%d, K=%d, nb_tiles=%d\n", H, W, dModel, K, nb_tiles);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, 2 * nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test_ss2d_tiled(); }
