// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// SUC only, tiled over dInner.
#include "helper.c"
#include "snax-simbacore-lib.h"

static inline void set_streamer_suc_tile(uint32_t ptr_dt_in, uint32_t ptr_dt_weight_1, uint32_t ptr_dt_bias,
                                         uint32_t ptr_dt_weight_2, uint32_t ptr_A, uint32_t ptr_BC, uint32_t ptr_D,
                                         uint32_t ptr_x, uint32_t ptr_z, uint32_t ptr_y) {
    set_streamer_csr(

        (uint32_t)0, 0, 0, 0, 0,                                            // osCore in (disabled)
        (uint32_t)0, 0, 0, 0, 0,                                            // osCore weight (disabled)
        (uint32_t)ptr_dt_in, M2_R2_ss, M2_R2_tb, M2_R2_ts, M2_R2_en,        // switchCore in (dt)
        (uint32_t)ptr_dt_weight_1, M2_R3_ss, M2_R3_tb, M2_R3_ts, M2_R3_en,  // switchCore weight 1
        (uint32_t)ptr_dt_bias, M2_R4_ss, M2_R4_tb, M2_R4_ts, M2_R4_en,      // switchCore bias
        (uint32_t)ptr_dt_weight_2, M2_R5_ss, M2_R5_tb, M2_R5_ts, M2_R5_en,  // switchCore weight 2
        (uint32_t)ptr_A, M2_R6_ss, M2_R6_tb, M2_R6_ts, M2_R6_en,            // SUC A
        (uint32_t)ptr_BC, M2_R7_ss, M2_R7_tb, M2_R7_ts, M2_R7_en,           // SUC BC (correct stride)
        (uint32_t)ptr_D, M2_R8_ss, M2_R8_tb, M2_R8_ts, M2_R8_en,            // SUC D
        (uint32_t)ptr_x, M2_R9_ss, M2_R9_tb, M2_R9_ts, M2_R9_en,            // SUC x
        (uint32_t)ptr_z, M2_R10_ss, M2_R10_tb, M2_R10_ts, M2_R10_en,        // SUC z (= osCore out, preloaded)
        (uint32_t)ptr_y, M2_R11_ss, M2_R11_tb, M2_R11_ts, M2_R11_en,        // isCore in = SUC y (unused, isCore off)
        (uint32_t)0, 0, 0, 0, 0,                                            // isCore weight (disabled)
        (uint32_t)0, 0, 0, 0, 0,                                            // isCore psum (disabled)

        (uint32_t)0, 0, 0, 0, 0,                                  // osCore out z (disabled)
        (uint32_t)0, 0, 0, 0, M2_W1_en,                           // disabled
        (uint32_t)ptr_y, M2_W2_ss, M2_W2_tb, M2_W2_ts, M2_W2_en,  // SUC y out
        (uint32_t)0, 0, 0, 0, 0                                   // isCore out (disabled)
    );
}

int test_suc_tiled() {
    int err = 0;

    // L3 staging for y (reserve 16 KiB first to skip putc_buffer).
    static uint8_t* ptr_y_l3 = NULL;
    if (snrt_global_core_idx() == 0) {
        (void)snrt_l3alloc(16 * 1024);
        ptr_y_l3 = (uint8_t*)snrt_l3alloc(M2_length_y);
    }
    snrt_cluster_hw_barrier();

    void* tcdm_base_ptr = snrt_l1_next();

#define _ALIGN64(p) ((uint8_t*)(((uintptr_t)(p) + 63u) & ~(uintptr_t)63u))

    // FULL switchCore input dt+BC (loaded once from golden).
    uint8_t* ptr_dt_in = (uint8_t*)tcdm_base_ptr;
    uint8_t* ptr_BC    = ptr_dt_in + M2_dt_to_BC_offset;

    uint8_t* pingpong_base_ptr = _ALIGN64(ptr_dt_in + M2_length_dt_BC);

    // ---- Per-dInner-tile ping-pong slots.
    uint8_t* ptr_dt_weight_1[2] = {
        pingpong_base_ptr,
        _ALIGN64(pingpong_base_ptr + M2_length_dt_weight_1_tile),
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
    uint8_t* ptr_x_tile[2] = {
        _ALIGN64(ptr_D[1] + M2_length_D_tile),
        _ALIGN64(_ALIGN64(ptr_D[1] + M2_length_D_tile) + M2_length_x_tile),
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

    const uint32_t K_i            = M2_dInner_tile / dInnerUnroll;
    uint32_t start_cycles         = 0;
    uint32_t simbacore_cycles_suc = 0;

    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: SUC-tiled (L=%d, dModel=%d, dInner=%d, nb_tiles=%d, K_i=%u, z from golden)\n\n",
               seqLen, dModel, dInner, nb_tiles, K_i);
        printf("Expected L1 TCDM usage: %u B (%u KiB)\n", (uint32_t)L1_TCDM_PEAK_BYTES,
               (uint32_t)(L1_TCDM_PEAK_BYTES / 1024));
        init_cycle_counter();
        start_cycles = snrt_mcycle();
        set_streamer_suc_tile((uint32_t)ptr_dt_in, (uint32_t)ptr_dt_weight_1[0], (uint32_t)ptr_dt_bias[0],
                              (uint32_t)ptr_dt_weight_2[0], (uint32_t)ptr_A[0], (uint32_t)ptr_BC, (uint32_t)ptr_D[0],
                              (uint32_t)ptr_x_tile[0], (uint32_t)ptr_z_tile[0], (uint32_t)ptr_y_tile[0]);
        set_simbacore_csr(M27_SUC_ONLY, seqLen, dModel, M2_dInner_tile, dtRank, dModel);
    }

    // Preload FULL dt+BC from golden.
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_dt_in, M2_dt_BC, M2_length_dt_BC);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    for (uint32_t i = 0; i < nb_tiles + 2; i++) {
        int buf = i % 2;

        // Stage I: load this tile's switchCore/SU-core inputs (incl. preloaded z) from golden.
        if (i < nb_tiles && snrt_is_dm_core()) {
            snrt_dma_start_1d(ptr_dt_weight_1[buf], M2_dt_weight_1 + i * M2_length_dt_weight_1_tile,
                              M2_length_dt_weight_1_tile);
            snrt_dma_start_1d(ptr_dt_weight_2[buf], M2_dt_weight_2 + i * M2_length_dt_weight_2_tile,
                              M2_length_dt_weight_2_tile);
            snrt_dma_start_1d(ptr_dt_bias[buf], M2_dt_bias + i * M2_length_dt_bias_tile, M2_length_dt_bias_tile);
            snrt_dma_start_1d(ptr_A[buf], M2_suc_A + i * M2_length_A_tile, M2_length_A_tile);
            snrt_dma_start_1d(ptr_D[buf], M2_suc_D + i * M2_length_D_tile, M2_length_D_tile);
            snrt_dma_start_1d(ptr_x_tile[buf], M2_suc_x + i * M2_length_x_tile, M2_length_x_tile);
            snrt_dma_start_1d(ptr_z_tile[buf], M2_oscore_expected + i * M2_length_z_tile, M2_length_z_tile);
        }

        // Stage II: compute tile
        if (i >= 1 && i <= nb_tiles && snrt_global_core_idx() == 0) {
            uint32_t tile = i - 1;

            // z is fully preloaded (no osCore producer), so no safe-to-start delay is needed.
            start_simbacore_and_streamers(M2_R10_en, 0, 0, 0);
            write_csr(STREAMER_START_CSR, 0);
            write_csr(SIMBACORE_START, 0);
            write_csr(DELAYED_START_READER_10, 0);

            if (tile != nb_tiles - 1) {
                int nbuf = (tile + 1) % 2;
                write_csr(BASE_PTR_READER_3_LOW, (uint32_t)ptr_dt_weight_1[nbuf]);
                write_csr(BASE_PTR_READER_4_LOW, (uint32_t)ptr_dt_bias[nbuf]);
                write_csr(BASE_PTR_READER_5_LOW, (uint32_t)ptr_dt_weight_2[nbuf]);
                write_csr(BASE_PTR_READER_6_LOW, (uint32_t)ptr_A[nbuf]);
                write_csr(BASE_PTR_READER_8_LOW, (uint32_t)ptr_D[nbuf]);
                write_csr(BASE_PTR_READER_9_LOW, (uint32_t)ptr_x_tile[nbuf]);
                write_csr(BASE_PTR_READER_10_LOW, (uint32_t)ptr_z_tile[nbuf]);
                write_csr(BASE_PTR_READER_11_LOW, (uint32_t)ptr_y_tile[nbuf]);
                write_csr(BASE_PTR_WRITER_2_LOW, (uint32_t)ptr_y_tile[nbuf]);
            }
            while (read_csr(SIMBACORE_BUSY));
            while (read_csr(STREAMER_BUSY_CSR));
            asm volatile("fence" ::: "memory");
            simbacore_cycles_suc += read_simbacore_perf_counter();
        }

        // Stage III: spill this tile's y to L3.
        if (i >= 2 && snrt_is_dm_core()) {
            uint32_t spill_tile = i - 2;
            int sbuf            = spill_tile % 2;
            snrt_dma_start_1d(ptr_y_l3 + spill_tile * M2_length_y_tile, ptr_y_tile[sbuf], M2_length_y_tile);
        }

        if (snrt_is_dm_core()) snrt_dma_wait_all();
        snrt_cluster_hw_barrier();
    }

    // --- Verification ---
    if (snrt_global_core_idx() == 0) {
        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Simbacore elapsed time: %u cycles\n", end_cycles, simbacore_cycles_suc);
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        err += check_result_sample(ptr_y_l3, M2_suc_expected, M2_test_samples_y,  //
                                   nb_test_samples, "SUC y (from L3)");

        printf("Test SUC-tiled: seqLen=%d, dModel=%d, dInner=%d, nb_tiles=%d\n", seqLen, dModel, dInner, nb_tiles);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test_suc_tiled(); }
