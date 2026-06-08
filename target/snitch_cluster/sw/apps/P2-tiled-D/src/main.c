// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// P2-tiled-D = Phase 2 of the Mamba main in isolation, dInner-tiled and double-buffered.
// The Phase-1 outputs that Phase 2 consumes (oscore_in, dt+BC, x) are preloaded from the golden
// reference instead of being produced upstream. osCore + SU-core + IS-core all run, with on-chip
// forwarding (z -> SU-core, y -> IS-core). z and y are staged through L3; iscore_out_P2 (the final
// out-proj) stays FULL in TCDM and accumulates in place. Dataflow: docs/dataflow/04_mamba_main.md.

#include "helper.c"
#include "snax-simbacore-lib.h"

int test_phase2_tiled() {
    int err = 0;

    // L3 staging for z and y (reserve 16 KiB first to skip putc_buffer).
    static uint8_t* ptr_z_l3 = NULL;
    static uint8_t* ptr_y_l3 = NULL;
    if (snrt_global_core_idx() == 0) {
        (void)snrt_l3alloc(16 * 1024);
        ptr_z_l3 = (uint8_t*)snrt_l3alloc(M2_length_z);
        ptr_y_l3 = (uint8_t*)snrt_l3alloc(M2_length_y);
    }
    snrt_cluster_hw_barrier();

    void* tcdm_base_ptr = snrt_l1_next();

#define _ALIGN64(p) ((uint8_t*)(((uintptr_t)(p) + 63u) & ~(uintptr_t)63u))

    // FULL buffers (preloaded from golden): osCore input, dt+BC (switchCore input), IS-core psum.
    uint8_t* ptr_oscore_in      = (uint8_t*)tcdm_base_ptr;
    uint8_t* ptr_dt_in          = _ALIGN64(ptr_oscore_in + M2_length_oscore_in);
    uint8_t* ptr_BC             = ptr_dt_in + M2_dt_to_BC_offset;
    uint16_t* ptr_iscore_out_P2 = (uint16_t*)_ALIGN64(ptr_dt_in + M2_length_dt_BC);

    uint8_t* pingpong_base_ptr = _ALIGN64((uint8_t*)ptr_iscore_out_P2 + M2_length_iscore_out);

    // ---- Phase 2 ping-pong slots.
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

    const uint32_t K_i               = M2_dInner_tile / dInnerUnroll;
    uint32_t start_cycles            = 0;
    uint32_t simbacore_cycles_phase2 = 0;

    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: P2-tiled-D (L=%d, dModel=%d, nb_tiles=%d, K_i=%u, x from golden, z/y via L3)\n\n",
               seqLen, dModel, nb_tiles, K_i);
        printf("Expected L1 TCDM usage: %u B (%u KiB)\n", (uint32_t)L1_TCDM_PEAK_BYTES,
               (uint32_t)(L1_TCDM_PEAK_BYTES / 1024));
        init_cycle_counter();
        start_cycles = snrt_mcycle();
        set_streamer_phase2((uint32_t)ptr_oscore_in, (uint32_t)ptr_oscore_weight_P2[0], (uint32_t)ptr_z_tile[0],
                            (uint32_t)ptr_dt_in, (uint32_t)ptr_dt_weight_1[0], (uint32_t)ptr_dt_weight_2[0],
                            (uint32_t)ptr_dt_bias[0], (uint32_t)ptr_x_tile[0], (uint32_t)ptr_A[0], (uint32_t)ptr_BC,
                            (uint32_t)ptr_D[0], (uint32_t)ptr_y_tile[0], (uint32_t)ptr_iscore_weight_P2[0],
                            (uint32_t)ptr_iscore_out_P2);
        set_simbacore_csr(M29_PHASE2_NO_REQUANT, seqLen, dModel, M2_dInner_tile, dtRank, dModel);
    }

    // Preload non-tiled inputs from golden: osCore in, dt+BC, IS-core bias into the psum buffer.
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_oscore_in, M2_oscore_in, M2_length_oscore_in);
        snrt_dma_start_1d(ptr_dt_in, M2_dt_BC, M2_length_dt_BC);
        snrt_dma_start_1d(ptr_iscore_out_P2, M2_iscore_bias, M2_length_iscore_out);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    for (uint32_t i = 0; i < nb_tiles + 2; i++) {
        int buf = i % 2;

        // Stage I: load next tile's weights (ping-pong) + x tile from golden.
        if (i < nb_tiles && snrt_is_dm_core()) {
            snrt_dma_start_1d(ptr_oscore_weight_P2[buf], M2_oscore_weight + i * M2_length_oscore_weight_tile,
                              M2_length_oscore_weight_tile);
            snrt_dma_start_1d(ptr_dt_weight_1[buf], M2_dt_weight_1 + i * M2_length_dt_weight_1_tile,
                              M2_length_dt_weight_1_tile);
            snrt_dma_start_1d(ptr_dt_weight_2[buf], M2_dt_weight_2 + i * M2_length_dt_weight_2_tile,
                              M2_length_dt_weight_2_tile);
            snrt_dma_start_1d(ptr_dt_bias[buf], M2_dt_bias + i * M2_length_dt_bias_tile, M2_length_dt_bias_tile);
            snrt_dma_start_1d(ptr_A[buf], M2_suc_A + i * M2_length_A_tile, M2_length_A_tile);
            snrt_dma_start_1d(ptr_D[buf], M2_suc_D + i * M2_length_D_tile, M2_length_D_tile);
            snrt_dma_start_1d(ptr_iscore_weight_P2[buf], M2_iscore_weight + i * M2_length_iscore_weight_tile,
                              M2_length_iscore_weight_tile);
            snrt_dma_start_1d(ptr_x_tile[buf], M2_suc_x + i * M2_length_x_tile, M2_length_x_tile);
        }

        // Stage II: compute tile
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
            asm volatile("fence" ::: "memory");
            simbacore_cycles_phase2 += read_simbacore_perf_counter();
            // printf("[%u cc] P2 tile %u/%d done\n", snrt_mcycle(), tile + 1, nb_tiles);
        }

        // Stage III: spill this tile's z and y to L3.
        if (i >= 2 && snrt_is_dm_core()) {
            uint32_t spill_tile = i - 2;
            int sbuf            = spill_tile % 2;
            snrt_dma_start_1d(ptr_z_l3 + spill_tile * M2_length_z_tile, ptr_z_tile[sbuf], M2_length_z_tile);
            snrt_dma_start_1d(ptr_y_l3 + spill_tile * M2_length_y_tile, ptr_y_tile[sbuf], M2_length_y_tile);
        }

        if (snrt_is_dm_core()) snrt_dma_wait_all();
        snrt_cluster_hw_barrier();
    }

    // --- Verification ---
    if (snrt_global_core_idx() == 0) {
        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Simbacore elapsed time: %u cycles\n", end_cycles, simbacore_cycles_phase2);
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        err += check_result_sample(ptr_z_l3, M2_oscore_expected, M2_test_samples_z,  //
                                   nb_test_samples, "z (osCore out)");
        err += check_result_sample(ptr_y_l3, M2_suc_expected, M2_test_samples_y,  //
                                   nb_test_samples, "SUC y");
        err += check_result_sample((uint8_t*)ptr_iscore_out_P2, M2_iscore_expected,  //
                                   M2_test_samples_iscore_out, nb_test_samples, "iscore_out");

        printf("Test P2-tiled-D: seqLen=%d, dModel=%d, dInner=%d, nb_tiles=%d\n", seqLen, dModel, dInner, nb_tiles);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, 3 * nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test_phase2_tiled(); }
