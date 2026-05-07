// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// Tiled, double-buffered, pipelined version of the Mamba main program. Tiling is along the dInner dimension and is
// applied to both Phase 1 and Phase 2. The fused test runs the two phases sequentially:
//
//     for phase in {phase1, phase2}:
//         for tile in [0, nb_tiles):
//             pipeline(transfer_in tile, compute tile-1)
//
// Per-phase pipeline:
//   stage 1 : DMA L3 -> L1   (transfer_in  weight/input tile)
//   stage 2 : compute        (simbacore: produce a slice of conv_out / z / y / accumulate IS GeMM psums)
//
// Notes on what is and is not tiled:
//   - Phase 1: oscore_in is loaded once (FULL). iscore_out (psum) is FULL: bias is preloaded, the IS GeMM psums
//     accumulate in-place across tiles via R13/W3 hitting the same address. conv_out is FULL: each tile writes
//     its own slice via W1.
//   - Phase 2: oscore_in, dt_in (== iscore_out from Phase 1), x (== conv_out from Phase 1) are FULL. iscore_out
//     accumulates in-place. z and y are FULL with per-tile slots.
//   - The streamer bounds are configured ONCE (per phase) using the per-tile bound values from data.h. Inside the
//     pipelined loop only the address CSRs change.

#include "helper.c"
#include "snax-simbacore-lib.h"

int test_phase1_and_2() {
    int err = 0;

    // ---------- TCDM allocation: shared P1+P2 layout ---------------------------
    // FULL (shared / P1 / P2):
    //   oscore_in (shared) | iscore_out_P1 (== dt_in for P2) | conv_out (== x for P2)
    //   z (P2 W0 out) | y (P2 W2 out) | iscore_out_P2
    // PING-PONG (P1):
    //   oscore_weight_P1, conv_weight, conv_bias, iscore_weight_P1
    // PING-PONG (P2):
    //   oscore_weight_P2, dt_weight_1, dt_weight_2, dt_bias, A, D, iscore_weight_P2
    void* tcdm_base_ptr = snrt_l1_next();

    // ---- Full / shared buffers
    uint8_t* ptr_oscore_in      = (uint8_t*)tcdm_base_ptr;
    uint8_t* ptr_iscore_out_P1  = ptr_oscore_in + M1_length_oscore_in;       // P1 psum (BF16), reused as dt_in
    uint8_t* ptr_conv_out       = ptr_iscore_out_P1 + M1_length_iscore_out;  // P1 W1 out, reused as P2 x
    uint8_t* ptr_z              = ptr_conv_out + M1_length_conv_out;         // P2 W0 out
    uint8_t* ptr_y              = ptr_z + M2_length_z;                       // P2 W2 out
    uint16_t* ptr_iscore_out_P2 = (uint16_t*)(ptr_y + M2_length_y);          // P2 psum (BF16)

    uint8_t* ptr_dt_in = ptr_iscore_out_P1;
    uint8_t* ptr_BC    = ptr_dt_in + M2_dt_to_BC_offset;
    uint8_t* ptr_x     = ptr_conv_out;

    // ---- Ping-pong region for P1
    uint8_t* pingpong_base_ptr       = (uint8_t*)ptr_iscore_out_P2 + M2_length_iscore_out;
    uint8_t* ptr_oscore_weight_P1[2] = {
        pingpong_base_ptr,
        pingpong_base_ptr + M1_length_oscore_weight_tile,
    };
    uint8_t* ptr_conv_weight[2] = {
        ptr_oscore_weight_P1[1] + M1_length_oscore_weight_tile,
        ptr_oscore_weight_P1[1] + M1_length_oscore_weight_tile + M1_length_conv_weight_tile,
    };
    uint8_t* ptr_conv_bias[2] = {
        ptr_conv_weight[1] + M1_length_conv_weight_tile,
        ptr_conv_weight[1] + M1_length_conv_weight_tile + M1_length_conv_bias_tile,
    };
    uint8_t* ptr_iscore_weight_P1[2] = {
        ptr_conv_bias[1] + M1_length_conv_bias_tile,
        ptr_conv_bias[1] + M1_length_conv_bias_tile + M1_length_iscore_weight_tile,
    };

    // ---- Ping-pong region for P2
    // Reuse the Phase 1 scratch region: Phase 1 fully completes before Phase 2 starts.
    uint8_t* ptr_oscore_weight_P2[2] = {
        pingpong_base_ptr,
        pingpong_base_ptr + M2_length_oscore_weight_tile,
    };
    uint8_t* ptr_dt_weight_1[2] = {
        ptr_oscore_weight_P2[1] + M2_length_oscore_weight_tile,
        ptr_oscore_weight_P2[1] + M2_length_oscore_weight_tile + M2_length_dt_weight_1_tile,
    };
    uint8_t* ptr_dt_weight_2[2] = {
        ptr_dt_weight_1[1] + M2_length_dt_weight_1_tile,
        ptr_dt_weight_1[1] + M2_length_dt_weight_1_tile + M2_length_dt_weight_2_tile,
    };
    uint8_t* ptr_dt_bias[2] = {
        ptr_dt_weight_2[1] + M2_length_dt_weight_2_tile,
        ptr_dt_weight_2[1] + M2_length_dt_weight_2_tile + M2_length_dt_bias_tile,
    };
    uint8_t* ptr_A[2] = {
        ptr_dt_bias[1] + M2_length_dt_bias_tile,
        ptr_dt_bias[1] + M2_length_dt_bias_tile + M2_length_A_tile,
    };
    uint8_t* ptr_D[2] = {
        ptr_A[1] + M2_length_A_tile,
        ptr_A[1] + M2_length_A_tile + M2_length_D_tile,
    };
    uint8_t* ptr_iscore_weight_P2[2] = {
        ptr_D[1] + M2_length_D_tile,
        ptr_D[1] + M2_length_D_tile + M2_length_iscore_weight_tile,
    };

    if (snrt_global_core_idx() == 0) init_cycle_counter();
    snrt_cluster_hw_barrier();

    // ---------- Preload Phase 1 non-tiled inputs -------------------------------
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_oscore_in, M1_oscore_in, M1_length_oscore_in);
        snrt_dma_start_1d(ptr_iscore_out_P1, M1_iscore_bias, M1_length_iscore_out);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    uint32_t start_cycles            = 0;
    uint32_t simbacore_cycles_phase1 = 0;
    uint32_t simbacore_cycles_phase2 = 0;

    // =========================================================================
    // Phase 1
    // =========================================================================
    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: Mamba main tiled (Phase1 then Phase2, nb_tiles=%d)\n\n", nb_tiles);
        start_cycles = snrt_mcycle();
        set_streamer_phase1((uint32_t)ptr_oscore_in, (uint32_t)ptr_oscore_weight_P1[0],  //
                            (uint32_t)ptr_conv_weight[0], (uint32_t)ptr_conv_bias[0],    //
                            (uint32_t)ptr_iscore_weight_P1[0], (uint32_t)ptr_iscore_out_P1, (uint32_t)ptr_conv_out);
        set_simbacore_csr(M1_PHASE1, seqLen, dModel, M1_dInner_tile, dtRank, xProjDim);
    }
    snrt_cluster_hw_barrier();

    for (uint32_t i = 0; i < nb_tiles + 1; i++) {
        int buf = i % 2;

        if (i < nb_tiles && snrt_is_dm_core()) {
            snrt_dma_start_1d(ptr_oscore_weight_P1[buf], M1_oscore_weight + i * M1_length_oscore_weight_tile,
                              M1_length_oscore_weight_tile);
            snrt_dma_start_1d(ptr_conv_weight[buf], M1_conv_weight + i * M1_length_conv_weight_tile,
                              M1_length_conv_weight_tile);
            snrt_dma_start_1d(ptr_conv_bias[buf], M1_conv_bias + i * M1_length_conv_bias_tile,
                              M1_length_conv_bias_tile);
            snrt_dma_start_1d(ptr_iscore_weight_P1[buf], M1_iscore_weight + i * M1_length_iscore_weight_tile,
                              M1_length_iscore_weight_tile);
        }

        if (i >= 1 && snrt_global_core_idx() == 0) {
            uint32_t tile = i - 1;
            int cbuf      = tile % 2;
            write_csr(BASE_PTR_READER_1_LOW, (uint32_t)ptr_oscore_weight_P1[cbuf]);
            write_csr(BASE_PTR_READER_3_LOW, (uint32_t)ptr_conv_weight[cbuf]);
            write_csr(BASE_PTR_READER_4_LOW, (uint32_t)ptr_conv_bias[cbuf]);
            write_csr(BASE_PTR_READER_12_LOW, (uint32_t)ptr_iscore_weight_P1[cbuf]);
            write_csr(BASE_PTR_WRITER_1_LOW, (uint32_t)(ptr_conv_out + tile * M1_length_conv_out_tile));
            // Non-final tiles must skip IS GeMM requant+transpose so R13 of the next tile re-reads the
            // BF16 partial sums. The final tile uses the full PHASE1 mode to produce convFormat FP8 output.
            write_csr(MODE, (tile == nb_tiles - 1) ? M1_PHASE1 : M28_PHASE1_NO_REQUANT);
            start_simbacore_and_streamers(M1_R10_en, 0, M1_R11_en, 0);
            wait_simbacore_and_streamer();
            simbacore_cycles_phase1 += read_simbacore_perf_counter();
        }

        if (snrt_is_dm_core()) snrt_dma_wait_all();
        snrt_cluster_hw_barrier();
    }

    // =========================================================================
    // Preload Phase 2 non-tiled bias (psum init). Other Phase 2 untiled tensors
    // (oscore_in, dt_in == iscore_out_P1, x == conv_out) are already in TCDM.
    // =========================================================================
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_iscore_out_P2, M2_iscore_bias, M2_length_iscore_out);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    // =========================================================================
    // Phase 2
    // =========================================================================
    if (snrt_global_core_idx() == 0) {
        set_streamer_phase2((uint32_t)ptr_oscore_in, (uint32_t)ptr_oscore_weight_P2[0],                 //
                            (uint32_t)ptr_z, (uint32_t)ptr_dt_in,                                       //
                            (uint32_t)ptr_dt_weight_1[0], (uint32_t)ptr_dt_weight_2[0],                 //
                            (uint32_t)ptr_dt_bias[0],                                                   //
                            (uint32_t)ptr_x, (uint32_t)ptr_A[0], (uint32_t)ptr_BC, (uint32_t)ptr_D[0],  //
                            (uint32_t)ptr_y, (uint32_t)ptr_iscore_weight_P2[0], (uint32_t)ptr_iscore_out_P2);
        set_simbacore_csr(M2_PHASE2, seqLen, dModel, M2_dInner_tile, dtRank, dModel);
    }
    snrt_cluster_hw_barrier();

    for (uint32_t i = 0; i < nb_tiles + 1; i++) {
        int buf = i % 2;

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
        }

        if (i >= 1 && snrt_global_core_idx() == 0) {
            uint32_t tile       = i - 1;
            int cbuf            = tile % 2;
            uint8_t* x_tile_ptr = ptr_x + tile * M2_length_x_tile;
            uint8_t* z_tile_ptr = ptr_z + tile * M2_length_z_tile;
            uint8_t* y_tile_ptr = ptr_y + tile * M2_length_y_tile;

            write_csr(BASE_PTR_READER_1_LOW, (uint32_t)ptr_oscore_weight_P2[cbuf]);
            write_csr(BASE_PTR_READER_3_LOW, (uint32_t)ptr_dt_weight_1[cbuf]);
            write_csr(BASE_PTR_READER_4_LOW, (uint32_t)ptr_dt_bias[cbuf]);
            write_csr(BASE_PTR_READER_5_LOW, (uint32_t)ptr_dt_weight_2[cbuf]);
            write_csr(BASE_PTR_READER_6_LOW, (uint32_t)ptr_A[cbuf]);
            write_csr(BASE_PTR_READER_8_LOW, (uint32_t)ptr_D[cbuf]);
            write_csr(BASE_PTR_READER_9_LOW, (uint32_t)x_tile_ptr);
            write_csr(BASE_PTR_READER_10_LOW, (uint32_t)z_tile_ptr);
            write_csr(BASE_PTR_READER_11_LOW, (uint32_t)y_tile_ptr);
            write_csr(BASE_PTR_READER_12_LOW, (uint32_t)ptr_iscore_weight_P2[cbuf]);
            write_csr(BASE_PTR_WRITER_0_LOW, (uint32_t)z_tile_ptr);
            write_csr(BASE_PTR_WRITER_2_LOW, (uint32_t)y_tile_ptr);

            // Non-final tiles skip IS GeMM requant; final tile produces requantized FP8 output.
            write_csr(MODE, (tile == nb_tiles - 1) ? M2_PHASE2 : M29_PHASE2_NO_REQUANT);
            start_simbacore_and_streamers(M2_R10_en, M2_R10_start_cnt, M2_R11_en, M2_R11_start_cnt);
            wait_simbacore_and_streamer();
            simbacore_cycles_phase2 += read_simbacore_perf_counter();
        }

        if (snrt_is_dm_core()) snrt_dma_wait_all();
        snrt_cluster_hw_barrier();
    }

    // ---------- Verify ---------------------------------------------------------
    if (snrt_global_core_idx() == 0) {
        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Simbacore Phase1 (sum over tiles): %u cycles\n", end_cycles, simbacore_cycles_phase1);
        printf("[%d cc] Simbacore Phase2 (sum over tiles): %u cycles\n", end_cycles, simbacore_cycles_phase2);
        printf("[%d cc] Simbacore total elapsed time: %u cycles\n", end_cycles,
               simbacore_cycles_phase1 + simbacore_cycles_phase2);
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        err += check_result_sample(ptr_z, M2_oscore_expected, M2_test_samples_z,  //
                                   nb_test_samples, "z (osCore out)");
        err += check_result_sample(ptr_y, M2_suc_expected, M2_test_samples_y,  //
                                   nb_test_samples, "SUC y");
        err += check_result_sample((uint8_t*)ptr_iscore_out_P2, M2_iscore_expected,  //
                                   M2_test_samples_iscore_out, nb_test_samples, "iscore_out");

        printf("Test Phase1+Phase2 tiled: seqLen=%d, dModel=%d, dInner=%d, nb_tiles=%d\n",  //
               seqLen, dModel, dInner, nb_tiles);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, 5 * nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() {
    int err = test_phase1_and_2();
    return err;
}
