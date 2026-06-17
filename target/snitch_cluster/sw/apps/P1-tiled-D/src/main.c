// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// P1-tiled-D = Phase 1 of the Mamba main in isolation, dInner-tiled and double-buffered.
// conv_out (= x for P2) is staged through L3; iscore_out_P1 (= dt+BC) stays FULL in TCDM and
// accumulates the IS-core psum in place (NO_REQUANT tiles, final tile requants+transposes).
// Dataflow: docs/dataflow/04_mamba_main.md (Phase 1 of main-tiled). Reuses main-tiled's data.

#include "helper.c"
#include "snax-simbacore-lib.h"

int test_phase1_tiled() {
    int err = 0;

    // L3 staging for conv_out. Reserve 16 KiB at the L3 base to skip putc_buffer.
    static uint8_t* ptr_conv_out_l3 = NULL;
    if (snrt_global_core_idx() == 0) {
        (void)snrt_l3alloc(16 * 1024);
        ptr_conv_out_l3 = (uint8_t*)snrt_l3alloc(M1_length_conv_out);
    }
    snrt_cluster_hw_barrier();

    void* tcdm_base_ptr = snrt_l1_next();

    uint8_t* ptr_oscore_in     = (uint8_t*)tcdm_base_ptr;
    uint8_t* ptr_iscore_out_P1 = ptr_oscore_in + M1_length_oscore_in;  // holds the psums

#define _ALIGN64(p) ((uint8_t*)(((uintptr_t)(p) + 63u) & ~(uintptr_t)63u))
    uint8_t* pingpong_base_ptr = _ALIGN64(ptr_iscore_out_P1 + M1_length_iscore_out);

    // ---- Phase 1 ping-pong slots. Last entry (conv_out_tile) is the W1 destination.
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
#undef _ALIGN64

    const uint32_t K_i               = M1_dInner_tile / dInnerUnroll;  // K-steps per DMA tile
    uint32_t start_cycles            = 0;
    uint32_t simbacore_cycles_phase1 = 0;

    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: P1-tiled-D (L=%d, dModel=%d, nb_tiles=%d, K_i=%u, conv_out via L3)\n\n", seqLen,
               dModel, nb_tiles, K_i);
        printf("Expected L1 TCDM usage: %u B (%u KiB)\n", (uint32_t)L1_TCDM_PEAK_BYTES,
               (uint32_t)(L1_TCDM_PEAK_BYTES / 1024));
        init_cycle_counter();
        start_cycles = snrt_mcycle();
        set_streamer_phase1((uint32_t)ptr_oscore_in, (uint32_t)ptr_oscore_weight_P1[0], (uint32_t)ptr_conv_weight[0],
                            (uint32_t)ptr_conv_bias[0], (uint32_t)ptr_iscore_weight_P1[0], (uint32_t)ptr_iscore_out_P1,
                            (uint32_t)ptr_conv_out_tile[0]);
        set_simbacore_csr(M28_PHASE1_NO_REQUANT, seqLen, dModel, M1_dInner_tile, dtRank, xProjDim);
    }

    // Preload non-tiled inputs: oscore_in (full); iscore bias into the psum buffer (where R13 reads).
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_oscore_in, M1_oscore_in, M1_length_oscore_in);
        snrt_dma_start_1d(ptr_iscore_out_P1, M1_iscore_bias, M1_length_iscore_out);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    for (uint32_t i = 0; i < nb_tiles + 2; i++) {
        int buf = i % 2;

        // Stage I: load next tile's data
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

        // Stage II: compute tile
        if (i >= 1 && i <= nb_tiles && snrt_global_core_idx() == 0) {
            uint32_t tile      = i - 1;
            bool is_final_tile = (tile == nb_tiles - 1);

            // Final tile flips MODE so HW's isCoreOutIsFinal applies requant + bank-transpose.
            if (is_final_tile) write_csr(MODE, M1_PHASE1);
            _set_streamer_start();
            _set_simbacore_start();
            write_csr(STREAMER_START_CSR, 0);
            write_csr(SIMBACORE_START, 0);
            if (!is_final_tile) {
                uint32_t next_tile = tile + 1;
                int nbuf           = next_tile % 2;
                write_csr(BASE_PTR_READER_1_LOW, (uint32_t)ptr_oscore_weight_P1[nbuf]);
                write_csr(BASE_PTR_READER_3_LOW, (uint32_t)ptr_conv_weight[nbuf]);
                write_csr(BASE_PTR_READER_4_LOW, (uint32_t)ptr_conv_bias[nbuf]);
                write_csr(BASE_PTR_READER_12_LOW, (uint32_t)ptr_iscore_weight_P1[nbuf]);
                write_csr(BASE_PTR_WRITER_1_LOW, (uint32_t)ptr_conv_out_tile[nbuf]);
            }
            while (read_csr(SIMBACORE_BUSY));
            while (read_csr(STREAMER_BUSY_CSR));
            simbacore_cycles_phase1 += read_simbacore_perf_counter();
        }

        // Stage III: spill this tile's conv_out to L3 (3-stage pipeline: load / compute / spill)
        if (i >= 2 && snrt_is_dm_core()) {
            uint32_t spill_tile = i - 2;
            int sbuf            = spill_tile % 2;
            snrt_dma_start_1d(ptr_conv_out_l3 + spill_tile * M1_length_conv_out_tile, ptr_conv_out_tile[sbuf],
                              M1_length_conv_out_tile);
        }

        if (snrt_is_dm_core()) snrt_dma_wait_all();
        snrt_cluster_hw_barrier();
    }

    // --- Verification ---
    if (snrt_global_core_idx() == 0) {
        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Simbacore elapsed time: %u cycles\n", end_cycles, simbacore_cycles_phase1);
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        err += check_result_sample(ptr_conv_out_l3, M1_conv_out, M1_test_samples_conv_out,  //
                                   nb_test_samples, "P1 conv_out (from L3)");
        err += check_result_sample(ptr_iscore_out_P1, M1_iscore_out, M1_test_samples_iscore_out,  //
                                   nb_test_samples, "P1 iscore_out");

        printf("Test P1-tiled-D: seqLen=%d, dModel=%d, dInner=%d, nb_tiles=%d\n", seqLen, dModel, dInner, nb_tiles);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, 2 * nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test_phase1_tiled(); }
