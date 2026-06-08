// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// main-tiled + async L-tiling of the os-core input
// iscore_out_P1/P2 stay full in TCDM

#include "helper.c"
#include "snax-simbacore-lib.h"

#define set_streamer_phase1_lTile(p_oi, p_ow, p_cw, p_cb, p_iw, p_io, p_co)                                           \
    set_streamer_csr((uint32_t)(p_oi), M1_R0_ss, M1_R0_tb_lTile, M1_R0_ts, M1_R0_en, (uint32_t)(p_ow), M1_R1_ss,      \
                     M1_R1_tb, M1_R1_ts, M1_R1_en, (uint32_t)0, 0, 0, 0, M1_R2_en, (uint32_t)(p_cw), M1_R3_ss,        \
                     M1_R3_tb, M1_R3_ts, M1_R3_en, (uint32_t)(p_cb), M1_R4_ss, M1_R4_tb, M1_R4_ts, M1_R4_en,          \
                     (uint32_t)0, 0, 0, 0, M1_R5_en, (uint32_t)0, 0, 0, 0, M1_R6_en, (uint32_t)0, 0, 0, 0, M1_R7_en,  \
                     (uint32_t)0, 0, 0, 0, M1_R8_en, (uint32_t)0, 0, 0, 0, M1_R9_en, (uint32_t)0, 0, 0, 0, M1_R10_en, \
                     (uint32_t)0, 0, 0, 0, M1_R11_en, (uint32_t)(p_iw), M1_R12_ss, M1_R12_tb, M1_R12_ts, M1_R12_en,   \
                     (uint32_t)(p_io), M1_R13_ss, M1_R13_tb, M1_R13_ts, M1_R13_en, (uint32_t)0, 0, 0, 0, M1_W0_en,    \
                     (uint32_t)(p_co), M1_W1_ss, M1_W1_tb, M1_W1_ts, M1_W1_en, (uint32_t)0, 0, 0, 0, M1_W2_en,        \
                     (uint32_t)(p_io), M1_W3_ss, M1_W3_tb, M1_W3_ts, M1_W3_en)

#define set_streamer_phase2_lTile(p_oi, p_ow, p_z, p_dt, p_dw1, p_dw2, p_db, p_x, p_A, p_BC, p_D, p_y, p_iw, p_io)     \
    set_streamer_csr((uint32_t)(p_oi), M2_R0_ss, M2_R0_tb_lTile, M2_R0_ts, M2_R0_en, (uint32_t)(p_ow), M2_R1_ss,       \
                     M2_R1_tb, M2_R1_ts, M2_R1_en, (uint32_t)(p_dt), M2_R2_ss, M2_R2_tb, M2_R2_ts, M2_R2_en,           \
                     (uint32_t)(p_dw1), M2_R3_ss, M2_R3_tb, M2_R3_ts, M2_R3_en, (uint32_t)(p_db), M2_R4_ss, M2_R4_tb,  \
                     M2_R4_ts, M2_R4_en, (uint32_t)(p_dw2), M2_R5_ss, M2_R5_tb, M2_R5_ts, M2_R5_en, (uint32_t)(p_A),   \
                     M2_R6_ss, M2_R6_tb, M2_R6_ts, M2_R6_en, (uint32_t)(p_BC), M2_R7_ss, M2_R7_tb, M2_R7_ts, M2_R7_en, \
                     (uint32_t)(p_D), M2_R8_ss, M2_R8_tb, M2_R8_ts, M2_R8_en, (uint32_t)(p_x), M2_R9_ss, M2_R9_tb,     \
                     M2_R9_ts, M2_R9_en, (uint32_t)(p_z), M2_R10_ss, M2_R10_tb, M2_R10_ts, M2_R10_en, (uint32_t)(p_y), \
                     M2_R11_ss, M2_R11_tb, M2_R11_ts, M2_R11_en, (uint32_t)(p_iw), M2_R12_ss, M2_R12_tb, M2_R12_ts,    \
                     M2_R12_en, (uint32_t)(p_io), M2_R13_ss, M2_R13_tb, M2_R13_ts, M2_R13_en, (uint32_t)(p_z),         \
                     M2_W0_ss, M2_W0_tb, M2_W0_ts, M2_W0_en, (uint32_t)0, 0, 0, 0, M2_W1_en, (uint32_t)(p_y),          \
                     M2_W2_ss, M2_W2_tb, M2_W2_ts, M2_W2_en, (uint32_t)(p_io), M2_W3_ss, M2_W3_tb, M2_W3_ts, M2_W3_en)

// Async oscore_in ring refill from L3, paced by R10 (osCore output-tile gauge).
// The SUC delayed readers are released from inside this loop the instant their own gauge crosses the threshold.
// A per-reader fallback handles the case where the threshold sits at/after the loop end.
static inline void oscore_in_refill_loop(uint8_t* slot_base, const uint8_t* l3_oscore_in,  //
                                         uint32_t r10_release_en, uint32_t r10_start_cnt,  //
                                         uint32_t r11_release_en, uint32_t r11_start_cnt) {
    const uint32_t N_visits   = (M1_dInner_tile / dInnerUnroll) * nb_l_tiles;
    const uint32_t gauge_step = M1_oscore_in_l_tile_gauge_step;
    const uint32_t len_l_tile = M1_length_oscore_in_l_tile;
    uint32_t r10_released     = 0;
    uint32_t r11_released     = 0;

    for (uint32_t r = 0; r < N_visits; r++) {
        // Snitch0 polls until L-tile r is consumed and its slot can be refilled.
        if (snrt_global_core_idx() == 0) {
            while (read_csr(R10_DELAY_GAUGE) < (r + 1) * gauge_step);

            // Release each delayed SUC reader
            // R10 has already been polled above, so reuse the loop index as its gauge value.
            if (r10_release_en && !r10_released && (r + 1) * gauge_step >= r10_start_cnt) {
                write_csr(DELAYED_START_READER_10, 1);
                r10_released = 1;
            }
            if (r11_release_en && !r11_released && read_csr(R11_DELAY_GAUGE) >= r11_start_cnt) {
                write_csr(DELAYED_START_READER_11, 1);
                r11_released = 1;
            }
        }

        // Make sure Snitch1 waits for the handover
        snrt_cluster_hw_barrier();

        // DMA refills slot (r % nb_slots) with the next L-tile for that slot
        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(slot_base + (r % nb_slots) * len_l_tile,
                              l3_oscore_in + ((r + nb_slots) % nb_l_tiles) * len_l_tile, len_l_tile);
        }
    }

    if (snrt_is_dm_core()) snrt_dma_wait_all();
    snrt_cluster_hw_barrier();

    // Fallback: a threshold sits at/after the loop end -> release here.
    if (snrt_global_core_idx() == 0) {
        if (r10_release_en && !r10_released) {
            printf("Fallback: delayed R10 threshold sits after refill loop end.\n");
            while (read_csr(R10_DELAY_GAUGE) < r10_start_cnt);
            write_csr(DELAYED_START_READER_10, 1);
        }
        if (r11_release_en && !r11_released) {
            printf("Fallback: delayed R11 threshold sits after refill loop end.\n");
            while (read_csr(R11_DELAY_GAUGE) < r11_start_cnt);
            write_csr(DELAYED_START_READER_11, 1);
        }
    }
}

int test_phase1_and_2() {
    int err = 0;

    // ---- L3 staging buffers. Reserve 16 KiB at the L3 base to skip putc_buffer.
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

    // ---- TCDM buffers.
    void* tcdm_base_ptr = snrt_l1_next();

    // B1: oscore_in is nb_slots ADJACENT ring slots of L_tile*dModel each (the R0 stride-0
    // wrap walks the slots contiguously, so they must abut with no gap). Slot s = base + s*len.
    uint8_t* ptr_oscore_in_base   = (uint8_t*)tcdm_base_ptr;
    uint32_t oscore_in_tcdm_bytes = nb_slots * M1_length_oscore_in_l_tile;

#define _ALIGN64(p) ((uint8_t*)(((uintptr_t)(p) + 63u) & ~(uintptr_t)63u))

    uint8_t* ptr_iscore_out_P1  = _ALIGN64(ptr_oscore_in_base + oscore_in_tcdm_bytes);
    uint8_t* ptr_dt_in          = ptr_iscore_out_P1;
    uint8_t* ptr_BC             = ptr_dt_in + M2_dt_to_BC_offset;
    uint16_t* ptr_iscore_out_P2 = (uint16_t*)(ptr_iscore_out_P1 + M1_length_iscore_out);

    uint8_t* pingpong_base_ptr = _ALIGN64((uint8_t*)ptr_iscore_out_P2 + M2_length_iscore_out);

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

    // ---- Phase 2 ping-pong slots. Overlays Phase 1's ping-pong region.
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

    // K-steps per DMA tile (P1, P2 share). N_visits = K_i * nb_l_tiles and gauge_step are
    // recomputed inside oscore_in_refill_loop from the same globals.
    const uint32_t K_i = M1_dInner_tile / dInnerUnroll;

    uint32_t start_cycles            = 0;
    uint32_t simbacore_cycles_phase1 = 0;
    uint32_t simbacore_cycles_phase2 = 0;

    if (snrt_global_core_idx() == 0) {
        printf(
            "\nStarting program: Mamba main tiled oscore (L=%d, dModel=%d, nb_tiles=%d, nb_l_tiles=%d, L_tile=%u, "
            "K_i=%u, oscore_in L-tiled async)\n\n",
            seqLen, dModel, nb_tiles, nb_l_tiles, L_tile, K_i);
        printf("Expected L1 TCDM usage: %u B (%u KiB)\n", (uint32_t)L1_TCDM_PEAK_BYTES,
               (uint32_t)(L1_TCDM_PEAK_BYTES / 1024));
        init_cycle_counter();
        start_cycles = snrt_mcycle();
        set_streamer_phase1_lTile((uint32_t)ptr_oscore_in_base, (uint32_t)ptr_oscore_weight_P1[0],
                                  (uint32_t)ptr_conv_weight[0], (uint32_t)ptr_conv_bias[0],
                                  (uint32_t)ptr_iscore_weight_P1[0], (uint32_t)ptr_iscore_out_P1,
                                  (uint32_t)ptr_conv_out_tile[0]);
        set_simbacore_csr(M28_PHASE1_NO_REQUANT, seqLen, dModel, M1_dInner_tile, dtRank, xProjDim);
    }

    if (snrt_is_dm_core()) {
        // Preload the first nb_slots L-tiles into the nb_slots ring slots
        for (uint32_t s = 0; s < nb_slots; s++)
            snrt_dma_start_1d(ptr_oscore_in_base + s * M1_length_oscore_in_l_tile,
                              M1_oscore_in + s * M1_oscore_in_l_offset, M1_length_oscore_in_l_tile);
        // Bias in (full) psum buffer
        snrt_dma_start_1d(ptr_iscore_out_P1, M1_iscore_bias, M1_length_iscore_out);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    /////////////////////////////////
    //////// Phase 1 ////////////////
    /////////////////////////////////

    // dInner loop
    for (uint32_t i = 0; i < nb_tiles + 2; i++) {
        int buf = i % 2;

        // EARLIEST-REFILL TEST: weight prefetch moved AFTER compute (below) so the refill owns the
        // DMA engine during compute and issues as early as physically possible (pace=5).

        // Compute tile (i-1). BOTH cores enter so they can run the refill loop together
        // (core 0 polls the gauge, DM core does the refill DMA).
        if (i >= 1 && i <= nb_tiles) {
            uint32_t tile      = i - 1;
            bool is_final_tile = (tile == nb_tiles - 1);

            if (snrt_global_core_idx() == 0) {
                if (is_final_tile) write_csr(MODE, M1_PHASE1);
                // printf("Starting simbacore and streamers for tile %d\n", tile);

                _set_streamer_start();
                _set_simbacore_start();
                // TODO why do we do this?
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
            }

            // Async oscore_in refill while the kernel computes. Must run immediately after simbacore start signal.
            // P1 has no delayed SUC readers (M1_R10_en == M1_R11_en == 0), so no mid-loop release.
            oscore_in_refill_loop(ptr_oscore_in_base, M1_oscore_in, M1_R10_en, 0, M1_R11_en, 0);

            if (snrt_global_core_idx() == 0) {
                // printf("Finished oscore_in refill for tile %d\n", tile);
                while (read_csr(SIMBACORE_BUSY));
                while (read_csr(STREAMER_BUSY_CSR));
                simbacore_cycles_phase1 += read_simbacore_perf_counter();
            }
        }

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

        // Load conv_out to L3
        if (i >= 2 && snrt_is_dm_core()) {
            uint32_t spill_tile = i - 2;
            int sbuf            = spill_tile % 2;
            snrt_dma_start_1d(ptr_conv_out_l3 + spill_tile * M1_length_conv_out_tile, ptr_conv_out_tile[sbuf],
                              M1_length_conv_out_tile);
        }

        if (snrt_is_dm_core()) snrt_dma_wait_all();
        snrt_cluster_hw_barrier();
    }

    if (snrt_global_core_idx() == 0) printf("[%u cc] P1 done, starting P2 bias preload\n", snrt_mcycle());

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d((uint8_t*)ptr_iscore_out_P2, M2_iscore_bias, M2_length_iscore_out);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    /////////////////////////////////
    //////// Phase 2 ////////////////
    /////////////////////////////////

    if (snrt_global_core_idx() == 0) {
        set_streamer_phase2_lTile((uint32_t)ptr_oscore_in_base, (uint32_t)ptr_oscore_weight_P2[0],
                                  (uint32_t)ptr_z_tile[0], (uint32_t)ptr_dt_in, (uint32_t)ptr_dt_weight_1[0],
                                  (uint32_t)ptr_dt_weight_2[0], (uint32_t)ptr_dt_bias[0], (uint32_t)ptr_x_tile[0],
                                  (uint32_t)ptr_A[0], (uint32_t)ptr_BC, (uint32_t)ptr_D[0], (uint32_t)ptr_y_tile[0],
                                  (uint32_t)ptr_iscore_weight_P2[0], (uint32_t)ptr_iscore_out_P2);
        set_simbacore_csr(M29_PHASE2_NO_REQUANT, seqLen, dModel, M2_dInner_tile, dtRank, dModel);
    }
    snrt_cluster_hw_barrier();

    for (uint32_t i = 0; i < nb_tiles + 2; i++) {
        int buf = i % 2;

        if (i >= 1 && i <= nb_tiles) {
            uint32_t tile      = i - 1;
            bool is_final_tile = (tile == nb_tiles - 1);

            if (snrt_global_core_idx() == 0) {
                if (is_final_tile) write_csr(MODE, M2_PHASE2);

                // Non-blocking start: do not use delayed start for the SUC z-reader: it blocks oscore refill loop
                _set_streamer_start();
                _set_simbacore_start();
                write_csr(STREAMER_START_CSR, 0);
                write_csr(SIMBACORE_START, 0);
            }

            // Both SUC readers (R10 z, R11 y) are released inside the refill loop the moment their
            // own gauge crosses M2_R1x_start_cnt.
            oscore_in_refill_loop(ptr_oscore_in_base, M2_oscore_in, M2_R10_en, M2_R10_start_cnt, M2_R11_en,
                                  M2_R11_start_cnt);

            // Clear both delayed-start CSRs to rearm for the next tile.
            if (snrt_global_core_idx() == 0) {
                write_csr(DELAYED_START_READER_10, 0);
                write_csr(DELAYED_START_READER_11, 0);
            }

            if (snrt_global_core_idx() == 0 && !is_final_tile) {
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

            // printf("Finished oscore_in refill for tile %d\n", tile);

            if (snrt_global_core_idx() == 0) {
                while (read_csr(SIMBACORE_BUSY));
                while (read_csr(STREAMER_BUSY_CSR));
                asm volatile("fence" ::: "memory");
                simbacore_cycles_phase2 += read_simbacore_perf_counter();
            }
        }

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
            snrt_dma_start_1d(ptr_x_tile[buf], ptr_conv_out_l3 + i * M2_length_x_tile, M2_length_x_tile);
        }

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
    if (snrt_global_core_idx() == 0) printf("[%u cc] P2 done, starting verification\n", snrt_mcycle());

    if (snrt_global_core_idx() == 0) {
        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Simbacore Phase1 (sum over tiles): %u cycles\n", end_cycles, simbacore_cycles_phase1);
        printf("[%d cc] Simbacore Phase2 (sum over tiles): %u cycles\n", end_cycles, simbacore_cycles_phase2);
        printf("[%d cc] Simbacore elapsed time: %u cycles\n", end_cycles,
               simbacore_cycles_phase1 + simbacore_cycles_phase2);
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        err += check_result_sample(ptr_conv_out_l3, M1_conv_out, M1_test_samples_conv_out,  //
                                   nb_test_samples, "P1 conv_out (= P2 x, from L3)");
        err += check_result_sample(ptr_iscore_out_P1, M1_iscore_out, M1_test_samples_iscore_out, nb_test_samples,
                                   "P1 iscore_out (= P2 dt+BC)");
        err += check_result_sample(ptr_z_l3, M2_oscore_expected, M2_test_samples_z,  //
                                   nb_test_samples, "z (osCore out)");
        err += check_result_sample(ptr_y_l3, M2_suc_expected, M2_test_samples_y,  //
                                   nb_test_samples, "SUC y");
        err += check_result_sample((uint8_t*)ptr_iscore_out_P2, M2_iscore_expected,  //
                                   M2_test_samples_iscore_out, nb_test_samples, "iscore_out");

        printf("Test Phase1+Phase2 oscore: seqLen=%d, dModel=%d, dInner=%d, nb_tiles=%d, nb_l_tiles=%d\n",  //
               seqLen, dModel, dInner, nb_tiles, nb_l_tiles);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, 5 * nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() {
    int err = test_phase1_and_2();
    return err;
}
