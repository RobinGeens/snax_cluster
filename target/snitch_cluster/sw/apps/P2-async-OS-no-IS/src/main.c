// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// P2-async-OS-no-IS = Phase 2 of the Mamba main with the IS-core (out-proj GEMM) EXCLUDED
// (mode M33_PHASE2_NO_ISCORE), the osCore input tensor (oscore_in) ASYNC L-tiled into a TCDM ring
// refilled from L3 during compute (B1), and the SU-core output y written straight to L3.
// osCore + SU-core run; z and y are staged through L3; x and dt+BC are preloaded from golden.
// Combines strategies A4 (drop IS-gemm, SUC->L3) + B1 (async OS input) from
// docs/dataflow/04_mamba_main.md; ring mechanics: docs/dataflow/09_async_tiling.md.
// Reuses main-tiled-oscore's data (M2_* + oscore_in ring scalars + the M33 mode constant).

#include "helper.c"
#include "snax-simbacore-lib.h"

// P2 streamer setup with osCore (R0 lTile ring) + switchCore + SU-core enabled, IS-core
// (R11/R12/R13/W3) DISABLED. R0 uses the lTile (nb_slots-slot stride-0 wrap) temporal bounds.
static inline void set_streamer_phase2_noIS_lTile(uint32_t p_oi, uint32_t p_ow, uint32_t p_z, uint32_t p_dt,
                                                  uint32_t p_dw1, uint32_t p_dw2, uint32_t p_db, uint32_t p_x,
                                                  uint32_t p_A, uint32_t p_BC, uint32_t p_D, uint32_t p_y) {
    set_streamer_csr(

        p_oi, M2_R0_ss, M2_R0_tb_lTile, M2_R0_ts, M2_R0_en,  // osCore in (async ring)
        p_ow, M2_R1_ss, M2_R1_tb, M2_R1_ts, M2_R1_en,        // osCore weight
        p_dt, M2_R2_ss, M2_R2_tb, M2_R2_ts, M2_R2_en,        // switchCore in (dt)
        p_dw1, M2_R3_ss, M2_R3_tb, M2_R3_ts, M2_R3_en,       // switchCore weight 1
        p_db, M2_R4_ss, M2_R4_tb, M2_R4_ts, M2_R4_en,        // switchCore bias
        p_dw2, M2_R5_ss, M2_R5_tb, M2_R5_ts, M2_R5_en,       // switchCore weight 2
        p_A, M2_R6_ss, M2_R6_tb, M2_R6_ts, M2_R6_en,         // SUC A
        p_BC, M2_R7_ss, M2_R7_tb, M2_R7_ts, M2_R7_en,        // SUC BC
        p_D, M2_R8_ss, M2_R8_tb, M2_R8_ts, M2_R8_en,         // SUC D
        p_x, M2_R9_ss, M2_R9_tb, M2_R9_ts, M2_R9_en,         // SUC x
        p_z, M2_R10_ss, M2_R10_tb, M2_R10_ts, M2_R10_en,     // SUC z (= osCore out)
        (uint32_t)0, 0, 0, 0, 0,                             // R11 isCore in (DISABLED)
        (uint32_t)0, 0, 0, 0, 0,                             // R12 isCore weight (DISABLED)
        (uint32_t)0, 0, 0, 0, 0,                             // R13 isCore psum (DISABLED)

        p_z, M2_W0_ss, M2_W0_tb, M2_W0_ts, M2_W0_en,  // osCore out = z
        (uint32_t)0, 0, 0, 0, M2_W1_en,               // W1 disabled
        p_y, M2_W2_ss, M2_W2_tb, M2_W2_ts, M2_W2_en,  // SUC y out
        (uint32_t)0, 0, 0, 0, 0                        // W3 isCore out (DISABLED)
    );
}

// Async oscore_in ring refill from L3, paced by R10 (osCore output-tile gauge). The SUC z-reader
// (DELAYED_START_READER_10) is released from inside this loop the instant its gauge crosses the
// threshold, with a post-loop fallback. No IS-core reader here (R11 disabled).
static inline void oscore_in_refill_loop(uint8_t* slot_base, const uint8_t* l3_oscore_in,  //
                                         uint32_t r10_release_en, uint32_t r10_start_cnt) {
    const uint32_t N_visits   = (M2_dInner_tile / dInnerUnroll) * nb_l_tiles;
    const uint32_t gauge_step = M2_oscore_in_l_tile_gauge_step;
    const uint32_t len_l_tile = M2_length_oscore_in_l_tile;
    uint32_t r10_released     = 0;

    for (uint32_t r = 0; r < N_visits; r++) {
        if (snrt_global_core_idx() == 0) {
            while (read_csr(R10_DELAY_GAUGE) < (r + 1) * gauge_step);
            if (r10_release_en && !r10_released && (r + 1) * gauge_step >= r10_start_cnt) {
                write_csr(DELAYED_START_READER_10, 1);
                r10_released = 1;
            }
        }
        snrt_cluster_hw_barrier();
        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(slot_base + (r % nb_slots) * len_l_tile,
                              l3_oscore_in + ((r + nb_slots) % nb_l_tiles) * len_l_tile, len_l_tile);
        }
    }

    if (snrt_is_dm_core()) snrt_dma_wait_all();
    snrt_cluster_hw_barrier();

    if (snrt_global_core_idx() == 0 && r10_release_en && !r10_released) {
        printf("Fallback: delayed R10 threshold sits after refill loop end.\n");
        while (read_csr(R10_DELAY_GAUGE) < r10_start_cnt);
        write_csr(DELAYED_START_READER_10, 1);
    }
}

int test_p2_async_no_is() {
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

    // B1: oscore_in is nb_slots ADJACENT ring slots (R0 stride-0 wrap walks them contiguously).
    uint8_t* ptr_oscore_in_base   = (uint8_t*)tcdm_base_ptr;
    uint32_t oscore_in_tcdm_bytes = nb_slots * M2_length_oscore_in_l_tile;

#define _ALIGN64(p) ((uint8_t*)(((uintptr_t)(p) + 63u) & ~(uintptr_t)63u))

    // FULL switchCore input dt+BC (preloaded from golden).
    uint8_t* ptr_dt_in = _ALIGN64(ptr_oscore_in_base + oscore_in_tcdm_bytes);
    uint8_t* ptr_BC    = ptr_dt_in + M2_dt_to_BC_offset;

    uint8_t* pingpong_base_ptr = _ALIGN64(ptr_dt_in + M2_length_dt_BC);

    // ---- Per-dInner-tile ping-pong slots (no iscore_weight: IS-core excluded).
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

    if (snrt_global_core_idx() == 0) init_cycle_counter();
    snrt_cluster_hw_barrier();

    // Preload: first nb_slots oscore_in L-tiles into the ring; FULL dt+BC from golden.
    if (snrt_is_dm_core()) {
        for (uint32_t s = 0; s < nb_slots; s++)
            snrt_dma_start_1d(ptr_oscore_in_base + s * M2_length_oscore_in_l_tile,
                              M2_oscore_in + s * M2_oscore_in_l_offset, M2_length_oscore_in_l_tile);
        snrt_dma_start_1d(ptr_dt_in, M2_dt_BC, M2_length_dt_BC);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    uint32_t start_cycles            = 0;
    uint32_t simbacore_cycles_phase2 = 0;
    const uint32_t K_i = M2_dInner_tile / dInnerUnroll;

    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: P2-async-OS-no-IS (L=%d, dModel=%d, nb_tiles=%d, nb_l_tiles=%d, L_tile=%u, "
               "nb_slots=%d, K_i=%u, no IS-core, oscore_in async)\n\n",
               seqLen, dModel, nb_tiles, nb_l_tiles, L_tile, nb_slots, K_i);
        start_cycles = snrt_mcycle();
        set_streamer_phase2_noIS_lTile((uint32_t)ptr_oscore_in_base, (uint32_t)ptr_oscore_weight_P2[0],
                                       (uint32_t)ptr_z_tile[0], (uint32_t)ptr_dt_in, (uint32_t)ptr_dt_weight_1[0],
                                       (uint32_t)ptr_dt_weight_2[0], (uint32_t)ptr_dt_bias[0], (uint32_t)ptr_x_tile[0],
                                       (uint32_t)ptr_A[0], (uint32_t)ptr_BC, (uint32_t)ptr_D[0], (uint32_t)ptr_y_tile[0]);
        set_simbacore_csr(M33_PHASE2_NO_ISCORE, seqLen, dModel, M2_dInner_tile, dtRank, dModel);
    }
    snrt_cluster_hw_barrier();

    for (uint32_t i = 0; i < nb_tiles + 2; i++) {
        int buf = i % 2;

        // Compute tile (i-1). BOTH cores enter so they can run the refill loop together.
        if (i >= 1 && i <= nb_tiles) {
            if (snrt_global_core_idx() == 0) {
                // Non-blocking start: SUC z-reader (R10) is released from inside the refill loop.
                _set_streamer_start();
                _set_simbacore_start();
                write_csr(STREAMER_START_CSR, 0);
                write_csr(SIMBACORE_START, 0);
            }

            oscore_in_refill_loop(ptr_oscore_in_base, M2_oscore_in, M2_R10_en, M2_R10_start_cnt);

            if (snrt_global_core_idx() == 0) {
                write_csr(DELAYED_START_READER_10, 0);  // rearm for next tile

                uint32_t tile = i - 1;
                if (tile != nb_tiles - 1) {
                    int nbuf = (tile + 1) % 2;
                    write_csr(BASE_PTR_READER_1_LOW, (uint32_t)ptr_oscore_weight_P2[nbuf]);
                    write_csr(BASE_PTR_READER_3_LOW, (uint32_t)ptr_dt_weight_1[nbuf]);
                    write_csr(BASE_PTR_READER_4_LOW, (uint32_t)ptr_dt_bias[nbuf]);
                    write_csr(BASE_PTR_READER_5_LOW, (uint32_t)ptr_dt_weight_2[nbuf]);
                    write_csr(BASE_PTR_READER_6_LOW, (uint32_t)ptr_A[nbuf]);
                    write_csr(BASE_PTR_READER_8_LOW, (uint32_t)ptr_D[nbuf]);
                    write_csr(BASE_PTR_READER_9_LOW, (uint32_t)ptr_x_tile[nbuf]);
                    write_csr(BASE_PTR_READER_10_LOW, (uint32_t)ptr_z_tile[nbuf]);
                    write_csr(BASE_PTR_WRITER_0_LOW, (uint32_t)ptr_z_tile[nbuf]);
                    write_csr(BASE_PTR_WRITER_2_LOW, (uint32_t)ptr_y_tile[nbuf]);
                }

                while (read_csr(SIMBACORE_BUSY));
                while (read_csr(STREAMER_BUSY_CSR));
                asm volatile("fence" ::: "memory");
                simbacore_cycles_phase2 += read_simbacore_perf_counter();
            }
        }

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
            snrt_dma_start_1d(ptr_x_tile[buf], M2_suc_x + i * M2_length_x_tile, M2_length_x_tile);
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
        printf("[%d cc] Simbacore Phase2-no-IS (sum over tiles): %u cycles\n", end_cycles, simbacore_cycles_phase2);
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        err += check_result_sample(ptr_z_l3, M2_oscore_expected, M2_test_samples_z,  //
                                   nb_test_samples, "z (osCore out)");
        err += check_result_sample(ptr_y_l3, M2_suc_expected, M2_test_samples_y,  //
                                   nb_test_samples, "SUC y (to L3)");

        printf("Test P2-async-OS-no-IS: seqLen=%d, dModel=%d, dInner=%d, nb_tiles=%d, nb_l_tiles=%d\n", seqLen, dModel,
               dInner, nb_tiles, nb_l_tiles);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, 2 * nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test_p2_async_no_is(); }
