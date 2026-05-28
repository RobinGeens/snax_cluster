// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>

#include "helper.c"
#include "snax-simbacore-lib.h"

// P1 streamer setup: same temporal bounds as v2's _nTile macro (R12/R13/W3 bounded
// to N_tile). R12 K-stride keeps full xProjDim (weight tile holds full N per K-row).
#define set_streamer_phase1_nTile(p_oi, p_ow, p_cw, p_cb, p_iw, p_io, p_co)                                          \
    set_streamer_csr((uint32_t)(p_oi), M1_R0_ss, M1_R0_tb_nTile, M1_R0_ts, M1_R0_en, (uint32_t)(p_ow), M1_R1_ss,     \
                     M1_R1_tb_nTile, M1_R1_ts, M1_R1_en, (uint32_t)0, 0, 0, 0, M1_R2_en, (uint32_t)(p_cw), M1_R3_ss, \
                     M1_R3_tb_nTile, M1_R3_ts, M1_R3_en, (uint32_t)(p_cb), M1_R4_ss, M1_R4_tb_nTile, M1_R4_ts,       \
                     M1_R4_en, (uint32_t)0, 0, 0, 0, M1_R5_en, (uint32_t)0, 0, 0, 0, M1_R6_en, (uint32_t)0, 0, 0, 0, \
                     M1_R7_en, (uint32_t)0, 0, 0, 0, M1_R8_en, (uint32_t)0, 0, 0, 0, M1_R9_en, (uint32_t)0, 0, 0, 0, \
                     M1_R10_en, (uint32_t)0, 0, 0, 0, M1_R11_en, (uint32_t)(p_iw), M1_R12_ss, M1_R12_tb_nTile,       \
                     M1_R12_ts, M1_R12_en, (uint32_t)(p_io), M1_R13_ss, M1_R13_tb_nTile, M1_R13_ts, M1_R13_en,       \
                     (uint32_t)0, 0, 0, 0, M1_W0_en, (uint32_t)(p_co), M1_W1_ss, M1_W1_tb_nTile, M1_W1_ts, M1_W1_en, \
                     (uint32_t)0, 0, 0, 0, M1_W2_en, (uint32_t)(p_io), M1_W3_ss, M1_W3_tb_nTile, M1_W3_ts, M1_W3_en)

// P2 streamer setup with N_tile-reduced R12/R13/W3 temporal bounds.
#define set_streamer_phase2_nTile(p_oi, p_ow, p_z, p_dt, p_dw1, p_dw2, p_db, p_x, p_A, p_BC, p_D, p_y, p_iw, p_io) \
    set_streamer_csr(                                                                                              \
        (uint32_t)(p_oi), M2_R0_ss, M2_R0_tb, M2_R0_ts, M2_R0_en, (uint32_t)(p_ow), M2_R1_ss, M2_R1_tb, M2_R1_ts,  \
        M2_R1_en, (uint32_t)(p_dt), M2_R2_ss, M2_R2_tb, M2_R2_ts, M2_R2_en, (uint32_t)(p_dw1), M2_R3_ss, M2_R3_tb, \
        M2_R3_ts, M2_R3_en, (uint32_t)(p_db), M2_R4_ss, M2_R4_tb, M2_R4_ts, M2_R4_en, (uint32_t)(p_dw2), M2_R5_ss, \
        M2_R5_tb, M2_R5_ts, M2_R5_en, (uint32_t)(p_A), M2_R6_ss, M2_R6_tb, M2_R6_ts, M2_R6_en, (uint32_t)(p_BC),   \
        M2_R7_ss, M2_R7_tb, M2_R7_ts, M2_R7_en, (uint32_t)(p_D), M2_R8_ss, M2_R8_tb, M2_R8_ts, M2_R8_en,           \
        (uint32_t)(p_x), M2_R9_ss, M2_R9_tb, M2_R9_ts, M2_R9_en, (uint32_t)(p_z), M2_R10_ss, M2_R10_tb, M2_R10_ts, \
        M2_R10_en, (uint32_t)(p_y), M2_R11_ss, M2_R11_tb, M2_R11_ts, M2_R11_en, (uint32_t)(p_iw), M2_R12_ss,       \
        M2_R12_tb_nTile, M2_R12_ts, M2_R12_en, (uint32_t)(p_io), M2_R13_ss, M2_R13_tb_nTile, M2_R13_ts, M2_R13_en, \
        (uint32_t)(p_z), M2_W0_ss, M2_W0_tb, M2_W0_ts, M2_W0_en, (uint32_t)0, 0, 0, 0, M2_W1_en, (uint32_t)(p_y),  \
        M2_W2_ss, M2_W2_tb, M2_W2_ts, M2_W2_en, (uint32_t)(p_io), M2_W3_ss, M2_W3_tb_nTile, M2_W3_ts, M2_W3_en)

int test_phase1_and_2() {
    int err = 0;

    // ---- L3 buffers. Reserve 16 KiB at L3 base to skip putc_buffer.
    static uint8_t* ptr_conv_out_l3 = NULL;
    static uint8_t* ptr_z_l3        = NULL;
    static uint8_t* ptr_y_l3        = NULL;
    // BF16 psum staging (re-DMA'd in/out every dInner step). Total size matches
    // M*_length_iscore_out (full per-phase BF16 footprint).
    static uint8_t* ptr_iscore_psum_P1_l3 = NULL;
    static uint8_t* ptr_iscore_psum_P2_l3 = NULL;
    // FP8 final outputs (written by the final dInner tile's kernel and spilled per-n).
    static uint8_t* ptr_iscore_out_P1_l3 = NULL;
    static uint8_t* ptr_iscore_out_P2_l3 = NULL;

    if (snrt_global_core_idx() == 0) {
        (void)snrt_l3alloc(16 * 1024);
        ptr_conv_out_l3       = (uint8_t*)snrt_l3alloc(M1_length_conv_out);
        ptr_z_l3              = (uint8_t*)snrt_l3alloc(M2_length_z);
        ptr_y_l3              = (uint8_t*)snrt_l3alloc(M2_length_y);
        ptr_iscore_psum_P1_l3 = (uint8_t*)snrt_l3alloc(M1_length_iscore_out);
        ptr_iscore_out_P1_l3  = (uint8_t*)snrt_l3alloc(M1_length_iscore_out);
        ptr_iscore_psum_P2_l3 = (uint8_t*)snrt_l3alloc(M2_length_iscore_out);
        ptr_iscore_out_P2_l3  = (uint8_t*)snrt_l3alloc(M2_length_iscore_out);
    }

    snrt_cluster_hw_barrier();

    // ---- TCDM buffers.
    void* tcdm_base_ptr = snrt_l1_next();

    uint8_t* ptr_oscore_in = (uint8_t*)tcdm_base_ptr;

    // Two psum ping-pong slots (sized for the larger of P1/P2). P1/P2 share these.
    uint32_t iscore_tile_bytes = (M1_length_iscore_out_psum_tile > M2_length_iscore_out_psum_tile)
                                     ? M1_length_iscore_out_psum_tile
                                     : M2_length_iscore_out_psum_tile;

#define _ALIGN64(p) ((uint8_t*)(((uintptr_t)(p) + 63u) & ~(uintptr_t)63u))

    uint8_t* ptr_iscore_psum_P1[2] = {
        _ALIGN64(ptr_oscore_in + M1_length_oscore_in),
        _ALIGN64(_ALIGN64(ptr_oscore_in + M1_length_oscore_in) + iscore_tile_bytes),
    };
    uint16_t* ptr_iscore_psum_P2[2] = {(uint16_t*)ptr_iscore_psum_P1[0], (uint16_t*)ptr_iscore_psum_P1[1]};

    uint8_t* pingpong_base_ptr = _ALIGN64(ptr_iscore_psum_P1[1] + iscore_tile_bytes);

    // ---- Phase 1 ping-pong slots (across dInner tiles).
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

    // P2 dt_in/BC: golden transposed vector DMA'd to TCDM before P2 (cross-N_tile assembly deferred).
    uint8_t* ptr_dt_in = _ALIGN64(ptr_y_tile[1] + M2_length_y_tile);
    uint8_t* ptr_BC    = ptr_dt_in + M2_dt_to_BC_offset;
#undef _ALIGN64

    if (snrt_global_core_idx() == 0) init_cycle_counter();
    snrt_cluster_hw_barrier();

    // Preload oscore_in (shared across all dInner / N iterations).
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_oscore_in, M1_oscore_in, M1_length_oscore_in);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    // Initialize per-N_tile psum staging in L3 with iscore_bias chunks (2D DMA per n).
    if (snrt_is_dm_core()) {
        for (uint32_t n = 0; n < nb_n_tiles; n++) {
            snrt_dma_start_2d(ptr_iscore_psum_P1_l3 + n * M1_length_iscore_out_psum_tile,
                              (uint8_t*)M1_iscore_bias + n * M1_iscore_bias_n_inner,  //
                              M1_iscore_bias_n_inner, M1_iscore_bias_n_inner,         //
                              M1_iscore_bias_n_src_stride, M1_iscore_bias_n_count);
        }
        for (uint32_t n = 0; n < nb_n_tiles; n++) {
            snrt_dma_start_2d(ptr_iscore_psum_P2_l3 + n * M2_length_iscore_out_psum_tile,
                              (uint8_t*)M2_iscore_bias + n * M2_iscore_bias_n_inner,  //
                              M2_iscore_bias_n_inner, M2_iscore_bias_n_inner,         //
                              M2_iscore_bias_n_src_stride, M2_iscore_bias_n_count);
        }
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    uint32_t start_cycles            = 0;
    uint32_t simbacore_cycles_phase1 = 0;
    uint32_t simbacore_cycles_phase2 = 0;

    const uint32_t K_i = M1_dInner_tile / dInnerUnroll;  // K-steps per DMA tile (P1, P2 share)

    /////////////////////////////////
    //////// Phase 1 ////////////////
    /////////////////////////////////

    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: Mamba main tiled v3 (L=%d, dModel=%d, nb_tiles=%d, nb_n_tiles=%d, K_i=%u)\n\n",
               seqLen, dModel, nb_tiles, nb_n_tiles, K_i);
        start_cycles = snrt_mcycle();
    }

    // Set streamer once for P1 (initial base ptrs; updated per (i, n) below).
    if (snrt_global_core_idx() == 0) {
        set_streamer_phase1_nTile((uint32_t)ptr_oscore_in,            //
                                  (uint32_t)ptr_oscore_weight_P1[0],  //
                                  (uint32_t)ptr_conv_weight[0],       //
                                  (uint32_t)ptr_conv_bias[0],         //
                                  (uint32_t)ptr_iscore_weight_P1[0],  //
                                  (uint32_t)ptr_iscore_psum_P1[0],    //
                                  (uint32_t)ptr_conv_out_tile[0]);
        set_simbacore_csr(M28_PHASE1_NO_REQUANT, seqLen, dModel, M1_dInner_tile, dtRank, xProjDim_tile);
    }
    snrt_cluster_hw_barrier();

    // Outer loop: dInner tile (3-stage pipeline on weight DMA / compute / conv_out spill).
    for (uint32_t i = 0; i < nb_tiles + 2; i++) {
        int wbuf = i % 2;  // DMA-in slot for dInner tile i

        // (1) DMA in dInner tile i's weights (oscore, conv, conv_bias, FULL iscore_weight over all N).
        if (i < nb_tiles && snrt_is_dm_core()) {
            snrt_dma_start_1d(ptr_oscore_weight_P1[wbuf], M1_oscore_weight + i * M1_length_oscore_weight_tile,
                              M1_length_oscore_weight_tile);
            snrt_dma_start_1d(ptr_conv_weight[wbuf], M1_conv_weight + i * M1_length_conv_weight_tile,
                              M1_length_conv_weight_tile);
            snrt_dma_start_1d(ptr_conv_bias[wbuf], M1_conv_bias + i * M1_length_conv_bias_tile,
                              M1_length_conv_bias_tile);
            snrt_dma_start_1d(ptr_iscore_weight_P1[wbuf], M1_iscore_weight + i * M1_length_iscore_weight_tile,
                              M1_length_iscore_weight_tile);
        }

        // (2) Compute: inner N loop for dInner tile (i-1).
        if (i >= 1 && i <= nb_tiles) {
            uint32_t tile      = i - 1;
            int tile_buf       = tile % 2;
            bool is_final_tile = (tile == nb_tiles - 1);

            if (snrt_global_core_idx() == 0) {
                if (is_final_tile) write_csr(MODE, M1_PHASE1);
                // SUC-side base ptrs (constant across inner n loop).
                write_csr(BASE_PTR_READER_1_LOW, (uint32_t)ptr_oscore_weight_P1[tile_buf]);
                write_csr(BASE_PTR_READER_3_LOW, (uint32_t)ptr_conv_weight[tile_buf]);
                write_csr(BASE_PTR_READER_4_LOW, (uint32_t)ptr_conv_bias[tile_buf]);
                write_csr(BASE_PTR_WRITER_1_LOW, (uint32_t)ptr_conv_out_tile[tile_buf]);
                // Initial R12/R13/W3 for n=0 (subsequent n's are preloaded during busy-wait).
                write_csr(BASE_PTR_READER_12_LOW, (uint32_t)ptr_iscore_weight_P1[tile_buf]);
                write_csr(BASE_PTR_READER_13_LOW, (uint32_t)ptr_iscore_psum_P1[0]);
                write_csr(BASE_PTR_WRITER_3_LOW, (uint32_t)ptr_iscore_psum_P1[0]);
            }

            // Inner N loop: 3-stage pipeline on psum (DMA-in[n] / compute[n-1] / DMA-out[n-2]).
            for (uint32_t n = 0; n < nb_n_tiles + 2; n++) {
                int n_in_buf  = n % 2;
                int n_out_buf = (n - 2) % 2;

                // (a) DMA-out psum[n-2] back to L3 (must come BEFORE DMA-in[n] which targets same slot).
                if (n >= 2 && snrt_is_dm_core()) {
                    uint32_t out_n = n - 2;
                    uint8_t* dst   = is_final_tile ? (ptr_iscore_out_P1_l3 + out_n * M1_length_iscore_out_psum_tile)
                                                   : (ptr_iscore_psum_P1_l3 + out_n * M1_length_iscore_out_psum_tile);
                    snrt_dma_start_1d(dst, ptr_iscore_psum_P1[n_out_buf], M1_length_iscore_out_psum_tile);
                }

                // (b) DMA-in psum[n] from L3 staging into ping-pong slot.
                if (n < nb_n_tiles && snrt_is_dm_core()) {
                    snrt_dma_start_1d(ptr_iscore_psum_P1[n_in_buf],
                                      ptr_iscore_psum_P1_l3 + n * M1_length_iscore_out_psum_tile,
                                      M1_length_iscore_out_psum_tile);
                }

                // (c) Compute kernel for n-1 (base ptrs were preloaded at the previous iter or
                // initial setup). Preload n's base ptrs during the busy-wait.
                if (n >= 1 && n <= nb_n_tiles && snrt_global_core_idx() == 0) {
                    _set_streamer_start();
                    _set_simbacore_start();
                    write_csr(STREAMER_START_CSR, 0);
                    write_csr(SIMBACORE_START, 0);

                    // Preload base ptrs for next compute (n's kernel = comp = n in the next iter).
                    // The next compute uses iscore_weight + n*offset and psum slot n%2.
                    if (n < nb_n_tiles) {
                        uint32_t next_off = n * M1_iscore_weight_n_offset;
                        write_csr(BASE_PTR_READER_12_LOW, (uint32_t)ptr_iscore_weight_P1[tile_buf] + next_off);
                        write_csr(BASE_PTR_READER_13_LOW, (uint32_t)ptr_iscore_psum_P1[n_in_buf]);
                        write_csr(BASE_PTR_WRITER_3_LOW, (uint32_t)ptr_iscore_psum_P1[n_in_buf]);
                    }

                    while (read_csr(SIMBACORE_BUSY));
                    while (read_csr(STREAMER_BUSY_CSR));
                    simbacore_cycles_phase1 += read_simbacore_perf_counter();
                }

                if (snrt_is_dm_core()) snrt_dma_wait_all();
                snrt_cluster_hw_barrier();
            }
        }

        // (3) Spill conv_out tile (i-2) to L3 — ONCE per dInner tile.
        if (i >= 2 && snrt_is_dm_core()) {
            uint32_t spill_tile = i - 2;
            int sbuf            = spill_tile % 2;
            snrt_dma_start_1d(ptr_conv_out_l3 + spill_tile * M1_length_conv_out_tile, ptr_conv_out_tile[sbuf],
                              M1_length_conv_out_tile);
            snrt_dma_wait_all();
        }

        snrt_cluster_hw_barrier();
    }

    if (snrt_global_core_idx() == 0) printf("[%u cc] P1 done, starting P2\n", snrt_mcycle());

    // DMA golden dt_in/BC (M1_iscore_out, transposed FP8) to TCDM for P2.
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_dt_in, M1_iscore_out, M1_length_iscore_out);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    /////////////////////////////////
    //////// Phase 2 ////////////////
    /////////////////////////////////

    if (snrt_global_core_idx() == 0) {
        set_streamer_phase2_nTile((uint32_t)ptr_oscore_in,            //
                                  (uint32_t)ptr_oscore_weight_P2[0],  //
                                  (uint32_t)ptr_z_tile[0],            //
                                  (uint32_t)ptr_dt_in,                //
                                  (uint32_t)ptr_dt_weight_1[0],       //
                                  (uint32_t)ptr_dt_weight_2[0],       //
                                  (uint32_t)ptr_dt_bias[0],           //
                                  (uint32_t)ptr_x_tile[0],            //
                                  (uint32_t)ptr_A[0],                 //
                                  (uint32_t)ptr_BC,                   //
                                  (uint32_t)ptr_D[0],                 //
                                  (uint32_t)ptr_y_tile[0],            //
                                  (uint32_t)ptr_iscore_weight_P2[0],  //
                                  (uint32_t)ptr_iscore_psum_P2[0]);
        set_simbacore_csr(M29_PHASE2_NO_REQUANT, seqLen, dModel, M2_dInner_tile, dtRank, dModel_tile);
    }
    snrt_cluster_hw_barrier();

    for (uint32_t i = 0; i < nb_tiles + 2; i++) {
        int wbuf = i % 2;

        // (1) DMA in dInner tile i's weights + x_tile.
        if (i < nb_tiles && snrt_is_dm_core()) {
            snrt_dma_start_1d(ptr_oscore_weight_P2[wbuf], M2_oscore_weight + i * M2_length_oscore_weight_tile,
                              M2_length_oscore_weight_tile);
            snrt_dma_start_1d(ptr_dt_weight_1[wbuf], M2_dt_weight_1 + i * M2_length_dt_weight_1_tile,
                              M2_length_dt_weight_1_tile);
            snrt_dma_start_1d(ptr_dt_weight_2[wbuf], M2_dt_weight_2 + i * M2_length_dt_weight_2_tile,
                              M2_length_dt_weight_2_tile);
            snrt_dma_start_1d(ptr_dt_bias[wbuf], M2_dt_bias + i * M2_length_dt_bias_tile, M2_length_dt_bias_tile);
            snrt_dma_start_1d(ptr_A[wbuf], M2_suc_A + i * M2_length_A_tile, M2_length_A_tile);
            snrt_dma_start_1d(ptr_D[wbuf], M2_suc_D + i * M2_length_D_tile, M2_length_D_tile);
            snrt_dma_start_1d(ptr_iscore_weight_P2[wbuf], M2_iscore_weight + i * M2_length_iscore_weight_tile,
                              M2_length_iscore_weight_tile);
            snrt_dma_start_1d(ptr_x_tile[wbuf], ptr_conv_out_l3 + i * M2_length_x_tile, M2_length_x_tile);
        }

        // (2) Compute: inner N loop for dInner tile (i-1).
        if (i >= 1 && i <= nb_tiles) {
            uint32_t tile      = i - 1;
            int tile_buf       = tile % 2;
            bool is_final_tile = (tile == nb_tiles - 1);

            if (snrt_global_core_idx() == 0) {
                if (is_final_tile) write_csr(MODE, M2_PHASE2);
                write_csr(BASE_PTR_READER_1_LOW, (uint32_t)ptr_oscore_weight_P2[tile_buf]);
                write_csr(BASE_PTR_READER_3_LOW, (uint32_t)ptr_dt_weight_1[tile_buf]);
                write_csr(BASE_PTR_READER_4_LOW, (uint32_t)ptr_dt_bias[tile_buf]);
                write_csr(BASE_PTR_READER_5_LOW, (uint32_t)ptr_dt_weight_2[tile_buf]);
                write_csr(BASE_PTR_READER_6_LOW, (uint32_t)ptr_A[tile_buf]);
                write_csr(BASE_PTR_READER_8_LOW, (uint32_t)ptr_D[tile_buf]);
                write_csr(BASE_PTR_READER_9_LOW, (uint32_t)ptr_x_tile[tile_buf]);
                write_csr(BASE_PTR_READER_10_LOW, (uint32_t)ptr_z_tile[tile_buf]);
                write_csr(BASE_PTR_READER_11_LOW, (uint32_t)ptr_y_tile[tile_buf]);
                write_csr(BASE_PTR_WRITER_0_LOW, (uint32_t)ptr_z_tile[tile_buf]);
                write_csr(BASE_PTR_WRITER_2_LOW, (uint32_t)ptr_y_tile[tile_buf]);
                // Initial R12/R13/W3 for n=0 (subsequent n's preloaded during busy-wait).
                write_csr(BASE_PTR_READER_12_LOW, (uint32_t)ptr_iscore_weight_P2[tile_buf]);
                write_csr(BASE_PTR_READER_13_LOW, (uint32_t)ptr_iscore_psum_P2[0]);
                write_csr(BASE_PTR_WRITER_3_LOW, (uint32_t)ptr_iscore_psum_P2[0]);
            }

            for (uint32_t n = 0; n < nb_n_tiles + 2; n++) {
                int n_in_buf  = n % 2;
                int n_out_buf = (n - 2) % 2;

                // (a) DMA-out psum[n-2]
                if (n >= 2 && snrt_is_dm_core()) {
                    uint32_t out_n = n - 2;
                    uint8_t* dst   = is_final_tile ? (ptr_iscore_out_P2_l3 + out_n * M2_length_iscore_out_psum_tile)
                                                   : (ptr_iscore_psum_P2_l3 + out_n * M2_length_iscore_out_psum_tile);
                    snrt_dma_start_1d(dst, (uint8_t*)ptr_iscore_psum_P2[n_out_buf], M2_length_iscore_out_psum_tile);
                }

                // (b) DMA-in psum[n]
                if (n < nb_n_tiles && snrt_is_dm_core()) {
                    snrt_dma_start_1d((uint8_t*)ptr_iscore_psum_P2[n_in_buf],
                                      ptr_iscore_psum_P2_l3 + n * M2_length_iscore_out_psum_tile,
                                      M2_length_iscore_out_psum_tile);
                }

                // (c) Compute for n-1 (base ptrs preloaded). Preload next n's base ptrs during busy-wait.
                if (n >= 1 && n <= nb_n_tiles && snrt_global_core_idx() == 0) {
                    start_simbacore_and_streamers(M2_R10_en, M2_R10_start_cnt, M2_R11_en, M2_R11_start_cnt);
                    write_csr(STREAMER_START_CSR, 0);
                    write_csr(SIMBACORE_START, 0);
                    write_csr(DELAYED_START_READER_10, 0);
                    write_csr(DELAYED_START_READER_11, 0);

                    if (n < nb_n_tiles) {
                        uint32_t next_off = n * M2_iscore_weight_n_offset;
                        write_csr(BASE_PTR_READER_12_LOW, (uint32_t)ptr_iscore_weight_P2[tile_buf] + next_off);
                        write_csr(BASE_PTR_READER_13_LOW, (uint32_t)ptr_iscore_psum_P2[n_in_buf]);
                        write_csr(BASE_PTR_WRITER_3_LOW, (uint32_t)ptr_iscore_psum_P2[n_in_buf]);
                    }

                    while (read_csr(SIMBACORE_BUSY));
                    while (read_csr(STREAMER_BUSY_CSR));
                    simbacore_cycles_phase2 += read_simbacore_perf_counter();
                }

                if (snrt_is_dm_core()) snrt_dma_wait_all();
                snrt_cluster_hw_barrier();
            }
        }

        // (3) Spill z/y tile (i-2) to L3 — once per dInner tile.
        if (i >= 2 && snrt_is_dm_core()) {
            uint32_t spill_tile = i - 2;
            int sbuf            = spill_tile % 2;
            snrt_dma_start_1d(ptr_z_l3 + spill_tile * M2_length_z_tile, ptr_z_tile[sbuf], M2_length_z_tile);
            snrt_dma_start_1d(ptr_y_l3 + spill_tile * M2_length_y_tile, ptr_y_tile[sbuf], M2_length_y_tile);
            snrt_dma_wait_all();
        }

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
        err += check_result_sample(ptr_z_l3, M2_oscore_expected, M2_test_samples_z,  //
                                   nb_test_samples, "z (osCore out)");
        err += check_result_sample(ptr_y_l3, M2_suc_expected, M2_test_samples_y,  //
                                   nb_test_samples, "SUC y");

        printf("Test Phase1+Phase2 N-tiled v3: seqLen=%d, dModel=%d, dInner=%d, nb_tiles=%d, nb_n_tiles=%d\n", seqLen,
               dModel, dInner, nb_tiles, nb_n_tiles);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, 3 * nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() {
    int err = test_phase1_and_2();
    return err;
}
