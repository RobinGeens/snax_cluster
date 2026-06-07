// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// Async tiling on IS-core output in P1. P2 is unchanged. The P1 output is not transposed or quantized to FP8.
// This was too difficult to implement because the output counter counts elements, not tiles.
// NOTE This requires a IS-core output tile counter CSR. This is added in RTL, but was not taped out.
//
// What stays full in TCDM: oscore_in, dt_in (loaded from L3), iscore_out_P2.
// What is tiled: conv_out (= P2 x) and z, y via L3; iscore_out_P1 via the async ring (nb_slots slots).

#include "helper.c"
#include "snax-simbacore-lib.h"

static inline void iscore_out_ring_loop(uint8_t* ring_base, uint8_t* l3_psum, uint32_t gauge_step, uint32_t len) {
    for (uint32_t r = 0; r < nb_l_tiles; r++) {
        if (snrt_global_core_idx() == 0)
            while (read_csr(ISCORE_TILE_CNT) < (r + 1) * gauge_step);

        snrt_cluster_hw_barrier();

        if (snrt_is_dm_core()) {
            uint32_t slot     = r % nb_slots;
            uint8_t* slot_ptr = ring_base + slot * len;
            snrt_dma_start_1d(l3_psum + r * len, slot_ptr, len);
            snrt_dma_start_1d(slot_ptr, l3_psum + ((r + nb_slots) % nb_l_tiles) * len, len);
        }
    }

    if (snrt_is_dm_core()) snrt_dma_wait_all();
    snrt_cluster_hw_barrier();
}

int test_phase1_and_2() {
    int err = 0;

    // ---- L3 staging buffers. Reserve 16 KiB at the L3 base to skip putc_buffer (see memory note).
    //   conv_out_l3        : P1 spills here, P2 fetches into x_tile slots.
    //   z_l3, y_l3         : P2 spills here per-tile (z and y stay tile-sized in TCDM, full in L3).
    //   iscore_out_P1_l3   : the FULL P1 psum; the TCDM P1 ring spills/reloads here, then it is
    //                        reassembled into the resident dt_in for Phase 2.
    static uint8_t* ptr_conv_out_l3      = NULL;
    static uint8_t* ptr_z_l3             = NULL;
    static uint8_t* ptr_y_l3             = NULL;
    static uint8_t* ptr_iscore_out_P1_l3 = NULL;
    if (snrt_global_core_idx() == 0) {
        (void)snrt_l3alloc(16 * 1024);
        ptr_conv_out_l3      = (uint8_t*)snrt_l3alloc(M1_length_conv_out);
        ptr_z_l3             = (uint8_t*)snrt_l3alloc(M2_length_z);
        ptr_y_l3             = (uint8_t*)snrt_l3alloc(M2_length_y);
        ptr_iscore_out_P1_l3 = (uint8_t*)snrt_l3alloc(M1_length_iscore_out);
    }
    snrt_cluster_hw_barrier();

    // ---- FULL TCDM buffers.
    void* tcdm_base_ptr = snrt_l1_next();

    uint8_t* ptr_oscore_in = (uint8_t*)tcdm_base_ptr;
    // Golden reference, from L3
    uint8_t* ptr_dt_in = ptr_oscore_in + M1_length_oscore_in;
    uint8_t* ptr_BC    = ptr_dt_in + M2_dt_to_BC_offset;
    // P1 ring overlays dt_in's region
    uint8_t* ptr_iscore_out_P1_ring   = ptr_dt_in;
    uint32_t iscore_out_P1_ring_bytes = nb_slots * M1_length_iscore_out_l_tile;
    uint32_t dt_in_region_bytes =
        M2_length_dt_BC > iscore_out_P1_ring_bytes ? M2_length_dt_BC : iscore_out_P1_ring_bytes;
    uint16_t* ptr_iscore_out_P2 = (uint16_t*)(ptr_dt_in + dt_in_region_bytes);

#define _ALIGN64(p) ((uint8_t*)(((uintptr_t)(p) + 63u) & ~(uintptr_t)63u))

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

    if (snrt_global_core_idx() == 0) init_cycle_counter();
    snrt_cluster_hw_barrier();

    // Preload Phase 1 non-tiled inputs. The isCore bias seeds the FULL P1 psum in L3 (every L-tile), then
    // the first nb_slots ring slots are loaded so the ring can start accumulating immediately.
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_oscore_in, M1_oscore_in, M1_length_oscore_in);
        snrt_dma_start_1d(ptr_iscore_out_P1_l3, M1_iscore_bias, M1_length_iscore_out);
        snrt_dma_wait_all();
        for (uint32_t s = 0; s < nb_slots; s++)
            snrt_dma_start_1d(ptr_iscore_out_P1_ring + s * M1_length_iscore_out_l_tile,
                              ptr_iscore_out_P1_l3 + s * M1_length_iscore_out_l_tile, M1_length_iscore_out_l_tile);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    uint32_t start_cycles            = 0;
    uint32_t simbacore_cycles_phase1 = 0;
    uint32_t simbacore_cycles_phase2 = 0;
    static uint32_t _p1_dma_done = 0, _p1_compute_done = 0;
    static uint32_t _p2_dma_done = 0, _p2_compute_done = 0;

    const uint32_t K_i = M1_dInner_tile / dInnerUnroll;  // K-steps per DMA tile (P1, P2 share)

    /////////////////////////////////
    //////// Phase 1 ////////////////
    /////////////////////////////////

    if (snrt_global_core_idx() == 0) {
        printf(
            "\nStarting program: Mamba main tiled iscore (L=%d, dModel=%d, nb_tiles=%d, K_i=%u, nb_l=%d, "
            "nb_slots=%d)\n\n",
            seqLen, dModel, nb_tiles, K_i, nb_l_tiles, nb_slots);
        printf("Expected L1 TCDM usage: %u KiB\n", (uint32_t)(L1_TCDM_PEAK_BYTES / 1024));

        start_cycles = snrt_mcycle();
        // Note: no requant, no transpose.
        set_streamer_phase1((uint32_t)ptr_oscore_in, (uint32_t)ptr_oscore_weight_P1[0], (uint32_t)ptr_conv_weight[0],
                            (uint32_t)ptr_conv_bias[0], (uint32_t)ptr_iscore_weight_P1[0],
                            (uint32_t)ptr_iscore_out_P1_ring, (uint32_t)ptr_conv_out_tile[0]);
        set_simbacore_csr(M28_PHASE1_NO_REQUANT, seqLen, dModel, M1_dInner_tile, dtRank, xProjDim);
    }

    snrt_cluster_hw_barrier();

    for (uint32_t i = 0; i < nb_tiles + 2; i++) {
        int buf = i % 2;

        // Compute tile i-
        if (i >= 1 && i <= nb_tiles) {
            uint32_t tile      = i - 1;
            bool is_final_tile = (tile == nb_tiles - 1);

            if (snrt_global_core_idx() == 0) {
                _set_streamer_start();
                _set_simbacore_start();
                write_csr(STREAMER_START_CSR, 0);
                write_csr(SIMBACORE_START, 0);
            }

            iscore_out_ring_loop(ptr_iscore_out_P1_ring, ptr_iscore_out_P1_l3, M1_iscore_out_l_tile_gauge_step,
                                 M1_length_iscore_out_l_tile);

            if (snrt_global_core_idx() == 0) {
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
                asm volatile("fence" ::: "memory");
                simbacore_cycles_phase1 += read_simbacore_perf_counter();
                if (i == 1) _p1_compute_done = snrt_mcycle();
            }
        }

        // Prefetch next tile's inputs after compute, so the ring loop owns the DMA engine during the
        // isCore phase (a prefetch issued before would queue ahead of the ring reloads and starve them).
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

    if (snrt_global_core_idx() == 0) printf("[%u cc] P1 done\n", snrt_mcycle());

    // Preload dt and BC golden reference from L3
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_dt_in, M1_iscore_out, M2_length_dt_BC);
        snrt_dma_start_1d((uint8_t*)ptr_iscore_out_P2, M2_iscore_bias, M2_length_iscore_out);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    /////////////////////////////////
    //////// Phase 2 ////////////////
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

        if (i >= 1 && i <= nb_tiles && snrt_global_core_idx() == 0) {
            uint32_t tile      = i - 1;
            bool is_final_tile = (tile == nb_tiles - 1);

            // Tile 0: base ptrs set by initial set_streamer_phase2. Tile > 0: preloaded in prev iter.
            if (is_final_tile) write_csr(MODE, M2_PHASE2);
            start_simbacore_and_streamers(M2_R10_en, M2_R10_start_cnt, M2_R11_en, M2_R11_start_cnt);
            write_csr(STREAMER_START_CSR, 0);
            write_csr(SIMBACORE_START, 0);
            write_csr(DELAYED_START_READER_10, 0);
            write_csr(DELAYED_START_READER_11, 0);

            // Preload next tile's base ptrs while HW computes.
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

    // --- Verification ---
    if (snrt_global_core_idx() == 0) printf("[%u cc] P2 done, starting verification\n", snrt_mcycle());

    if (snrt_global_core_idx() == 0) {
        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Simbacore Phase1 (sum over tiles): %u cycles\n", end_cycles, simbacore_cycles_phase1);
        printf("[%d cc] Simbacore Phase2 (sum over tiles): %u cycles\n", end_cycles, simbacore_cycles_phase2);
        printf("[%d cc] Simbacore elapsed time: %u cycles\n", end_cycles,
               simbacore_cycles_phase1 + simbacore_cycles_phase2);
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);
        printf("DMA latency hiding: P1=%s, P2=%s\n", _p1_dma_done < _p1_compute_done ? "ok" : "STALL",
               _p2_dma_done < _p2_compute_done ? "ok" : "STALL");

        // P1 conv_out (= P2 x)
        err += check_result_sample(ptr_conv_out_l3, M1_conv_out, M1_test_samples_conv_out,  //
                                   nb_test_samples, "P1 conv_out (= P2 x, from L3)");
        // P1 isCore psum (raw BF16, un-transposed)
        err += check_result_sample_u16((uint16_t*)ptr_iscore_out_P1_l3, M1_iscore_out_bf16_untransposed,
                                       M1_test_samples_iscore_out, nb_test_samples, "P1 psum (BF16)");

        err += check_result_sample(ptr_z_l3, M2_oscore_expected, M2_test_samples_z,  //
                                   nb_test_samples, "z (osCore out)");
        err += check_result_sample(ptr_y_l3, M2_suc_expected, M2_test_samples_y,  //
                                   nb_test_samples, "SUC y");
        err += check_result_sample((uint8_t*)ptr_iscore_out_P2, M2_iscore_expected,  //
                                   M2_test_samples_iscore_out, nb_test_samples, "iscore_out P2");

        printf("Test Phase1+Phase2 tiled-iscore: seqLen=%d, dModel=%d, dInner=%d, nb_tiles=%d, nb_l=%d, nb_slots=%d\n",
               seqLen, dModel, dInner, nb_tiles, nb_l_tiles, nb_slots);
        // Sample checks: conv_out, P1 psum, z, y, iscore_out_P2 = 5 * nb_test_samples.
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, 5 * nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() {
    int err = test_phase1_and_2();
    return err;
}
