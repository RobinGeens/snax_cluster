// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// Tiled, double-buffered, three-stage pipelined version of OSGeMM.
//
// Pipeline:
//   stage 1 : DMA L3  -> L1   (transfer_in  B-tile)
//   stage 2 : compute         (osCore: A * B-tile -> D-tile)
//   stage 3 : DMA L1  -> L1   (transfer_out D-tile into the contiguous result buffer)
//
// Tile in dInner dimension (dim2). Input A is loaded once and shared across tiles. B and D are double-buffered and
// ping-ponged. The transfer_out destination is a contiguous TCDM buffer (sized at runtime from M3_length_d) instead
// of L3, since M3_D in L3 is the read-only golden reference. The pipelined stages and synchronization pattern are
// identical to an L1<->L3 deployment.

#include "data.h"
#include "snax-simbacore-lib.h"

// Number of pipeline stages (transfer_in, compute, transfer_out). This is a property of the algorithm, do not change
#define NB_STAGES 3

int test_osgemm_tiled() {
    int err = 0;

    // TCDM allocation: [ A (full) | B0 | B1 | D0 | D1 | D_full ]
    void* tcdm_base_ptr = snrt_l1_next();
    uint8_t* ptr_a      = (uint8_t*)tcdm_base_ptr;
    uint8_t* ptr_b[2]   = {
        ptr_a + M3_length_a,                     // Ping
        ptr_a + M3_length_a + M3_length_b_tile,  // Pong
    };
    uint8_t* ptr_d[2] = {
        ptr_b[1] + M3_length_b_tile,                     // Ping
        ptr_b[1] + M3_length_b_tile + M3_length_d_tile,  // Pong
    };
    uint8_t* ptr_d_out = ptr_d[1] + M3_length_d_tile;  // emulate off-chip transfer out(sized M3_length_d)

    if (snrt_global_core_idx() == 0) init_cycle_counter();
    snrt_cluster_hw_barrier();

    // Preload A: shared across all tiles, loaded once.
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_a, M3_A, M3_length_a);
        snrt_dma_wait_all();
    }

    // One-time streamer setup
    if (snrt_global_core_idx() == 0) {
        set_osgemm_streamer_csr((uint32_t)ptr_a, M3_R0_ss, M3_R0_tb, M3_R0_ts,      //
                                (uint32_t)ptr_b[0], M3_R1_ss, M3_R1_tb, M3_R1_ts,   //
                                (uint32_t)ptr_d[0], M3_W0_ss, M3_W0_tb, M3_W0_ts);  //
        set_simbacore_csr(M3_OSGEMM, dim0, dim1, M3_dim2_tile, 1, 1);
    }

    snrt_cluster_hw_barrier();

    uint32_t start_cycles           = 0;
    uint32_t simbacore_cycles_total = 0;
    static uint32_t _dma_done = 0, _compute_done = 0;

    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: OSGeMM tiled (nb_tiles=%d)\n\n", nb_tiles);
        start_cycles = snrt_mcycle();
    }

    // -------------------------------------------------------------------------
    // Pipelined loop: nb_tiles + NB_STAGES - 1 iterations.
    //
    //   i in [0, nb_tiles)       -> transfer_in tile i        (DM core)
    //   i in [1, nb_tiles + 1)   -> compute     tile i-1      (core 0)
    //   i in [2, nb_tiles + 2)   -> transfer_out tile i-2     (DM core)
    // -------------------------------------------------------------------------
    for (uint32_t i = 0; i < nb_tiles + NB_STAGES - 1; i++) {
        // Stage 1: transfer_in B-tile i to ping pong
        int buf = i % 2;
        if (i < nb_tiles) {
            if (snrt_is_dm_core()) {
                snrt_dma_start_1d(ptr_b[buf], M3_B + i * M3_length_b_tile, M3_length_b_tile);
            }
        }

        // Stage 2: compute tile i-1
        if (i >= 1 && i < nb_tiles + 1) {
            uint32_t tile = i - 1;
            int buf       = tile % 2;

            if (snrt_global_core_idx() == 0) {
                // Only the B and D base pointers vary between tiles; bounds, strides, A's pointer and the simbacore
                // CSRs are constant and were configured once before the loop.
                write_csr(BASE_PTR_READER_1_LOW, (uint32_t)ptr_b[buf]);
                write_csr(BASE_PTR_WRITER_0_LOW, (uint32_t)ptr_d[buf]);
                start_simbacore_and_streamers(M3_R10_en, 0, M3_R11_en, 0);
                wait_simbacore_and_streamer();
                simbacore_cycles_total += read_simbacore_perf_counter();

                // First iteration: check if time(DMA) < time(compute) for latency hiding
                if (i == 1) _compute_done = snrt_mcycle();
            }
        }

        // Stage 3: transfer_out D-tile i-2  ->  ptr_d_out + tile_offset
        if (i >= 2) {
            uint32_t tile = i - 2;
            int buf       = tile % 2;
            if (snrt_is_dm_core()) {
                snrt_dma_start_1d(ptr_d_out + tile * M3_length_d_tile, ptr_d[buf], M3_length_d_tile);
            }
        }

        // sync(): wait for all DMAs of this iteration, then cluster barrier.
        if (snrt_is_dm_core()) {
            snrt_dma_wait_all();
            // First iteration: check if time(DMA) < time(compute) for latency hiding
            if (i == 1) _dma_done = snrt_mcycle();
        }

        snrt_cluster_hw_barrier();
    }

    // --- Verification ---
    if (snrt_global_core_idx() == 0) {
        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Simbacore elapsed time: %u cycles\n", end_cycles, simbacore_cycles_total);
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);
        printf("DMA latency hiding: tile=%s\n", _dma_done < _compute_done ? "hidden" : "STALL");

        err += check_result_sample(ptr_d_out, M3_D, M3_test_samples_D, nb_test_samples, "out");

        printf("Test OSGeMM tiled: dim0=%d, dim1=%d, dim2=%d, nb_tiles=%d\n", dim0, dim1, dim2, nb_tiles);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test_osgemm_tiled(); }
