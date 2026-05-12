// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// Tiled, double-buffered, two-stage pipelined version of ISGeMM.
//
// K-tile (accumulating): each tile invokes the iscore on a K_tile-sized slice of the
// dInner (= IS-core reduction) axis and accumulates into the shared CD buffer in TCDM
// via R13 (read) -> W3 (write) hitting the same address. The C input for tile 0 is the
// bias preloaded into the CD buffer; thereafter R13 reads the running BF16 psum from
// the previous tile's W3.
//
// Non-final tiles run in M5_ISGEMM_NO_REQUANT so the psum stays in BF16. The final tile
// switches to M4_ISGEMM, applying the FP8 requant to produce the final output (which
// occupies the low byte of each BF16 slot, with high bits zeroed - matches `M4_D`).
//
// Pipeline (2 stages; the compute stages are serial because they share the CD
// accumulator, so we only overlap DMA-in with compute):
//   stage 1 : DMA L3 -> L1   (transfer_in A-tile and B-tile, ping/pong)
//   stage 2 : compute        (isCore: D := C_running + A_tile * B_tile)

#include "data.h"
#include "snax-simbacore-lib.h"

// Number of pipeline stages (transfer_in, compute). Do not change.
#define NB_STAGES 2

int test_isgemm_tiled() {
    int err = 0;

    // TCDM layout: [ A0 | A1 | B0 | B1 | CD ]
    void* tcdm_base_ptr = snrt_l1_next();
    uint8_t* ptr_a[2]   = {
        (uint8_t*)tcdm_base_ptr,
        (uint8_t*)tcdm_base_ptr + M4_length_a_tile,
    };
    uint8_t* ptr_b[2] = {
        ptr_a[1] + M4_length_a_tile,
        ptr_a[1] + M4_length_a_tile + M4_length_b_tile,
    };
    uint16_t* ptr_cd = (uint16_t*)(ptr_b[1] + M4_length_b_tile);

    if (snrt_global_core_idx() == 0) init_cycle_counter();
    snrt_cluster_hw_barrier();

    // Preload C (bias) into the CD buffer. CD is shared across all tile invocations and
    // accumulates the partial sums in place.
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d((uint8_t*)ptr_cd, M4_C, M4_length_cd);
        snrt_dma_wait_all();
    }

    // One-time streamer setup. R13/W3 base ptrs and bounds are constant across tiles
    // (FULL output, accumulates in place). R11/R12 use tile 0's ping-buffer pointers as
    // a placeholder; their base ptrs are rewritten in each compute stage.
    if (snrt_global_core_idx() == 0) {
        set_isgemm_streamer_csr((uint32_t)ptr_a[0], M4_R11_ss, M4_R11_tb, M4_R11_ts,  // A
                                (uint32_t)ptr_b[0], M4_R12_ss, M4_R12_tb, M4_R12_ts,  // B
                                (uint32_t)ptr_cd, M4_W3_ss, M4_W3_tb, M4_W3_ts);      // C/D
        set_simbacore_csr(M4_ISGEMM, dim0, 1, M4_dInner_tile, 1, dim2);
    }

    snrt_cluster_hw_barrier();

    uint32_t start_cycles           = 0;
    uint32_t simbacore_cycles_total = 0;

    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: ISGeMM tiled (nb_tiles=%d)\n\n", nb_tiles);
        start_cycles = snrt_mcycle();
    }

    // -------------------------------------------------------------------------
    // Pipelined loop: nb_tiles + NB_STAGES - 1 iterations.
    //
    //   i in [0, nb_tiles)       -> transfer_in tile i        (DM core)
    //   i in [1, nb_tiles + 1)   -> compute     tile i-1      (core 0)
    // -------------------------------------------------------------------------
    for (uint32_t i = 0; i < nb_tiles + NB_STAGES - 1; i++) {
        int buf = i % 2;

        // Stage 1: transfer_in tile i (A-tile + B-tile) into ping-pong
        if (i < nb_tiles) {
            if (snrt_is_dm_core()) {
                snrt_dma_start_1d(ptr_a[buf], M4_A + i * M4_length_a_tile, M4_length_a_tile);
                snrt_dma_start_1d(ptr_b[buf], M4_B + i * M4_length_b_tile, M4_length_b_tile);
            }
        }

        // Stage 2: compute tile i-1
        if (i >= 1 && i < nb_tiles + 1) {
            uint32_t tile = i - 1;
            int cbuf      = tile % 2;
            if (snrt_global_core_idx() == 0) {
#ifdef VERBOSE
                printf("[%d cc] Compute tile %u (buf %d)\n", snrt_mcycle(), tile, cbuf);
#endif
                // Only A and B base pointers vary between tiles; everything else (bounds,
                // strides, CD pointer, simbacore CSRs) is constant and was set once.
                write_csr(BASE_PTR_READER_11_LOW, (uint32_t)ptr_a[cbuf]);
                write_csr(BASE_PTR_READER_12_LOW, (uint32_t)ptr_b[cbuf]);
                // Non-final tiles must keep the psum in BF16 (no requant) so the next
                // tile's R13 can read it back. Only the final tile applies the FP8
                // requant to produce the result.
                write_csr(MODE, (tile == nb_tiles - 1) ? M4_ISGEMM : M5_ISGEMM_NO_REQUANT);
                start_simbacore_and_streamers(M4_R10_en, 0, M4_R11_en, 0);
                wait_simbacore_and_streamer();
                simbacore_cycles_total += read_simbacore_perf_counter();
            }
        }

        // sync(): wait for all DMAs of this iteration, then cluster barrier.
        if (snrt_is_dm_core()) snrt_dma_wait_all();
        snrt_cluster_hw_barrier();
    }

    // Verify and report
    if (snrt_global_core_idx() == 0) {
        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Simbacore total elapsed time (sum over tiles): %u cycles\n", end_cycles,
               simbacore_cycles_total);
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        err += check_result_sample((uint8_t*)ptr_cd, M4_D, M4_test_samples_D, nb_test_samples, "out");

        printf("Test ISGeMM tiled: dim0=%d, dim1=%d, dim2=%d, nb_tiles=%d\n", dim0, dim1, dim2, nb_tiles);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test_isgemm_tiled(); }
