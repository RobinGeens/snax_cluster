// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// Tiled, double-buffered parallel OSGEMM + ISGEMM.
// Both cores share a single simbacore kernel call per tile. Tiles along dInner:
//   OSGEMM: dInner is the output N dimension (each tile = different columns)
//   ISGEMM: dInner is the reduction K dimension (each tile accumulates into CD)
//
// Pipeline (3 stages):
//   stage 1 : DMA transfer_in  (B_os tile, A_is tile, B_is tile)
//   stage 2 : compute          (oscore + iscore in parallel)
//   stage 3 : DMA transfer_out (D_os tile -> L3)

#include "data.h"
#include "snax-simbacore-lib.h"

#define NB_STAGES 3

int test_iosgemm_tiled() {
    int err = 0;

    static uint8_t* l3_d_os = NULL;
    if (snrt_global_core_idx() == 0) {
        (void)snrt_l3alloc(16 * 1024);  // reserve putc_buffer region
        l3_d_os = (uint8_t*)snrt_l3alloc(M3_length_d);
    }
    snrt_cluster_hw_barrier();

    void* tcdm_base_ptr = snrt_l1_next();

    // OSGEMM buffers: A_os shared across tiles, B_os and D_os ping-ponged
    uint8_t* ptr_a_os    = (uint8_t*)tcdm_base_ptr;
    uint8_t* ptr_b_os[2] = {
        ptr_a_os + M3_length_a,
        ptr_a_os + M3_length_a + M3_length_b_tile,
    };
    uint8_t* ptr_d_os[2] = {
        ptr_b_os[1] + M3_length_b_tile,
        ptr_b_os[1] + M3_length_b_tile + M3_length_d_tile,
    };

    // ISGEMM buffers: A_is and B_is ping-ponged, CD_is full (accumulator)
    uint8_t* ptr_a_is[2] = {
        ptr_d_os[1] + M3_length_d_tile,
        ptr_d_os[1] + M3_length_d_tile + M4_length_a_tile,
    };
    uint8_t* ptr_b_is[2] = {
        ptr_a_is[1] + M4_length_a_tile,
        ptr_a_is[1] + M4_length_a_tile + M4_length_b_tile,
    };
    uint16_t* ptr_cd_is = (uint16_t*)(ptr_b_is[1] + M4_length_b_tile);

    if (snrt_global_core_idx() == 0) init_cycle_counter();
    snrt_cluster_hw_barrier();

    // Preload: OSGEMM A (shared across all tiles) and ISGEMM bias (into CD accumulator)
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_a_os, M3_A, M3_length_a);
        snrt_dma_start_1d((uint8_t*)ptr_cd_is, M4_C, M4_length_cd);
        snrt_dma_wait_all();
    }

    // One-time streamer setup: OSGEMM uses R0/R1/W0, ISGEMM uses R11/R12/R13/W3
    if (snrt_global_core_idx() == 0) {
        set_streamer_csr((uint32_t)ptr_a_os, M3_R0_ss, M3_R0_tb, M3_R0_ts, M3_R0_en,         // R0: oscore A
                         (uint32_t)ptr_b_os[0], M3_R1_ss, M3_R1_tb, M3_R1_ts, M3_R1_en,      // R1: oscore B
                         (uint32_t)0, 0, 0, 0, 0,                                            // R2: disabled
                         (uint32_t)0, 0, 0, 0, 0,                                            // R3: disabled
                         (uint32_t)0, 0, 0, 0, 0,                                            // R4: disabled
                         (uint32_t)0, 0, 0, 0, 0,                                            // R5: disabled
                         (uint32_t)0, 0, 0, 0, 0,                                            // R6: disabled
                         (uint32_t)0, 0, 0, 0, 0,                                            // R7: disabled
                         (uint32_t)0, 0, 0, 0, 0,                                            // R8: disabled
                         (uint32_t)0, 0, 0, 0, 0,                                            // R9: disabled
                         (uint32_t)0, 0, 0, 0, 0,                                            // R10: disabled
                         (uint32_t)ptr_a_is[0], M4_R11_ss, M4_R11_tb, M4_R11_ts, M4_R11_en,  // R11: iscore A
                         (uint32_t)ptr_b_is[0], M4_R12_ss, M4_R12_tb, M4_R12_ts, M4_R12_en,  // R12: iscore B
                         (uint32_t)ptr_cd_is, M4_R13_ss, M4_R13_tb, M4_R13_ts, M4_R13_en,    // R13: iscore CD
                         (uint32_t)ptr_d_os[0], M3_W0_ss, M3_W0_tb, M3_W0_ts, M3_W0_en,      // W0: oscore D
                         (uint32_t)0, 0, 0, 0, 0,                                            // W1: disabled
                         (uint32_t)0, 0, 0, 0, 0,                                            // W2: disabled
                         (uint32_t)ptr_cd_is, M4_W3_ss, M4_W3_tb, M4_W3_ts, M4_W3_en         // W3: iscore CD
        );
        set_simbacore_csr(IS_OSGEMM_NO_REQUANT, seqLen, dModel, dInner_tile, 1, dModel);
    }

    snrt_cluster_hw_barrier();

    uint32_t start_cycles           = 0;
    uint32_t simbacore_cycles_total = 0;
    static uint32_t _dma_done = 0, _compute_done = 0;

    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: IS+OSGeMM tiled (nb_tiles=%d)\n\n", nb_tiles);
        printf("Expected L1 TCDM usage: %u B (%u KiB)\n", (uint32_t)L1_TCDM_PEAK_BYTES,
               (uint32_t)(L1_TCDM_PEAK_BYTES / 1024));
        start_cycles = snrt_mcycle();
    }

    // Pipelined loop: nb_tiles + NB_STAGES - 1 iterations.
    //   i in [0, nb_tiles)       -> transfer_in tile i        (DM core)
    //   i in [1, nb_tiles + 1)   -> compute     tile i-1      (core 0)
    //   i in [2, nb_tiles + 2)   -> transfer_out tile i-2     (DM core)
    for (uint32_t i = 0; i < nb_tiles + NB_STAGES - 1; i++) {
        int buf = i % 2;

        // Stage 1: transfer_in B_os tile, A_is tile, B_is tile
        if (i < nb_tiles) {
            if (snrt_is_dm_core()) {
                snrt_dma_start_1d(ptr_b_os[buf], M3_B + i * M3_length_b_tile, M3_length_b_tile);
                snrt_dma_start_1d(ptr_a_is[buf], M4_A + i * M4_length_a_tile, M4_length_a_tile);
                snrt_dma_start_1d(ptr_b_is[buf], M4_B + i * M4_length_b_tile, M4_length_b_tile);
            }
        }

        // Stage 2: compute tile i-1 (both cores in parallel).
        if (i >= 1 && i < nb_tiles + 1) {
            uint32_t tile = i - 1;
            if (snrt_global_core_idx() == 0) {
                write_csr(MODE, (tile == nb_tiles - 1) ? IS_OSGEMM : IS_OSGEMM_NO_REQUANT);
                start_simbacore_and_streamers(0, 0, M4_R11_en, 0);
                write_csr(STREAMER_START_CSR, 0);
                write_csr(SIMBACORE_START, 0);
                write_csr(DELAYED_START_READER_10, 0);
                write_csr(DELAYED_START_READER_11, 0);

                // CSR pre-load
                if (tile < nb_tiles - 1) {
                    int nbuf = (tile + 1) % 2;
                    write_csr(BASE_PTR_READER_1_LOW, (uint32_t)ptr_b_os[nbuf]);
                    write_csr(BASE_PTR_WRITER_0_LOW, (uint32_t)ptr_d_os[nbuf]);
                    write_csr(BASE_PTR_READER_11_LOW, (uint32_t)ptr_a_is[nbuf]);
                    write_csr(BASE_PTR_READER_12_LOW, (uint32_t)ptr_b_is[nbuf]);
                }
                while (read_csr(SIMBACORE_BUSY));
                while (read_csr(STREAMER_BUSY_CSR));
                simbacore_cycles_total += read_simbacore_perf_counter();
                if (i == 1) _compute_done = snrt_mcycle();
            }
        }

        // Stage 3: spill D_os tile i-2 to L3 (write-only, not consumed on-chip)
        if (i >= 2) {
            uint32_t tile = i - 2;
            int sbuf      = tile % 2;
            if (snrt_is_dm_core()) {
                snrt_dma_start_1d(l3_d_os + tile * M3_length_d_tile, ptr_d_os[sbuf], M3_length_d_tile);
            }
        }

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
        printf("DMA latency hiding: tile=%s\n", _dma_done < _compute_done ? "ok" : "STALL");
        err += check_result_sample(l3_d_os, M3_D, M3_test_samples_D, nb_test_samples, "osgemm_out");
        err += check_result_sample((uint8_t*)ptr_cd_is, M4_D, M4_test_samples_D, nb_test_samples, "isgemm_out");

        printf("Test IS+OSGeMM tiled: seqLen=%d, dModel=%d, dInner=%d, nb_tiles=%d\n", seqLen, dModel, dInner,
               nb_tiles);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, 2 * nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test_iosgemm_tiled(); }
