// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// K-tiled, accumulating, double-buffered ISGeMM. See docs/dataflow/02_iscore_kernels.md
// §2.2 for dataflow, K-vs-N tiling rationale, pipeline pattern, and the
// final-tile requant trick.

#include "data.h"
#include "snax-simbacore-lib.h"

#define NB_STAGES 2  // transfer_in, compute

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

    // Preload C (bias) into CD; CD is the FULL accumulator (R13/W3 same address every tile).
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d((uint8_t*)ptr_cd, M4_C, M4_length_cd);
        snrt_dma_wait_all();
    }

    // One-time streamer setup. R11/R12 base ptrs are rewritten per tile.
    if (snrt_global_core_idx() == 0) {
        set_isgemm_streamer_csr((uint32_t)ptr_a[0], M4_R11_ss, M4_R11_tb, M4_R11_ts, (uint32_t)ptr_b[0], M4_R12_ss,
                                M4_R12_tb, M4_R12_ts, (uint32_t)ptr_cd, M4_W3_ss, M4_W3_tb, M4_W3_ts);
        set_simbacore_csr(M4_ISGEMM, dim0, 1, M4_dInner_tile, 1, dim2);
    }

    snrt_cluster_hw_barrier();

    uint32_t start_cycles           = 0;
    uint32_t simbacore_cycles_total = 0;
    static uint32_t _dma_done = 0, _compute_done = 0;

    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: ISGeMM tiled (nb_tiles=%d)\n\n", nb_tiles);
        start_cycles = snrt_mcycle();
    }

    // Pipeline: transfer_in tile i in [0, nb_tiles), compute tile i-1 in [1, nb_tiles + 1).
    for (uint32_t i = 0; i < nb_tiles + NB_STAGES - 1; i++) {
        int buf = i % 2;

        if (i < nb_tiles) {
            if (snrt_is_dm_core()) {
                snrt_dma_start_1d(ptr_a[buf], M4_A + i * M4_length_a_tile, M4_length_a_tile);
                snrt_dma_start_1d(ptr_b[buf], M4_B + i * M4_length_b_tile, M4_length_b_tile);
            }
        }

        if (i >= 1 && i < nb_tiles + 1) {
            uint32_t tile = i - 1;
            int cbuf      = tile % 2;
            if (snrt_global_core_idx() == 0) {
                write_csr(BASE_PTR_READER_11_LOW, (uint32_t)ptr_a[cbuf]);
                write_csr(BASE_PTR_READER_12_LOW, (uint32_t)ptr_b[cbuf]);
                write_csr(MODE, (tile == nb_tiles - 1) ? M4_ISGEMM : M5_ISGEMM_NO_REQUANT);
                start_simbacore_and_streamers(M4_R10_en, 0, M4_R11_en, 0);
                wait_simbacore_and_streamer();
                simbacore_cycles_total += read_simbacore_perf_counter();
                if (i == 1) _compute_done = snrt_mcycle();
            }
        }

        if (snrt_is_dm_core()) {
            snrt_dma_wait_all();
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

        err += check_result_sample((uint8_t*)ptr_cd, M4_D, M4_test_samples_D, nb_test_samples, "out");

        printf("Test ISGeMM tiled: dim0=%d, dim1=%d, dim2=%d, nb_tiles=%d\n", dim0, dim1, dim2, nb_tiles);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test_isgemm_tiled(); }
