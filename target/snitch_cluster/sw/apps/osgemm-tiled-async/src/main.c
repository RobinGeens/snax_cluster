// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// osgemm-tiled-async = minimal single-osCore GEMM whose A input (oscore_in) is L-tiled into an
// nb_slots-slot TCDM ring and refilled asynchronously from L3 during compute, paced by R10.
// Design: docs/dataflow/09_async_tiling.md (input-side ring).

#include "data.h"
#include "snax-simbacore-lib.h"

int test_osgemm_async() {
    int err = 0;

    void* tcdm_base_ptr = snrt_l1_next();
    uint8_t* ptr_a      = (uint8_t*)tcdm_base_ptr;                // A ring base = slot 0
    uint8_t* ptr_b      = ptr_a + nb_slots * M3_length_a_l_tile;  // one B-tile (reloaded per dInner tile)
    uint8_t* ptr_d      = ptr_b + M3_length_b_tile;               // full D (osCore writes tile slices)

    uint32_t start_cycles           = 0;
    uint32_t simbacore_cycles_total = 0;

    if (snrt_global_core_idx() == 0) {
        printf(
            "\nStarting program: OSGeMM async (seqLen=%d dModel=%d dInner=%d nb_tiles=%d nb_l_tiles=%d nb_slots=%d "
            "L_tile=%d)\n\n",
            dim0, dim1, dim2, nb_tiles, nb_l_tiles, nb_slots, dim0 / nb_l_tiles);
        printf("Expected L1 TCDM usage: %u B (%u KiB)\n", (uint32_t)L1_TCDM_PEAK_BYTES,
               (uint32_t)(L1_TCDM_PEAK_BYTES / 1024));
        init_cycle_counter();
        start_cycles = snrt_mcycle();
        set_osgemm_streamer_csr((uint32_t)ptr_a, M3_R0_ss, M3_R0_tb, M3_R0_ts,   //
                                (uint32_t)ptr_b, M3_R1_ss, M3_R1_tb, M3_R1_ts,   //
                                (uint32_t)ptr_d, M3_W0_ss, M3_W0_tb, M3_W0_ts);  //
        set_simbacore_csr(M3_OSGEMM, dim0, dim1, M3_dim2_tile, 1, 1);
    }

    // Preload the first nb_slots A L-tiles into the ring (B is loaded per dInner tile below).
    if (snrt_is_dm_core()) {
        for (uint32_t s = 0; s < nb_slots; s++)
            snrt_dma_start_1d(ptr_a + s * M3_length_a_l_tile, M3_A + s * M3_oscore_in_l_offset, M3_length_a_l_tile);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    const uint32_t N_visits   = nb_l_tiles;  // one full A pass per dInner-tile invocation
    const uint32_t gauge_step = M3_oscore_in_l_tile_gauge_step;

    for (uint32_t tile = 0; tile < nb_tiles; tile++) {
        // Load this dInner tile's B (blocking) -> keeps the DM core free during the A refill.
        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(ptr_b, M3_B + tile * M3_length_b_tile, M3_length_b_tile);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();

        // Kick the osCore for this tile (non-blocking, start_cnt=0). R10 resets per invocation.
        // The refill loop below MUST poll R10 immediately after this, with no slow op in between,
        // or the osCore runs ahead and the refills land too late (see docs/dataflow/09_async_tiling.md).
        if (snrt_global_core_idx() == 0) {
            write_csr(BASE_PTR_READER_1_LOW, (uint32_t)ptr_b);
            write_csr(BASE_PTR_WRITER_0_LOW, (uint32_t)(ptr_d + tile * M3_length_d_tile));
            start_simbacore_and_streamers(M3_R10_en, 0, M3_R11_en, 0);
        }

        // Async A-ring refill during compute, paced by R10 (osCore output-tile gauge).
        for (uint32_t r = 0; r < N_visits; r++) {
            if (snrt_global_core_idx() == 0) {
                while (read_snax_csr_safe(R10_DELAY_GAUGE) < (r + 1) * gauge_step);
            }
            snrt_cluster_hw_barrier();
            if (snrt_is_dm_core()) {
                snrt_dma_start_1d(ptr_a + (r % nb_slots) * M3_length_a_l_tile,
                                  M3_A + ((r + nb_slots) % nb_l_tiles) * M3_oscore_in_l_offset, M3_length_a_l_tile);
            }
        }
        if (snrt_is_dm_core()) snrt_dma_wait_all();
        snrt_cluster_hw_barrier();

        if (snrt_global_core_idx() == 0) {
            wait_simbacore_and_streamer();
            simbacore_cycles_total += read_simbacore_perf_counter();
            asm volatile("fence" ::: "memory");
        }
        snrt_cluster_hw_barrier();
    }

    // Verify the full osCore output D against golden (FP8, +-1 LSB tolerance; signed-zero 0==128).
    if (snrt_global_core_idx() == 0) {
        uint32_t end_cycles = snrt_mcycle();
        printf("[%u cc] Simbacore elapsed time: %u cycles\n", end_cycles, simbacore_cycles_total);
        printf("[%u cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        const int16_t TOL   = 1;
        uint32_t total_fail = 0;
        for (uint32_t idx = 0; idx < M3_length_d; idx++) {
            uint8_t o = ptr_d[idx], g = M3_D[idx];
            int16_t d = (int16_t)o - (int16_t)g;
            int ok    = ((o == 0 && g == 128) || (o == 128 && g == 0)) || (d >= -TOL && d <= TOL);
            if (!ok) total_fail++;
        }
        err = (total_fail > 0);
        printf("Test OSGeMM async: seqLen=%d dModel=%d dInner=%d nb_tiles=%d nb_l_tiles=%d nb_slots=%d\n", dim0, dim1,
               dim2, nb_tiles, nb_l_tiles, nb_slots);
        printf("%s: %u / %u D elements wrong.\n", total_fail ? "FAIL" : "PASS", total_fail, M3_length_d);
    }
    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test_osgemm_async(); }
