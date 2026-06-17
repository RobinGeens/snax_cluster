// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// is-osgemm-tiled-async = osCore and isCore run concurrently (IS_OSGEMM_NO_REQUANT) with BOTH async
// rings active at once: the osCore A INPUT ring (refill) and the isCore PSUM OUTPUT ring (spill+reload).
// One DMA loop double-paces them with two independent gauge cursors (see the design doc for why).
// Design: docs/dataflow/09_async_tiling.md ("Both rings at once (dual-core, double-pacing)").

#include "data.h"
#include "snax-simbacore-lib.h"

int test_is_osgemm_tiled_async() {
    int err = 0;

    // ---- L3: full isCore psum accumulator. Reserve 16 KiB to skip putc_buffer (snrt_l3alloc overlap).
    static uint8_t* ptr_psum_l3 = NULL;
    if (snrt_global_core_idx() == 0) {
        (void)snrt_l3alloc(16 * 1024);
        ptr_psum_l3 = (uint8_t*)snrt_l3alloc(M4_length_cd);
    }
    snrt_cluster_hw_barrier();

    // ---- TCDM: [ A_os ring(nb_slots) | B_os tile | D_os full | A_is ktile | B_is ktile | psum ring(nb_slots) ]
    void* tcdm_base   = snrt_l1_next();
    uint8_t* ptr_a_os = (uint8_t*)tcdm_base;                       // osCore A ring base (slot 0)
    uint8_t* ptr_b_os = ptr_a_os + nb_slots * M3_length_a_l_tile;  // osCore B (one dInner tile)
    uint8_t* ptr_d_os = ptr_b_os + M3_length_b_tile;               // osCore D (full, resident; tile-sliced)
    uint8_t* ptr_a_is = ptr_d_os + M3_length_d;                    // isCore A (one K-step)
    uint8_t* ptr_b_is = ptr_a_is + M4_length_a_ktile;              // isCore B (one K-step)
    uint8_t* ptr_ring = ptr_b_is + M4_length_b_ktile;              // isCore psum ring base (slot 0)

    uint32_t start_cycles           = 0;
    uint32_t simbacore_cycles_total = 0;

    if (snrt_global_core_idx() == 0) {
        printf(
            "\nStarting program: IS+OSGeMM tiled async (seqLen=%d dModel=%d dInner=%d nb_l_tiles=%d nb_slots=%d "
            "nb_inv=%d L_tile=%d)\n\n",
            seqLen, dModel, dInner, nb_l_tiles, nb_slots, M4_nb_k_tiles, M4_L_tile);
        printf("Expected L1 TCDM usage: %u B (%u KiB)\n", (uint32_t)L1_TCDM_PEAK_BYTES,
               (uint32_t)(L1_TCDM_PEAK_BYTES / 1024));
        init_cycle_counter();
        start_cycles = snrt_mcycle();
        set_streamer_csr((uint32_t)ptr_a_os, M3_R0_ss, M3_R0_tb, M3_R0_ts, M3_R0_en,      // R0:  osCore A ring
                         (uint32_t)ptr_b_os, M3_R1_ss, M3_R1_tb, M3_R1_ts, M3_R1_en,      // R1:  osCore B
                         (uint32_t)0, 0, 0, 0, 0,                                         // R2:  disabled
                         (uint32_t)0, 0, 0, 0, 0,                                         // R3:  disabled
                         (uint32_t)0, 0, 0, 0, 0,                                         // R4:  disabled
                         (uint32_t)0, 0, 0, 0, 0,                                         // R5:  disabled
                         (uint32_t)0, 0, 0, 0, 0,                                         // R6:  disabled
                         (uint32_t)0, 0, 0, 0, 0,                                         // R7:  disabled
                         (uint32_t)0, 0, 0, 0, 0,                                         // R8:  disabled
                         (uint32_t)0, 0, 0, 0, 0,                                         // R9:  disabled
                         (uint32_t)0, 0, 0, 0, 0,                                         // R10: disabled
                         (uint32_t)ptr_a_is, M4_R11_ss, M4_R11_tb, M4_R11_ts, M4_R11_en,  // R11: isCore A
                         (uint32_t)ptr_b_is, M4_R12_ss, M4_R12_tb, M4_R12_ts, M4_R12_en,  // R12: isCore B
                         (uint32_t)ptr_ring, M4_R13_ss, M4_R13_tb, M4_R13_ts, M4_R13_en,  // R13: psum ring read
                         (uint32_t)ptr_d_os, M3_W0_ss, M3_W0_tb, M3_W0_ts, M3_W0_en,      // W0:  osCore D
                         (uint32_t)0, 0, 0, 0, 0,                                         // W1:  disabled
                         (uint32_t)0, 0, 0, 0, 0,                                         // W2:  disabled
                         (uint32_t)ptr_ring, M4_W3_ss, M4_W3_tb, M4_W3_ts, M4_W3_en);     // W3:  psum ring write
        set_simbacore_csr(IS_OSGEMM_NO_REQUANT, seqLen, dModel, dInnerUnroll, 1, dModel);
    }

    // Preload both rings: osCore A first nb_slots L-tiles; isCore psum bias C -> L3 then first nb_slots slots.
    if (snrt_is_dm_core()) {
        for (uint32_t s = 0; s < nb_slots; s++)
            snrt_dma_start_1d(ptr_a_os + s * M3_length_a_l_tile, M3_A + s * M3_oscore_in_l_offset, M3_length_a_l_tile);
        snrt_dma_start_1d(ptr_psum_l3, M4_C, M4_length_cd);
        snrt_dma_wait_all();
        for (uint32_t s = 0; s < nb_slots; s++)
            snrt_dma_start_1d(ptr_ring + s * M4_length_psum_l_tile, ptr_psum_l3 + s * M4_length_psum_l_tile,
                              M4_length_psum_l_tile);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    const uint32_t nb_inv  = M4_nb_k_tiles;  // = dInner / dInnerUnroll (same for both cores)
    const uint32_t os_step = M3_oscore_in_l_tile_gauge_step;
    const uint32_t is_step = M4_iscore_out_l_tile_gauge_step;

    // SW-outer loop = the isCore K reduction; each invocation is also one osCore dInner N-slice.
    for (uint32_t inv = 0; inv < nb_inv; inv++) {
        // Load this invocation's osCore B and isCore A/B (blocking) -> frees the DM core for both rings.
        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(ptr_b_os, M3_B + inv * M3_length_b_tile, M3_length_b_tile);
            snrt_dma_start_1d(ptr_a_is, M4_A + inv * M4_length_a_ktile, M4_length_a_ktile);
            snrt_dma_start_1d(ptr_b_is, M4_B + inv * M4_length_b_ktile, M4_length_b_ktile);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();

        // Kick BOTH cores (non-blocking, start_cnt=0). The combined ring loop below MUST poll the gauges
        // immediately after, with no slow op in between (see docs/dataflow/09_async_tiling.md).
        if (snrt_global_core_idx() == 0) {
            write_csr(BASE_PTR_READER_1_LOW, (uint32_t)ptr_b_os);
            write_csr(BASE_PTR_WRITER_0_LOW, (uint32_t)(ptr_d_os + inv * M3_length_d_tile));
            write_csr(BASE_PTR_READER_11_LOW, (uint32_t)ptr_a_is);
            write_csr(BASE_PTR_READER_12_LOW, (uint32_t)ptr_b_is);
            start_simbacore_and_streamers(0, 0, M4_R11_en, 0);
        }

        // Double-paced async rings: each ring has its OWN cursor advanced by its OWN gauge, so the
        // faster core's refills are never held back by the slower one (avoids the pace-on-slower
        // starvation). Core 0 spins until at least one ring has a consumed slot, publishes which are
        // ready, and the DM core refills only those — so a visit may touch one ring or both.
        uint32_t r_os = 0, r_is = 0;
        static volatile uint32_t do_os = 0, do_is = 0;
        while (r_os < nb_l_tiles || r_is < nb_l_tiles) {
            if (snrt_global_core_idx() == 0) {
                uint32_t ready_os, ready_is;
                do {
                    ready_os = (r_os < nb_l_tiles) && (read_csr(R10_DELAY_GAUGE) >= (r_os + 1) * os_step);
                    ready_is = (r_is < nb_l_tiles) && (read_csr(ISCORE_TILE_CNT) >= (r_is + 1) * is_step);
                } while (!ready_os && !ready_is);
                do_os = ready_os;
                do_is = ready_is;
            }
            snrt_cluster_hw_barrier();  // publish do_os/do_is to all cores
            if (snrt_is_dm_core()) {
                if (do_os) {
                    // osCore A ring: refill the freed slot with the next L-tile (refill only).
                    snrt_dma_start_1d(ptr_a_os + (r_os % nb_slots) * M3_length_a_l_tile,
                                      M3_A + ((r_os + nb_slots) % nb_l_tiles) * M3_oscore_in_l_offset,
                                      M3_length_a_l_tile);
                }
                if (do_is) {
                    // isCore psum ring: spill this L-tile's updated running psum to L3, then reload the next.
                    snrt_dma_start_1d(ptr_psum_l3 + r_is * M4_length_psum_l_tile,
                                      ptr_ring + (r_is % nb_slots) * M4_length_psum_l_tile, M4_length_psum_l_tile);
                    snrt_dma_start_1d(ptr_ring + (r_is % nb_slots) * M4_length_psum_l_tile,
                                      ptr_psum_l3 + ((r_is + nb_slots) % nb_l_tiles) * M4_length_psum_l_tile,
                                      M4_length_psum_l_tile);
                }
            }
            if (do_os) r_os++;
            if (do_is) r_is++;
            snrt_cluster_hw_barrier();  // all cores done reading do_*; safe for core 0 to recompute
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

    // ---- Verify osCore FP8 output (resident in TCDM) and isCore BF16 psum (accumulated in L3).
    if (snrt_global_core_idx() == 0) {
        uint32_t end_cycles = snrt_mcycle();
        printf("[%u cc] Simbacore elapsed time: %u cycles\n", end_cycles, simbacore_cycles_total);
        printf("[%u cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);
        err += check_result_sample(ptr_d_os, M3_D, M3_test_samples_D, nb_test_samples, "osgemm_out");
        err += check_result_sample_u16((uint16_t*)ptr_psum_l3, M4_D_no_requant, M4_test_samples_D_no_requant,
                                       nb_test_samples, "iscore_psum");
        printf("Test IS+OSGeMM tiled async: seqLen=%d dModel=%d dInner=%d nb_l_tiles=%d nb_slots=%d nb_inv=%d\n",
               seqLen, dModel, dInner, nb_l_tiles, nb_slots, nb_inv);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, 2 * nb_test_samples);
    }
    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test_is_osgemm_tiled_async(); }
