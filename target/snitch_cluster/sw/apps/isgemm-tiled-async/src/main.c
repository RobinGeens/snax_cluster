// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// IS-GEMM with async tiling on the psum output.

#include "data.h"
#include "snax-simbacore-lib.h"

int test_isgemm_tiled_async() {
    int err = 0;

    // ---- L3: full psum accumulator. Reserve 16 KiB to skip putc_buffer (snrt_l3alloc overlap).
    static uint8_t* ptr_psum_l3 = NULL;
    if (snrt_global_core_idx() == 0) {
        (void)snrt_l3alloc(16 * 1024);
        ptr_psum_l3 = (uint8_t*)snrt_l3alloc(M4_length_cd);
    }
    snrt_cluster_hw_barrier();

    // ---- TCDM: [ A_k(2) | B_k(2) | psum ring (nb_slots adjacent slots) ]
    void* tcdm_base   = snrt_l1_next();
    uint8_t* ptr_a[2] = {(uint8_t*)tcdm_base, (uint8_t*)tcdm_base + M4_length_a_ktile};
    uint8_t* ptr_b[2] = {ptr_a[1] + M4_length_a_ktile, ptr_a[1] + M4_length_a_ktile + M4_length_b_ktile};
    uint8_t* ptr_ring = ptr_b[1] + M4_length_b_ktile;  // slot s = ptr_ring + s * M4_length_psum_l_tile

    if (snrt_global_core_idx() == 0) init_cycle_counter();
    snrt_cluster_hw_barrier();

    // Init the full psum in L3 with the bias C, then preload the first nb_slots ring slots from it.
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_psum_l3, M4_C, M4_length_cd);
        snrt_dma_wait_all();
        for (uint32_t s = 0; s < nb_slots; s++)
            snrt_dma_start_1d(ptr_ring + s * M4_length_psum_l_tile, ptr_psum_l3 + s * M4_length_psum_l_tile,
                              M4_length_psum_l_tile);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    // One-time streamer + simbacore config. R13/W3 base = ring base (fixed; the wrap walks the
    // slots). R11 (A) / R12 (B) base ptrs are rewritten per K-step.
    if (snrt_global_core_idx() == 0) {
        set_isgemm_streamer_csr((uint32_t)ptr_a[0], M4_R11_ss, M4_R11_tb, M4_R11_ts,  // A
                                (uint32_t)ptr_b[0], M4_R12_ss, M4_R12_tb, M4_R12_ts,  // B
                                (uint32_t)ptr_ring, M4_W3_ss, M4_W3_tb, M4_W3_ts);    // psum ring (R13 + W3)
        set_simbacore_csr(M5_ISGEMM_NO_REQUANT, dim0, 1, dInnerUnroll, 1, dim2);
        printf(
            "\nStarting program: ISGeMM tiled async (seqLen=%d dInner=%d dModel=%d nb_l_tiles=%d nb_slots=%d "
            "nb_k_tiles=%d L_tile=%d)\n\n",
            dim0, dim1, dim2, nb_l_tiles, nb_slots, M4_nb_k_tiles, M4_L_tile);
    }
    snrt_cluster_hw_barrier();

    const uint32_t gauge_step = M4_iscore_out_l_tile_gauge_step;

    uint32_t start_cycles           = 0;
    uint32_t simbacore_cycles_total = 0;
    if (snrt_global_core_idx() == 0) start_cycles = snrt_mcycle();

    // K reduction = SW-outer loop over the nb_k_tiles single-K-step invocations.
    for (uint32_t k = 0; k < M4_nb_k_tiles; k++) {
        int buf = k % 2;

        // Load this K-step's A and B (blocking) -> the DM core is free for the psum ring during compute.
        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(ptr_a[buf], M4_A + k * M4_length_a_ktile, M4_length_a_ktile);
            snrt_dma_start_1d(ptr_b[buf], M4_B + k * M4_length_b_ktile, M4_length_b_ktile);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();

        // Kick the isCore for this K-step (non-blocking, start_cnt=0). The ring loop below MUST
        // poll the gauge immediately after, with no slow op in between, or the isCore runs ahead
        // and the spills/reloads land too late (see osgemm-tiled-async / main-tiled-oscore).
        if (snrt_global_core_idx() == 0) {
            write_csr(BASE_PTR_READER_11_LOW, (uint32_t)ptr_a[buf]);
            write_csr(BASE_PTR_READER_12_LOW, (uint32_t)ptr_b[buf]);
            start_simbacore_and_streamers(M4_R10_en, 0, M4_R11_en, 0);
        }

        // Async psum ring: spill the just-updated running psum and reload the next L-tile's psum,
        // paced by ISCORE_TILE_CNT (gauge_step ticks per L-tile).
        for (uint32_t r = 0; r < nb_l_tiles; r++) {
            if (snrt_global_core_idx() == 0) {
                while (read_csr(ISCORE_TILE_CNT) < (r + 1) * gauge_step);
            }
            snrt_cluster_hw_barrier();
            if (snrt_is_dm_core()) {
                uint32_t slot = r % nb_slots;
                // (1) spill L-tile r's updated running psum to L3
                snrt_dma_start_1d(ptr_psum_l3 + r * M4_length_psum_l_tile, ptr_ring + slot * M4_length_psum_l_tile,
                                  M4_length_psum_l_tile);
                // (2) reload the next L-tile's running psum into the freed slot. The (r+nb_slots)
                // wrap pre-loads slots 0..nb_slots-1 at the tail for the next K-step invocation.
                snrt_dma_start_1d(ptr_ring + slot * M4_length_psum_l_tile,
                                  ptr_psum_l3 + ((r + nb_slots) % nb_l_tiles) * M4_length_psum_l_tile,
                                  M4_length_psum_l_tile);
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

    // ---- Verify the full accumulated BF16 psum (now in L3) against golden D_no_requant.
    if (snrt_global_core_idx() == 0) {
        uint32_t end_cycles = snrt_mcycle();
        printf("[%u cc] Simbacore elapsed time: %u cycles\n", end_cycles, simbacore_cycles_total);
        printf("[%u cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);
        err += check_result_sample_u16((uint16_t*)ptr_psum_l3, M4_D_no_requant, M4_test_samples_D_no_requant,
                                       nb_test_samples, "psum (no-requant)");
        printf("Test ISGeMM tiled async: dim0=%d dim1=%d dim2=%d nb_l_tiles=%d nb_slots=%d nb_k_tiles=%d\n", dim0, dim1,
               dim2, nb_l_tiles, nb_slots, M4_nb_k_tiles);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, nb_test_samples);
    }
    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test_isgemm_tiled_async(); }
