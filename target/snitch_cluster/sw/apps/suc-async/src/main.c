// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// Isolated SUC with async tiling on BC. dt is kept in full

#include "helper.c"
#include "snax-simbacore-lib.h"

static inline void set_streamer_suc_async(uint32_t p_dt, uint32_t p_dw1, uint32_t p_dw2, uint32_t p_db, uint32_t p_A,
                                          uint32_t p_BC, uint32_t p_D, uint32_t p_x, uint32_t p_z, uint32_t p_y) {
    set_streamer_csr((uint32_t)0, 0, 0, 0, 0,                                     // R0 osCore in (disabled)
                     (uint32_t)0, 0, 0, 0, 0,                                     // R1 osCore weight (disabled)
                     p_dt, M2_R2_ss, M2_R2_tb_packed, M2_R2_ts_packed, M2_R2_en,  // R2 switchCore in (dt, PACKED)
                     p_dw1, M2_R3_ss, M2_R3_tb, M2_R3_ts, M2_R3_en,               // R3 switchCore weight 1
                     p_db, M2_R4_ss, M2_R4_tb, M2_R4_ts, M2_R4_en,                // R4 switchCore bias
                     p_dw2, M2_R5_ss, M2_R5_tb, M2_R5_ts, M2_R5_en,               // R5 switchCore weight 2
                     p_A, M2_R6_ss, M2_R6_tb, M2_R6_ts, M2_R6_en,                 // R6 SUC A
                     p_BC, M2_R7_ss, M2_R7_tb_ring, M2_R7_ts_ring, M2_R7_en,      // R7 SUC BC (async ring)
                     p_D, M2_R8_ss, M2_R8_tb, M2_R8_ts, M2_R8_en,                 // R8 SUC D
                     p_x, M2_R9_ss, M2_R9_tb, M2_R9_ts, M2_R9_en,                 // R9 SUC x
                     p_z, M2_R10_ss, M2_R10_tb, M2_R10_ts, M2_R10_en,             // R10 SUC z (preloaded)
                     p_y, M2_R11_ss, M2_R11_tb, M2_R11_ts, M2_R11_en,             // R11 iscore in = SUC y
                     (uint32_t)0, 0, 0, 0, 0,                                     // R12 isCore weight (disabled)
                     (uint32_t)0, 0, 0, 0, 0,                                     // R13 isCore psum (disabled)

                     (uint32_t)0, 0, 0, 0, 0,                      // W0 osCore out (disabled)
                     (uint32_t)0, 0, 0, 0, M2_W1_en,               // W1 disabled
                     p_y, M2_W2_ss, M2_W2_tb, M2_W2_ts, M2_W2_en,  // W2 SUC y out
                     (uint32_t)0, 0, 0, 0, 0                       // W3 isCore out (disabled)
    );
}

// BC ring refill for one dInner-tile invocation, paced by SUC output elements CSR
static inline void bc_ring_refill(uint8_t* bc_slot_base, const uint8_t* l3_dt_BC) {
    const uint32_t bc_visits = M2_BC_n_visits;
    const uint32_t bc_step   = M2_BC_gauge_step;

    for (uint32_t r = 0; r < bc_visits; r++) {
        if (snrt_global_core_idx() == 0)
            while (read_csr(R11_DELAY_GAUGE) < (r + 1) * bc_step);

        snrt_cluster_hw_barrier();
        if (snrt_is_dm_core()) {
            uint32_t lt = (r + nb_slots) % nb_l_tiles;
            snrt_dma_start_2d(bc_slot_base + (r % nb_slots) * M2_length_BC_l_tile,
                              l3_dt_BC + lt * M2_BC_windows_per_l_tile * M2_dtBC_window_src_stride + M2_BC_l3_offset,
                              M2_BC_window_bytes, M2_BC_window_bytes, M2_dtBC_window_src_stride,
                              M2_BC_windows_per_l_tile);
        }
    }

    if (snrt_is_dm_core()) snrt_dma_wait_all();
    snrt_cluster_hw_barrier();
}

int main() {
    int err = 0;

    // L3 staging for the full y output (reserve 16 KiB first to skip putc_buffer).
    static uint8_t* ptr_y_l3 = NULL;
    if (snrt_global_core_idx() == 0) {
        (void)snrt_l3alloc(16 * 1024);
        ptr_y_l3 = (uint8_t*)snrt_l3alloc(M2_length_y);
    }
    snrt_cluster_hw_barrier();

    void* tcdm_base_ptr = snrt_l1_next();
#define _ALIGN64(p) ((uint8_t*)(((uintptr_t)(p) + 63u) & ~(uintptr_t)63u))

    // Shared (L-only) across all dInner tiles: switchCore dt PACKED full + BC nb_slots-slot async ring,
    // both extracted from the combined dt_BC L3 buffer ([dt|BC] per window). Per-dInner-tile single
    // buffers (reloaded each invocation): dt weights/bias, A, D, x, z, y.
    uint8_t* ptr_dt_packed = (uint8_t*)tcdm_base_ptr;
    uint8_t* ptr_BC_ring   = _ALIGN64(ptr_dt_packed + M2_length_dt_packed);
    uint8_t* ptr_dt_w1     = _ALIGN64(ptr_BC_ring + nb_slots * M2_length_BC_l_tile);
    uint8_t* ptr_dt_w2     = _ALIGN64(ptr_dt_w1 + M2_length_dt_weight_1_tile);
    uint8_t* ptr_dt_bias   = _ALIGN64(ptr_dt_w2 + M2_length_dt_weight_2_tile);
    uint8_t* ptr_A         = _ALIGN64(ptr_dt_bias + M2_length_dt_bias_tile);
    uint8_t* ptr_D         = _ALIGN64(ptr_A + M2_length_A_tile);
    uint8_t* ptr_x         = _ALIGN64(ptr_D + M2_length_D_tile);
    uint8_t* ptr_z         = _ALIGN64(ptr_x + M2_length_x_tile);
    uint8_t* ptr_y         = _ALIGN64(ptr_z + M2_length_z_tile);
#undef _ALIGN64

    uint32_t start_cycles     = 0;
    uint32_t simbacore_cycles = 0;

    if (snrt_global_core_idx() == 0) {
        printf(
            "\nStarting program: suc-async (L=%d, dModel=%d, dInner=%d, nb_tiles=%d, nb_l_tiles=%d, nb_slots=%d, "
            "dInner_tile=%u, L_tile=%u)\n\n",
            seqLen, dModel, dInner, nb_tiles, nb_l_tiles, nb_slots, M2_dInner_tile, L_tile);
        init_cycle_counter();
        start_cycles = snrt_mcycle();
        set_streamer_suc_async((uint32_t)ptr_dt_packed, (uint32_t)ptr_dt_w1, (uint32_t)ptr_dt_w2, (uint32_t)ptr_dt_bias,
                               (uint32_t)ptr_A, (uint32_t)ptr_BC_ring, (uint32_t)ptr_D, (uint32_t)ptr_x,
                               (uint32_t)ptr_z, (uint32_t)ptr_y);
        set_simbacore_csr(M27_SUC_ONLY, seqLen, dModel, M2_dInner_tile, dtRank, dModel);
    }

    // Preload the shared, L-only inputs once: packed dt (all L windows) + first nb_slots BC L-tiles.
    if (snrt_is_dm_core()) {
        snrt_dma_start_2d(ptr_dt_packed, M2_dt_BC, M2_dt_pack_window_bytes, M2_dt_pack_window_bytes,
                          M2_dtBC_window_src_stride, M2_dt_windows_total);
        for (uint32_t s = 0; s < nb_slots; s++)
            snrt_dma_start_2d(ptr_BC_ring + s * M2_length_BC_l_tile,
                              M2_dt_BC + s * M2_BC_windows_per_l_tile * M2_dtBC_window_src_stride + M2_BC_l3_offset,
                              M2_BC_window_bytes, M2_BC_window_bytes, M2_dtBC_window_src_stride,
                              M2_BC_windows_per_l_tile);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    for (uint32_t tile = 0; tile < nb_tiles; tile++) {
        // Stage I: (re)load this dInner tile's per-tile inputs (incl. preloaded z) from golden, blocking,
        // so the DM core is then free for the BC ring during compute.
        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(ptr_dt_w1, M2_dt_weight_1 + tile * M2_length_dt_weight_1_tile,
                              M2_length_dt_weight_1_tile);
            snrt_dma_start_1d(ptr_dt_w2, M2_dt_weight_2 + tile * M2_length_dt_weight_2_tile,
                              M2_length_dt_weight_2_tile);
            snrt_dma_start_1d(ptr_dt_bias, M2_dt_bias + tile * M2_length_dt_bias_tile, M2_length_dt_bias_tile);
            snrt_dma_start_1d(ptr_A, M2_suc_A + tile * M2_length_A_tile, M2_length_A_tile);
            snrt_dma_start_1d(ptr_D, M2_suc_D + tile * M2_length_D_tile, M2_length_D_tile);
            snrt_dma_start_1d(ptr_x, M2_suc_x + tile * M2_length_x_tile, M2_length_x_tile);
            snrt_dma_start_1d(ptr_z, M2_oscore_expected + tile * M2_length_z_tile, M2_length_z_tile);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();

        // Stage II: kick the SUC (non-blocking, R10 delayed-start released immediately since z is resident)
        // so the BC refill loop can run during compute. No slow op between start and the first gauge poll.
        if (snrt_global_core_idx() == 0) start_simbacore_and_streamers(M2_R10_en, 0, 0, 0);
        snrt_cluster_hw_barrier();

        bc_ring_refill(ptr_BC_ring, M2_dt_BC);

        if (snrt_global_core_idx() == 0) {
            wait_simbacore_and_streamer();
            asm volatile("fence" ::: "memory");
            simbacore_cycles += read_simbacore_perf_counter();
        }
        snrt_cluster_hw_barrier();

        // Stage III: spill this dInner tile's y to L3.
        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(ptr_y_l3 + tile * M2_length_y_tile, ptr_y, M2_length_y_tile);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();
    }

    if (snrt_global_core_idx() == 0) {
        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Simbacore elapsed time: %u cycles\n", end_cycles, simbacore_cycles);
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        err += check_result_sample(ptr_y_l3, M2_suc_expected, M2_test_samples_y, nb_test_samples, "SUC y (from L3)");

        uint32_t y_bad = 0, y_first = 0xffffffff;
        for (uint32_t k = 0; k < M2_length_y; k++) {
            int d = (int)ptr_y_l3[k] - (int)M2_suc_expected[k];
            if (d < -1 || d > 1) {
                if (y_first == 0xffffffff) y_first = k;
                y_bad++;
            }
        }
        printf("FULLCHK y: %u/%u bad (first@%d)\n", y_bad, M2_length_y, (int)y_first);

        printf("Test suc-async: %s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, nb_test_samples);
    }
    snrt_cluster_hw_barrier();
    return err;
}
