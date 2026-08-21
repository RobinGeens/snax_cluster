// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// Isolated SUC with async tiling on dt, BC, x, z and y. dt uses a coarser ring (refilled every
// dt_group visits). See 12_suc_async.md (09_async_tiling.md for the ring working principle).

#include "helper.c"
#include "snax-simbacore-lib.h"

static inline void set_streamer_suc_async(uint32_t p_dt, uint32_t p_dw1, uint32_t p_dw2, uint32_t p_db, uint32_t p_A,
                                          uint32_t p_BC, uint32_t p_D, uint32_t p_x, uint32_t p_z, uint32_t p_y) {
    set_streamer_csr((uint32_t)0, 0, 0, 0, 0,                                     // R0 osCore in (disabled)
                     (uint32_t)0, 0, 0, 0, 0,                                     // R1 osCore weight (disabled)
                     p_dt, M2_R2_ss, M2_R2_tb_dt_ring, M2_R2_ts_dt_ring, M2_R2_en,  // R2 switchCore in (dt async ring)
                     p_dw1, M2_R3_ss, M2_R3_tb, M2_R3_ts, M2_R3_en,               // R3 switchCore weight 1
                     p_db, M2_R4_ss, M2_R4_tb, M2_R4_ts, M2_R4_en,                // R4 switchCore bias
                     p_dw2, M2_R5_ss, M2_R5_tb, M2_R5_ts, M2_R5_en,               // R5 switchCore weight 2
                     p_A, M2_R6_ss, M2_R6_tb, M2_R6_ts, M2_R6_en,                 // R6 SUC A
                     p_BC, M2_R7_ss, M2_R7_tb_ring, M2_R7_ts_ring, M2_R7_en,      // R7 SUC BC (async ring)
                     p_D, M2_R8_ss, M2_R8_tb, M2_R8_ts, M2_R8_en,                 // R8 SUC D
                     p_x, M2_R9_ss, M2_xzy_tb_ring, M2_xzy_ts_ring, M2_R9_en,     // R9 SUC x (async ring)
                     p_z, M2_R10_ss, M2_xzy_tb_ring, M2_xzy_ts_ring, M2_R10_en,   // R10 SUC z (async ring)
                     (uint32_t)0, 0, 0, 0, 0,                                     // R11 iscore in (disabled)
                     (uint32_t)0, 0, 0, 0, 0,                                     // R12 isCore weight (disabled)
                     (uint32_t)0, 0, 0, 0, 0,                                     // R13 isCore psum (disabled)

                     (uint32_t)0, 0, 0, 0, 0,                                  // W0 osCore out (disabled)
                     (uint32_t)0, 0, 0, 0, M2_W1_en,                           // W1 disabled
                     p_y, M2_W2_ss, M2_xzy_tb_ring, M2_xzy_ts_ring, M2_W2_en,  // W2 SUC y out (async ring)
                     (uint32_t)0, 0, 0, 0, 0                                   // W3 isCore out (disabled)
    );
}

// L3 byte offset of the (L-tile lt, broadcast index bc) compact subtile chunk inside one dInner-tile's
// x/z/y tensor (ConvFormat). bc decomposes col-block-major into (col-block, subtile): the chunk is
// win_per_l_tile windows of M2_xzy_subtile_bytes at L3 stride M2_xzy_window_src_stride.
static inline uint32_t xzy_l3_offset(uint32_t lt, uint32_t bc) {
    uint32_t cb = bc / M2_xzy_subtiles_per_colblock;
    uint32_t s  = bc % M2_xzy_subtiles_per_colblock;
    return cb * M2_xzy_colblock_src_stride + lt * M2_BC_windows_per_l_tile * M2_xzy_window_src_stride +
           s * M2_xzy_subtile_bytes;
}

static inline void suc_ring_loop(uint8_t* dt_slot, const uint8_t* l3_dt,    //
                                 uint8_t* bc_slot, const uint8_t* l3_dt_BC,  //
                                 uint8_t* x_slot, const uint8_t* x_l3_tile,  //
                                 uint8_t* z_slot, const uint8_t* z_l3_tile,  //
                                 uint8_t* y_slot, uint8_t* y_l3_tile) {
    const uint32_t visits = M2_BC_n_visits;    // nb_l_tiles * broadcast
    const uint32_t step   = M2_BC_gauge_step;  // L_tile * delaySU (SUC outputs per visit)
    const uint32_t wpt    = M2_BC_windows_per_l_tile;
    const uint32_t sb     = M2_xzy_subtile_bytes;

    for (uint32_t r = 0; r < visits; r++) {
        if (snrt_global_core_idx() == 0)
            while (read_csr(R11_DELAY_GAUGE) < (r + 1) * step);

        snrt_cluster_hw_barrier();

        if (snrt_is_dm_core()) {
            uint32_t slot = r % nb_slots;

            // x/z refill: gather the future visit's subtile chunk (guarded to this dInner tile).
            if (r + nb_slots < visits) {
                uint32_t off = xzy_l3_offset((r + nb_slots) % nb_l_tiles, (r + nb_slots) / nb_l_tiles);
                snrt_dma_start_2d(x_slot + slot * M2_length_xzy_l_tile, x_l3_tile + off, sb, sb,
                                  M2_xzy_window_src_stride, wpt);
                snrt_dma_start_2d(z_slot + slot * M2_length_xzy_l_tile, z_l3_tile + off, sb, sb,
                                  M2_xzy_window_src_stride, wpt);
            }

            // y spill: visit r's subtile chunk (just written) scatters out to y's ConvFormat slot in L3.
            snrt_dma_start_2d(y_l3_tile + xzy_l3_offset(r % nb_l_tiles, r / nb_l_tiles),
                              y_slot + slot * M2_length_xzy_l_tile, sb, M2_xzy_window_src_stride, sb, wpt);

            // BC refill: re-read whole L-tile for the future visit (cycles into the next tile too).
            uint32_t bc_lt = (r + nb_slots) % nb_l_tiles;
            snrt_dma_start_2d(bc_slot + slot * M2_length_BC_l_tile,
                              l3_dt_BC + bc_lt * M2_BC_windows_per_l_tile * M2_dtBC_window_src_stride + M2_BC_l3_offset,
                              M2_BC_window_bytes, M2_BC_window_bytes, M2_dtBC_window_src_stride,
                              M2_BC_windows_per_l_tile);

            // dt refill (lower frequency): only at a dt-tile boundary (every dt_group visits), for the
            // future dt-tile. At this r the gauge wait above (>= (r+1)*step) exactly meets the dt-slot
            // safe threshold, so no extra wait is needed. dt is re-read across broadcast, so it wraps
            // like BC (no end guard).
            if ((r + 1) % M2_dt_group == 0) {
                uint32_t dr      = (r + 1) / M2_dt_group + nb_dt_slots - 1;  // future dt-visit to refill
                uint32_t dt_lt   = dr % nb_dt_tiles;
                snrt_dma_start_2d(dt_slot + (dr % nb_dt_slots) * M2_length_dt_slot,
                                  l3_dt + dt_lt * M2_dt_slot_windows * M2_dtBC_window_src_stride,
                                  M2_dt_pack_window_bytes, M2_dt_pack_window_bytes, M2_dtBC_window_src_stride,
                                  M2_dt_slot_windows);
            }
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

    // Shared (L-only) across all dInner tiles
    uint8_t* ptr_dt_ring = (uint8_t*)tcdm_base_ptr;
    uint8_t* ptr_BC_ring = _ALIGN64(ptr_dt_ring + nb_dt_slots * M2_length_dt_slot);
    uint8_t* ptr_dt_w1     = _ALIGN64(ptr_BC_ring + nb_slots * M2_length_BC_l_tile);
    uint8_t* ptr_dt_w2     = _ALIGN64(ptr_dt_w1 + M2_length_dt_weight_1_tile);
    uint8_t* ptr_dt_bias   = _ALIGN64(ptr_dt_w2 + M2_length_dt_weight_2_tile);
    uint8_t* ptr_A         = _ALIGN64(ptr_dt_bias + M2_length_dt_bias_tile);
    uint8_t* ptr_D         = _ALIGN64(ptr_A + M2_length_A_tile);
    uint8_t* ptr_x_ring    = _ALIGN64(ptr_D + M2_length_D_tile);
    uint8_t* ptr_z_ring    = _ALIGN64(ptr_x_ring + nb_slots * M2_length_xzy_l_tile);
    uint8_t* ptr_y_ring    = _ALIGN64(ptr_z_ring + nb_slots * M2_length_xzy_l_tile);
#undef _ALIGN64

    uint32_t start_cycles     = 0;
    uint32_t simbacore_cycles = 0;

    if (snrt_global_core_idx() == 0) {
        printf(
            "\nStarting program: suc-async-dt (L=%d, dModel=%d, dInner=%d, nb_tiles=%d, nb_l_tiles=%d, nb_slots=%d, "
            "dInner_tile=%u, L_tile=%u)\n\n",
            seqLen, dModel, dInner, nb_tiles, nb_l_tiles, nb_slots, M2_dInner_tile, L_tile);
        printf("Expected L1 TCDM usage: %u B (%u KiB)\n", (uint32_t)L1_TCDM_PEAK_BYTES,
               (uint32_t)(L1_TCDM_PEAK_BYTES / 1024));
        init_cycle_counter();
        start_cycles = snrt_mcycle();
        set_streamer_suc_async((uint32_t)ptr_dt_ring, (uint32_t)ptr_dt_w1, (uint32_t)ptr_dt_w2, (uint32_t)ptr_dt_bias,
                               (uint32_t)ptr_A, (uint32_t)ptr_BC_ring, (uint32_t)ptr_D, (uint32_t)ptr_x_ring,
                               (uint32_t)ptr_z_ring, (uint32_t)ptr_y_ring);
        set_simbacore_csr(M27_SUC_ONLY, seqLen, dModel, M2_dInner_tile, dtRank, dModel);
    }

    // Preload once: first nb_dt_slots dt-tiles + first nb_slots BC L-tiles.
    if (snrt_is_dm_core()) {
        for (uint32_t s = 0; s < nb_dt_slots; s++)
            snrt_dma_start_2d(ptr_dt_ring + s * M2_length_dt_slot,
                              M2_dt_BC + s * M2_dt_slot_windows * M2_dtBC_window_src_stride, M2_dt_pack_window_bytes,
                              M2_dt_pack_window_bytes, M2_dtBC_window_src_stride, M2_dt_slot_windows);
        for (uint32_t s = 0; s < nb_slots; s++)
            snrt_dma_start_2d(ptr_BC_ring + s * M2_length_BC_l_tile,
                              M2_dt_BC + s * M2_BC_windows_per_l_tile * M2_dtBC_window_src_stride + M2_BC_l3_offset,
                              M2_BC_window_bytes, M2_BC_window_bytes, M2_dtBC_window_src_stride,
                              M2_BC_windows_per_l_tile);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    for (uint32_t tile = 0; tile < nb_tiles; tile++) {
        const uint8_t* x_l3_tile = M2_suc_x + tile * M2_length_x_tile;
        const uint8_t* z_l3_tile = M2_oscore_expected + tile * M2_length_z_tile;

        // Stage I: load this tile's weights (blocking) and preload the first nb_slots x/z ring slots
        // (subtile 0 of L-tiles 0..nb_slots-1) so R9/R10 can start immediately.
        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(ptr_dt_w1, M2_dt_weight_1 + tile * M2_length_dt_weight_1_tile,
                              M2_length_dt_weight_1_tile);
            snrt_dma_start_1d(ptr_dt_w2, M2_dt_weight_2 + tile * M2_length_dt_weight_2_tile,
                              M2_length_dt_weight_2_tile);
            snrt_dma_start_1d(ptr_dt_bias, M2_dt_bias + tile * M2_length_dt_bias_tile, M2_length_dt_bias_tile);
            snrt_dma_start_1d(ptr_A, M2_suc_A + tile * M2_length_A_tile, M2_length_A_tile);
            snrt_dma_start_1d(ptr_D, M2_suc_D + tile * M2_length_D_tile, M2_length_D_tile);
            for (uint32_t s = 0; s < nb_slots; s++) {
                uint32_t off = xzy_l3_offset(s, 0);  // visit s: L-tile s, broadcast 0 (col-block 0, subtile 0)
                snrt_dma_start_2d(ptr_x_ring + s * M2_length_xzy_l_tile, x_l3_tile + off, M2_xzy_subtile_bytes,
                                  M2_xzy_subtile_bytes, M2_xzy_window_src_stride, M2_BC_windows_per_l_tile);
                snrt_dma_start_2d(ptr_z_ring + s * M2_length_xzy_l_tile, z_l3_tile + off, M2_xzy_subtile_bytes,
                                  M2_xzy_subtile_bytes, M2_xzy_window_src_stride, M2_BC_windows_per_l_tile);
            }
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();

        // Stage II: kick the SUC
        if (snrt_global_core_idx() == 0) start_simbacore_and_streamers(M2_R10_en, 0, 0, 0);
        snrt_cluster_hw_barrier();

        // Ring loop: refill BC + x + z, spill y. y is fully spilled to L3, so no separate Stage III spill.
        suc_ring_loop(ptr_dt_ring, M2_dt_BC, ptr_BC_ring, M2_dt_BC, ptr_x_ring, x_l3_tile, ptr_z_ring, z_l3_tile,
                      ptr_y_ring, ptr_y_l3 + tile * M2_length_y_tile);

        if (snrt_global_core_idx() == 0) {
            wait_simbacore_and_streamer();
            asm volatile("fence" ::: "memory");
            simbacore_cycles += read_simbacore_perf_counter();
        }

        snrt_cluster_hw_barrier();
    }

    if (snrt_global_core_idx() == 0) {
        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Simbacore elapsed time: %u cycles\n", end_cycles, simbacore_cycles);
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        err += check_result_sample(ptr_y_l3, M2_suc_expected, M2_test_samples_y, nb_test_samples, "SUC y (from L3)");

        printf("Test suc-async-dt: %s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, nb_test_samples);
    }
    snrt_cluster_hw_barrier();
    return err;
}
