// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// Mamba P2 only, without IS-core. Async tiling on oscore_in and BC

#include "helper.c"
#include "snax-simbacore-lib.h"

static inline void set_streamer_phase2_noIS_lTile(uint32_t p_oi, uint32_t p_ow, uint32_t p_z, uint32_t p_dt,
                                                  uint32_t p_dw1, uint32_t p_dw2, uint32_t p_db, uint32_t p_x,
                                                  uint32_t p_A, uint32_t p_BC, uint32_t p_D, uint32_t p_y) {
    set_streamer_csr(

        p_oi, M2_R0_ss, M2_R0_tb_lTile, M2_R0_ts, M2_R0_en,          // osCore in (async ring)
        p_ow, M2_R1_ss, M2_R1_tb, M2_R1_ts, M2_R1_en,                // osCore weight
        p_dt, M2_R2_ss, M2_R2_tb_packed, M2_R2_ts_packed, M2_R2_en,  // switchCore in (dt, PACKED full)
        p_dw1, M2_R3_ss, M2_R3_tb, M2_R3_ts, M2_R3_en,               // switchCore weight 1
        p_db, M2_R4_ss, M2_R4_tb, M2_R4_ts, M2_R4_en,                // switchCore bias
        p_dw2, M2_R5_ss, M2_R5_tb, M2_R5_ts, M2_R5_en,               // switchCore weight 2
        p_A, M2_R6_ss, M2_R6_tb, M2_R6_ts, M2_R6_en,                 // SUC A
        p_BC, M2_R7_ss, M2_R7_tb_ring, M2_R7_ts_ring, M2_R7_en,      // SUC BC (async ring)
        p_D, M2_R8_ss, M2_R8_tb, M2_R8_ts, M2_R8_en,                 // SUC D
        p_x, M2_R9_ss, M2_R9_tb, M2_R9_ts, M2_R9_en,                 // SUC x
        p_z, M2_R10_ss, M2_R10_tb, M2_R10_ts, M2_R10_en,             // SUC z (= osCore out)
        (uint32_t)0, 0, 0, 0, 0,                                     // R11 isCore in (DISABLED)
        (uint32_t)0, 0, 0, 0, 0,                                     // R12 isCore weight (DISABLED)
        (uint32_t)0, 0, 0, 0, 0,                                     // R13 isCore psum (DISABLED)

        p_z, M2_W0_ss, M2_W0_tb, M2_W0_ts, M2_W0_en,  // osCore out = z
        (uint32_t)0, 0, 0, 0, M2_W1_en,               // W1 disabled
        p_y, M2_W2_ss, M2_W2_tb, M2_W2_ts, M2_W2_en,  // SUC y out
        (uint32_t)0, 0, 0, 0, 0                       // W3 isCore out (DISABLED)
    );
}

// Refill loop for oscore_in and BC
static inline void refill_loop(uint8_t* os_slot_base, const uint8_t* l3_oscore_in,  //
                               uint8_t* bc_slot_base, const uint8_t* l3_dt_BC,      //
                               uint32_t r10_release_en, uint32_t r10_start_cnt) {
    const uint32_t os_visits = (M2_dInner_tile / dInnerUnroll) * nb_l_tiles;
    const uint32_t os_step   = M2_oscore_in_l_tile_gauge_step;
    const uint32_t os_len    = M2_length_oscore_in_l_tile;
    const uint32_t bc_visits = M2_BC_n_visits;
    const uint32_t bc_step   = M2_BC_gauge_step;
    uint32_t r_os = 0, r_bc = 0, r10_released = 0;
    static volatile uint32_t do_os = 0, do_bc = 0;

    while (r_os < os_visits || r_bc < bc_visits) {
        if (snrt_global_core_idx() == 0) {
            uint32_t rdy_os, rdy_bc;
            do {
                rdy_os = (r_os < os_visits) && (read_snax_csr_safe(R10_DELAY_GAUGE) >= (r_os + 1) * os_step);
                rdy_bc = (r_bc < bc_visits) && (read_snax_csr_safe(R11_DELAY_GAUGE) >= (r_bc + 1) * bc_step);
                if (r10_release_en && !r10_released && read_snax_csr_safe(R10_DELAY_GAUGE) >= r10_start_cnt) {
                    // release the SUC -> it pipelines behind the osCore
                    write_csr(DELAYED_START_READER_10, 1);
                    r10_released = 1;
                }
            } while (!rdy_os && !rdy_bc);
            do_os = rdy_os;
            do_bc = rdy_bc;
        }

        // barrier 1: publish do_os/do_bc to all cores
        snrt_cluster_hw_barrier();
        if (snrt_is_dm_core()) {
            // issue oscore_in first so a queued BC DMA never delays it
            if (do_os)
                snrt_dma_start_1d(os_slot_base + (r_os % nb_slots) * os_len,
                                  l3_oscore_in + ((r_os + nb_slots) % nb_l_tiles) * os_len, os_len);
            if (do_bc) {
                uint32_t lt = (r_bc + nb_slots) % nb_l_tiles;
                snrt_dma_start_2d(
                    bc_slot_base + (r_bc % nb_slots) * M2_length_BC_l_tile,
                    l3_dt_BC + lt * M2_BC_windows_per_l_tile * M2_dtBC_window_src_stride + M2_BC_l3_offset,
                    M2_BC_window_bytes, M2_BC_window_bytes, M2_dtBC_window_src_stride, M2_BC_windows_per_l_tile);
            }
        }
        if (do_os) r_os++;
        if (do_bc) r_bc++;
        // barrier 2: all cores done reading do_*; safe for core 0 to recompute
        snrt_cluster_hw_barrier();
    }

    if (snrt_is_dm_core()) snrt_dma_wait_all();
    snrt_cluster_hw_barrier();

    if (snrt_global_core_idx() == 0 && r10_release_en && !r10_released) {
        while (read_snax_csr_safe(R10_DELAY_GAUGE) < r10_start_cnt);
        write_csr(DELAYED_START_READER_10, 1);
    }
}

int test_p2_async_no_is() {
    int err = 0;

    // L3 staging for z and y (reserve 16 KiB first to skip putc_buffer).
    static uint8_t* ptr_z_l3 = NULL;
    static uint8_t* ptr_y_l3 = NULL;
    if (snrt_global_core_idx() == 0) {
        (void)snrt_l3alloc(16 * 1024);
        ptr_z_l3 = (uint8_t*)snrt_l3alloc(M2_length_z);
        ptr_y_l3 = (uint8_t*)snrt_l3alloc(M2_length_y);
    }
    snrt_cluster_hw_barrier();

    void* tcdm_base_ptr = (void*)(((uintptr_t)snrt_l1_next() + 2047u) & ~(uintptr_t)2047u);

    // B1: oscore_in is nb_slots ADJACENT ring slots (R0 stride-0 wrap walks them contiguously).
    uint8_t* ptr_oscore_in_base   = (uint8_t*)tcdm_base_ptr;
    uint32_t oscore_in_tcdm_bytes = nb_slots * M2_length_oscore_in_l_tile;

#define _ALIGN64(p) ((uint8_t*)(((uintptr_t)(p) + 63u) & ~(uintptr_t)63u))

    // switchCore dt input: full but packed (per-window dt only) -> small. BC: nb_slots-slot async ring.
    uint8_t* ptr_dt_packed      = _ALIGN64(ptr_oscore_in_base + oscore_in_tcdm_bytes);
    // BC ring on a 2 KiB boundary: datagen pre-swizzles the BC windows against ring-relative
    // positions (bc_swizzle, docs/dataflow/22_agu_xor_swizzle.md).
    uint8_t* ptr_BC_ring = (uint8_t*)(((uintptr_t)(ptr_dt_packed + M2_length_dt_packed) + 2047u) & ~(uintptr_t)2047u);
    uint32_t bc_ring_tcdm_bytes = nb_slots * M2_length_BC_l_tile;

    uint8_t* pingpong_base_ptr = _ALIGN64(ptr_BC_ring + bc_ring_tcdm_bytes);

    // ---- Per-dInner-tile SINGLE buffers (no ping-pong). x/z/y are full-L per tile and at L=3136
    // their ping-pong pair overflowed TCDM; single-buffering them (and the small weights) fits. The
    // dInner-tile loop is SERIALIZED (load -> compute -> spill) so one buffer per operand is safe.
    uint8_t* ptr_oscore_weight = pingpong_base_ptr;
    uint8_t* ptr_dt_weight_1   = _ALIGN64(ptr_oscore_weight + M2_length_oscore_weight_tile);
    uint8_t* ptr_dt_weight_2   = _ALIGN64(ptr_dt_weight_1 + M2_length_dt_weight_1_tile);
    uint8_t* ptr_dt_bias       = _ALIGN64(ptr_dt_weight_2 + M2_length_dt_weight_2_tile);
    uint8_t* ptr_A             = _ALIGN64(ptr_dt_bias + M2_length_dt_bias_tile);
    uint8_t* ptr_D             = _ALIGN64(ptr_A + M2_length_A_tile);
    uint8_t* ptr_x             = _ALIGN64(ptr_D + M2_length_D_tile);
    uint8_t* ptr_z             = _ALIGN64(ptr_x + M2_length_x_tile);
    uint8_t* ptr_y             = _ALIGN64(ptr_z + M2_length_z_tile);
#undef _ALIGN64

    const uint32_t K_i               = M2_dInner_tile / dInnerUnroll;
    uint32_t start_cycles            = 0;
    uint32_t simbacore_cycles_phase2 = 0;

    if (snrt_global_core_idx() == 0) {
        printf(
            "\nStarting program: P2-async-OS-no-IS (L=%d, dModel=%d, nb_tiles=%d, nb_l_tiles=%d, L_tile=%u, "
            "nb_slots=%d, K_i=%u, no IS-core, oscore_in async)\n\n",
            seqLen, dModel, nb_tiles, nb_l_tiles, L_tile, nb_slots, K_i);
        printf("Expected L1 TCDM usage: %u B (%u KiB)\n", (uint32_t)L1_TCDM_PEAK_BYTES,
               (uint32_t)(L1_TCDM_PEAK_BYTES / 1024));
        init_cycle_counter();
        start_cycles = snrt_mcycle();
        set_streamer_phase2_noIS_lTile((uint32_t)ptr_oscore_in_base, (uint32_t)ptr_oscore_weight, (uint32_t)ptr_z,
                                       (uint32_t)ptr_dt_packed, (uint32_t)ptr_dt_weight_1, (uint32_t)ptr_dt_weight_2,
                                       (uint32_t)ptr_dt_bias, (uint32_t)ptr_x, (uint32_t)ptr_A, (uint32_t)ptr_BC_ring,
                                       (uint32_t)ptr_D, (uint32_t)ptr_y);
        set_simbacore_csr(M33_PHASE2_NO_ISCORE, seqLen, dModel, M2_dInner_tile, dtRank, dModel);
        if (bc_swizzle) write_csr(ADDR_REMAP_INDEX_READER_7, 1);  // BC ring read through the XOR swizzle
    }

    // Preload: first nb_slots oscore_in + BC L-tiles into their rings; PACKED full dt extracted
    // from the combined dt_BC L3 buffer (drop per-window BC bytes).
    if (snrt_is_dm_core()) {
        for (uint32_t s = 0; s < nb_slots; s++) {
            snrt_dma_start_1d(ptr_oscore_in_base + s * M2_length_oscore_in_l_tile,
                              M2_oscore_in + s * M2_oscore_in_l_offset, M2_length_oscore_in_l_tile);
            snrt_dma_start_2d(ptr_BC_ring + s * M2_length_BC_l_tile,
                              M2_dt_BC + s * M2_BC_windows_per_l_tile * M2_dtBC_window_src_stride + M2_BC_l3_offset,
                              M2_BC_window_bytes, M2_BC_window_bytes, M2_dtBC_window_src_stride,
                              M2_BC_windows_per_l_tile);
        }
        snrt_dma_start_2d(ptr_dt_packed, M2_dt_BC, M2_dt_pack_window_bytes, M2_dt_pack_window_bytes,
                          M2_dtBC_window_src_stride, M2_dt_windows_total);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    for (uint32_t t = 0; t < nb_tiles; t++) {
        // Stage I: load this tile's weights + x into the single buffers.
        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(ptr_oscore_weight, M2_oscore_weight + t * M2_length_oscore_weight_tile,
                              M2_length_oscore_weight_tile);
            snrt_dma_start_1d(ptr_dt_weight_1, M2_dt_weight_1 + t * M2_length_dt_weight_1_tile,
                              M2_length_dt_weight_1_tile);
            snrt_dma_start_1d(ptr_dt_weight_2, M2_dt_weight_2 + t * M2_length_dt_weight_2_tile,
                              M2_length_dt_weight_2_tile);
            snrt_dma_start_1d(ptr_dt_bias, M2_dt_bias + t * M2_length_dt_bias_tile, M2_length_dt_bias_tile);
            snrt_dma_start_1d(ptr_A, M2_suc_A + t * M2_length_A_tile, M2_length_A_tile);
            snrt_dma_start_1d(ptr_D, M2_suc_D + t * M2_length_D_tile, M2_length_D_tile);
            snrt_dma_start_1d(ptr_x, M2_suc_x + t * M2_length_x_tile, M2_length_x_tile);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();

        // Non-blocking start (SUC z-reader held by DELAYED_START_READER_10 until released below).
        if (snrt_global_core_idx() == 0) {
            _set_streamer_start();
            _set_simbacore_start();
            write_csr(STREAMER_START_CSR, 0);
            write_csr(SIMBACORE_START, 0);
        }
        snrt_cluster_hw_barrier();

        // Refill loop for oscore_in and BC
        refill_loop(ptr_oscore_in_base, M2_oscore_in, ptr_BC_ring, M2_dt_BC, 1, M2_R10_start_cnt);

        if (snrt_global_core_idx() == 0) {
            while (read_csr(SIMBACORE_BUSY));
            while (read_csr(STREAMER_BUSY_CSR));
            asm volatile("fence" ::: "memory");  // ensure W0(z)/W2(y) visible to the spill DMA
            simbacore_cycles_phase2 += read_simbacore_perf_counter();
            write_csr(DELAYED_START_READER_10, 0);  // rearm for next tile
        }
        snrt_cluster_hw_barrier();

        // Stage III: spill this tile's z and y to L3 (after compute + fence; single buffers).
        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(ptr_z_l3 + t * M2_length_z_tile, ptr_z, M2_length_z_tile);
            snrt_dma_start_1d(ptr_y_l3 + t * M2_length_y_tile, ptr_y, M2_length_y_tile);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();
    }

    // --- Verification ---
    if (snrt_global_core_idx() == 0) {
        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Simbacore elapsed time: %u cycles\n", end_cycles, simbacore_cycles_phase2);
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        err += check_result_sample(ptr_z_l3, M2_oscore_expected, M2_test_samples_z,  //
                                   nb_test_samples, "z (osCore out)");
        err += check_result_sample(ptr_y_l3, M2_suc_expected, M2_test_samples_y,  //
                                   nb_test_samples, "SUC y (to L3)");

        uint32_t z_bad = 0, z_first = 0xffffffff, y_bad = 0, y_first = 0xffffffff;
        for (uint32_t k = 0; k < M2_length_z; k++) {
            int d = (int)ptr_z_l3[k] - (int)M2_oscore_expected[k];
            if (d < -1 || d > 1) {
                if (z_first == 0xffffffff) z_first = k;
                z_bad++;
            }
        }
        for (uint32_t k = 0; k < M2_length_y; k++) {
            int d = (int)ptr_y_l3[k] - (int)M2_suc_expected[k];
            if (d < -1 || d > 1) {
                if (y_first == 0xffffffff) y_first = k;
                y_bad++;
            }
        }
        printf("FULLCHK z: %u/%u bad (first@%d)  y: %u/%u bad (first@%d)\n", z_bad, M2_length_z, (int)z_first, y_bad,
               M2_length_y, (int)y_first);

        printf("Test P2-async-OS-no-IS: seqLen=%d, dModel=%d, dInner=%d, nb_tiles=%d, nb_l_tiles=%d\n", seqLen, dModel,
               dInner, nb_tiles, nb_l_tiles);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, 2 * nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test_p2_async_no_is(); }
