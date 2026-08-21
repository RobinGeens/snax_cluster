// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Stand-alone SUC, sequence-tiled with the SSM hidden state handed across L-tiles through the isCore
// in_c/out_d streamer ports (R13/W3).
//  See docs/dataflow/13_suc_carry.md.

#include "helper.c"
#include "snax-simbacore-lib.h"

// L3 byte offset of L-tile `lt`'s x/z/y ConvFormat block (col-block 0). Successive col-blocks follow at
// stride M2_xzy_colblock_src_stride; each is M2_xzy_colblock_bytes of contiguous ConvFormat data.
static inline uint32_t xzy_l3_tile_base(uint32_t lt) {
    return lt * M2_BC_windows_per_l_tile * M2_xzy_window_src_stride;
}

// DM-core: gather L-tile `lt`'s dt + BC (one L-tile each) + x/z into the slot. x/z land in ConvFormat
// ([col-block][window][subtile]); the streamer transposes to SUCFormat, so each tensor is one 2D DMA of
// M2_xzy_n_colblocks contiguous col-block bursts. dt/BC are interleaved per window in dt_BC.
static inline void gather_tile(uint32_t lt, uint8_t* dt_s, uint8_t* bc_s, uint8_t* x_s, uint8_t* z_s) {
    uint32_t dtbc_base = lt * M2_BC_windows_per_l_tile * M2_dtBC_window_src_stride;
    snrt_dma_start_2d(dt_s, M2_dt_BC + dtbc_base, M2_dt_pack_window_bytes, M2_dt_pack_window_bytes,
                      M2_dtBC_window_src_stride, M2_BC_windows_per_l_tile);
    snrt_dma_start_2d(bc_s, M2_dt_BC + dtbc_base + M2_BC_l3_offset, M2_BC_window_bytes, M2_BC_window_bytes,
                      M2_dtBC_window_src_stride, M2_BC_windows_per_l_tile);
    uint32_t base = xzy_l3_tile_base(lt);
    snrt_dma_start_2d(x_s, M2_suc_x + base, M2_xzy_colblock_bytes, M2_xzy_colblock_bytes, M2_xzy_colblock_src_stride,
                      M2_xzy_n_colblocks);
    snrt_dma_start_2d(z_s, M2_oscore_expected + base, M2_xzy_colblock_bytes, M2_xzy_colblock_bytes,
                      M2_xzy_colblock_src_stride, M2_xzy_n_colblocks);
}

// DM-core: spill L-tile `lt`'s y (ConvFormat in the slot) to L3 as one 2D DMA of col-block bursts.
static inline void spill_tile(uint32_t lt, uint8_t* y_s, uint8_t* y_l3) {
    uint32_t base = xzy_l3_tile_base(lt);
    snrt_dma_start_2d(y_l3 + base, y_s, M2_xzy_colblock_bytes, M2_xzy_colblock_src_stride, M2_xzy_colblock_bytes,
                      M2_xzy_n_colblocks);
}

static inline void set_streamer_suc_carry(uint32_t p_dt, uint32_t p_dw1, uint32_t p_dw2, uint32_t p_db, uint32_t p_A,
                                          uint32_t p_BC, uint32_t p_D, uint32_t p_x, uint32_t p_z, uint32_t p_y,
                                          uint32_t p_state, bool load_en, bool save_en) {
    set_streamer_csr((uint32_t)0, 0, 0, 0, 0,                                              // R0 (disabled)
                     (uint32_t)0, 0, 0, 0, 0,                                              // R1 (disabled)
                     p_dt, M2_R2_ss, M2_R2_tb_sync, M2_R2_ts_sync, M2_R2_en,               // R2 switchCore in (dt)
                     p_dw1, M2_R3_ss, M2_R3_tb, M2_R3_ts, M2_R3_en,                        // R3 switchCore weight 1
                     p_db, M2_R4_ss, M2_R4_tb, M2_R4_ts, M2_R4_en,                         // R4 switchCore bias
                     p_dw2, M2_R5_ss, M2_R5_tb, M2_R5_ts, M2_R5_en,                        // R5 switchCore weight 2
                     p_A, M2_R6_ss, M2_R6_tb, M2_R6_ts, M2_R6_en,                          // R6 SUC A
                     p_BC, M2_R7_ss, M2_R7_tb_sync, M2_R7_ts_sync, M2_R7_en,               // R7 SUC BC
                     p_D, M2_R8_ss, M2_R8_tb, M2_R8_ts, M2_R8_en,                          // R8 SUC D
                     p_x, M2_R9_ss, M2_R9_tb_sync, M2_R9_ts_sync, M2_R9_en,                // R9 SUC x
                     p_z, M2_R10_ss, M2_R10_tb_sync, M2_R10_ts_sync, M2_R10_en,            // R10 SUC z
                     (uint32_t)0, 0, 0, 0, 0,                                              // R11 (disabled)
                     (uint32_t)0, 0, 0, 0, 0,                                              // R12 (disabled)
                     p_state, M2_R13_state_ss, M2_R13_state_tb, M2_R13_state_ts, load_en,  // R13 SUC state IN

                     (uint32_t)0, 0, 0, 0, 0,                                          // W0 (disabled)
                     (uint32_t)0, 0, 0, 0, 0,                                          // W1 (disabled)
                     p_y, M2_W2_ss, M2_W2_tb_sync, M2_W2_ts_sync, M2_W2_en,            // W2 SUC y out
                     p_state, M2_W3_state_ss, M2_W3_state_tb, M2_W3_state_ts, save_en  // W3 SUC state OUT
    );
}

int main() {
    int err = 0;

    static uint8_t* ptr_y_l3 = NULL;
    if (snrt_global_core_idx() == 0) {
        (void)snrt_l3alloc(16 * 1024);
        ptr_y_l3 = (uint8_t*)snrt_l3alloc(M2_length_y);
    }
    snrt_cluster_hw_barrier();

    const uint32_t slot_bc  = M2_length_BC_l_tile;
    const uint32_t slot_xzy = M2_BC_broadcast * M2_length_xzy_l_tile;

    void* tcdm_base_ptr = snrt_l1_next();
#define _ALIGN64(p) ((uint8_t*)(((uintptr_t)(p) + 63u) & ~(uintptr_t)63u))
    // weights / A / D / state are single-resident; dt / BC / x / z / y are double-buffered (2 slots) so the
    // DM core can gather the next tile (and spill the previous) while the accelerator computes the current one.
    uint8_t* ptr_dt_w1   = (uint8_t*)tcdm_base_ptr;
    uint8_t* ptr_dt_w2   = _ALIGN64(ptr_dt_w1 + M2_length_dt_weight_1);
    uint8_t* ptr_dt_bias = _ALIGN64(ptr_dt_w2 + M2_length_dt_weight_2);
    uint8_t* ptr_A       = _ALIGN64(ptr_dt_bias + M2_length_dt_bias);
    uint8_t* ptr_D       = _ALIGN64(ptr_A + M2_length_A);
    uint8_t* ptr_state   = _ALIGN64(ptr_D + M2_length_D);
    uint8_t *ptr_dt[2], *ptr_BC[2], *ptr_x[2], *ptr_z[2], *ptr_y[2];
    uint8_t* p = _ALIGN64(ptr_state + M2_length_state_buf);
    for (int s = 0; s < 2; s++) {
        ptr_dt[s] = p;
        p         = _ALIGN64(p + M2_length_dt_l_tile);
        // BC slots on a 2 KiB boundary: datagen pre-swizzles the BC windows against
        // phase-0 slots (bc_swizzle, docs/dataflow/22_agu_xor_swizzle.md).
        ptr_BC[s] = (uint8_t*)(((uintptr_t)p + 2047u) & ~(uintptr_t)2047u);
        p         = _ALIGN64(ptr_BC[s] + slot_bc);
        ptr_x[s]  = p;
        p         = _ALIGN64(p + slot_xzy);
        ptr_z[s]  = p;
        p         = _ALIGN64(p + slot_xzy);
        ptr_y[s]  = p;
        p         = _ALIGN64(p + slot_xzy);
    }
#undef _ALIGN64

    if (snrt_global_core_idx() == 0) init_cycle_counter();

    // Load the L-independent operands once, and pre-gather tile 0 into slot 0.
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_dt_w1, M2_dt_weight_1, M2_length_dt_weight_1);
        snrt_dma_start_1d(ptr_dt_w2, M2_dt_weight_2, M2_length_dt_weight_2);
        snrt_dma_start_1d(ptr_dt_bias, M2_dt_bias, M2_length_dt_bias);
        snrt_dma_start_1d(ptr_A, M2_suc_A, M2_length_A);
        snrt_dma_start_1d(ptr_D, M2_suc_D, M2_length_D);
        gather_tile(0, ptr_dt[0], ptr_BC[0], ptr_x[0], ptr_z[0]);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    uint32_t start_cycles = 0, simbacore_cycles = 0;
    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: suc-carry (L=%d, dModel=%d, dInner=%d, NB=%u, L_tile=%u, bc=%u, dbuf)\n\n", seqLen,
               dModel, dInner, M2_NB_L_TILES, M2_L_tile, M2_BC_broadcast);
        if (bc_swizzle) write_csr(ADDR_REMAP_INDEX_READER_7, 1);  // BC read through the XOR swizzle
        start_cycles = snrt_mcycle();
    }

    for (uint32_t lt = 0; lt < M2_NB_L_TILES; lt++) {
        uint32_t cur = lt & 1u;

        // core 0: configure + (non-blocking) start the accelerator on the current slot.
        if (snrt_global_core_idx() == 0) {
            bool load_en  = (lt != 0);
            bool save_en  = (lt != M2_NB_L_TILES - 1u);
            uint32_t mode = (lt == 0)                    ? M34_SUC_ONLY_STATE_SAVE
                            : (lt == M2_NB_L_TILES - 1u) ? M36_SUC_ONLY_STATE_LOAD
                                                         : M35_SUC_ONLY_STATE_CARRY;
            set_streamer_suc_carry((uint32_t)ptr_dt[cur], (uint32_t)ptr_dt_w1, (uint32_t)ptr_dt_w2,
                                   (uint32_t)ptr_dt_bias, (uint32_t)ptr_A, (uint32_t)ptr_BC[cur], (uint32_t)ptr_D,
                                   (uint32_t)ptr_x[cur], (uint32_t)ptr_z[cur], (uint32_t)ptr_y[cur],
                                   (uint32_t)ptr_state, load_en, save_en);
            set_simbacore_csr(mode, M2_L_tile, dModel, dInner, dtRank, dModel);
            start_simbacore_and_streamers(M2_R10_en, 0, 0, 0);
        }

        // DM core (concurrent with compute): gather next tile + spill previous tile.
        if (snrt_is_dm_core()) {
            if (lt + 1u < M2_NB_L_TILES)
                gather_tile(lt + 1u, ptr_dt[cur ^ 1u], ptr_BC[cur ^ 1u], ptr_x[cur ^ 1u], ptr_z[cur ^ 1u]);
            if (lt >= 1u) spill_tile(lt - 1u, ptr_y[cur ^ 1u], ptr_y_l3);
            snrt_dma_wait_all();
        }

        // core 0: wait for the accelerator to finish this tile.
        if (snrt_global_core_idx() == 0) {
            wait_simbacore_and_streamer();
            asm volatile("fence" ::: "memory");
            simbacore_cycles += read_simbacore_perf_counter();
        }
        snrt_cluster_hw_barrier();
    }

    // Epilogue: spill the last tile's y.
    if (snrt_is_dm_core()) {
        spill_tile(M2_NB_L_TILES - 1u, ptr_y[(M2_NB_L_TILES - 1u) & 1u], ptr_y_l3);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    if (snrt_global_core_idx() == 0) {
        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Simbacore elapsed time: %u cycles\n", end_cycles, simbacore_cycles);
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);
        err += check_result_sample(ptr_y_l3, M2_suc_expected, M2_test_samples_y, nb_test_samples, "suc-carry y");
        printf("Test suc-carry: %s: %u/%d errors (L=%d, dModel=%d)\n", err ? "FAIL" : "PASS", err, nb_test_samples,
               seqLen, dModel);
    }
    snrt_cluster_hw_barrier();
    return err;
}
