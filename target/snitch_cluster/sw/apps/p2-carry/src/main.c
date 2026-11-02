// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Mamba P2 (no isCore) sequence-tiled with on-chip SSM state carry, fused with the output projection.
// Replaces the (P2-async-OS-no-IS + separate out-proj) two-program pipeline with one program.
// Phase A skews the osCore one L-tile ahead of the SUC so neither waits on the other.
// See docs/dataflow/15_p2_carry.md.

#include "helper.c"
#include "snax-simbacore-lib.h"

// DM-core: gather L-tile `lt`'s dt (packed) + BC into their slots (dt/BC interleaved per window in dt_BC).
static inline void gather_dt_bc(uint32_t lt, uint8_t* dt_s, uint8_t* bc_s) {
    uint32_t dtbc_base = lt * M2_BC_windows_per_l_tile * M2_dtBC_window_src_stride;
    snrt_dma_start_2d(dt_s, M2_dt_BC + dtbc_base, M2_dt_pack_window_bytes, M2_dt_pack_window_bytes,
                      M2_dtBC_window_src_stride, M2_BC_windows_per_l_tile);
    snrt_dma_start_2d(bc_s, M2_dt_BC + dtbc_base + M2_BC_l3_offset, M2_BC_window_bytes, M2_BC_window_bytes,
                      M2_dtBC_window_src_stride, M2_BC_windows_per_l_tile);
}

// DM-core: gather a convFormat L-tile (one contiguous per-d3-colblock block each) from `src_l3` into `dst`.
static inline void gather_conv_tile(uint32_t lt, uint8_t* dst, const uint8_t* src_l3) {
    snrt_dma_start_2d(dst, src_l3 + lt * M2_conv_l_tile_l3_offset, M2_conv_tcdm_colblock_stride,
                      M2_conv_tcdm_colblock_stride, M2_conv_l3_colblock_stride, M2_conv_colblock_count);
}

// DM-core: spill a convFormat L-tile from `src` back to the full-L convFormat `dst_l3`.
static inline void spill_conv_tile(uint32_t lt, const uint8_t* src, uint8_t* dst_l3) {
    snrt_dma_start_2d(dst_l3 + lt * M2_conv_l_tile_l3_offset, src, M2_conv_tcdm_colblock_stride,
                      M2_conv_l3_colblock_stride, M2_conv_tcdm_colblock_stride, M2_conv_colblock_count);
}

// osCore(z) + switchCore(dt) + SUC(y), with state carry through R13(in)/W3(out). z is osCore-produced
// (W0) and read back by the SUC (R10) one launch later; x/y ride the convFormat conv<->suc streamers.
// Full program, written once; set_launch() then switches slots and ports per launch.
static inline void set_streamer_p2_carry(uint32_t p_oi, uint32_t p_ow, uint32_t p_dt, uint32_t p_dw1, uint32_t p_dw2,
                                         uint32_t p_db, uint32_t p_A, uint32_t p_BC, uint32_t p_D, uint32_t p_x,
                                         uint32_t p_z, uint32_t p_y, uint32_t p_state) {
    set_streamer_csr(p_oi, M2_R0_ss, M2_R0_lt_tb, M2_R0_lt_ts, M2_R0_en,      // R0 osCore in
                     p_ow, M2_R1_ss, M2_R1_lt_tb, M2_R1_lt_ts, M2_R1_en,      // R1 osCore weight
                     p_dt, M2_R2_ss, M2_R2_tb_sync, M2_R2_ts_sync, M2_R2_en,  // R2 switchCore in (dt)
                     p_dw1, M2_R3_ss, M2_R3_tb, M2_R3_ts, M2_R3_en,           // R3 switchCore weight 1
                     p_db, M2_R4_ss, M2_R4_tb, M2_R4_ts, M2_R4_en,            // R4 switchCore bias
                     p_dw2, M2_R5_ss, M2_R5_tb, M2_R5_ts, M2_R5_en,           // R5 switchCore weight 2
                     p_A, M2_R6_ss, M2_R6_tb, M2_R6_ts, M2_R6_en,             // R6 SUC A
                     p_BC, M2_R7_ss, M2_R7_tb_sync, M2_R7_ts_sync, M2_R7_en,  // R7 SUC BC
                     p_D, M2_R8_ss, M2_R8_tb, M2_R8_ts, M2_R8_en,             // R8 SUC D
                     p_x, M2_R9_ss, M2_R9_lt_tb, M2_R9_lt_ts, M2_R9_en,       // R9 SUC x (convFormat)
                     p_z, M2_R10_ss, M2_R10_lt_tb, M2_R10_lt_ts, M2_R10_en,   // R10 SUC z (= osCore out, convFormat)
                     (uint32_t)0, 0, 0, 0, 0,                                 // R11 (disabled)
                     (uint32_t)0, 0, 0, 0, 0,                                 // R12 (disabled)
                     p_state, M2_R13_state_ss, M2_R13_state_tb, M2_R13_state_ts, true,  // R13 SUC state IN

                     p_z, M2_W0_ss, M2_W0_lt_tb, M2_W0_lt_ts, M2_W0_en,               // W0 osCore out (z)
                     (uint32_t)0, 0, 0, 0, 0,                                         // W1 (disabled)
                     p_y, M2_W2_ss, M2_W2_lt_tb, M2_W2_lt_ts, M2_W2_en,               // W2 SUC y out (convFormat)
                     p_state, M2_W3_state_ss, M2_W3_state_tb, M2_W3_state_ts, true);  // W3 SUC state OUT
}

// Streamer program of launch l: osCore tile l, switchCore+SUC tile l-1 (zero outer temporal bound = port off).
static void set_launch(uint32_t l, uint8_t* const* oi, uint8_t* const* dt, uint8_t* const* bc, uint8_t* const* x,
                       uint8_t* const* z, uint8_t* const* y) {
    const bool os_on  = l < M2_NB_L_TILES;
    const bool suc_on = l >= 1u;
    const int o       = l & 1;        // tile l slot
    const int s       = (l + 1) & 1;  // tile l-1 slot

    write_csr(BASE_PTR_READER_0_LOW, (uint32_t)oi[o]);
    write_csr(BASE_PTR_WRITER_0_LOW, (uint32_t)z[o]);
    write_csr(T_BOUND_BASE_READER_0, os_on ? M2_R0_lt_tb[0] : 0);
    write_csr(T_BOUND_BASE_READER_1, os_on ? M2_R1_lt_tb[0] : 0);
    write_csr(T_BOUND_BASE_WRITER_0, os_on ? M2_W0_lt_tb[0] : 0);

    write_csr(BASE_PTR_READER_2_LOW, (uint32_t)dt[s]);
    write_csr(BASE_PTR_READER_7_LOW, (uint32_t)bc[s]);
    write_csr(BASE_PTR_READER_9_LOW, (uint32_t)x[s]);
    write_csr(BASE_PTR_READER_10_LOW, (uint32_t)z[s]);
    write_csr(BASE_PTR_WRITER_2_LOW, (uint32_t)y[s]);
    write_csr(T_BOUND_BASE_READER_2, suc_on ? M2_R2_tb_sync[0] : 0);
    write_csr(T_BOUND_BASE_READER_3, suc_on ? M2_R3_tb[0] : 0);
    write_csr(T_BOUND_BASE_READER_4, suc_on ? M2_R4_tb[0] : 0);
    write_csr(T_BOUND_BASE_READER_5, suc_on ? M2_R5_tb[0] : 0);
    write_csr(T_BOUND_BASE_READER_6, suc_on ? M2_R6_tb[0] : 0);
    write_csr(T_BOUND_BASE_READER_7, suc_on ? M2_R7_tb_sync[0] : 0);
    write_csr(T_BOUND_BASE_READER_8, suc_on ? M2_R8_tb[0] : 0);
    write_csr(T_BOUND_BASE_READER_9, suc_on ? M2_R9_lt_tb[0] : 0);
    write_csr(T_BOUND_BASE_READER_10, suc_on ? M2_R10_lt_tb[0] : 0);
    write_csr(T_BOUND_BASE_WRITER_2, suc_on ? M2_W2_lt_tb[0] : 0);
    write_csr(T_BOUND_BASE_READER_13, l >= 2u ? M2_R13_state_tb[0] : 0);                        // load: tile l-1 not first
    write_csr(T_BOUND_BASE_WRITER_3, (suc_on && l < M2_NB_L_TILES) ? M2_W3_state_tb[0] : 0);  // save: tile l-1 not last
}

int main() {
    int err = 0;

    // PHASE2_NO_ISCORE + hidden-state carry: the two state bits (isolated by XOR-ing the SUC_ONLY carry
    // modes against the plain SUC_ONLY encoding) are OR-ed onto M33; the datapath decodes the mode word as
    // a bitfield so the composed word behaves as a PHASE2_NO_ISCORE launch that also loads/saves state.
    const uint32_t state_save_bit = M34_SUC_ONLY_STATE_SAVE ^ M27_SUC_ONLY;
    const uint32_t state_load_bit = M36_SUC_ONLY_STATE_LOAD ^ M27_SUC_ONLY;

    static uint8_t* ptr_z_l3   = NULL;
    static uint8_t* ptr_y_l3   = NULL;
    static uint8_t* ptr_out_l3 = NULL;
    if (snrt_global_core_idx() == 0) {
        (void)snrt_l3alloc(16 * 1024);
        ptr_z_l3   = (uint8_t*)snrt_l3alloc(M2_length_z);
        ptr_y_l3   = (uint8_t*)snrt_l3alloc(M2_length_y);
        ptr_out_l3 = (uint8_t*)snrt_l3alloc(M2_length_iscore_out);
    }
    snrt_cluster_hw_barrier();

    void* tcdm_base_ptr = snrt_l1_next();
#define _ALIGN64(p) ((uint8_t*)(((uintptr_t)(p) + 63u) & ~(uintptr_t)63u))
    // Resident (L-independent) operands.
    uint8_t* ptr_dt_w1     = (uint8_t*)tcdm_base_ptr;
    uint8_t* ptr_dt_w2     = _ALIGN64(ptr_dt_w1 + M2_length_dt_weight_1);
    uint8_t* ptr_dt_bias   = _ALIGN64(ptr_dt_w2 + M2_length_dt_weight_2);
    uint8_t* ptr_A         = _ALIGN64(ptr_dt_bias + M2_length_dt_bias);
    uint8_t* ptr_D         = _ALIGN64(ptr_A + M2_length_A);
    uint8_t* ptr_os_weight = _ALIGN64(ptr_D + M2_length_D);
    uint8_t* ptr_state     = _ALIGN64(ptr_os_weight + M2_length_oscore_weight);
    // Double-buffered per-L-tile operands: oscore_in / dt / BC / x gathered, z osCore-produced, y SUC-produced.
    uint8_t *ptr_oi[2], *ptr_dt[2], *ptr_BC[2], *ptr_x[2], *ptr_z[2], *ptr_y[2];
    uint8_t* p = _ALIGN64(ptr_state + M2_length_state_buf);
    for (int s = 0; s < 2; s++) {
        ptr_oi[s] = p;
        p         = _ALIGN64(p + M2_length_oscore_in_l_tile);
        ptr_dt[s] = p;
        p         = _ALIGN64(p + M2_length_dt_l_tile);
        // BC slots on a 2 KiB boundary: datagen pre-swizzles the BC windows against
        // phase-0 slots (bc_swizzle, docs/dataflow/22_agu_xor_swizzle.md).
        ptr_BC[s] = (uint8_t*)(((uintptr_t)p + 2047u) & ~(uintptr_t)2047u);
        p         = _ALIGN64(ptr_BC[s] + M2_length_BC_l_tile);
        ptr_x[s]  = p;
        p         = _ALIGN64(p + M2_length_conv_l_tile);
        ptr_z[s]  = p;
        p         = _ALIGN64(p + M2_length_conv_l_tile);
        ptr_y[s]  = p;
        p         = _ALIGN64(p + M2_length_conv_l_tile);
    }
#undef _ALIGN64

    if (snrt_global_core_idx() == 0) init_cycle_counter();

    // Load L-independent operands once.
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_dt_w1, M2_dt_weight_1, M2_length_dt_weight_1);
        snrt_dma_start_1d(ptr_dt_w2, M2_dt_weight_2, M2_length_dt_weight_2);
        snrt_dma_start_1d(ptr_dt_bias, M2_dt_bias, M2_length_dt_bias);
        snrt_dma_start_1d(ptr_A, M2_suc_A, M2_length_A);
        snrt_dma_start_1d(ptr_D, M2_suc_D, M2_length_D);
        snrt_dma_start_1d(ptr_os_weight, M2_oscore_weight, M2_length_oscore_weight);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    uint32_t start_cycles = 0, simbacore_cycles = 0;
    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: p2-carry (L=%d, dModel=%d, dInner=%d, NB=%u, L_tile=%u, dbuf)\n\n", seqLen, dModel,
               dInner, M2_NB_L_TILES, M2_L_tile);
        if (bc_swizzle) write_csr(ADDR_REMAP_INDEX_READER_7, 1);  // BC read through the XOR swizzle
        set_streamer_p2_carry((uint32_t)ptr_oi[0], (uint32_t)ptr_os_weight, (uint32_t)ptr_dt[0], (uint32_t)ptr_dt_w1,
                              (uint32_t)ptr_dt_w2, (uint32_t)ptr_dt_bias, (uint32_t)ptr_A, (uint32_t)ptr_BC[0],
                              (uint32_t)ptr_D, (uint32_t)ptr_x[0], (uint32_t)ptr_z[0], (uint32_t)ptr_y[0],
                              (uint32_t)ptr_state);
        set_launch(0, ptr_oi, ptr_dt, ptr_BC, ptr_x, ptr_z, ptr_y);
        set_simbacore_csr(M3_OSGEMM, M2_L_tile, dModel, dInner, dtRank, dModel);  // shape; MODE rewritten per launch
        start_cycles = snrt_mcycle();
    }

    // ---- Phase A: sequence-tiled P2 (no isCore) with state carry. Launch l = osCore tile l + SUC tile l-1;
    // iteration i runs launch i-1 while the DM core loads for launch i and spills what launch i-2 produced. ----
    for (uint32_t i = 0; i < M2_NB_L_TILES + 3u; i++) {
        if (snrt_is_dm_core()) {
            if (i < M2_NB_L_TILES)
                snrt_dma_start_1d(ptr_oi[i & 1u], M2_oscore_in + i * M2_oscore_in_l_tile_offset,
                                  M2_length_oscore_in_l_tile);
            if (i >= 1u && i <= M2_NB_L_TILES) {
                uint32_t t = i - 1u;
                gather_dt_bc(t, ptr_dt[t & 1u], ptr_BC[t & 1u]);
                gather_conv_tile(t, ptr_x[t & 1u], M2_suc_x);
            }
            if (i >= 2u && i <= M2_NB_L_TILES + 1u) spill_conv_tile(i - 2u, ptr_z[i & 1u], ptr_z_l3);
            if (i >= 3u) spill_conv_tile(i - 3u, ptr_y[(i - 3u) & 1u], ptr_y_l3);
        }

        if (i >= 1u && i <= M2_NB_L_TILES + 1u && snrt_global_core_idx() == 0) {
            uint32_t l    = i - 1u;
            bool suc_on   = l >= 1u;
            uint32_t mode = !suc_on ? M3_OSGEMM : (l < M2_NB_L_TILES ? M33_PHASE2_NO_ISCORE : M27_SUC_ONLY);
            if (l >= 2u) mode |= state_load_bit;
            if (suc_on && l < M2_NB_L_TILES) mode |= state_save_bit;
            write_csr(MODE, mode);
            _set_streamer_start();
            _set_simbacore_start();
            if (suc_on) write_csr(DELAYED_START_READER_10, 1);  // z slot complete: release at once
            write_csr(STREAMER_START_CSR, 0);
            write_csr(SIMBACORE_START, 0);
            write_csr(DELAYED_START_READER_10, 0);
            // Next launch's program (latched at its START) while the accelerator runs.
            if (l < M2_NB_L_TILES) set_launch(l + 1u, ptr_oi, ptr_dt, ptr_BC, ptr_x, ptr_z, ptr_y);
            wait_simbacore_and_streamer();
            asm volatile("fence" ::: "memory");
            simbacore_cycles += read_simbacore_perf_counter();
        }

        if (snrt_is_dm_core()) snrt_dma_wait_all();
        snrt_cluster_hw_barrier();
    }
    if (snrt_global_core_idx() == 0) printf("Phase A done (z/y spilled to L3), starting out-proj (K-tiled)\n");

    // ---- Phase B: output projection out = y @ iscore_weight (+ bias), K-tiled isgemm. The out psum stays
    // FULL in TCDM and accumulates in place; y/weight are loaded one contiguous K-tile at a time (K is
    // convFormat-outermost). Non-final K-tiles run NO_REQUANT; the last applies the FP8 requant. Reuse all
    // TCDM (Phase A is done). Fail loudly if the full out psum + one K-tile does not fit.
    uint16_t* ptr_out = (uint16_t*)tcdm_base_ptr;
    uint8_t* ptr_y_kt;
    uint8_t* ptr_w_kt;
    {
#define _ALIGN64(x) ((uint8_t*)(((uintptr_t)(x) + 63u) & ~(uintptr_t)63u))
        ptr_y_kt = _ALIGN64((uint8_t*)ptr_out + M2_length_iscore_out);
        ptr_w_kt = _ALIGN64(ptr_y_kt + M2_length_op_a_tile);
#undef _ALIGN64
    }
    uint32_t outproj_bytes = (uint32_t)((ptr_w_kt + M2_length_op_b_tile) - (uint8_t*)tcdm_base_ptr);
    if (outproj_bytes > 480u * 1024u) {
        if (snrt_global_core_idx() == 0)
            printf(
                "FATAL: out-proj footprint %u B exceeds TCDM budget (L=%d, dModel=%d). Raise nb_op_k_tiles "
                "or tile the output.\n",
                outproj_bytes, seqLen, dModel);
        return 1;
    }

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d((uint8_t*)ptr_out, M2_iscore_bias, M2_length_iscore_out);  // psum bias init (full)
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    uint32_t simbacore_cycles_outproj = 0;
    for (uint32_t kt = 0; kt < M2_NB_OP_K_TILES; kt++) {
        if (snrt_is_dm_core()) {  // y / weight K-tiles are contiguous slices of the full L3 tensors
            snrt_dma_start_1d(ptr_y_kt, ptr_y_l3 + kt * M2_length_op_a_tile, M2_length_op_a_tile);
            snrt_dma_start_1d(ptr_w_kt, M2_iscore_weight + kt * M2_length_op_b_tile, M2_length_op_b_tile);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();
        if (snrt_global_core_idx() == 0) {
            set_isgemm_streamer_csr((uint32_t)ptr_y_kt, M2_R11_ss, M2_R11_kt_tb, M2_R11_kt_ts,  //
                                    (uint32_t)ptr_w_kt, M2_R12_ss, M2_R12_kt_tb, M2_R12_kt_ts,  //
                                    (uint32_t)ptr_out, M2_W3_ss, M2_W3_kt_tb, M2_W3_kt_ts);
            set_simbacore_csr((kt == M2_NB_OP_K_TILES - 1u) ? M4_ISGEMM : M5_ISGEMM_NO_REQUANT, seqLen, 1,
                              M2_op_dInner_tile, 1, dModel);
            start_simbacore_and_streamers(0, 0, 1, 0);  // release isCore input reader R11
            wait_simbacore_and_streamer();
            asm volatile("fence" ::: "memory");
            simbacore_cycles_outproj += read_simbacore_perf_counter();
        }
        snrt_cluster_hw_barrier();
    }

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_out_l3, (uint8_t*)ptr_out, M2_length_iscore_out);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    if (snrt_global_core_idx() == 0) {
        uint32_t end_cycles = snrt_mcycle();
        simbacore_cycles += simbacore_cycles_outproj;
        printf("[%d cc] Simbacore Phase A (sum over tiles): %u cycles\n", end_cycles,
               simbacore_cycles - simbacore_cycles_outproj);
        printf("[%d cc] Simbacore out-proj: %u cycles\n", end_cycles, simbacore_cycles_outproj);
        printf("[%d cc] Simbacore elapsed time: %u cycles\n", end_cycles, simbacore_cycles);
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);
        err += check_result_sample(ptr_z_l3, M2_oscore_expected, M2_test_samples_z, nb_test_samples, "z (osCore)");
        err += check_result_sample(ptr_y_l3, M2_suc_expected, M2_test_samples_y, nb_test_samples, "y (SUC)");
        err += check_result_sample((uint8_t*)ptr_out_l3, M2_iscore_expected, M2_test_samples_iscore_out,
                                   nb_test_samples, "out (projection)");
        printf("Test p2-carry: %s: %u/%d errors (L=%d, dModel=%d)\n", err ? "FAIL" : "PASS", err, 3 * nb_test_samples,
               seqLen, dModel);
    }
    snrt_cluster_hw_barrier();
    return err;
}
