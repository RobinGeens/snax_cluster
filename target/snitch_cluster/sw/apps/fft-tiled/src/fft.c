// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// Tiled version of the FFT program. Reduces TCDM peak usage by keeping the
// Phase A working set per-tile and the Phase B working set per-K-tile.
//
// (Naming caveat: "L1" here means the TCDM scratchpad, not the FFT inner
// butterfly axis L1. The FFT axes are L1 and L2 = inner/outer butterfly.)
//
// Tile structure (different axis per phase, by IS-core capability):
//   Phase A (partition1 + hadamard + reorder): dModel-axis tiled. `in` lives
//     in L3 as [L2*dModel][L1] col-major with d outer in the N axis, so a
//     dModel-slice is a contiguous L3 chunk. Each tile DMAs its slice in,
//     runs partition1 + hadamard + reorder against per-tile TCDM slots, then
//     DMAs the per-tile reorder output into the [d][l]-col-major
//     hadamard_reordered_l3 region.
//   Phase B (partition2): K-AXIS tiled, mirroring isgemm-tiled. The IS-core
//     requantizes its "last K iteration" to produce the final output; tiling
//     N corrupts that requant timing (each tile sees an artificial "last
//     iteration"), so we tile K instead and accumulate in place. Each tile
//     consumes one K-macro of weight2 and one K-macro of hadamard_reordered
//     (re or im), with R13 (read previous psum) → W3 (write new psum)
//     hitting the SAME FULL partition2_out address in TCDM. Non-final tiles
//     use M30_ISGEMM_SQ_NO_REQUANT (keep psum in BF16); the final tile uses
//     M6_ISGEMM_SQ to apply the requant on the fully accumulated psum.
//
// TCDM layout (Phase B working set overlays Phase A working set):
//   Always live: weight1, weight2 (FULL, sliced by BASE_PTR update per tile),
//                twiddles
//   Phase A working set (single-buffered, reused across tiles):
//     in_tile, partition1_out_tile, hadamard_out_tile, had_reord_a_tile
//   Phase B working set (overlays the Phase A region after barrier):
//     had_reord_b_ktile (single K-macro: reals OR imags),
//     partition2_out (FULL, psum accumulator across K-tiles)
//
// L3 layout (allocated via snrt_l3alloc):
//   hadamard_reordered_l3 (16 KiB, [reals | imags] each [d][l] col-major)
//
// Phase A reorder writes the tile's [reals | imags] contiguously to its TCDM
// tile slot. The per-tile DMA-out is two 1D DMAs (one per re/im) into the
// matching position of the L3 [d][l] col-major layout — dModel tiling makes
// each tile's contribution a contiguous d-row range, so no scatter is
// required.
//
// IS GeMM psum init: partition1_out_tile is an accumulator and MUST be zero
// before each tile's GEMM (otherwise we accumulate into uninitialized memory
// or prior tile data, which triggers vsim RegWriteKnown asserts). We DMA
// snrt_zero_memory_ptr() into it (and into the SIMD intermediate slots, as a
// defensive measure) at the start of each tile's DMA-in stage.
//
// Memory-layout references used to design this structure:
//   docs/memory_layouts/09_fft.md   — partitioned DFT layout, especially §9.6
//                                     on the [d][l] col-major reorder output.
//   docs/memory_layouts/10_simd.md  — SIMD CMul lane layout.

#include "../data/data.h"
#include "snax-simbacore-lib.h"

static inline uint32_t align64(uint32_t x) { return (x + 63u) & ~63u; }

int test() {
    int err = 0;

    // ---------- L3 buffers --------------------------------------------------
    // Static so all cores see the same allocated address (snrt_l3alloc updates
    // a shared global allocator, but the returned address must be communicated
    // to other cores; locals on the stack would not be shared across cores).
    static uint8_t* ptr_hadamard_reordered_l3 = NULL;
    if (snrt_global_core_idx() == 0) {
        ptr_hadamard_reordered_l3 = (uint8_t*)snrt_l3alloc(M6_length_hadamard_reordered);
    }
    snrt_cluster_hw_barrier();

    // ---------- TCDM allocation --------------------------------------------
    // Always-live region. ptr_twiddles_tiled holds nb_tiles packed-pair blocks
    // (256 B each for L1=16) that the SUC can read directly during hadamard.
    void* tcdm_base_ptr        = snrt_l1_next();
    uint8_t* ptr_weight1        = (uint8_t*)tcdm_base_ptr;
    uint8_t* ptr_weight2        = ptr_weight1 + align64(M6_length_weight1);
    uint8_t* ptr_twiddles_tiled = ptr_weight2 + align64(M6_length_weight2);

    // Working region: shared by Phase A and Phase B (overlay after barrier).
    uint8_t* ptr_work_base = ptr_twiddles_tiled + align64(M6_length_twiddles_tiled_total);

    // ---- Phase A working slots ----
    uint8_t* ptr_in_tile             = ptr_work_base;
    uint8_t* ptr_partition1_out_tile = ptr_in_tile + align64(M6_length_in_tile);
    uint8_t* ptr_hadamard_out_tile   = ptr_partition1_out_tile + align64(M6_length_partition1_out_tile);
    uint8_t* ptr_had_reord_a_tile    = ptr_hadamard_out_tile + align64(M6_length_hadamard_out_tile);
    // Phase A peak end:
    // (uint8_t*)ptr_had_reord_a_tile + M6_length_hadamard_reordered_tile

    // ---- Phase B working slots (overlay starting at ptr_work_base) ----
    // K-axis tiled: per-tile B slot (one K-macro = reals OR imags of had_reord),
    // FULL partition2_out psum accumulator (R13/W3 hit the same address each tile).
    uint8_t* ptr_had_reord_b_ktile = ptr_work_base;
    uint8_t* ptr_partition2_out    = ptr_had_reord_b_ktile + align64(M6_length_hadamard_reordered_ktile);

    if (snrt_global_core_idx() == 0) init_cycle_counter();
    snrt_cluster_hw_barrier();

    // ---------- Preload always-live small inputs: weights + twiddles ----
    // FFT twiddles only depend on (l1, l2), not on d. With dModel tiling, all
    // tiles share the SAME twiddle table — no relayout needed.
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_weight1, M6_dft_weight1, M6_length_weight1);
        snrt_dma_start_1d(ptr_weight2, M6_dft_weight2, M6_length_weight2);
        snrt_dma_start_1d(ptr_twiddles_tiled, M6_twiddles, M6_length_twiddles);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    uint32_t start_cycles = 0;
    uint32_t simbacore_cycles_phaseA = 0;
    uint32_t simbacore_cycles_phaseB = 0;
    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: tiled FFT (nb_tiles=%d, L3-backed intermediates)\n\n", nb_tiles);
        start_cycles = snrt_mcycle();
    }

    // ========================================================================
    // Phase A tile loop: partition1 + hadamard + reorder, tile along L2
    // ========================================================================
    for (uint32_t tile = 0; tile < nb_tiles; tile++) {
        // ---- DMA-in `in` tile slice + zero-init the partition 1 psum slot ----
        // partition1_out_tile is the IS GeMM psum accumulator; it MUST be zero
        // before each tile's GEMM (otherwise we accumulate into prior tile data
        // or, on the first tile, into uninitialized TCDM).
        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(ptr_in_tile, M6_dft_in + tile * M6_length_in_tile, M6_length_in_tile);
            snrt_dma_start_1d(ptr_partition1_out_tile, (void*)snrt_zero_memory_ptr(),
                              M6_length_partition1_out_tile);
            // Defensive: zero-init the SIMD intermediate slots so residual
            // bytes from a prior tile cannot leak into the current tile's output.
            snrt_dma_start_1d(ptr_hadamard_out_tile, (void*)snrt_zero_memory_ptr(),
                              M6_length_hadamard_out_tile);
            snrt_dma_start_1d(ptr_had_reord_a_tile, (void*)snrt_zero_memory_ptr(),
                              M6_length_hadamard_reordered_tile);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();

        if (snrt_global_core_idx() == 0) {
            // Twiddles are shared across tiles (depend only on (l1, l2), not d).
            uint8_t* ptr_twiddles_tile = ptr_twiddles_tiled;

            // ---- Step 1: partition 1 (per-tile slice along L2) ----
            set_isgemm_streamer_csr((uint32_t)ptr_weight1, M6_R11_1_ss, M6_R11_1_tb, M6_R11_1_ts,            // A
                                    (uint32_t)ptr_in_tile, M6_R12_1_ss, M6_R12_1_tb, M6_R12_1_ts,           // B
                                    (uint32_t)ptr_partition1_out_tile, M6_W3_1_ss, M6_W3_1_tb, M6_W3_1_ts); // C/D
            set_simbacore_csr(M7_ISGEMM_SQ_TRANSPOSE, 2 * L1, 1, L1_padded, 1, M6_N_1_tile);
            start_simbacore_and_streamers(M6_R10_en, 0, 1, 0);
            wait_simbacore_and_streamer();
            simbacore_cycles_phaseA += read_simbacore_perf_counter();

            // ---- Step 2: Hadamard (CMUL on per-tile slice) ----
            set_simd_streamer_csr((uint32_t)ptr_partition1_out_tile, M6_R7_2_ss, M6_R7_2_tb, M6_R7_2_ts,
                                  (uint32_t)ptr_twiddles_tile, M6_R13_2_ss, M6_R13_2_tb, M6_R13_2_ts,
                                  (uint32_t)ptr_hadamard_out_tile, M6_W3_2_ss, M6_W3_2_tb, M6_W3_2_ts);
            set_simbacore_csr(M20_SIMD_CMUL_FP8, 0, 0, 0, 0, 0);
            start_simbacore_and_streamers(0, 0, 0, 0);
            wait_simbacore_and_streamer();
            simbacore_cycles_phaseA += read_simbacore_perf_counter();

            // ---- Step 2B: reorder. Writes [tile_reals|tile_imags] to TCDM tile slot ----
            set_simd_streamer_no_b((uint32_t)ptr_hadamard_out_tile, M6_R7_2B_ss, M6_R7_2B_tb, M6_R7_2B_ts,
                                   (uint32_t)ptr_had_reord_a_tile, M6_W3_2B_ss, M6_W3_2B_tb, M6_W3_2B_ts);
            set_simbacore_csr(M23_SIMD_NOOP_FP8, 0, 0, 0, 0, 0);
            start_simbacore_and_streamers(0, 0, 0, 0);
            wait_simbacore_and_streamer();
            simbacore_cycles_phaseA += read_simbacore_perf_counter();
        }
        snrt_cluster_hw_barrier();

        // ---- DMA-out tile slot to L3 standard layout [d][l] col-major ----
        // dModel tiling = contiguous d-row slice in L3. Tile k contributes d-rows
        // [k*dModel_tile, (k+1)*dModel_tile), each L bytes, contiguous. Per re/im
        // = 1D contiguous DMA of dModel_tile*L bytes.
        if (snrt_is_dm_core()) {
            uint8_t* reals_dst = ptr_hadamard_reordered_l3
                                 + tile * M6_phaseA_dma_per_tile_per_part;
            uint8_t* imags_dst = reals_dst + (M6_length_hadamard_reordered / 2);
            uint8_t* reals_src = ptr_had_reord_a_tile;
            uint8_t* imags_src = reals_src + M6_length_hadamard_reordered_tile_re;

            snrt_dma_start_1d(reals_dst, reals_src, M6_phaseA_dma_per_tile_per_part);
            snrt_dma_start_1d(imags_dst, imags_src, M6_phaseA_dma_per_tile_per_part);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();
    }

    // ========================================================================
    // Phase B: partition 2, K-axis tiled (mirrors isgemm-tiled).
    //
    // Streamer CSRs are set ONCE outside the loop with per-K-tile bounds. Each
    // tile updates only:
    //   - BASE_PTR_READER_11 → weight2 + tile * length_weight2_ktile   (A K-slice)
    //   - BASE_PTR_READER_12 → ptr_had_reord_b_ktile                    (B is loaded fresh)
    //   - MODE              → NO_REQUANT (non-final) or ISGEMM_SQ (final)
    //
    // Per-tile DMA-in: copy the corresponding K-slice (tile 0 = reals,
    // tile 1 = imags) of hadamard_reordered from L3 into TCDM at the SAME
    // address (overwriting the previous tile's data). The streamer reads
    // K_2_t=1 macros starting from BASE_PTR_READER_12.
    //
    // The psum (partition2_out FULL in TCDM) accumulates in place across
    // tiles. We zero-init it once before the loop.
    // ========================================================================
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_partition2_out, (void*)snrt_zero_memory_ptr(), M6_length_partition2_out);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    if (snrt_global_core_idx() == 0) {
        // One-time streamer + IS-core setup. R11/R12 base pointers and MODE are
        // overridden per tile; everything else (bounds, strides, W3 base, dInner_tile)
        // is constant.
        set_isgemm_streamer_csr((uint32_t)ptr_weight2,             M6_R11_3_ss, M6_R11_3_tb, M6_R11_3_ts,
                                (uint32_t)ptr_had_reord_b_ktile,   M6_R12_3_ss, M6_R12_3_tb, M6_R12_3_ts,
                                (uint32_t)ptr_partition2_out,      M6_W3_3_ss,  M6_W3_3_tb,  M6_W3_3_ts);
        set_simbacore_csr(M6_ISGEMM_SQ, 2 * L2, 1, M6_dInner_2_tile, 1, dModel * L1);
    }
    snrt_cluster_hw_barrier();

    for (uint32_t tile = 0; tile < nb_tiles; tile++) {
        // ---- DMA-in this K-tile's slice of hadamard_reordered ----
        // L3 layout: bytes [0, len/2) = reals (K-macro 0), bytes [len/2, len) = imags (K-macro 1).
        if (snrt_is_dm_core()) {
            snrt_dma_start_1d(ptr_had_reord_b_ktile,
                              ptr_hadamard_reordered_l3 + tile * M6_length_hadamard_reordered_ktile,
                              M6_length_hadamard_reordered_ktile);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();

        if (snrt_global_core_idx() == 0) {
            // Update BASE_PTR_READER_11 to this tile's K-slice of weight2.
            // (weight2 is K-major contiguous; per-tile slice = nb_tiles-th chunk.)
            write_csr(BASE_PTR_READER_11_LOW, (uint32_t)(ptr_weight2 + tile * M6_length_weight2_ktile));
            write_csr(BASE_PTR_READER_12_LOW, (uint32_t)ptr_had_reord_b_ktile);
            // Non-final tiles keep psum in BF16; final tile applies requant.
            write_csr(MODE, (tile == nb_tiles - 1) ? M6_ISGEMM_SQ : M30_ISGEMM_SQ_NO_REQUANT);
            start_simbacore_and_streamers(M6_R10_en, 0, 1, 0);
            wait_simbacore_and_streamer();
            simbacore_cycles_phaseB += read_simbacore_perf_counter();
        }
        snrt_cluster_hw_barrier();
    }

    // ---------- Verify ------------------------------------------------------
    if (snrt_global_core_idx() == 0) {
        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Simbacore Phase A (sum over tiles, 3 steps each): %u cycles\n", end_cycles,
               simbacore_cycles_phaseA);
        printf("[%d cc] Simbacore Phase B (sum over tiles): %u cycles\n", end_cycles, simbacore_cycles_phaseB);
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        err += check_result_sample(ptr_hadamard_reordered_l3, M6_hadamard_reordered,
                                   M6_test_samples_expected, nb_test_samples, "hadamard_reordered (L3)");
        err += check_result_sample(ptr_partition2_out, M6_partition2_expected,
                                   M6_test_samples_expected, nb_test_samples, "partition2_out (TCDM)");

        printf("Test FFT tiled (Phase A dModel-tiled, Phase B K-tiled): (%d x %d), nb_tiles=%d\n", seqLen,
               dModel, nb_tiles);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, 2 * nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test(); }
