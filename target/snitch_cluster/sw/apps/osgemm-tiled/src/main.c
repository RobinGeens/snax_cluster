// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// Energy-optimised GEMM on the OS-core alone. See docs/dataflow/01_oscore_kernels.md.

#include "data.h"
#include "snax-simbacore-lib.h"

// Number of pipeline stages (transfer_in, compute, transfer_out). This is a property of the algorithm, do not change
#define NB_STAGES 3
#define MU 16u                                // seqLenUnroll: tokens per M-block
#define NU 24u                                // dInnerUnroll: channels per N-block
#define TILE_BYTES (MU * NU)                  // one FP8 output tile
#define ALIGN64(x) (((uint32_t)(x) + 63u) & ~63u)

// Same split as split() in data/memory_model.py
static inline uint32_t part_size(uint32_t total, uint32_t n, uint32_t i) { return total / n + (i < total % n); }
static inline uint32_t part_offset(uint32_t total, uint32_t n, uint32_t i) {
    uint32_t r = total % n;
    return i * (total / n) + (i < r ? i : r);
}

static uint32_t MB, NB;  // M-blocks and N-blocks of the call

static inline uint32_t chunk_blocks(uint32_t c) { return part_size(MB, nb_l_tiles, c); }
static inline uint32_t tile_blocks(uint32_t j) { return part_size(NB, nb_tiles, j); }

// Launch t -> activation chunk c, weight tile j
static inline void launch_of(uint32_t t, uint32_t* c, uint32_t* j) {
    uint32_t n_inner = a_stationary ? nb_tiles : nb_l_tiles;
    uint32_t outer = t / n_inner, inner = t % n_inner;
    *c = a_stationary ? outer : inner;
    *j = a_stationary ? inner : outer;
}

static inline void dma_in_a(uint8_t* dst, uint32_t c) {
    snrt_dma_start_1d(dst, M3_A + part_offset(MB, nb_l_tiles, c) * dim1 * MU, chunk_blocks(c) * dim1 * MU);
}
static inline void dma_in_b(uint8_t* dst, uint32_t j) {
    snrt_dma_start_1d(dst, M3_B + part_offset(NB, nb_tiles, j) * dim1 * NU, tile_blocks(j) * dim1 * NU);
}
static inline void dma_in_outer(uint8_t* dst, uint32_t o) { a_stationary ? dma_in_a(dst, o) : dma_in_b(dst, o); }
static inline void dma_in_inner(uint8_t* dst, uint32_t s) { a_stationary ? dma_in_b(dst, s) : dma_in_a(dst, s); }

static inline void dma_out_d(uint8_t* l3_d, uint8_t* src, uint32_t t) {
    uint32_t c, j;
    launch_of(t, &c, &j);
    uint32_t row = chunk_blocks(c) * TILE_BYTES;
    uint8_t* dst = l3_d + (part_offset(NB, nb_tiles, j) * MB + part_offset(MB, nb_l_tiles, c)) * TILE_BYTES;
    snrt_dma_start_2d(dst, src, row, MB * TILE_BYTES, row, tile_blocks(j));
}

int test_osgemm_tiled() {
    int err = 0;
    MB = dim0 / MU;
    NB = dim2 / NU;
    const uint32_t n_inner = a_stationary ? nb_tiles : nb_l_tiles;
    const uint32_t T       = nb_tiles * nb_l_tiles;  // launches

    static uint8_t* l3_d = NULL;
    if (snrt_global_core_idx() == 0) {
        (void)snrt_l3alloc(16 * 1024);  // reserve putc_buffer region
        l3_d = (uint8_t*)snrt_l3alloc(M3_length_d);
    }

    const uint32_t len_outer = a_stationary ? M3_length_a_chunk : M3_length_b_tile;
    const uint32_t len_inner = a_stationary ? M3_length_b_tile : M3_length_a_chunk;
    uint8_t* ptr_outer       = (uint8_t*)ALIGN64(snrt_l1_next());
    uint8_t* ptr_inner[2];
    ptr_inner[0] = (uint8_t*)ALIGN64(ptr_outer + len_outer);
    ptr_inner[1] = M3_nb_inner_bufs > 1 ? (uint8_t*)ALIGN64(ptr_inner[0] + len_inner) : ptr_inner[0];
    uint8_t* ptr_d[2];
    ptr_d[0] = (uint8_t*)ALIGN64(ptr_inner[1] + len_inner);
    ptr_d[1] = M3_nb_d_bufs > 1 ? (uint8_t*)ALIGN64(ptr_d[0] + M3_length_d_tile) : ptr_d[0];
#define INNER_BUF(t) ptr_inner[n_inner > 1 ? (t) % 2 : 0]
#define D_BUF(t) ptr_d[(t) % M3_nb_d_bufs]

    if (snrt_global_core_idx() == 0) init_cycle_counter();
    snrt_cluster_hw_barrier();

    // Set up for launch 0, the largest
    uint32_t cur_mb = part_size(MB, nb_l_tiles, 0), cur_nb = part_size(NB, nb_tiles, 0);
    if (snrt_global_core_idx() == 0) {
        set_osgemm_streamer_csr((uint32_t)ptr_outer, M3_R0_ss, M3_R0_tb, M3_R0_ts,      //
                                (uint32_t)ptr_outer, M3_R1_ss, M3_R1_tb, M3_R1_ts,      //
                                (uint32_t)ptr_d[0], M3_W0_ss, M3_W0_tb, M3_W0_ts);  //
        set_simbacore_csr(M3_OSGEMM, cur_mb * MU, dim1, cur_nb * NU, 1, 1);
    }
    snrt_cluster_hw_barrier();

    uint32_t start_cycles           = 0;
    uint32_t simbacore_cycles_total = 0;
    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: OSGeMM tiled (%s stationary, nb_l_tiles=%d, nb_tiles=%d)\n\n",
               a_stationary ? "activation" : "weight", nb_l_tiles, nb_tiles);
        printf("Expected L1 TCDM usage: %u B (%u KiB)\n", (uint32_t)L1_TCDM_PEAK_BYTES,
               (uint32_t)(L1_TCDM_PEAK_BYTES / 1024));
        start_cycles = snrt_mcycle();
    }

    for (uint32_t i = 0; i < T + NB_STAGES - 1; i++) {
        // Launch i-1 opens a new resident chunk/tile; launch i-2 was the last to read the old one
        if (i >= 2 && i <= T && (i - 1) % n_inner == 0) {
            if (snrt_is_dm_core()) {
                dma_in_outer(ptr_outer, (i - 1) / n_inner);
                snrt_dma_wait_all();
            }
            snrt_cluster_hw_barrier();
        }

        // Stage 1: transfer_in launch i
        if (snrt_is_dm_core()) {
            if (i == 0) dma_in_outer(ptr_outer, 0);
            if (i < T && (n_inner > 1 || i == 0)) dma_in_inner(INNER_BUF(i), i % n_inner);
        }

        // Stage 2: compute launch i-1
        if (i >= 1 && i <= T && snrt_global_core_idx() == 0) {
            uint32_t t = i - 1, c, j;
            launch_of(t, &c, &j);
            uint32_t mb = chunk_blocks(c), nb = tile_blocks(j);
            if (mb != cur_mb || nb != cur_nb) {  // a smaller remainder chunk/tile: rewrite the bounds
                write_csr(T_BOUND_BASE_READER_0 + 1, mb);
                write_csr(T_BOUND_BASE_READER_0 + 2, nb);
                write_csr(T_BOUND_BASE_READER_1 + 1, mb);
                write_csr(T_BOUND_BASE_READER_1 + 2, nb);
                write_csr(T_BOUND_BASE_WRITER_0 + 0, M3_d_beats * mb * nb);
                set_simbacore_csr(M3_OSGEMM, mb * MU, dim1, nb * NU, 1, 1);
                cur_mb = mb;
                cur_nb = nb;
            }
            write_csr(BASE_PTR_READER_0_LOW, (uint32_t)(a_stationary ? ptr_outer : INNER_BUF(t)));
            write_csr(BASE_PTR_READER_1_LOW, (uint32_t)(a_stationary ? INNER_BUF(t) : ptr_outer));
            write_csr(BASE_PTR_WRITER_0_LOW, (uint32_t)D_BUF(t));
            start_simbacore_and_streamers(M3_R10_en, 0, M3_R11_en, 0);
            wait_simbacore_and_streamer();
            simbacore_cycles_total += read_simbacore_perf_counter();
        }

        // Stage 3: transfer_out launch i-2
        if (i >= 2 && snrt_is_dm_core()) dma_out_d(l3_d, D_BUF(i - 2), i - 2);

        if (snrt_is_dm_core()) snrt_dma_wait_all();
        snrt_cluster_hw_barrier();
    }

    // --- Verification ---
    if (snrt_global_core_idx() == 0) {
        uint32_t end_cycles = snrt_mcycle();
        printf("[%d cc] Simbacore elapsed time: %u cycles\n", end_cycles, simbacore_cycles_total);
        printf("[%d cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        err += check_result_sample(l3_d, M3_D, M3_test_samples_D, nb_test_samples, "out");

        printf("Test OSGeMM tiled: dim0=%d, dim1=%d, dim2=%d, nb_l_tiles=%d, nb_tiles=%d, a_stationary=%d\n", dim0,
               dim1, dim2, nb_l_tiles, nb_tiles, a_stationary);
        printf("%s: %u/%d errors.\n", err ? "FAIL" : "PASS", err, nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test_osgemm_tiled(); }
