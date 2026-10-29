// Copyright 2025 KU Leuven.
// Not released under license. All rights reserved.
//
// Author: Robin Geens <robin.geens@kuleuven.be>
//
// Element-wise FP8 add out = a + b over nb_tiles tiles, double-buffered through TCDM: one SIMD_ADD_FP8
// pass per tile. Every tile streams the same golden tile from L3.
// See docs/dataflow/23_add_tiled.md.

#include <stdint.h>

#include "../data/data.h"
#include "snax-simbacore-lib.h"

// Inlined start/wait as in rmsnorm-tiled: a SIMD pass never uses the R10/R11 delayed streamers.
// START is a pulse -- the streamer latches its config at the rising edge.
static inline void simd_start(uint32_t mode) {
    write_csr(MODE, mode);  // MODE is read in real time, must be set before START (never preloaded)
    _set_streamer_start();
    _set_simbacore_start();
    write_csr(STREAMER_START_CSR, 0);
    write_csr(SIMBACORE_START, 0);
}
static inline void simd_wait(void) {
    while (read_csr(SIMBACORE_BUSY));
    while (read_csr(STREAMER_BUSY_CSR));
}

int test() {
    int err = 0;

    // TCDM layout: two a slots (the sum is written in place) + two b slots.
    void* base = snrt_l1_next();
    uint8_t* ptr_a[2];
    ptr_a[0] = (uint8_t*)base;
    ptr_a[1] = ptr_a[0] + M8_length_tile;
    // +32 B: shift the b slots' bank group off the a slots' so R13 does not contend with R7/W3
    uint8_t* ptr_b[2];
    ptr_b[0] = ptr_a[1] + M8_length_tile + 32;
    ptr_b[1] = ptr_b[0] + M8_length_tile;

    uint32_t start_cycles           = 0;
    uint32_t simbacore_cycles_total = 0;

    // L3 staging buffer for the output
    static uint8_t* l3_out;

    if (snrt_global_core_idx() == 0) {
        printf("\nStarting program: FP8 add tiled (nb_tiles=%u x %u B)\n\n", nb_tiles, M8_length_tile);
        printf("Expected L1 TCDM usage: %u B (%u KiB)\n", (uint32_t)L1_TCDM_PEAK_BYTES,
               (uint32_t)(L1_TCDM_PEAK_BYTES / 1024));
        init_cycle_counter();
        start_cycles = snrt_mcycle();

        // Reserve 16 KiB so snrt_l3alloc does not overlap the runtime's putc_buffer.
        (void)snrt_l3alloc(16 * 1024);
        l3_out = (uint8_t*)snrt_l3alloc((uint32_t)nb_tiles * M8_length_tile);
    }

    // First tile in
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(ptr_a[0], (uint8_t*)M8_in_a_fp8, M8_length_tile);
        snrt_dma_start_1d(ptr_b[0], (uint8_t*)M8_in_b_fp8, M8_length_tile);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    // Double-buffered pipeline. While the SIMD core adds tile n (slot n&1), the DM core
    // (concurrently) DMAs tile n-1's result out and tile n+1's inputs in (both via slot (n+1)&1).
    for (uint32_t n = 0; n < nb_tiles; n++) {
        uint32_t s     = n & 1;
        uint32_t other = s ^ 1;

        if (snrt_is_dm_core()) {
            if (n >= 1) snrt_dma_start_1d(l3_out + (n - 1) * M8_length_tile, ptr_a[other], M8_length_tile);
            if (n + 1 < nb_tiles) {
                snrt_dma_start_1d(ptr_a[other], (uint8_t*)M8_in_a_fp8, M8_length_tile);
                snrt_dma_start_1d(ptr_b[other], (uint8_t*)M8_in_b_fp8, M8_length_tile);
            }
        }

        if (snrt_global_core_idx() == 0) {
            set_simd_streamer_csr((uint32_t)ptr_a[s], M8_R7_a_ss, M8_R7_a_tb, M8_R7_a_ts,     //
                                  (uint32_t)ptr_b[s], M8_R13_b_ss, M8_R13_b_tb, M8_R13_b_ts,  //
                                  (uint32_t)ptr_a[s], M8_W3_out_ss, M8_W3_out_tb, M8_W3_out_ts);
            simd_start(M16_SIMD_ADD_FP8);
            simd_wait();
            // Drain the SIMD write buffer to TCDM before the DM core DMAs this slot out.
            asm volatile("fence" ::: "memory");
            simbacore_cycles_total += read_simbacore_perf_counter();
        }

        snrt_cluster_hw_barrier();                   // compute tile n done
        if (snrt_is_dm_core()) snrt_dma_wait_all();  // tile n+1 inputs ready, tile n-1 output drained
        snrt_cluster_hw_barrier();
    }

    // Flush the last tile's output (issued after the loop since it has no successor iteration).
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(l3_out + (nb_tiles - 1) * M8_length_tile, ptr_a[(nb_tiles - 1) & 1], M8_length_tile);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    if (snrt_global_core_idx() == 0) {
        uint32_t end_cycles = snrt_mcycle();
        printf("[%u cc] Simbacore elapsed time: %u cycles\n", end_cycles, simbacore_cycles_total);
        printf("[%u cc] Snitch elapsed time: %u cycles\n", end_cycles, end_cycles - start_cycles);

        for (uint32_t n = 0; n < nb_tiles; n++)
            err += check_result_sample(l3_out + n * M8_length_tile, M8_add_out_fp8, M8_test_samples_out,
                                       nb_test_samples, "out");

        printf("Test FP8 add tiled: numElem=%u per tile, nb_tiles=%u\n", numElem, nb_tiles);
        printf("%s: %u/%u errors.\n", err ? "FAIL" : "PASS", err, nb_tiles * nb_test_samples);
    }

    snrt_cluster_hw_barrier();
    return err;
}

int main() { return test(); }
