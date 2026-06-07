// Copyright 2026 KU Leuven. memsim — per-cycle TCDM fabric + streamer engine.
// The real cycle-accurate memory model: 32-bank TCDM with per-bank arbitration,
// DMA-vs-streamer superbank preemption, and reader/writer streamers with AGUs +
// FIFOs. Bank conflicts, FIFO backpressure, and DMA preemption are produced by
// stepping this every cycle. See docs/dataflow/10_memsim.md.
#pragma once
#include <cstdint>
#include <vector>

#include "machine.hpp"  // Agu

// ---- 32-bank TCDM fabric with per-bank round-robin arbitration --------------
struct Fabric {
    static constexpr int NB = 32;          // banks
    static constexpr int SB = 8;           // banks per superbank (4 superbanks)
    uint32_t dma_owned = 0;                // bitmask of banks a DMA beat owns this cycle
    uint8_t rr[NB] = {0};                  // per-bank round-robin pointer (last winner+1)

    // requests posted this cycle: bank -> list of requester tokens
    int req_bank[64];                      // up to 64 simultaneous lane requests
    int req_tok[64];                       // requester token (encodes port<<4 | lane)
    int n_req = 0;

    void begin_cycle(uint32_t dma_mask) { dma_owned = dma_mask; n_req = 0; }
    void post(int bank, int tok) {
        if (n_req < 64) { req_bank[n_req] = bank & (NB - 1); req_tok[n_req] = tok; n_req++; }
    }
    // Resolve: returns a bitset (by request index) of which requests were GRANTED.
    // One winner per bank (round-robin); DMA-owned banks grant nobody.
    void arbitrate(bool granted[64]) {
        for (int i = 0; i < n_req; i++) granted[i] = false;
        for (int b = 0; b < NB; b++) {
            if (dma_owned & (1u << b)) continue;            // DMA preempts this bank
            // collect requesters for bank b
            int best = -1, best_rank = 1 << 30;
            for (int i = 0; i < n_req; i++) {
                if (req_bank[i] != b) continue;
                int rank = (req_tok[i] - rr[b] + 256) & 0xff;  // round-robin distance
                if (rank < best_rank) { best_rank = rank; best = i; }
            }
            if (best >= 0) { granted[best] = true; rr[b] = (uint8_t)(req_tok[best] + 1); }
        }
    }
};

// ---- Reader streamer: AGU (4 temporal dims + spatial lanes) + FIFO -----------
// Models: num_channel parallel lanes/cycle; a lane stalls (re-proposes next cycle)
// if it loses bank arbitration; the AGU group advances only when ALL its lanes are
// granted (so a 2-banks-for-4-lanes group takes 2 cycles). stride-0 INNER dim =
// reuse: the group is read once and replayed `reuse` times to the accelerator with
// NO new TCDM reads. Reader fills the FIFO up to fifo_depth ahead of the consumer.
struct CycReader {
    // config
    uint64_t base = 0;
    int32_t ts[4] = {0, 0, 0, 0};          // temporal strides (bytes)
    int eb[4] = {1, 1, 1, 1};              // effective temporal bounds (stride-0 dim0 -> 1)
    int num_channel = 1;
    int n_spatial = 1;
    int32_t sstride[2] = {8, 0};
    int sbound[2] = {1, 1};
    int fifo_depth = 4;
    int reuse = 1;                          // replays per group (stride-0 dim0 bound)

    // state
    int ti[4] = {0, 0, 0, 0};
    bool lane_done[8] = {false};           // granted lanes of the current group
    int port = 0;                          // for arbiter token
    bool active = false;
    bool done = false;
    int fifo_occ = 0;                       // beats buffered for the consumer
    int outstanding = 0;                    // granted reads in flight (+1cc)
    int pending_push = 0;                   // reads granted this cycle -> push next cycle
    int reuse_left = 0;                     // remaining replays of the buffered group

    // BIST: tag each delivered group with its production index so the consumer can
    // verify it receives groups in AGU order (a delay/reorder bug breaks the order).
    bool track_idx = false;                 // enable index tagging (BIST only)
    int prod_idx = 0;                       // next group's production index
    int pend_idx = -1;                      // index granted this cycle, lands next
    int idx_ring[64] = {0};                 // delivered indices, FIFO order
    int ir_head = 0, ir_tail = 0, ir_cnt = 0;
    bool pop_idx(int& v) {                  // BIST consumer pops one delivered index
        if (ir_cnt == 0) return false;
        v = idx_ring[ir_head]; ir_head = (ir_head + 1) & 63; ir_cnt--; return true;
    }

    void configure(const Agu& a, int nch, int nsp, int fd, int prt);
    // lane byte address at the current temporal position
    uint64_t lane_addr(int lane) const;
    // Phase 1 of a cycle: post bank requests for ungranted lanes (if FIFO has room).
    void propose(Fabric& f);
    // Phase 2: apply arbiter grants (granted[] indexed as posted), advance AGU.
    void commit(Fabric& f, const bool granted[64], int& grant_idx);
    // consumer pops one beat (returns true if a beat was available)
    bool pop();
    // call at top of cycle: land last cycle's granted reads into the FIFO
    void land_reads() {
        if (track_idx && pending_push && pend_idx >= 0) {
            idx_ring[ir_tail] = pend_idx; ir_tail = (ir_tail + 1) & 63; ir_cnt++; pend_idx = -1;
        }
        fifo_occ += pending_push; outstanding -= pending_push; pending_push = 0;
    }
    int total_read_groups() const { return eb[0] * eb[1] * eb[2] * eb[3]; }
    void reset() {                          // restart the AGU (continuous stream)
        for (int i = 0; i < 4; i++) ti[i] = 0;
        for (int l = 0; l < 8; l++) lane_done[l] = false;
        done = false; active = true;
    }
};

// ---- Writer streamer: AGU + output FIFO (drains the core's output to TCDM) -----
// The producing subcore pushes output groups into the FIFO; each cycle the writer
// proposes the current AGU group's lane addresses to the Fabric and, when all lanes
// are granted, advances the AGU and pops one group. A write that loses bank arbitration
// retries, so the post-compute output drain (the writer finishing after the core) is
// produced by stepping the writer, not added as a fixed cost.
struct CycWriter {
    uint64_t base = 0;
    int32_t ts[4] = {0, 0, 0, 0};
    int eb[4] = {1, 1, 1, 1};
    int num_channel = 1;
    int n_spatial = 1;
    int32_t sstride[2] = {8, 0};
    int sbound[2] = {1, 1};
    int port = 0;
    int ti[4] = {0, 0, 0, 0};
    bool lane_done[8] = {false};
    long fifo_occ = 0;     // groups produced by the core, waiting to be written
    long written = 0;      // groups committed to TCDM
    bool done = false;
    void configure(const Agu& a, int nch, int nsp, int prt);
    uint64_t lane_addr(int lane) const;
    void push(long groups) { fifo_occ += groups; }
    void propose(Fabric& f);
    void commit(Fabric& f, const bool granted[64], int& grant_idx);
    long total_beats() const { return (long)eb[0] * eb[1] * eb[2] * eb[3]; }
};

// Cycle-stepped GEMM invocation (osCore or isCore): the array consumes one group from
// each input reader per cycle (stalling if a reader's FIFO is empty from a bank
// conflict), accumulates K_i steps, and emits one output tile every K_i steps into the
// writer's FIFO; the writer drains to TCDM through the same Fabric. Returns the
// MambaCore-busy cycle count (perf counter = array active + output drain), both produced
// by stepping. dma_mask = banks a concurrent DMA owns. n_out_tiles = M_i*N_i; K_i =
// reduction steps per tile.
struct GemmResult { uint64_t busy; uint64_t end; };
GemmResult cyc_gemm(const Agu* in_readers, const int* rd_nch, const int* rd_nsp, int n_readers,
                    const Agu& out_writer, int w_nch, int w_nsp,
                    long n_out_tiles, long K_i, uint32_t dma_mask = 0);

// Bank-conflict probe (used by test/suc_grid_test.cpp): drive a reader (e.g. captured R7)
// per-cycle with a consumer that pops one beat every `consume_period` cycles, for `n_beats`
// beats. Returns the cycles until all beats are consumed -- a bank conflict slows delivery.
uint64_t cyc_reader_test(const Agu& r7_agu, int num_channel, int n_spatial, int fifo_depth,
                         int consume_period, int n_beats, uint32_t dma_mask = 0);

// Per-cycle SUC busy duration (models StateUpdateCore.scala): produces seqLen*dInner_tile
// outputs at 1/cycle, needing a fresh BC refresh (4 R7
// reads = B(VecN)+C(VecN)) every delaySU iterations. R7's reads go through the
// per-cycle 32-bank fabric, so the bank conflict (0.5 vs 1.0 read/cyc) and whether
// it starves the SUC come from arbitration on R7's real strides.
// dma_mask = banks a concurrent DMA owns (superbank preemption), 0 if none.
uint64_t cyc_suc_duration(const Agu& r7_agu, int seqLen, int dInner_tile, uint32_t dma_mask = 0);

// Timing-coupled BIST for the SUC dt_BC delivery path. The SUC's BC data flows
// through the per-cycle fabric+FIFO; each group's data is valid only AFTER its +1cc
// TCDM response lands. The BIST asserts three genuine, falsifiable properties:
//   (1) liveness/completeness: the SUC consumes every BC group with no deadlock
//       (a dropped/under-delivered group would stall it forever).
//   (2) delay-respect: with the correct +1cc latency, the consumer never reads an
//       un-landed (poison) slot.
//   (3) delay-sensitivity (the "a wrong delay -> error" guarantee): when the model
//       is perturbed to count an IN-FLIGHT (not-yet-landed) read as ready — i.e. a
//       delay modelled one cycle too SHORT — the consumer reads poison and it is
//       detected. This proves a wrong delay surfaces as an error.
// The data FIFO (landed-value ring) is the single occupancy ledger, so a producer/
// consumer count desync also surfaces as a poison read.
struct SucBistResult {
    bool clean_complete;     // clean run finished all outputs (no deadlock)
    bool clean_poison_free;  // clean run never read un-landed data
    bool fault_detected;     // too-short-delay perturbation -> poison read, caught
    long groups;             // BC groups consumed in the clean run
    int clean_poison_count;  // # un-landed reads in the clean run (should be 0)
    int fault_poison_count;  // # un-landed reads the injected too-short delay caused
    int n_idx;               // # of recorded offending BC-group indices below
    int first_idx[8];        // first offending BC-group (production) indices
};
SucBistResult cyc_suc_bist(const Agu& r7_agu, int seqLen, int dInner_tile);
