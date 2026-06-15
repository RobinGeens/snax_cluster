// Copyright 2026 KU Leuven. memsim — per-cycle TCDM fabric + streamer engine.
// The real cycle-accurate memory model: 32-bank TCDM with per-bank arbitration,
// DMA-vs-streamer superbank preemption, and reader/writer streamers with AGUs +
// FIFOs. Bank conflicts, FIFO backpressure, and DMA preemption are produced by
// stepping this every cycle. See docs/dataflow/10_memsim.md.
#pragma once
#include <array>
#include <cstdint>
#include <vector>

#include "machine.hpp"  // Agu

// One row of the per-cycle streamer-FIFO occupancy trace: index = port (R0..R13 = 0..13,
// W0..W3 = 14..17), value = elements in that port's FIFO this cycle, or -1 if the port is
// absent from this invocation. Matches AccelEngine's EngineResult::fifo so all stepper paths
// (engine P1/P2, cyc_gemm, cyc_simd) feed the same .fifo.csv plot.
using FifoRow = std::array<int16_t, 18>;

// Per-port streamer FIFO depth — the SINGLE source of truth for every streamer depth in the model
// (CycReader data_depth=addr_depth=fd, CycWriter.depth, and the R7 hand-path r7_fifo_d_). Values are
// verbatim from the cluster config cfg/snax_simbacore_cluster.hjson:
//   data_reader_params.fifo_depth = [8,3,8,1,6,3,2,4,1,6,7,2,6,4]   (R0..R13)
//   data_writer_params.fifo_depth = [4,8,3,4]                       (W0..W3)
// (addressBufferDepth and dataBufferDepth are both this value per port — StreamParamGen mirrors the
// hjson.) Index = streamer port: readers R0..R13 = 0..13, writers W0..W3 = 14..17. The depths are
// heterogeneous (osCore weight R1 = 3, dt_BC R7 = 4, isCore weight R12 = 6), so a uniform depth
// over-hides contention on the shallow ports. KEEP THIS ARRAY IN SYNC WITH THE HJSON.
inline int snax_streamer_depth(int port_idx) {
    static const int D[18] = {8, 3, 8, 1, 6, 3, 2, 4, 1, 6, 7, 2, 6, 4,  // R0..R13  (reader fifo_depth)
                              4, 8, 3, 4};                                // W0..W3   (writer fifo_depth)
    return (port_idx >= 0 && port_idx < 18) ? D[port_idx] : 4;
}

// ---- 32-bank TCDM fabric with per-bank round-robin arbitration --------------
struct Fabric {
    static constexpr int NB = 32;          // banks
    static constexpr int SB = 8;           // banks per superbank (4 superbanks)
    uint32_t dma_owned = 0;                // bitmask of banks a DMA beat owns this cycle
    uint8_t rr[NB] = {0};                  // per-bank round-robin pointer (last winner+1)

    // requests posted this cycle: bank -> list of requester tokens
    int req_bank[64];                      // up to 64 simultaneous lane requests
    int req_tok[64];                       // requester token (encodes port<<4 | lane)
    int req_prio[64];                      // 1 = urgent (reader FIFO near-empty / writer near-full)
    int n_req = 0;

    // --- optional per-bank contention histogram (MEMSIM_BANKHIST), debug-only ---
    static bool h_on;
    static long h_req[NB];                  // total lane-requests landing on each bank
    static long h_conf[NB];                 // requests that lost arbitration on each bank
    static long h_portbank[18][NB];         // per-port (R0..W3) request count per bank
    static void hist_reset();
    static void hist_dump(const char* tag);

    void begin_cycle(uint32_t dma_mask) { dma_owned = dma_mask; n_req = 0; }
    void post(int bank, int tok, int prio = 0) {
        if (n_req < 64) {
            req_bank[n_req] = bank & (NB - 1);
            req_tok[n_req]  = tok;
            req_prio[n_req] = prio;
            n_req++;
        }
    }
    // Resolve: returns a bitset (by request index) of which requests were GRANTED. One winner per
    // bank: the real RTL is a PriorityRoundRobinArbiter (SparseInterconnect.scala) — the highest
    // priority masks the rest, then round-robin among ties. Priority is the streamers' buffer-urgency
    // QoS (ComplexQueue.scala:63: reader near-empty / writer near-full). DMA-owned banks grant nobody.
    void arbitrate(bool granted[64]) {
        for (int i = 0; i < n_req; i++) granted[i] = false;
        for (int b = 0; b < NB; b++) {
            if (dma_owned & (1u << b)) continue;            // DMA preempts this bank
            int best = -1, best_rank = 1 << 30, best_prio = -1;
            for (int i = 0; i < n_req; i++) {
                if (req_bank[i] != b) continue;
                int rank = (req_tok[i] - rr[b] + 256) & 0xff;  // round-robin distance
                if (req_prio[i] > best_prio || (req_prio[i] == best_prio && rank < best_rank)) {
                    best_prio = req_prio[i];
                    best_rank = rank;
                    best      = i;
                }
            }
            if (best >= 0) { granted[best] = true; rr[b] = (uint8_t)(req_tok[best] + 1); }
            if (h_on) {
                int c = 0;
                for (int i = 0; i < n_req; i++) {
                    if (req_bank[i] != b) continue;
                    c++;
                    int p = req_tok[i] >> 4;
                    if (p >= 0 && p < 18) h_portbank[p][b]++;
                }
                if (c > 0 && !(dma_owned & (1u << b))) { h_req[b] += c; if (c > 1) h_conf[b] += c - 1; }
            }
        }
    }
};

// ---- Resident DMA beat engine ------------------------------------------------
// Drains queued cluster-DMA transfers one 64 B beat at a time and exposes, per cycle, the
// TCDM superbank its current beat occupies. The real mem_wide_narrow_mux (fixed priority,
// DMA wins) grounds the 8 narrow streamer q_ready on the superbank the DMA presents a beat
// to (snitch_cluster.sv); a 64 B beat = 8 banks = exactly one superbank, advancing by one
// superbank per beat as the address streams. Stepped on the SAME clock + SAME Fabric as the
// accelerator streamers, so DMA<->compute contention is produced by arbitration, not a
// per-app dma_cycles parameter -> the refill-ring stall and async-ring contention emerge.
// Each transfer becomes drainable only at its submit cycle (at_rel), so DMAs issued mid-
// invocation collide only from when the SW actually launched them. period = cycles the
// backend holds a superbank per beat (L3 read-bound vs TCDM bus-bound; derived from the DMA
// trace, see docs/dataflow/10_memsim.md).
struct DmaEngine {
    struct Xfer { uint32_t tcdm_addr; uint32_t beats; int period; uint64_t at_rel; };
    std::vector<Xfer> q;            // FIFO of transfers, in submit order
    size_t head = 0;               // next undrained transfer
    uint32_t done_cnt = 0;         // completed transfers (dma_completed_id analogue)
    // current transfer
    bool active = false;
    uint32_t addr = 0, beats_left = 0;
    int period = 1, timer = 0;

    int beat_period_l3 = 4;        // cycles/64B beat, L3<->TCDM (read-bound)
    int beat_period_tcdm = 1;      // cycles/64B beat, TCDM<->TCDM (bus-bound)

    void clear() { q.clear(); head = 0; done_cnt = 0; active = false; }
    void rewind() { head = 0; done_cnt = 0; active = false; }  // re-drain the same queue (re-run)
    void enqueue(uint32_t tcdm_addr, uint32_t bytes, bool l3, uint64_t at_rel) {
        uint32_t beats = (bytes + 63) / 64;
        if (!beats) return;
        q.push_back({tcdm_addr, beats, l3 ? beat_period_l3 : beat_period_tcdm, at_rel});
    }
    bool busy() const { return active || head < q.size(); }
    // Advance one cycle (relative cycle `cyc`); return the dma_owned superbank mask (0 = idle).
    // The backend presents a TCDM beat for ONE cycle per 64 B, then is read-bound on L3 for the
    // remaining (period-1) cycles during which it grounds nothing (measured: ~1 beat / ~4.4 cyc for
    // L3, so it occupies a given superbank only ~1/period of the time). Each beat advances the
    // address by 64 B = +1 superbank, so a streaming DMA walks the 4 superbanks.
    uint32_t step(uint64_t cyc) {
        if (!active) {
            if (head >= q.size() || q[head].at_rel > cyc) return 0;  // not launched yet
            Xfer& x = q[head++];
            active = true; addr = x.tcdm_addr; beats_left = x.beats; period = x.period; timer = x.period;
        }
        uint32_t mask = (period > 0 && timer == period) ? (0xFFu << (((addr >> 6) & 3) * 8)) : 0u;  // beat this cyc
        if (--timer <= 0) {                            // beat done -> advance address, next beat
            addr += 64;
            if (--beats_left == 0) { active = false; done_cnt++; }
            timer = period;
        }
        return mask;
    }
};

// ---- Reader streamer: AGU + per-lane PIPELINED issue (request + data FIFOs) ----
// Each of num_channel lanes runs its OWN address generator and issues reads independently: a lane
// keeps issuing the next group's address as soon as it wins arbitration, up to addr_depth requests
// outstanding (the request-side address FIFO) and data_depth groups ahead of the consumer (the
// response-side data FIFO). Lanes do NOT wait for each other — a lane blocked by bank/DMA contention
// re-proposes while the others run ahead; a group becomes consumable only once ALL lanes have landed
// it (min over lanes). This is the real streamer: the read-ahead hides the +1cc latency and sparse
// contention, so a one-cycle DMA block does not stall the whole reader. stride-0 inner dim = reuse:
// one TCDM read replayed `reuse` times to the accelerator. RTL depths: addr_depth=4, data_depth=8.
struct CycReader {
    // config
    uint64_t base = 0;
    int32_t ts[4] = {0, 0, 0, 0};          // temporal strides (bytes)
    int eb[4] = {1, 1, 1, 1};              // effective temporal bounds (stride-0 dim0 -> 1)
    int num_channel = 1;
    int n_spatial = 1;
    int32_t sstride[2] = {8, 0};
    int sbound[2] = {1, 1};
    int data_depth = 8;                     // response-side data FIFO (read-ahead vs consumer)
    int addr_depth = 4;                     // request-side address FIFO (outstanding/unlanded)
    int reuse = 1;                          // replays per group (stride-0 dim0 bound)
    int port = 0;

    // per-lane pipelined state
    long lane_issued[8] = {0};             // groups this lane has had granted
    long lane_landed[8] = {0};             // groups landed for this lane (+1cc after grant)
    int  pend[8] = {0};                    // granted this cycle -> land next
    long consumed = 0;                      // groups the consumer has popped
    int  reuse_left = 0;
    bool active = false, done = false;
    int  fifo_occ = 0;                      // consumer-visible available groups = min(landed)-consumed
    // Compat for the safe-to-start stale check: the temporal position + count of the read GRANTED this
    // cycle (lane 0). word_off(ts, ti) gives the word a 1-lane reader just accessed.
    int  ti[4] = {0, 0, 0, 0};
    int  pending_push = 0;

    // BIST: deliver group production indices in order so the consumer can verify ordering.
    bool track_idx = false;
    long prod_delivered = 0;
    int idx_ring[64] = {0};
    int ir_head = 0, ir_tail = 0, ir_cnt = 0;
    bool pop_idx(int& v) {
        if (ir_cnt == 0) return false;
        v = idx_ring[ir_head]; ir_head = (ir_head + 1) & 63; ir_cnt--; return true;
    }

    long total() const { return (long)eb[0] * eb[1] * eb[2] * eb[3]; }
    long min_landed() const {
        long m = lane_landed[0];
        for (int l = 1; l < num_channel; l++) if (lane_landed[l] < m) m = lane_landed[l];
        return m;
    }
    void set_ti(long g) { long r = g % (total() ? total() : 1); for (int d = 0; d < 4; d++) { ti[d] = (int)(r % eb[d]); r /= eb[d]; } }

    void configure(const Agu& a, int nch, int nsp, int fd, int prt);
    uint64_t lane_addr(long group, int lane) const;     // byte address of `lane` at temporal `group`
    void propose(Fabric& f);
    void commit(Fabric& f, const bool granted[64], int& grant_idx);
    bool pop();
    // call at top of cycle: land last cycle's granted reads; refresh consumer occupancy + BIST order
    void land_reads() {
        for (int l = 0; l < num_channel; l++) { lane_landed[l] += pend[l]; pend[l] = 0; }
        long m = min_landed();
        if (track_idx) while (ir_cnt < 64 && prod_delivered < m) { idx_ring[ir_tail] = (int)prod_delivered++; ir_tail = (ir_tail + 1) & 63; ir_cnt++; }
        fifo_occ = (reuse > 1) ? (m > consumed ? 1 : 0) : (int)(m - consumed);
    }
    int total_read_groups() const { return (int)total(); }
    void reset() {                          // restart the AGU (continuous stream)
        for (int l = 0; l < 8; l++) { lane_issued[l] = lane_landed[l] = 0; pend[l] = 0; }
        consumed = 0; reuse_left = 0; fifo_occ = 0; pending_push = 0; prod_delivered = 0;
        ir_head = ir_tail = ir_cnt = 0; done = false; active = true;
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
    int depth = 8;         // output-FIFO depth: a near-full writer (>= depth-1) asserts TCDM priority
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

// NOTE: the standalone cyc_gemm (osgemm/isgemm) and cyc_simd steppers were removed — every accelerator
// mode (osCore/isCore/SUC-P2/P1/SIMD) is now stepped by the single AccelEngine on ONE shared fabric with
// the live DmaEngine, so DMA<->streamer contention emerges from arbitration uniformly (no per-app scalar
// dma_cycles, no per-port private fabrics). See cyc_engine.{hpp,cpp}.

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
