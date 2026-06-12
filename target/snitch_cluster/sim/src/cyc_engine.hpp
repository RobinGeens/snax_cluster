// Copyright 2026 KU Leuven. memsim — the single cycle-exact accelerator engine.
//
// ONE resident per-cycle co-simulator that replaces the per-phase steppers (cyc_phase1 / cyc_phase2
// / cyc_gemm). Driven by SimWorld::advance_to one cycle at a time, on ONE shared 32-bank Fabric,
// with the DMA beat engine stepped in the SAME loop. MODE (en_osCore/suCore/isCore + switchCoreMode)
// selects which units are active and how they connect — not "phases", one machine configured
// differently:
//
//   osCore : R0,R1                 -> array(K=dModel) -> W0 (z, TCDM)   OR on-chip stream (P1)
//   switch : R2,R3,R5 (dt-matmul, P2)  /  R3,R4 (conv1d, P1)            -> on-chip dt_delta/conv_out
//   SUC    : R7(dt_BC),R10(z)      -> scan            -> W2 (y, TCDM)                     (P2)
//   isCore : R11(y),R12(w),R13(psum) -> array(K=osN)  -> W3
//
// Every reader/writer arbitrates on the one Fabric each cycle; the DMA beat engine grounds the
// bank(s) it touches the SAME cycle. So contention, FIFO-slack DMA-hiding, inter-stage overlap and
// pipeline fill/drain all EMERGE. The per-word producer-commit (W0/W2 grant) and consumer-read
// (R10/R11 grant) cycles are exact, so the safe-to-start sweep (smallest start_cnt with zero stale
// reads) returns the EXACT minimum delay. See docs/dataflow/11_memsim_cycle_accuracy_audit.md (§3.3).
#pragma once
#include <array>
#include <cstdint>
#include <vector>

#include "cyc.hpp"       // Fabric, CycReader, CycWriter, DmaEngine
#include "machine.hpp"   // Agu, SimbacoreCfg

struct EngineResult {
    uint64_t busy = 0;                         // MambaCore-busy cycle count (perf counter)
    uint64_t osc_end = 0, suc_start = 0, suc_end = 0, isc_start = 0, isc_end = 0;
    uint64_t sw_start = 0, sw_end = 0;
    long stale_z = 0, stale_y = 0;             // reads of a word its producer had not committed
    std::vector<uint32_t> r10_fire, r11_fire;  // cycle of each osCore-tile / SUC-y fire (real gauges)
    // Per-stepped-cycle streamer-FIFO occupancy (rec_fifo): row k = the FIFO fullness at the end of
    // relative cycle k+1. Index into the row = port (readers R0..R13 = 0..13, writers W0..W3 = 14..17);
    // value = elements currently in that port's FIFO (reader data-FIFO landed-not-consumed, writer
    // output-FIFO pushed-not-drained), or -1 if the port is not part of this invocation.
    std::vector<std::array<int16_t, 18>> fifo;
};

// The single engine. configure() once from the decoded ports + MODE, then step() each cycle.
class AccelEngine {
   public:
    void configure(const Agu* ports, const SimbacoreCfg& cfg, long r10_start_cnt, long r11_start_cnt,
                   long sw_cyc);
    bool active() const { return running_; }
    bool step();                 // advance one cycle on the shared fabric+DMA; returns running_
    void dma_enqueue(uint32_t tcdm_addr, uint32_t bytes, bool l3, uint64_t at_rel) {
        dma_.enqueue(tcdm_addr, bytes, l3, at_rel);
    }
    uint32_t g_r10() const { return (uint32_t)g_r10_; }
    uint32_t g_r11() const { return (uint32_t)g_r11_; }
    uint32_t g_iscore() const { return (uint32_t)g_iscore_; }
    void release_r10() { rel_r10_ = true; }    // SNAX_DELAYED_START_R10 latch (real-time SW release)
    void release_r11() { rel_r11_ = true; }
    uint64_t cyc() const { return cyc_; }
    const EngineResult& result() const { return res_; }
    bool rec_fires = false;      // record per-fire gauge cycles (set before stepping the app config)
    bool rec_fifo  = false;      // record per-cycle streamer-FIFO occupancy into res_.fifo

   private:
    void sample_fifo();                         // append this cycle's per-port FIFO occupancy to res_.fifo
    int r7bank(long st, int lane) const;       // dt_BC bank for SUC R7 lane at temporal step st
    void configure_p1();                        // set up the P1 / IS_OSGEMM pipeline
    bool step_p1();                             // one cycle of the P1 pipeline on the shared fabric

    // ---- config / shape ----
    static const int NPORT = 18;  // 14 readers (R0..R13) + 4 writers (W0..W3)
    Agu p_[NPORT];
    int seqLen_ = 0, dInner_ = 0, dModel_ = 0, dFinal_ = 0;
    bool en_os_ = false, en_suc_ = false, en_isc_ = false;
    int sw_mode_ = 0;
    long r10_cnt_ = 0, r11_cnt_ = 0;
    long M_i_ = 0, osN_ = 0, K_os_ = 0, K_is_ = 0, n_tiles_os_ = 0, n_tiles_is_ = 0, iters_ = 0;
    long wbpt_os_ = 1, wbpt_is_ = 1, Wz_ = 0, Wy_ = 0;
    long sw_cyc_ = 0;
    int  kind_ = 0;          // 1 = P2 (osCore->SUC->isCore), 2 = P1/IS_OSGEMM (osCore->conv->isCore)
    bool has_conv_ = false;  // P1: switchCore conv stage present (PHASE1) vs absent (IS_OSGEMM)

    // ---- osCore: R0,R1 -> array -> W0(z, P2) or W1(P1) ----
    CycReader r0_, r1_;
    CycWriter w0_;
    long as_os_ = 0, tiles_os_ = 0;
    long os_owed_ = 0;  // osCore output groups computed but not yet pushed into the writer FIFO (back-pressure)
    std::vector<uint8_t> zwr_;
    // ---- P1 extra: switchCore conv (R3,R4), isCore psum (R13), W1, on-chip stream cursors ----
    CycWriter w1_;
    CycReader r3c_, r4c_, r13_;
    bool w1_en_ = false, r13_en_ = false, p1sw_en_ = false;
    bool w0p1_en_ = false, r11p1_en_ = false;  // IS_OSGEMM: osCore spills via W0, isCore reads A via R11
    long osc_elem_ = 0, sw_elem_ = 0;   // elements osCore produced / switchCore produced (on-chip)

    // ---- SUC: R7 dt_BC (manual per-lane) + R10(z) -> scan -> W2(y) ----
    static const int NCH7 = 4, RPR = 4, ADDR_D = 4, DATA_D = 8, delaySU = 4;
    int32_t r7ts_[4] = {0, 0, 0, 0};
    int r7eb_[4] = {1, 1, 1, 1};
    long issued_[NCH7] = {0}, landed_[NCH7] = {0};
    int pend7_[NCH7] = {0};
    long agu7_ = 0, consumed7_ = 0;
    int k7_ = 0;
    bool have_bc_ = false;
    CycReader r10_;
    CycWriter w2_;
    long out_su_ = 0, y_in_word_ = 0;
    std::vector<uint8_t> ywr_;

    // ---- switchCore (P2 dt-matmul, co-active with SUC) ----
    CycReader r2sw_, r3sw_, r5sw_;
    bool sw_en_ = false;
    long sw_dt_ = 0, sw_acc_ = 0, sw_first_ = 0, sw_last_ = 0;
    static const long SWBUF = 8;

    // ---- isCore: R11(y),R12(w) -> array -> W3 ----
    CycReader r11_, r12_;
    CycWriter w3_;
    long as_is_ = 0, tiles_is_ = 0;
    long is_owed_ = 0;  // isCore output groups computed but not yet pushed into W3 (back-pressure)

    // ---- control / time ----
    bool running_ = false, suc_started_ = false, isc_started_ = false;
    uint64_t cyc_ = 0, cap_ = 0;
    long g_r10_ = 0, g_r11_ = 0, g_iscore_ = 0;
    bool rel_r10_ = false, rel_r11_ = false;
    DmaEngine dma_;
    Fabric fabric_;
    EngineResult res_;
};
