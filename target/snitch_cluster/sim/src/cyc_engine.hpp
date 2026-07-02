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
// reads) returns the EXACT minimum delay.
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
    void configure_os();                        // kind_=3: standalone osCore (osgemm) array
    bool step_os();                             // one cycle of the osCore-only array on the shared fabric
    void configure_is();                        // kind_=4: standalone isCore (isgemm) array
    bool step_is();                             // one cycle of the isCore-only array on the shared fabric
    void configure_simd();                      // kind_=5: SIMD readers->core->writers
    bool step_simd();                           // one cycle of the SIMD pass on the shared fabric

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
    static const int NCH7 = 4, RPR = 4, delaySU = 4;   // R7 lanes, BC-refresh groups, SUC FMA delay
    // R7 dt_BC FIFOs, split exactly as the RTL (and cyc_suc_duration ADDR_D/DATA_D): the request-side
    // ADDRESS FIFO = fifo_depth[7] (snax_simbacore_cluster.hjson) bounds OUTSTANDING requests -> this is
    // what exposes concurrent-DMA blocks on the SUC's critical refresh (keep it shallow). The response-
    // side DATA read-ahead = responser + dataBuffer = 2x that (how far landed data leads the consumer) ->
    // buffering, deeper. The earlier flat DATA_D=8 regressed suc-async because it ALSO deepened the
    // outstanding window; splitting addr(4)/data(8) keeps DMA exposure while matching RTL buffering.
    int r7_fifo_d_ = 4;   // addr FIFO (outstanding requests) = fifo_depth[7]
    int r7_data_d_ = 8;   // data read-ahead = responser + dataBuffer = 2 * fifo_depth[7]
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
    // isCore is input-stationary (VersaCore K_M_N): A=y (R11) serial-loaded once per (k,m) tile over
    // serDesA_ beats (Mu*Ku fp8 / 8 B/beat = 48) with the ARRAY IDLE, then reused for dFinal output
    // columns; B=weight (R12) + psum (R13/W3 RMW) every compute step. isc_aload_ = remaining A-load
    // beats; isc_since_a_ = compute steps into the current A-tile (reload at dFinal). (StaticVersaCoreParams
    // serDesFactorA; VersaCore.scala address advances every serDesA_+dFinal cycles.)
    long serDesA_ = 0, isc_aload_ = 0, isc_since_a_ = 0;

    // ---- SIMD (kind_=5): arbitrary enabled reader/writer ports -> implicit SimdCore -> writers ----
    // Generic per-port streamers (indexed by port), distinct from the named GEMM/SUC members above so a
    // SIMD pass can drive any subset of R0..R13 / W0..W3 on the one shared fabric with the live DMA.
    CycReader rd_[14];
    CycWriter wr_[4];
    bool simd_r_en_[14] = {false}, simd_w_en_[4] = {false};
    long simd_r_beats_[14] = {0}, simd_r_pops_[14] = {0};
    bool is_simd_ = false;       // this invocation is a SIMD pass (perf counter stays 0)

    // ---- control / time ----
    bool running_ = false, suc_started_ = false, isc_started_ = false;
    uint64_t cyc_ = 0, cap_ = 0;
    long g_r10_ = 0, g_r11_ = 0, g_iscore_ = 0;
    bool rel_r10_ = false, rel_r11_ = false;
    DmaEngine dma_;
    Fabric fabric_;
    EngineResult res_;
};
