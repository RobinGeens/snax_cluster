// Copyright 2026 KU Leuven. memsim — SimWorld: functional datapath (real data
// movement) + cycle-accurate timing. See docs/dataflow/10_memsim.md.
#pragma once
#include <cstdint>
#include <vector>

#include "machine.hpp"
#include "snax_csr.hpp"

// Streamer ports: 14 readers R0..R13 then 4 writers W0..W3.
static constexpr int N_READERS = 14;
static constexpr int N_WRITERS = 4;
static constexpr int N_PORTS   = N_READERS + N_WRITERS;

struct PortLayout {
    uint32_t base_csr;  // CSR of BASE_PTR_*_LOW
    int n_s;            // number of spatial strides (2 for R7, else 1)
};

class SimWorld : public World {
   public:
    SimWorld();

    // ---- DMA (functional byte copy + timing) ----
    uint32_t dma_submit(const DmaDesc& d, uint64_t at) override;
    uint32_t dma_busy(uint64_t at) override { return at < dma_busy_until_ ? 1 : 0; }
    uint32_t dma_completed_id(uint64_t at) override { return dma_txid_; }

    // ---- SNAX CSR window ----
    void snax_write(uint32_t csr, uint32_t val, uint64_t at) override;
    uint32_t snax_read(uint32_t csr, uint64_t at, bool& is_poll) override;
    bool snax_write_serializes(uint32_t csr, uint64_t at) const override;

    // ---- time / events ----
    uint64_t next_event_cycle() const override;

    // ---- layout verification (integer cross-check) ----
    void set_verify(bool v) { verify_ = v; }
    bool layout_ok() const { return !verify_ran_ || layout_pass_; }
    // App golden buffers (ELF .data addresses, resolved in main.cpp). 0 = absent.
    void set_goldens(uint32_t z, uint32_t y, uint32_t isc) {
        golden_z_   = z;
        golden_y_   = y;
        golden_isc_ = isc;
    }
    // SSM shape params (read from ELF .data in main.cpp) for the FP32 SUC datapath.
    void set_ssm(uint32_t dState, uint32_t xProjDim, uint32_t bc_pad, uint32_t dtRankU) {
        dState_       = dState;
        xProjDim_     = xProjDim;
        bc_pad_       = bc_pad;
        dtRankUnroll_ = dtRankU;
    }
    // P1-output goldens (dt_BC matrix, conv_out=x): the model doesn't compute P1, so the
    // SUC sources its dt/B/C and x from these instead of the (zero) TCDM buffers.
    void set_p1out(uint32_t dtbc, uint32_t sucx) {
        golden_dtbc_ = dtbc;
        golden_sucx_ = sucx;
    }

    // ---- timeline trace (optional, --timeline <file>) ----
    // Each segment is a cycle window [start,end) during which one engine is active, plus
    // `ideal` = the shortest possible cycle count for that block's compute (its MAC-group
    // count at peak, conflict- and drain-free). Hardware utilization = ideal/(end-start):
    // a conflict-free array is ~1.0, the SUC's bank conflict drops it (~0.57 at bc_pad=0),
    // a GEMM's output drain and the DMA first-beat latency also pull it below 1.0.
    // The engines tile the accelerator window per invocation (osCore/SUC/isCore serialize in
    // P2; osCore/switchCore/isCore overlap in P1); DMA windows come from dma_submit. main.cpp
    // dumps these to CSV for plot_timeline.py.
    // TR_TCDM is not an engine bar: its `ideal` field carries the TCDM word-access count over
    // [start,end] (streamer accesses per invocation + DMA beats*8), so plot_timeline.py can draw
    // TCDM bandwidth utilization (= words/cyc / 32 banks) as a 0-100% line.
    enum TraceEngine { TR_OSCORE = 0, TR_ISCORE, TR_SUC, TR_SWITCHCORE, TR_DMA, TR_TCDM, TR_N };
    struct TraceSeg {
        int engine;
        uint64_t start, end, ideal;
    };
    void set_trace(bool v) { trace_on_ = v; }
    const std::vector<TraceSeg>& trace() const { return trace_; }

    // Optimal safe-to-start delays
    //  -1 if the sweep didn't run (not a full P2 osCore->SUC->isCore app). Surfaced so main.cpp can hand them to
    //  plot_timeline.py for the schedule plot.
    long s2s_opt_r10() const { return s2s_opt_r10_; }
    long s2s_opt_r11() const { return s2s_opt_r11_; }
    long s2s_total_r10() const { return s2s_total_r10_; }
    long s2s_total_r11() const { return s2s_total_r11_; }

   private:
    void rec(int eng, uint64_t s, uint64_t e, uint64_t ideal) {
        if (trace_on_ && e > s) trace_.push_back({eng, s, e, ideal});
    }
    bool trace_on_ = false;
    std::vector<TraceSeg> trace_;

    PortLayout layout_[N_PORTS];
    int csr_to_port_[256];     // (csr-960) -> port index, or -1
    uint32_t raw_[256] = {0};  // streamer config CSR shadow, index = csr-960
    Agu ports_[N_PORTS];

    SimbacoreCfg cfg_;
    uint32_t dma_txid_ = 0;

    // ---- accelerator timing state (per invocation) ----
    uint64_t accel_start_   = 0;  // cycle SIMBACORE_START was written
    uint64_t accel_end_     = 0;  // cycle the core returns to sIDLE (busy=0)
    uint64_t t_oscore_done_ = 0;  // R10 gauge reaches its total here
    uint64_t t_suc_done_    = 0;  // R11 gauge reaches its total here
    uint32_t g_r10_total_ = 0, g_r11_total_ = 0, g_iscore_total_ = 0;
    uint64_t osc_dur_ = 0, suc_dur_ = 0, isc_dur_ = 0;
    uint32_t perf_ = 0;  // last invocation busy-cycle count
    bool phase2_   = false;

    // ---- DMA timing state ----
    uint64_t dma_busy_until_ = 0;

    // ---- timing constants ----
    // Timing constants. Each value is derived from an RTL register/FIFO depth (file:line
    // cited) or marked residual where a latency is data-dependent rather than a static depth.
    // Full rationale + the constants table live in docs/dataflow/10_memsim.md; cross-check the
    // cited RTL before changing a value.
    uint32_t dma_first_beat_l3_ = 19;    // iDMA backend req->AR->first-R/W (axi_dma_backend.sv) +
                                         // 3 FE stages; 11 of 16 is the simulated L3+xbar latency.
    uint32_t dma_first_beat_tcdm_ = 10;  // TCDM<->TCDM
    // PHASE1 pipeline fill (osCore->switchCore conv->isCore). RESIDUAL, measured from vsim: it
    // scales with the seqLen-tile count M_i (each row-tile pays an inter-stage handoff bubble the
    // pipeline doesn't fully overlap) plus a constant 3-stage lead-in/drain. fill = M_i*31 + 180,
    // fit to P1-tiled-D vsim at M_i=8 (fill 428) and M_i=49 (fill 1686), cross-checked vs main-tiled
    // M_i=12 (~560). The earlier flat 562 only matched M_i=12 and undershot large seqLen by ~31/M_i.
    uint32_t fill_p1_base_      = 180;  // constant 3-stage lead-in + drain
    uint32_t fill_p1_per_mtile_ = 31;   // per-seqLen-tile inter-stage handoff bubble
    // IS_OSGEMM (both cores chained but NO switchCore conv) is a different pipeline; keep its flat
    // residual unchanged (no vsim point to re-fit it).
    uint32_t fill_is_osgemm_ = 562;
    uint32_t fill_osc_p2_    = 261;  // ~2 derived (osCore pipelineDelay 1 + output ser 1 [Array.scala:206])
                                     // + ~259 residual (osCore->SUC handoff: W0 drain + R6 fetch +
                                     // delayBtoC=6 [SimbaCore.scala:129] + SUC warmup).
    uint32_t fill_isc_p2_ = 118;     // ~56 derived (s2p warmup 48 [isCoreSerDes, MambaCoreParams.scala:85]
                                     // + array pipelineDelay 5 [Array.scala:140] + 3 acc/io) + ~62 residual.
    // The SUC dt_BC bank conflict (geometry and magnitude) is produced by cyc_suc_duration.
    // Safe-to-start commit-pipe depths (MEMSIM_S2S): the RTL commit pipe of the first output
    // element (the boundary itself comes from the per-element schedule, not these constants):
    //   z (W0): osCore array out 1 + output ser 1 + W0 first-write 1 = ~3 cc.
    //   y (W2): SU-core output datapath delayTotal 17 (StateUpdateCore.scala:135) + W2 commit ~3 = ~20
    //           (delayBtoC=6, SimbaCore.scala:129, is B-C input sync, not the output pipe).
    double s2s_lat_z_ = 3;   // z commit-pipe; boundary = K_i + this (R10=2)
    double s2s_lat_y_ = 20;  // y commit-pipe; boundary 5397, inside vsim's bracket (5300 hang/5400 safe)
    // Sweep optima (gauge units), captured by verify_datapath; -1 = sweep didn't run.
    long s2s_opt_r10_ = -1, s2s_opt_r11_ = -1;
    long s2s_total_r10_ = 0, s2s_total_r11_ = 0;

    // ---- layout verification state ----
    bool verify_       = false;                                           // run the integer layout cross-check
    bool verify_ran_   = false;                                           // datapath verify has run once
    bool layout_pass_  = true;                                            // running result
    uint32_t golden_z_ = 0, golden_y_ = 0, golden_isc_ = 0;               // app FP8 goldens (.data)
    uint32_t dState_ = 0, xProjDim_ = 0, bc_pad_ = 0, dtRankUnroll_ = 0;  // SSM shape
    uint32_t golden_dtbc_ = 0, golden_sucx_ = 0;                          // P1-output goldens (dt_BC, conv_out=x)

    void decode_ports();               // raw_ -> ports_
    void run_invocation(uint64_t at);  // compute durations + functional data movement
    uint32_t gauge_at(uint64_t now, uint64_t t0, uint64_t t1,
                      uint32_t total) const;  // ramped gauge value
    void verify_layout();                     // golden-free AGU structural audit (all apps, located)
    void verify_datapath();                   // golden-aware datapath + safe-to-start + BIST checks

   public:
    uint32_t n_layout_errs() const { return n_layout_errs_; }
    int layout_invocations() const { return layout_invs_; }

   private:
    uint32_t n_layout_errs_  = 0;   // total located AGU layout errors over the run
    int layout_invs_         = 0;   // invocations the AGU audit ran on
    int layout_print_budget_ = 16;  // cap on per-error location prints
};
