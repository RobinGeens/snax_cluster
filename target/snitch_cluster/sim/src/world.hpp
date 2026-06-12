// Copyright 2026 KU Leuven. memsim — SimWorld: functional datapath (real data
// movement) + cycle-accurate timing. See docs/dataflow/10_memsim.md.
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "cyc.hpp"  // DmaEngine
#include "cyc_engine.hpp"  // AccelEngine
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
    void advance_to(uint64_t cycle) override;  // step the engine (MEMSIM_ENGINE) up to `cycle`
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
    // Phase-2 safe-to-start thresholds (M2_R10/R11_start_cnt from ELF .data). The app polls the
    // R10/R11 gauges to these before releasing the SUC/isCore; modeling the overlap they enable.
    void set_s2s(uint32_t r10_cnt, uint32_t r11_cnt) {
        r10_start_cnt_ = r10_cnt;
        r11_start_cnt_ = r11_cnt;
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

    // Per-cycle streamer-FIFO occupancy, accumulated across every stepped invocation: the engine
    // (P1/P2), cyc_gemm (osgemm/isgemm) and cyc_simd (SIMD passes) all append via append_fifo. Each
    // entry is (absolute cycle, occupancy of each of the 18 ports: readers R0..R13 = 0..13, writers
    // W0..W3 = 14..17; -1 = port absent that cycle). main.cpp dumps it to a `.fifo.csv` for
    // plot_timeline.py. fifo_capped() => the soft row cap was hit (the run is truncated).
    const std::vector<std::pair<uint64_t, std::array<int16_t, 18>>>& fifo_trace() const { return fifo_trace_; }
    bool fifo_capped() const { return fifo_capped_; }

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
    // Splice a stepper's per-cycle FIFO trace (relative rows, row k = relative cycle k+1) into the
    // run-wide trace at absolute `start`. Shared by the engine (P1/P2), cyc_gemm and cyc_simd paths.
    void append_fifo(const std::vector<FifoRow>& rows, uint64_t start) {
        if (!trace_on_) return;
        for (size_t k = 0; k < rows.size(); k++) {
            if (fifo_trace_.size() >= FIFO_ROW_CAP) { fifo_capped_ = true; return; }
            fifo_trace_.push_back({start + k + 1, rows[k]});
        }
    }
    bool trace_on_ = false;
    std::vector<TraceSeg> trace_;
    // Per-cycle FIFO occupancy trace (see fifo_trace()). Soft-capped so a full-size multi-tile run
    // can't balloon to a multi-GB CSV; the cap is reported, never silent.
    static const size_t FIFO_ROW_CAP = 4000000;
    std::vector<std::pair<uint64_t, std::array<int16_t, 18>>> fifo_trace_;
    bool fifo_capped_ = false;

    PortLayout layout_[N_PORTS];
    int csr_to_port_[256];     // (csr-960) -> port index, or -1
    uint32_t raw_[256] = {0};  // streamer config CSR shadow, index = csr-960
    Agu ports_[N_PORTS];

    SimbacoreCfg cfg_;
    uint32_t dma_txid_ = 0;

    // ---- accelerator timing state (per invocation) ----
    uint64_t accel_start_   = 0;  // cycle SIMBACORE_START was written
    uint64_t accel_end_     = 0;  // cycle the core returns to sIDLE (busy=0)
    uint64_t t_oscore_done_ = 0;  // R10 gauge reaches its total here (osCore window end)
    uint64_t t_suc_start_   = 0;  // SUC release cycle (P2 safe-to-start overlap); R11 gauge window start
    uint64_t t_suc_done_    = 0;  // R11 gauge reaches its total here
    uint32_t g_r10_total_ = 0, g_r11_total_ = 0, g_iscore_total_ = 0;
    // Safe-to-start thresholds the app ships via SNAX_DELAYED_START_R10/R11. They release the
    // SUC (when the R10 osCore gauge hits r10_start_cnt) and the isCore (R11 SUC gauge hits
    // r11_start_cnt), overlapping the P2 stages. 0 = unset -> full serialize. See run_invocation.
    uint32_t r10_start_cnt_ = 0, r11_start_cnt_ = 0;
    // Stepped P2 gauges: cycle (relative to accel_start_) of each osCore-tile / SUC-y fire, from the
    // co-sim. snax_read returns the count <= now, so the SW poll resolves on the real (bursty)
    // fire-counter instead of gauge_at's linear ramp. Empty -> fall back to gauge_at.
    std::vector<uint32_t> g_r10_fire_, g_r11_fire_;
    uint32_t gauge_stepped(const std::vector<uint32_t>& fire, uint64_t now) const {
        if (now <= accel_start_) return 0;
        uint32_t rel = (uint32_t)(now - accel_start_);
        return (uint32_t)(std::upper_bound(fire.begin(), fire.end(), rel) - fire.begin());
    }
    uint64_t osc_dur_ = 0, suc_dur_ = 0, isc_dur_ = 0;
    uint32_t perf_ = 0;  // last invocation busy-cycle count
    bool phase2_   = false;

    // ---- per-invocation DMA contention (cycle-accurate, no per-app parameter) ----
    // Every DMA the SW launches DURING a P2 invocation (e.g. the oscore_in refill ring) is recorded
    // into inv_dma_ with its real TCDM address + submit cycle. run_invocation computes a provisional
    // timeline (the gauges the refill loop polls); the first SIMBACORE_BUSY read after the SW has
    // issued those DMAs re-steps the P2 co-sim with inv_dma_ on the shared fabric, so the contention
    // emerges by arbitration. Apps with no mid-invocation DMA record nothing -> provisional == final.
    // ---- the single cycle-exact engine (MEMSIM_ENGINE): real-time advance_to driving ----
    // When engine_active_, run_invocation has configured engine_ for THIS invocation and advance_to
    // steps it one cycle at a time (DMA enqueued live by dma_submit). accel_end_/perf_/gauges are read
    // from the engine. Gated behind MEMSIM_ENGINE while it's brought up alongside the legacy path.
    AccelEngine engine_;
    bool engine_active_ = false;  // engine is mid-invocation (advance_to steps it)
    bool engine_used_   = false;  // this invocation used the engine (read gauges/perf from it)

    DmaEngine inv_dma_;
    bool dma_engine_on_ = false;  // per-cycle DMA-contention engine enabled (MEMSIM_DMA_PERIOD>0); off by default
    int  inv_kind_      = 0;       // current invocation: 0=none, 1=P2 (cyc_phase2), 2=P1/IS_OSGEMM (cyc_phase1)
    bool inv_finalized_ = false;   // the re-step has folded in inv_dma_ for this invocation
    bool has_conv_      = false;   // P1: conv stage present (PHASE1) vs absent (IS_OSGEMM) — for the re-step
    long sw_cyc_        = 0;       // switchCore cycles for this invocation (P2 re-step)
    long dma_ov_        = 0;       // prefetch-at-start DMA overlap (cycling model); preserved into the re-step
    void finalize_inv();           // re-step the right co-sim with the recorded DMA engine on first BUSY read

    // ---- DMA timing state ----
    uint64_t dma_busy_until_ = 0;
    uint32_t dma_busy_addr_ = 0;  // TCDM-side address of the in-flight DMA (for the engine prefetch tail)

    // ---- timing constants ----
    // Timing constants. Each value is derived from an RTL register/FIFO depth (file:line
    // cited) or marked residual where a latency is data-dependent rather than a static depth.
    // Full rationale + the constants table live in docs/dataflow/10_memsim.md; cross-check the
    // cited RTL before changing a value.
    uint32_t dma_first_beat_l3_ = 19;    // iDMA backend req->AR->first-R/W (axi_dma_backend.sv) +
                                         // 3 FE stages; 11 of 16 is the simulated L3+xbar latency.
    uint32_t dma_first_beat_tcdm_ = 10;  // TCDM<->TCDM
    // PHASE1 + IS_OSGEMM timing now emerge from the per-cycle pipeline co-sim (cyc_phase1.cpp):
    // the lead-in/drain and cross-stage contention are stepped, so the old fitted fills
    // (fill_p1_base/per_mtile, fill_is_osgemm) are gone.
    // The Phase-2 timing + safe-to-start boundary are produced by the per-cycle co-sim
    // (cyc_phase2.cpp): the osCore->SUC->isCore overlap, the handoff bubbles, and the stale-read
    // boundary all emerge from stepping. The old fitted P2 constants (fill_osc_p2_, fill_isc_p2_,
    // s2s_lat_z_, s2s_lat_y_) are gone — the SUC dt_BC bank conflict still comes from cyc_suc_duration.
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
