// Copyright 2026 KU Leuven. memsim — SimWorld implementation.
#include "world.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <unordered_set>
#include <vector>

#include "cyc.hpp"
#include "cyc_engine.hpp"
#include "fp.hpp"

// TCDM narrow channels per streamer port (R0..R13, W0..W3); a port issues this many 8-byte
// accesses per temporal step. Used for the TCDM bandwidth trace and the layout audit.
static const int PORT_NCH[N_PORTS] = {2, 4, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 4, 4, 1, 1, 1, 4};

SimWorld::SimWorld() {
    for (int i = 0; i < 256; i++) csr_to_port_[i] = -1;
    uint32_t c = SNAX_STREAMER_CFG_LO;  // 960
    for (int p = 0; p < N_PORTS; p++) {
        int n_s    = (p == 7) ? 2 : 1;  // only R7 has 2 spatial strides
        layout_[p] = {c, n_s};
        int block  = 2 + n_s + 4 + 4;  // base_lo/hi + s_stride[n_s] + t_bound[4] + t_stride[4]
        for (int k = 0; k < block; k++) csr_to_port_[(c - SNAX_STREAMER_CFG_LO) + k] = p;
        c += block;
    }
    // c should now equal SNAX_DELAYED_START_R10 (1159).
}

void SimWorld::decode_ports() {
    for (int p = 0; p < N_PORTS; p++) {
        int i   = layout_[p].base_csr - SNAX_STREAMER_CFG_LO;
        int n_s = layout_[p].n_s;
        Agu& a  = ports_[p];
        a.base  = ((uint64_t)raw_[i + 1] << 32) | raw_[i + 0];
        a.n_s   = n_s;
        for (int k = 0; k < n_s; k++) a.s_stride[k] = (int32_t)raw_[i + 2 + k];
        for (int k = 0; k < 4; k++) a.t_bound[k] = (int32_t)raw_[i + 2 + n_s + k];
        for (int k = 0; k < 4; k++) a.t_stride[k] = (int32_t)raw_[i + 2 + n_s + 4 + k];
        a.enabled = (a.t_bound[0] != 0);  // T_BOUND_BASE==0 is the disable convention
    }
}

uint32_t SimWorld::dma_submit(const DmaDesc& d, uint64_t at) {
    uint32_t src = (uint32_t)d.src, dst = (uint32_t)d.dst;
    // Functional: move the bytes now (data correct immediately).
    std::vector<uint8_t> buf(d.size);
    uint32_t rows = d.is_2d ? d.repeat : 1;
    for (uint32_t r = 0; r < rows; r++) {
        mem->read(src + r * d.src_stride, buf.data(), d.size);
        mem->write(dst + r * d.dst_stride, buf.data(), d.size);
    }
    // Timing: single channel, 64 B/beat. First transfer of a chain pays the
    // first-beat latency; back-to-back transfers pipeline (bus-bound on beats).
    uint64_t beats      = ((uint64_t)d.size * rows + 63) / 64;
    bool l3             = d.src_is_l3 || d.dst_is_l3;
    uint32_t first_beat = l3 ? dma_first_beat_l3_ : dma_first_beat_tcdm_;
    uint64_t seg_start  = (at >= dma_busy_until_) ? at : dma_busy_until_;
    if (at >= dma_busy_until_)  // engine was idle
        dma_busy_until_ = at + first_beat + beats;
    else  // pipelined behind outstanding work
        dma_busy_until_ += beats;
    rec(TR_DMA, seg_start, dma_busy_until_, beats);  // ideal = pure bus beats (no first-beat latency)
    // TCDM traffic: each 64B beat = 8 narrow (8B) words on each TCDM-resident side of the copy.
    int tcdm_sides = (d.src_is_l3 ? 0 : 1) + (d.dst_is_l3 ? 0 : 1);
    rec(TR_TCDM, seg_start, dma_busy_until_, beats * 8 * (uint64_t)tcdm_sides);
    if (tcdm_sides > 0) dma_busy_addr_ = d.dst_is_l3 ? src : dst;  // remember the in-flight TCDM-side addr
    // Record DMAs launched DURING a P2 invocation for the cycle-accurate contention re-step. Only
    // the TCDM-resident side contends on the banks; an L3->L3 copy (tcdm_sides==0) doesn't.
    if (dma_engine_on_ && inv_kind_ && !inv_finalized_ && at >= accel_start_ && tcdm_sides > 0) {
        uint32_t tcdm_addr = d.dst_is_l3 ? src : dst;     // the side that lands in TCDM
        inv_dma_.enqueue(tcdm_addr, d.size * rows, l3, at - accel_start_);
    }
    // Engine path: enqueue the DMA live; advance_to steps it in the same loop as compute so the
    // contention (and its FIFO-slack hiding) emerges. at_rel is relative to the invocation start.
    if (engine_active_ && at >= accel_start_ && tcdm_sides > 0) {
        advance_to(at);
        uint32_t ta = d.dst_is_l3 ? src : dst;
        engine_.dma_enqueue(ta, d.size * rows, l3, at - accel_start_);
        if (std::getenv("MEMSIM_ENGDBG"))
            std::fprintf(stderr, "  [ENQ] +%lld dst=%08x sb%d bytes=%u l3=%d\n",
                         (long long)(at - accel_start_), ta, (int)((ta >> 6) & 3), d.size * rows, (int)l3);
    }
    ++dma_txid_;
    if (std::getenv("MEMSIM_DMA") && dma_txid_ <= 8) {
        std::fprintf(stderr, "  dma#%u src=0x%08x dst=0x%08x size=%u 2d=%d rep=%u beats=%llu busy_until=%llu\n",
                     dma_txid_, src, dst, d.size, d.is_2d, d.repeat, (unsigned long long)beats,
                     (unsigned long long)dma_busy_until_);
    }
    return dma_txid_;
}

void SimWorld::snax_write(uint32_t csr, uint32_t val, uint64_t at) {
    if (csr >= SNAX_STREAMER_CFG_LO && csr <= SNAX_STREAMER_CFG_HI) {
        raw_[csr - SNAX_STREAMER_CFG_LO] = val;
        return;
    }
    switch (csr) {
        case SNAX_STREAMER_START:
            break;  // functional: run happens at simbacore start
        case SNAX_DELAYED_START_R10:
            // Engine path: latch the SUC release at the ACTUAL SW-write cycle (advance the engine to it
            // first so the gauge it polled is consistent). Else release is gauge-based in the co-sim.
            if (val && engine_active_) { advance_to(at); engine_.release_r10(); }
            break;
        case SNAX_DELAYED_START_R11:
            if (val && engine_active_) { advance_to(at); engine_.release_r11(); }
            break;
        case SNAX_MODE:
            cfg_.mode = val;
            break;
        case SNAX_SEQ_LEN:
            cfg_.seqLen = val;
            break;
        case SNAX_D_MODEL:
            cfg_.dModel = val;
            break;
        case SNAX_DT_RANK:
            cfg_.dtRank = val;
            break;
        case SNAX_D_INNER:
            cfg_.dInner = val;
            break;
        case SNAX_D_FINAL:
            cfg_.dFinal = val;
            break;
        case SNAX_SIMBACORE_START:
            if (val) run_invocation(at);
            break;
        default:
            break;
    }
}

// CSR pre-loading (docs/dataflow/08): the app issues the NEXT tile's base-ptr/stride/
// bound writes AFTER asserting START, while the accelerator is still busy, so the offload
// latency of those writes is hidden under compute. A streamer-config write (range
// [960,1158]) issued while the core is busy (at < accel_end_) is therefore FREE; the
// MODE/dim/START writes (outside that range) and any config write while idle (first-tile
// setup, post-poll re-launch) are on the critical path and charged.
bool SimWorld::snax_write_serializes(uint32_t csr, uint64_t at) const {
    bool config = (csr >= SNAX_STREAMER_CFG_LO && csr <= SNAX_STREAMER_CFG_HI);
    bool busy   = at < accel_end_;
    return !(config && busy);  // overlapped preload -> not on the critical path
}

uint32_t SimWorld::gauge_at(uint64_t now, uint64_t t0, uint64_t t1, uint32_t total) const {
    if (now <= t0 || total == 0) return 0;
    if (now >= t1 || t1 <= t0) return total;
    return (uint32_t)((unsigned long long)(now - t0) * total / (t1 - t0));
}

uint32_t SimWorld::snax_read(uint32_t csr, uint64_t at, bool& is_poll) {
    is_poll = false;
    // The first BUSY poll happens after the SW has issued the invocation's DMAs (the refill loop
    // runs before the wait_simbacore), so this is the point to fold the recorded DMA contention into
    // the P2 timeline. Apps with no mid-invocation DMA recorded nothing -> no re-step, no change.
    if (inv_kind_ && !inv_finalized_ && (csr == SNAX_SIMBACORE_BUSY || csr == SNAX_STREAMER_BUSY)) {
        inv_finalized_ = true;
        if (inv_dma_.busy()) finalize_inv();
    }
    switch (csr) {
        case SNAX_STREAMER_BUSY:
            is_poll = true;
            return (engine_active_ || at < accel_end_) ? 1 : 0;
        case SNAX_SIMBACORE_BUSY:
            is_poll = true;
            return (engine_active_ || at < accel_end_) ? 1 : 0;
        case SNAX_STREAMER_PERF:
            return 0;
        case SNAX_SIMBACORE_PERF:
            return perf_;
        case SNAX_R10_GAUGE:
            is_poll = true;
            // Engine path: the live osCore-tile fire-counter (advance_to stepped it to `at`). Else the
            // co-sim fire-vector / linear ramp.
            if (engine_used_) return engine_.g_r10();
            return !g_r10_fire_.empty() ? gauge_stepped(g_r10_fire_, at)
                                        : gauge_at(at, accel_start_, t_oscore_done_, g_r10_total_);
        case SNAX_R11_GAUGE:
            is_poll = true;
            if (engine_used_) return engine_.g_r11();
            return !g_r11_fire_.empty() ? gauge_stepped(g_r11_fire_, at)
                                        : gauge_at(at, t_suc_start_, t_suc_done_, g_r11_total_);
        case SNAX_ISCORE_TILE_CNT:
            is_poll = true;
            if (engine_used_) return engine_.g_iscore();
            // P2: isCore runs last (after SUC); P1: overlaps osCore from start.
            return gauge_at(at, phase2_ ? t_suc_done_ : accel_start_, accel_end_, g_iscore_total_);
        default:
            if (csr >= SNAX_STREAMER_CFG_LO && csr <= SNAX_STREAMER_CFG_HI) return raw_[csr - SNAX_STREAMER_CFG_LO];
            return 0;
    }
}

void SimWorld::run_invocation(uint64_t at) {
    decode_ports();
    // Fresh per-invocation DMA-contention state: DMAs the SW launches during this invocation
    // (recorded in dma_submit) are folded in by finalize_p2() on the first SIMBACORE_BUSY read.
    inv_dma_.clear();
    // The per-cycle DMA-contention engine is OPT-IN (MEMSIM_DMA_PERIOD=<beats period>). Verification
    // showed P2's SUC is switchCore-bound (sw_cyc gating already captures the oscore gap to +0.5%), so
    // the engine's bank contention is mostly hidden and over-counts if always on. Off by default ->
    // the (accurate) provisional timeline is unchanged. See docs/dataflow/10_memsim.md.
    // The per-cycle DMA-contention engine is ALWAYS ON: all timing emerges from stepping, no closed
    // form. beat_period_l3 = measured L3 rate (~4 cyc/64B beat). MEMSIM_DMA_PERIOD overrides the period;
    // =0 disables the engine (debug only -> falls back to the dma_cycles approximation).
    dma_engine_on_ = true;
    inv_dma_.beat_period_l3 = 4;
    if (const char* e = std::getenv("MEMSIM_DMA_PERIOD")) {
        int p = atoi(e);
        if (p > 0) inv_dma_.beat_period_l3 = p;
        else dma_engine_on_ = false;
    }
    inv_kind_     = 0;
    inv_finalized_ = false;
    engine_active_ = false;
    engine_used_   = false;
    // ---- compute per-invocation busy-cycle durations (see docs/dataflow/10_memsim.md
    // and the MambaCore FSM). cfg_.dInner is the per-tile dInner (M*_dInner_tile). ----
    const uint32_t Mu = 16, Nu = 24;
    uint32_t M_i = cfg_.seqLen / Mu;                            // seqLen tiles
    uint32_t osN = (cfg_.dInner >= Nu) ? cfg_.dInner / Nu : 0;  // osCore N tiles per invocation
    uint32_t K_i = osN;                                         // isCore K steps per invocation
    // Decode the MODE bitfield (SimbaCoreCtrlBundle, width 20, MSB first): bit19 en_osCore,
    // bit18 en_suCore, bit17 en_isCore. MODE -- not the streamer ports -- distinguishes a SUC
    // scan from a GEMM/SIMD op: the ports (R6/R7) are reused across modes, so keying off them
    // mis-fires (an FFT SIMD pass enables R6/R7 with all core bits 0). See docs.
    bool en_osCore = (cfg_.mode >> 19) & 1;
    bool en_suCore = (cfg_.mode >> 18) & 1;
    bool en_isCore = (cfg_.mode >> 17) & 1;
    bool is_simd   = !(en_osCore || en_suCore || en_isCore);  // SIMD: no GEMM subcore enabled
    osc_dur_       = en_osCore ? (uint64_t)M_i * osN * cfg_.dModel : 0;
    isc_dur_       = en_isCore ? (uint64_t)M_i * cfg_.dFinal * K_i : 0;
    phase2_        = en_suCore;  // SUC scan active <=> en_suCore (not the ports)
    Fabric::h_on   = std::getenv("MEMSIM_BANKHIST") != nullptr;  // per-bank contention diag
    if (Fabric::h_on && (en_suCore || en_osCore || en_isCore)) Fabric::hist_reset();  // scope to this invocation

    // SUC duration comes from the per-cycle R7 dt_BC fabric sim (cyc_suc_duration), which
    // includes the bank-conflict magnitude; the SUC compute-pipeline fill is modelled
    // separately (the s2s commit-pipe, ~20cc). See cyc.cpp / docs.
    suc_dur_ = 0;
    if (en_suCore) {
        suc_dur_ = cyc_suc_duration(ports_[7], cfg_.seqLen, cfg_.dInner, 0);
        if (std::getenv("MEMSIM_ACC"))
            std::fprintf(stderr,
                         "  SUC per-cycle: suc=%llu "
                         "(seqLen=%u dInner_tile=%u)\n",
                         (unsigned long long)suc_dur_, cfg_.seqLen, cfg_.dInner);
    }

    accel_start_ = at;
    if (is_simd) {
        // SIMD pass (SimdCore): no GEMM subcore runs. The enabled reader/writer ports are stepped by the
        // single engine on the ONE shared fabric with the live DMA beat engine, so strided-gather bank
        // conflicts AND any concurrent ring DMA emerge (was cyc_simd + a scalar dma_cycles prefetch tail).
        // perf counter stays 0 (SIMD is outside the MambaCore counter); accel_end_ from advance_to.
        engine_.configure(ports_, cfg_, r10_start_cnt_, r11_start_cnt_, 0);
        engine_.rec_fires = false;
        engine_.rec_fifo  = trace_on_;
        engine_active_ = engine_used_ = engine_.active();
        if (engine_active_) {
            accel_end_     = accel_start_ + 1000000000ULL;  // sentinel until advance_to completes it
            t_oscore_done_ = t_suc_start_ = t_suc_done_ = accel_start_;
            if (dma_busy_until_ > accel_start_ && dma_busy_addr_) {
                long ovb = ((long)(dma_busy_until_ - accel_start_) + 3) / 4;
                engine_.dma_enqueue(dma_busy_addr_, (uint32_t)(ovb * 64), true, 0);
            }
        } else {  // degenerate (no enabled ports)
            t_oscore_done_ = t_suc_done_ = accel_end_ = at;
        }
    } else if (en_suCore) {
        // Phase 2 with safe-to-start overlap. The app releases the SUC when the R10 (osCore z)
        // gauge reaches r10_start_cnt, and the isCore when the R11 (SUC y) gauge reaches
        // r11_start_cnt (shipped via SNAX_DELAYED_START_R10/R11). With full-serialize start_cnts
        // (== gauge totals, the conservative default used by `main`) this reduces to strict
        // osCore->SUC->isCore; a tiled app that paces them low overlaps the stages, which is why
        // modeling it matters — strict-serialize over-predicts P2 by ~40% on overlapped configs.
        // Per-stage fills (osCore+261, isCore+118) are the handoff bubble from gauge-trigger to the
        // consumer's real start; the SUC bank conflict is already inside suc_dur_ (cyc_suc_duration).
        // Cycle-accurate co-sim: step osCore->z->SUC->y->isCore on the fabric with the app's
        // start_cnts; the overlapped busy count and the per-stage windows EMERGE from stepping
        // (no gauge fractions, no fill constants). Validated: main R11 boundary 5377 sits inside
        // the vsim 5300-hang/5400-safe bracket, R10=2 exact. See cyc_phase2.cpp.
        // A prefetch/spill DMA still in flight at accel start preempts a cycling superbank for its
        // remaining beats (double-buffered apps overlap the next tile's DMA with this compute).
        long dma_ov = (dma_busy_until_ > accel_start_) ? (long)(dma_busy_until_ - accel_start_) : 0;
        // switchCore dt projection (Matmul) cycles: dt_delta = dt·Wᵀ streams ON-CHIP to the SUC, which
        // can't scan past it -> they are co-active. Cycles = seqLen·dInner·dtRank/(convUnroll·dtRankUnroll).
        long sw_cyc = (((cfg_.mode >> 8) & 0x3) == 2 && dtRankUnroll_ && cfg_.dtRank)
                          ? (long)((uint64_t)cfg_.seqLen * cfg_.dInner * cfg_.dtRank / (4u * dtRankUnroll_))
                          : 0;
        // Single cycle-exact engine (MEMSIM_ENGINE): configure now; advance_to steps it one cycle at a
        // time with DMA enqueued live (dma_submit). accel_end_/perf_/gauges come from the engine once it
        // completes; the DMA hides naturally via the reader FIFO slack -> no over-count. Sentinel
        // accel_end_ keeps BUSY polls "busy" until advance_to finishes the engine.
        {  // the single per-cycle engine — the only P1/P2 stepper
            engine_.configure(ports_, cfg_, r10_start_cnt_, r11_start_cnt_, sw_cyc);
            engine_.rec_fires = true;
            engine_.rec_fifo  = trace_on_;  // per-cycle FIFO occupancy for the schedule plot
            engine_active_ = engine_used_ = engine_.active();
            if (engine_active_) {
                accel_end_     = accel_start_ + 1000000000ULL;  // sentinel until advance_to completes it
                t_oscore_done_ = t_suc_start_ = t_suc_done_ = accel_start_;
                // DMA still in flight at invocation start (the dma_ov prefetch tail): ground its real
                // superbank for the remaining cycles, so initial DMA contention emerges like live DMA.
                if (dma_busy_until_ > accel_start_ && dma_busy_addr_) {
                    long ovb = ((long)(dma_busy_until_ - accel_start_) + 3) / 4;  // L3 beat period = 4
                    engine_.dma_enqueue(dma_busy_addr_, (uint32_t)(ovb * 64), true, 0);
                }
            }
        }
        (void)dma_ov;
        if (!engine_active_) {  // degenerate config (M_i<=0 etc.): strict serialize from stage durations
            t_oscore_done_ = at + osc_dur_;
            t_suc_start_   = t_oscore_done_;
            t_suc_done_    = t_suc_start_ + suc_dur_;
            uint64_t t_isc_start = t_suc_done_, t_isc_done = t_isc_start + isc_dur_;
            accel_end_     = t_isc_done;
            if (en_osCore) rec(TR_OSCORE, accel_start_, accel_start_ + osc_dur_, osc_dur_);
            rec(TR_SUC, t_suc_start_, t_suc_done_, (uint64_t)cfg_.seqLen * cfg_.dInner);
            if (en_isCore) rec(TR_ISCORE, t_isc_start, t_isc_done, isc_dur_);
        }
    } else if (en_osCore && !en_isCore) {
        // OSGEMM: standalone osCore array (R0=A, R1=B -> array(K=dModel) -> W0), now stepped by the single
        // engine on the shared fabric with the live DMA beat engine — so the async oscore-in ring's refill
        // DMA contends by arbitration, exactly like the P2/SUC path (was cyc_gemm + a scalar dma_cycles
        // prefetch tail that ignored mid-invocation refills). accel_end_/perf_/gauges come from advance_to.
        engine_.configure(ports_, cfg_, r10_start_cnt_, r11_start_cnt_, 0);
        engine_.rec_fires = true;
        engine_.rec_fifo  = trace_on_;
        engine_active_ = engine_used_ = engine_.active();
        if (engine_active_) {
            accel_end_     = accel_start_ + 1000000000ULL;  // sentinel until advance_to completes it
            t_oscore_done_ = t_suc_start_ = t_suc_done_ = accel_start_;
            if (dma_busy_until_ > accel_start_ && dma_busy_addr_) {  // prefetch tail in flight at start
                long ovb = ((long)(dma_busy_until_ - accel_start_) + 3) / 4;  // L3 beat period = 4
                engine_.dma_enqueue(dma_busy_addr_, (uint32_t)(ovb * 64), true, 0);
            }
        } else {  // degenerate (M_i<=0 etc.): strict serialize from the stage duration
            t_oscore_done_ = t_suc_done_ = at;
            accel_end_                   = at + osc_dur_;
            rec(TR_OSCORE, at, accel_end_, (uint64_t)M_i * osN * cfg_.dModel);
        }
    } else if (en_isCore && !en_osCore) {
        // ISGEMM: standalone isCore array draining through W3, now stepped by the single engine on the
        // shared fabric with the live DMA beat engine — so the async psum-ring spill/reload DMA contends
        // on W3's superbank by arbitration (was cyc_gemm + scalar dma_cycles). accel_end_/perf_ from advance_to.
        engine_.configure(ports_, cfg_, r10_start_cnt_, r11_start_cnt_, 0);
        engine_.rec_fires = true;
        engine_.rec_fifo  = trace_on_;
        engine_active_ = engine_used_ = engine_.active();
        if (engine_active_) {
            accel_end_     = accel_start_ + 1000000000ULL;  // sentinel until advance_to completes it
            t_oscore_done_ = t_suc_start_ = t_suc_done_ = accel_start_;
            if (dma_busy_until_ > accel_start_ && dma_busy_addr_) {
                long ovb = ((long)(dma_busy_until_ - accel_start_) + 3) / 4;
                engine_.dma_enqueue(dma_busy_addr_, (uint32_t)(ovb * 64), true, 0);
            }
        } else {  // degenerate
            t_oscore_done_ = t_suc_done_ = at;
            accel_end_                   = at + isc_dur_;
            rec(TR_ISCORE, at, accel_end_, (uint64_t)M_i * cfg_.dFinal * K_i);
        }
    } else {
        // both-core chained (PHASE1, IS_OSGEMM): osCore -> switchCore conv -> isCore pipeline,
        // stepped per cycle on the shared fabric (cyc_phase1) — the slowest stage, the lead-in/drain
        // and the cross-stage contention all emerge, no max-of-formulas, no fitted fill.
        int sw_mode    = (cfg_.mode >> 8) & 0x3;  // m_switchCoreMode (1=Conv -> PHASE1)
        long dma_ov    = (dma_busy_until_ > at) ? (long)(dma_busy_until_ - at) : 0;
        // Single cycle-exact engine (MEMSIM_ENGINE): the P1 pipeline runs real-time with DMA enqueued
        // live, so contention emerges (no deferred cycling-superbank approximation). Else legacy cyc_phase1.
        {  // the single per-cycle engine — the only P1/P2 stepper
            engine_.configure(ports_, cfg_, r10_start_cnt_, r11_start_cnt_, /*sw_cyc=*/0);
            engine_.rec_fires = false;
            engine_.rec_fifo  = trace_on_;  // per-cycle FIFO occupancy for the schedule plot
            engine_active_ = engine_used_ = engine_.active();
            if (engine_active_ && dma_busy_until_ > accel_start_ && dma_busy_addr_) {
                long ovb = ((long)(dma_busy_until_ - accel_start_) + 3) / 4;  // prefetch tail (dma_ov)
                engine_.dma_enqueue(dma_busy_addr_, (uint32_t)(ovb * 64), true, 0);
            }
        }
        if (engine_active_) {
            accel_end_ = accel_start_ + 1000000000ULL;  // sentinel until advance_to completes the engine
        } else {
            accel_end_ = at + osc_dur_ + isc_dur_;  // degenerate (engine inactive): strict serialize
        }
        (void)dma_ov;
        has_conv_  = (sw_mode == 1);
        t_oscore_done_ = at + osc_dur_;
        t_suc_done_    = t_oscore_done_;
        // Trace rows: the three chained stages are co-active across the whole invocation
        // [at, accel_end_], so each bar spans the true stepped duration; `ideal` stays the
        // conflict/drain-free work count, so util = ideal/(accel_end_-at) reflects the
        // shared-fabric bank contention + pipeline lead-in/drain (osCore is the bottleneck;
        // the shorter conv/isCore stages read lower because they idle within the window).
        if (!engine_active_) {  // engine path: accel_end_ is a sentinel until advance_to completes it
            rec(TR_OSCORE, at, accel_end_, osc_dur_);
            if (sw_mode == 1) {
                uint64_t conv_dur = (uint64_t)cfg_.seqLen * cfg_.dInner / 4;
                rec(TR_SWITCHCORE, at, accel_end_, conv_dur);
            }
            rec(TR_ISCORE, at, accel_end_, isc_dur_);
        }
    }
    // The perf counter counts MambaCore busy (globalState != sIDLE, MambaCore.scala:159). The
    // SimdCore is outside the MambaCore, so a SIMD pass costs wall-clock (accel_end_) but does
    // not tick the counter -> perf_ = GEMM/SUC busy only, 0 for a SIMD-only invocation.
    // engine path: accel_end_/perf_ are set by advance_to when the engine completes (sentinel for now).
    if (!engine_active_) perf_ = is_simd ? 0u : (uint32_t)(accel_end_ - accel_start_);
    g_r10_total_    = M_i * osN;
    g_r11_total_    = cfg_.seqLen * cfg_.dInner;
    g_iscore_total_ = M_i * cfg_.dFinal * K_i;

    // TCDM traffic for the bandwidth line: every enabled streamer port issues PORT_NCH narrow
    // (8B) accesses per temporal step, so its total accesses = (product of temporal bounds)*NCH.
    // Summed over all ports active this invocation and spread over [accel_start_, accel_end_], this
    // is the average TCDM word demand; plot_timeline.py divides by the 32-bank peak. (In P2 the
    // serialized stages are averaged together — a single invocation-average, not per-stage peaks.)
    if (trace_on_ && !engine_active_) {  // engine path: accel_end_ is a sentinel here -> deferred to advance_to
        uint64_t tcdm_words = 0;
        for (int p = 0; p < N_PORTS; p++) {
            if (!ports_[p].enabled) continue;
            uint64_t steps = 1;
            for (int d = 0; d < 4; d++) steps *= ports_[p].t_bound[d] ? (uint64_t)ports_[p].t_bound[d] : 1;
            tcdm_words += steps * PORT_NCH[p];
        }
        rec(TR_TCDM, accel_start_, accel_end_, tcdm_words);
    }

    // Golden-free AGU structural audit: runs on EVERY invocation of ANY app (located
    // bounds + producer->consumer containment). The detailed mamba osCore round-trip below
    // runs once on the first P2 invocation (FP-aware, mamba-only).
    if (verify_) verify_layout();
    if (verify_ && phase2_ && !verify_ran_) {
        verify_ran_ = true;
        verify_datapath();
    }

    if (std::getenv("MEMSIM_ACC"))
        std::fprintf(stderr,
                     "  accel mode=%u P%d seqLen=%u dModel=%u dInner=%u dFinal=%u "
                     "osc=%llu suc=%llu isc=%llu dur=%u busy=%llu\n",
                     cfg_.mode, phase2_ ? 2 : 1, cfg_.seqLen, cfg_.dModel, cfg_.dInner, cfg_.dFinal,
                     (unsigned long long)osc_dur_, (unsigned long long)suc_dur_, (unsigned long long)isc_dur_, perf_,
                     (unsigned long long)(accel_end_ - accel_start_));
}

// Re-step the P2 co-sim with the DMAs the SW issued during this invocation (inv_dma_), now stepping
// on the shared fabric so DMA<->streamer contention emerges by arbitration. Updates the (previously
// provisional) accel_end_/perf_/gauges. ports_/cfg_ still describe THIS invocation (decode_ports runs
// only at START; the SW's next-tile config writes land in raw_ but aren't decoded until the next START).
void SimWorld::finalize_inv() {
    // Obsolete deferred DMA re-step: the single engine enqueues DMA live (dma_submit -> engine_.dma_enqueue)
    // and steps it on the same fabric as compute, so there is nothing to fold in afterwards. inv_kind_ is
    // never set anymore, so this is never reached; kept as a no-op for the existing call site.
}

// Scale-normalized FP32-vs-golden compare for the functional datapath. A correct
// kernel matches the FP8 golden up to a single global requant scale (normalized out
// via the median model/golden ratio) plus per-element fp8/LUT rounding; a layout/
// stale-read error gives wrong OPERANDS -> large residual on many elements. An element
// is "bad" only if off by > TOL relative AND > ABS absolute (the ABS floor ignores
// near-zero fp8 round-to-0 noise). Reports count + first offending indices.
static bool cmp_fp32_golden(const std::vector<double>& model, const std::vector<double>& gold, const char* label,
                            int cols, double TOL, double ABS, double max_bad_frac) {
    std::vector<double> ratios;
    for (size_t i = 0; i < gold.size(); i++)
        if (std::fabs(gold[i]) > 1e-3) ratios.push_back(model[i] / gold[i]);
    double scale = 1.0;
    if (!ratios.empty()) {
        std::sort(ratios.begin(), ratios.end());
        scale = ratios[ratios.size() / 2];
    }
    int bad = 0, nfm = 0, fm[8];
    for (size_t i = 0; i < gold.size(); i++) {
        double ref = scale * gold[i], d = std::fabs(model[i] - ref);
        if (d > ABS && d / (std::fabs(ref) + 1e-3) > TOL) {
            if (nfm < 8) fm[nfm++] = (int)i;
            bad++;
        }
    }
    // A scale far from 1 means the model collapsed (~0) or blew up — not a real match,
    // even if residuals look "within tol" (0 vs 0). The requant scale is ~1 for all kernels.
    bool collapsed = std::fabs(scale) < 0.05 || std::fabs(scale) > 20.0;
    bool ok        = !collapsed && bad <= (int)(max_bad_frac * gold.size());
    std::fprintf(stderr, "FP32 %s vs golden: requant scale=%.4g, %d/%zu within %.0f%% -> %s%s\n", label, scale,
                 (int)gold.size() - bad, gold.size(), TOL * 100, ok ? "MATCH" : "MISMATCH",
                 collapsed ? " (scale collapse!)" : "");
    for (int i = 0; i < nfm; i++)
        std::fprintf(stderr, "    %s @ idx %d (r=%d,c=%d): model=%.3f scale*golden=%.3f\n", label, fm[i], fm[i] / cols,
                     fm[i] % cols, model[fm[i]], scale * gold[fm[i]]);
    return ok;
}

// Golden-free AGU structural audit, run on every invocation of every app. Three located,
// pure-address checks (no golden needed); see docs/dataflow/10_memsim.md:
//   (1) bounds: each streamer's address extent stays inside the 512 KiB TCDM.
//   (2) producer->consumer containment: a reader overlapping a writer reads only within it.
//   (3) writer no-alias: a writer never writes one word twice (a permutation within bounds).
// Catches gross AGU bugs (OOB, overrun) and in-bounds output permutations. A bijective-but-
// wrong order still needs the golden datapath check (verify_datapath, mamba only).
void SimWorld::verify_layout() {
    static const int NCH[N_PORTS] = {2, 4, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 4, 4, 1, 1, 1, 4};
    const uint32_t LO = 0x10000000u, HI = 0x10000000u + 0x80000u;  // TCDM [base, base+512KiB)
    auto extent = [&](int p, uint32_t& lo, uint32_t& hi) {
        const Agu& a = ports_[p];
        int64_t mn = 0, mx = 0;
        for (int d = 0; d < 4; d++) {  // temporal extent (signed strides)
            int b     = a.t_bound[d] ? a.t_bound[d] : 1;
            int64_t s = (int64_t)(b - 1) * a.t_stride[d];
            (s >= 0 ? mx : mn) += s;
        }
        int ns = a.n_s ? a.n_s : 1;  // spatial extent
        if (ns == 2) {
            for (int s = 0; s < 2; s++) {
                int64_t v = a.s_stride[s];
                (v >= 0 ? mx : mn) += v;
            }
        } else {
            int64_t v = (int64_t)(NCH[p] - 1) * a.s_stride[0];
            (v >= 0 ? mx : mn) += v;
        }
        lo = (uint32_t)((int64_t)a.base + mn);
        hi = (uint32_t)((int64_t)a.base + mx);
    };
    uint32_t rlo[N_PORTS] = {0}, rhi[N_PORTS] = {0};
    for (int p = 0; p < N_PORTS; p++)
        if (ports_[p].enabled) extent(p, rlo[p], rhi[p]);
    layout_invs_++;
    auto report = [&](int p, const char* kind, uint32_t lo, uint32_t hi) {
        n_layout_errs_++;
        layout_pass_ = false;
        if (layout_print_budget_-- > 0)
            std::fprintf(stderr, "  LAYOUT ERROR (inv %d): %s%d %s addr [%08x,%08x]\n", layout_invs_,
                         p < N_READERS ? "R" : "W", p < N_READERS ? p : p - N_READERS, kind, lo, hi);
    };
    for (int p = 0; p < N_PORTS; p++)  // (1) bounds
        if (ports_[p].enabled && (rlo[p] < LO || rhi[p] >= HI))
            report(p, "out-of-bounds (outside TCDM)", rlo[p], rhi[p]);
    for (int r = 0; r < N_READERS; r++) {  // (2) producer->consumer containment
        if (!ports_[r].enabled) continue;
        for (int w = N_READERS; w < N_PORTS; w++) {
            if (!ports_[w].enabled) continue;
            bool overlap = rlo[r] < rhi[w] && rlo[w] < rhi[r];  // strict (ignore boundary touch)
            if (overlap && (rlo[r] < rlo[w] || rhi[r] > rhi[w]))
                report(r, "reads beyond producer's written range", rlo[r], rhi[r]);
        }
    }
    // (3) Writer no-alias: enumerate each writer's emitted address sequence and flag the first
    // word written twice -- a permuted stride within the (correct) extent that the bounds check
    // can't see. Stride-0 dims are collapsed first (the accumulation pass, e.g. the isCore
    // K-reduction, legitimately holds the output address). Writer-only: readers re-read via
    // stride-0 reuse, and no writer has n_s==2 (R7 is the only 2-spatial port, a reader).
    auto alias_scan = [&](const Agu& a, long nch, int wport, bool do_report) -> bool {
        long b[4];  // collapse stride-0 (accumulation) dims
        for (int d = 0; d < 4; d++) b[d] = a.t_stride[d] == 0 ? 1 : (a.t_bound[d] ? a.t_bound[d] : 1);
        long ns    = a.s_stride[0] == 0 ? 1 : (nch ? nch : 1);
        long total = b[0] * b[1] * b[2] * b[3] * ns;
        if (total > 4000000) {  // never silently cap: say what's skipped
            std::fprintf(stderr, "  AGU no-alias: W%d skipped (%ld words > 4M cap)\n", wport - N_READERS, total);
            return false;
        }
        std::unordered_set<uint32_t> seen;
        seen.reserve((size_t)total * 2);
        long iter = 0;
        for (long i0 = 0; i0 < b[0]; i0++)
            for (long i1 = 0; i1 < b[1]; i1++)
                for (long i2 = 0; i2 < b[2]; i2++)
                    for (long i3 = 0; i3 < b[3]; i3++)
                        for (long s = 0; s < ns; s++, iter++) {
                            uint32_t addr = (uint32_t)((int64_t)a.base + i0 * a.t_stride[0] + i1 * a.t_stride[1] +
                                                       i2 * a.t_stride[2] + i3 * a.t_stride[3] + s * a.s_stride[0]);
                            if (!seen.insert(addr).second) {  // already written -> alias
                                if (do_report) {
                                    n_layout_errs_++;
                                    layout_pass_ = false;
                                    if (layout_print_budget_-- > 0)
                                        std::fprintf(stderr,
                                                     "  LAYOUT ERROR (inv %d): W%d ALIASES "
                                                     "word %08x (written twice by emit #%ld) -- permutation "
                                                     "within bounds\n",
                                                     layout_invs_, wport - N_READERS, addr, iter);
                                }
                                return true;  // one located alias per writer is enough
                            }
                        }
        return false;
    };
    for (int w = N_READERS; w < N_PORTS; w++)
        if (ports_[w].enabled && ports_[w].n_s <= 1) alias_scan(ports_[w], NCH[w], w, true);
    // Liveness self-test (MEMSIM_LAYOUT_FAULT): a deliberately broken AGU (outer stride
    // pushed far past TCDM) must be located, proving "0 errors" means the audit is live.
    // Synthetic; does not touch the real audit state.
    if (std::getenv("MEMSIM_LAYOUT_FAULT") && layout_invs_ == 1) {
        int r = -1;
        for (int p = 0; p < N_READERS; p++)
            if (ports_[p].enabled) {
                r = p;
                break;
            }
        if (r >= 0) {
            int b       = ports_[r].t_bound[1] > 1 ? ports_[r].t_bound[1] - 1 : 1;
            int64_t hi  = (int64_t)ports_[r].base + (int64_t)b * (ports_[r].t_stride[1] + 0x100000);
            bool caught = (uint64_t)hi >= HI;
            std::fprintf(stderr, "  AGU audit self-test: R%d outer-stride+0x100000 -> hi=%08lx -> %s\n", r,
                         (unsigned long)(uint32_t)hi, caught ? "LOCATED (audit is live)" : "MISSED (audit DEAD)");
        }
        // no-alias self-test: a synthetic writer with two EQUAL non-zero temporal strides
        // (base + 8*(i0+i1)) revisits the same word for every (i0,i1) with equal i0+i1 -- a
        // permutation collision among moving dims. The collapsed check must catch it (proves
        // it is live, and that stride-0 collapse didn't neuter it).
        Agu fake;
        fake.t_stride[0] = 8;
        fake.t_bound[0]  = 4;  // both non-zero, equal -> aliasing
        fake.t_stride[1] = 8;
        fake.t_bound[1]  = 4;
        bool caught      = alias_scan(fake, 1, N_READERS, false);
        std::fprintf(stderr, "  AGU no-alias self-test: equal-stride writer -> %s\n",
                     caught ? "LOCATED (no-alias check is live)" : "MISSED (DEAD)");
    }
}

// Golden-aware datapath verification (mamba P2, runs once): integer osCore GEMM round-trip,
// FP32 osCore/isCore/SUC cross-checks vs the app's FP8 goldens (MEMSIM_DATAPATH), the
// safe-to-start sweep, and the SUC dt_BC delivery BIST. See docs/dataflow/10_memsim.md.
void SimWorld::verify_datapath() {
    const int Mu = 16, Nu = 24, CONV = 4;
    int M = cfg_.seqLen, K = cfg_.dModel, N = cfg_.dInner;  // dInner = per-tile N
    if (M % Mu || N % Nu) {
        return;
    }                                            // not an osCore GEMM shape
    uint32_t Abase = (uint32_t)ports_[0].base;   // R0 oscore_in
    uint32_t Bbase = (uint32_t)ports_[1].base;   // R1 oscore_weight
    uint32_t Zbase = (uint32_t)ports_[14].base;  // W0 z output

    auto flatA = [&](int m, int k) -> uint32_t {  // documented flattenA (bytes)
        return (uint32_t)((m / Mu) * (K * Mu) + k * Mu + (m % Mu));
    };
    auto convfmt = [&](int m, int n) -> uint32_t {  // ConvFormat linear (bytes, fp8)
        int l2 = m / Mu, l1 = m % Mu, d3 = n / Nu, rem = n % Nu, d2 = rem / CONV, c = rem % CONV;
        return (uint32_t)(((((d3) * (M / Mu) + l2) * (Nu / CONV) + d2) * Mu + l1) * CONV + c);
    };

    auto rd8 = [&](uint32_t a) { return (int)(int8_t)mem->ld8(a); };

    // (1) R0 input AGU implements the documented flattenA: temporal k-stride =
    // Mu, m-tile-stride = K*Mu; spatial = serial-width contiguous over the Mu
    // seq lanes (the 2 narrow lanes * 8B serial form 16 = Mu contiguous bytes).
    const Agu& r0 = ports_[0];
    bool r0_ok = r0.t_stride[0] == Mu && r0.t_stride[1] == K * Mu && (r0.s_stride[0] == 0 || r0.s_stride[0] * 2 == Mu);

    // (2) W0 output AGU writes z contiguously in ConvFormat (serial-width 8B/cyc).
    bool w0_ok = ports_[14].t_stride[0] == 8;

    // (3) Element-wise integer GEMM check with MISMATCH REPORTING. For every (m,n):
    // compute z=A.B through the documented flattenA layout, write it via ConvFormat,
    // read it back, and compare to a freshly recomputed reference -> round-trip
    // integrity (catches any write/gather/address inconsistency), reported as a count
    // + the first offending indices. A layout-FAULT self-test (a permuted A-gather)
    // then shows the check localises WHERE values go wrong: it reports how many (m,n)
    // the wrong layout corrupts, and the first few (got vs expected).
    auto gemm = [&](int m, int n, bool scramble) {
        int acc = 0;
        for (int k = 0; k < K; k++) {
            int b = rd8(Bbase + (uint32_t)(k * N + n));
            acc += rd8(Abase + (scramble ? flatA(k % M, (m * 7 + k) % K) : flatA(m, k))) * b;
        }
        return acc & 0xff;
    };
    int rt_checked = 0, rt_bad = 0;                        // round-trip integrity mismatches
    int fault_bad = 0;                                     // layout-fault self-test mismatches
    int first_rt[8][3], first_ft[8][3], nrt = 0, nft = 0;  // {flat_idx, got, expected}
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            int ref = gemm(m, n, false);
            mem->st8(Zbase + convfmt(m, n), (uint8_t)ref);  // datapath write
            int got = rd8(Zbase + convfmt(m, n)) & 0xff;    // read back
            rt_checked++;
            if (got != ref) {
                if (nrt < 8) {
                    first_rt[nrt][0] = m * N + n;
                    first_rt[nrt][1] = got;
                    first_rt[nrt][2] = ref;
                    nrt++;
                }
                rt_bad++;
            }
            if (gemm(m, n, true) != ref) {  // fault self-test
                if (nft < 8) {
                    first_ft[nft][0] = m * N + n;
                    first_ft[nft][1] = gemm(m, n, true);
                    first_ft[nft][2] = ref;
                    nft++;
                }
                fault_bad++;
            }
        }
    bool discriminates = fault_bad > 0;  // the fault must corrupt some values

    // Producer/consumer: R10 (SUC z reader) base must match W0 (z writer) base,
    // i.e. the consumer reads the buffer the producer wrote.
    bool pc_ok = !ports_[10].enabled || (ports_[10].base == ports_[14].base);

    layout_pass_ = r0_ok && w0_ok && rt_bad == 0 && discriminates && pc_ok;
    std::fprintf(stderr,
                 "Layout check (osCore GEMM, M=%d K=%d N=%d): R0 flattenA=%s  W0 ConvFormat=%s  "
                 "producer->consumer(W0->R10)=%s\n"
                 "  round-trip: %d/%d correct, %d mismatch%s\n"
                 "  layout-fault self-test: %d/%d values corrupted by a permuted A-gather (check is %s). %s\n",
                 M, K, N, r0_ok ? "PASS" : "FAIL", w0_ok ? "PASS" : "FAIL", pc_ok ? "PASS" : "FAIL",
                 rt_checked - rt_bad, rt_checked, rt_bad, rt_bad ? "" : " (clean)", fault_bad, rt_checked,
                 discriminates ? "live" : "DEAD (no-op!)", layout_pass_ ? "LAYOUT PASS" : "LAYOUT FAIL");
    for (int i = 0; i < nrt; i++)
        std::fprintf(stderr, "    round-trip mismatch @ idx %d: got %d, expected %d\n", first_rt[i][0], first_rt[i][1],
                     first_rt[i][2]);
    if (std::getenv("MEMSIM_ACC"))
        for (int i = 0; i < nft; i++)
            std::fprintf(stderr, "    [fault] idx %d (m=%d,n=%d): permuted=%d vs correct=%d\n", first_ft[i][0],
                         first_ft[i][0] / N, first_ft[i][0] % N, first_ft[i][1], first_ft[i][2]);

    // FP32 osCore GEMM vs the app's FP8 golden (M2_oscore_expected, tile 0): decode A/B
    // fp8_alt->FP32, compute z=A.B, compare to the golden. A global requant scale is
    // normalized out (median ratio) so only wrong operands (layout/stale read) flag; the
    // tolerance covers fp8(e5m2) rounding. See docs/dataflow/10_memsim.md.
    // Gated on MEMSIM_DATAPATH like the isCore/SUC golden checks below: the flattenB/convfmt
    // layout assumed here is the non-tiled `main` layout, so a TILED app's tile-0 golden does
    // not match (scale collapses to ~0) and would false-fail every tiled config. The structural
    // round-trip + AGU audit above already cover layout for tiled apps without a golden.
    if (std::getenv("MEMSIM_DATAPATH") && golden_z_) {
        auto rdf = [&](uint32_t a) { return fp8_alt_to_f32((uint8_t)mem->ld8(a)); };
        std::vector<double> model, gold;
        // oscore_weight is flattenB(N_M_K, Ku=1, Nu=24): col-major tile grid, so element
        // (k,n) is at ((n/24)*K + k)*24 + (n%24). (For a single N-tile, N<=24, this reduces
        // to row-major — which is why a per-tile config masked the difference.)
        auto flatB_os = [&](int k, int n) { return (uint32_t)(((n / 24) * K + k) * 24 + (n % 24)); };
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                float acc = 0.f;
                for (int k = 0; k < K; k++) acc += rdf(Abase + flatA(m, k)) * rdf(Bbase + flatB_os(k, n));
                model.push_back(acc);
                gold.push_back(rdf(golden_z_ + convfmt(m, n)));
            }
        std::vector<double> ratios;
        for (size_t i = 0; i < gold.size(); i++)
            if (std::fabs(gold[i]) > 1e-3) ratios.push_back(model[i] / gold[i]);
        double scale = 1.0;
        if (!ratios.empty()) {
            std::sort(ratios.begin(), ratios.end());
            scale = ratios[ratios.size() / 2];
        }
        const double TOL = 0.30;    // generous: FP32 vs fp8 (2 mantissa bits)
        const double ABS = 0.0625;  // floor: ignore near-zero fp8 round-to-0 noise
        int bad = 0, nfm = 0, fm[8];
        for (size_t i = 0; i < gold.size(); i++) {
            double ref = scale * gold[i], diff = std::fabs(model[i] - ref);
            if (diff > ABS && diff / (std::fabs(ref) + 1e-3) > TOL) {  // not tiny AND off
                if (nfm < 8) fm[nfm++] = (int)i;
                bad++;
            }
        }
        bool z_ok    = bad <= (int)(0.05 * gold.size());  // tolerate a few fp8-rounding outliers
        layout_pass_ = layout_pass_ && z_ok;
        std::fprintf(stderr, "FP32 osCore vs golden (tile0, %dx%d): requant scale=%.4g, %d/%zu within %.0f%% -> %s\n",
                     M, N, scale, (int)gold.size() - bad, gold.size(), TOL * 100, z_ok ? "MATCH" : "MISMATCH");
        for (int i = 0; i < nfm; i++)
            std::fprintf(stderr, "    z mismatch @ (m=%d,n=%d): model=%.3f scale*golden=%.3f\n", fm[i] / N, fm[i] % N,
                         model[fm[i]], scale * gold[fm[i]]);
    }

    // FP32 isCore output GEMM+requant vs golden (M2_iscore_expected): z_is[m][o] =
    // sum_k y[m][k]*W[k][o]; y from the SUC golden (so isCore validates independently of the
    // SUC), W=isCore_weight (flattenB K_M_N), output flattenCD K_M_N. Gated by MEMSIM_DATAPATH
    // + non-tiled shape (the full inter-stage tensor must fit one invocation; run on `main`).
    if (std::getenv("MEMSIM_DATAPATH") && golden_isc_ && golden_y_ && cfg_.seqLen % 16 == 0 && cfg_.dInner % 24 == 0) {
        const int KU = 24;                                            // dInnerUnroll (Mu=16, CONV=4 from fn scope)
        int isM = cfg_.seqLen, isK = cfg_.dInner, isN = cfg_.dFinal;  // main: 64, 96, 48
        uint32_t Wb = (uint32_t)ports_[12].base;
        auto rdf    = [&](uint32_t a) { return fp8_alt_to_f32((uint8_t)mem->ld8(a)); };
        auto convA  = [&](int m, int k) {  // ConvFormat of y over [seqLen][dInner]
            int l2 = m / Mu, l1 = m % Mu, d3 = k / KU, rem = k % KU, d2 = rem / CONV, c = rem % CONV;
            return (uint32_t)(((((d3) * (isM / Mu) + l2) * (KU / CONV) + d2) * Mu + l1) * CONV + c);
        };
        auto flatB  = [&](int k, int o) { return (uint32_t)(((k / KU) * isN + o) * KU + (k % KU)); };
        auto flatCD = [&](int m, int o) { return (uint32_t)(((m / Mu) * isN + o) * Mu + (m % Mu)); };
        std::vector<double> model, gold;
        for (int m = 0; m < isM; m++)
            for (int o = 0; o < isN; o++) {
                float acc = 0.f;
                for (int k = 0; k < isK; k++) acc += rdf(golden_y_ + convA(m, k)) * rdf(Wb + flatB(k, o));
                model.push_back(acc);
                gold.push_back(rdf(golden_isc_ + flatCD(m, o)));
            }
        layout_pass_ = layout_pass_ && cmp_fp32_golden(model, gold, "isCore (out GEMM)", isN, 0.30, 0.0625, 0.05);
    }

    // FP32 SUC selective scan vs golden (M2_suc_expected): recompute the full Mamba
    // state-update in FP32 through the actual layouts:
    //   dt_delta = dt·Wᵀ + bias (switchCore);  dsp = softplus(dt_delta)
    //   h[d][n] = h[d][n]·exp(A[d][n]·dsp) + (B·dsp)·x ;  y = silu(z)·(Σ_n h·C + x·D)
    // dt/B/C from the bank-transposed dt_BC matrix; W from the split delta-weight; A row-major;
    // x/z ConvFormat. softplus/silu/exp are FP32 stand-ins for the HW LUTs (looser tolerance).
    // Validates the whole datapath + every layout at once (a wrong gather blows up the recurrence).
    if (std::getenv("MEMSIM_DATAPATH") && golden_y_ && golden_z_ && golden_dtbc_ && golden_sucx_ && dState_ &&
        cfg_.seqLen % 16 == 0 && cfg_.dInner % 24 == 0) {
        const int Lq = cfg_.seqLen, Din = cfg_.dInner, dtR = cfg_.dtRank, dS = (int)dState_;
        const int XP = (int)xProjDim_, KU = 24;  // Mu=16, CONV=4 from fn scope
        const int Ku = (int)dtRankUnroll_, convU = 4, dConv = 4, nTilesRow = dtR / (int)dtRankUnroll_;
        const int paddedMat = 128 + (int)bc_pad_ * 8;
        // dt_BC and x (=conv_out) are P1 OUTPUTS the model doesn't compute (0 in TCDM);
        // source them from the goldens. A/D/weights are DMA'd inputs (TCDM is valid).
        uint32_t dtbc = golden_dtbc_, Ab = (uint32_t)ports_[6].base;
        uint32_t Db = (uint32_t)ports_[8].base, xb = golden_sucx_;
        uint32_t w1 = (uint32_t)ports_[3].base, w2 = (uint32_t)ports_[5].base, bb = (uint32_t)ports_[4].base;
        auto rdf   = [&](uint32_t a) { return (double)fp8_alt_to_f32((uint8_t)mem->ld8(a)); };
        auto dtbcB = [&](int m, int col) {  // dt_BC byte for xProj[m][col]
            int flat = ((m / Mu) * XP + col) * Mu + (m % Mu), t = flat / 16, w = flat % 16;
            return (uint32_t)((t / 8) * paddedMat + (w / 2) * 16 + (w % 2) * 8 + (t % 8));
        };
        auto dtAt = [&](int m, int r) { return rdf(dtbc + dtbcB(m, r)); };
        auto BAt  = [&](int m, int n) { return rdf(dtbc + dtbcB(m, dtR + (n / 16) * 32 + (n % 16))); };
        auto CAt  = [&](int m, int n) { return rdf(dtbc + dtbcB(m, dtR + (n / 16) * 32 + 16 + (n % 16))); };
        auto cvx  = [&](uint32_t base, int m, int d) {  // ConvFormat x/z gather
            int l2 = m / Mu, l1 = m % Mu, d3 = d / KU, rem = d % KU, d2 = rem / CONV, c = rem % CONV;
            return rdf(base + (uint32_t)(((((d3) * (Lq / Mu) + l2) * (KU / CONV) + d2) * Mu + l1) * CONV + c));
        };
        auto WAt = [&](int R, int d) {  // split delta-weight W[R][d]
            int i = R / Ku, r = R % Ku, j = d / convU, conv = d % convU;
            return r < dConv ? rdf(w1 + ((j * nTilesRow + i) * convU + conv) * dConv + r)
                             : rdf(w2 + ((j * nTilesRow + i) * convU + conv) * (Ku - dConv) + (r - dConv));
        };
        auto softplus = [](double x) { return x > 20 ? x : std::log1p(std::exp(x)); };
        auto silu     = [](double x) { return x / (1.0 + std::exp(-x)); };
        std::vector<double> h((size_t)Din * dS, 0.0), model, gold;
        std::vector<double> dsp((size_t)Lq * Din);
        for (int i = 0; i < Lq; i++)
            for (int d = 0; d < Din; d++) {
                double acc = rdf(bb + d);
                for (int R = 0; R < dtR; R++) acc += dtAt(i, R) * WAt(R, d);
                dsp[(size_t)i * Din + d] = softplus(acc);
            }
        for (int i = 0; i < Lq; i++)
            for (int d = 0; d < Din; d++) {
                double dd = dsp[(size_t)i * Din + d], xid = cvx(xb, i, d), hC = 0;
                for (int n = 0; n < dS; n++) {
                    double hv = h[(size_t)d * dS + n] * std::exp(rdf(Ab + d * dS + n) * dd) + (BAt(i, n) * dd) * xid;
                    h[(size_t)d * dS + n] = hv;
                    hC += hv * CAt(i, n);
                }
                double zid = cvx(golden_z_, i, d);  // z = osCore output (silu gate)
                model.push_back(silu(zid) * (xid * rdf(Db + d) + hC));
                gold.push_back(cvx(golden_y_, i, d));
            }
        layout_pass_ = layout_pass_ && cmp_fp32_golden(model, gold, "SUC (scan)", Din, 0.40, 0.125, 0.10);
    }

    // Safe-to-start boundary (R10 z-gate osCore->SUC, R11 y-gate SUC->isCore): NO analytic model.
    // The cycle-accurate co-sim steps osCore->z->SUC->y->isCore on the fabric and reports the
    // smallest start_cnt for which the consumer never reads a word its producer has not committed
    // (the stale/hazard check over the real per-cycle write + read schedules). Validated against
    // the vsim brackets (main R10=2, R11=5377 in 5300-hang/5400-safe). See cyc_phase2.cpp.
    if (phase2_) {
        long sw_cyc = (((cfg_.mode >> 8) & 0x3) == 2 && dtRankUnroll_ && cfg_.dtRank)
                          ? (long)((uint64_t)cfg_.seqLen * cfg_.dInner * cfg_.dtRank / (4u * dtRankUnroll_))
                          : 0;
        bool en_osCore = (cfg_.mode >> 19) & 1;
        bool en_isCore = (cfg_.mode >> 17) & 1;
        // Engine-based sweep (single stepper): run the engine at fixed (r10,r11) WITHOUT real-time
        // release -> it gates the SUC on g_r10>=r10 and the isCore on g_r11>=r11; result().stale_z/y
        // count reads of words their producer had not committed. bsearch the smallest zero-stale count.
        long r10_total = (long)(cfg_.seqLen / 16) * (cfg_.dInner >= 24 ? cfg_.dInner / 24 : 0);
        long r11_total = (long)cfg_.seqLen * cfg_.dInner;
        struct SR { long z, y; };
        auto run = [&](long r10c, long r11c) -> SR {
            AccelEngine eng;
            eng.configure(ports_, cfg_, r10c, r11c, sw_cyc);
            if (!eng.active()) return {-1, -1};
            long guard = 0;
            while (eng.step() && ++guard < 500000000) {}
            return {eng.result().stale_z, eng.result().stale_y};
        };
        if (en_osCore && r10_total > 0 && r11_total > 0) {
            long min_r10 = r10_total, lo = 1, hi = r10_total;   // z-gate: isCore released last (r11=total)
            while (lo <= hi) { long mid = lo + (hi - lo) / 2;
                if (run(mid, r11_total).z == 0) { min_r10 = mid; hi = mid - 1; } else lo = mid + 1; }
            long min_r11 = r11_total;                            // y-gate: SUC released at the z-safe point
            if (en_isCore) { lo = 1; hi = r11_total;
                while (lo <= hi) { long mid = lo + (hi - lo) / 2;
                    if (run(min_r10, mid).y == 0) { min_r11 = mid; hi = mid - 1; } else lo = mid + 1; } }
            SR app = run(r10_start_cnt_ ? (long)r10_start_cnt_ : r10_total,
                         r11_start_cnt_ ? (long)r11_start_cnt_ : r11_total);
            s2s_total_r10_ = r10_total; s2s_total_r11_ = r11_total;
            s2s_opt_r10_ = min_r10; s2s_opt_r11_ = min_r11;
            std::fprintf(stderr,
                         "safe-to-start (cycle-accurate engine): R10 (z) optimal=%ld/%ld | R11 (y) optimal=%ld/%ld | "
                         "at app(r10=%u,r11=%u) stale_z=%ld stale_y=%ld%s\n",
                         min_r10, r10_total, min_r11, r11_total, r10_start_cnt_, r11_start_cnt_,
                         app.z, app.y, (app.z > 0 || app.y > 0) ? "  <-- UNSAFE (reads uncommitted data)" : "");
            // An app start_cnt releasing a consumer before its producer committed is a predicted
            // wrong-output fault (the gross iscore_out/SUC-y vsim failures) -> fail the run.
            if (app.z > 0 || app.y > 0) layout_pass_ = false;

            // Full hazard-vs-gate sweep for the plot's safe-to-start subplot, recorded once (first P2
            // tile; all tiles share the shape). R10 (z, osCore->SUC) always; R11 (y, SUC->isCore)
            // only when the isCore runs. ~100 points each, every point a fresh co-sim.
            if (s2s_sweep_r10_.empty()) {
                long step10 = std::max(1L, r10_total / 100);
                for (long r = 0; r <= r10_total; r += step10)
                    s2s_sweep_r10_.push_back({r, run(r, r11_total).z});   // z-gate: isCore held at total
                if (en_isCore) {
                    long step11 = std::max(1L, r11_total / 100);
                    for (long r = 0; r <= r11_total; r += step11)
                        s2s_sweep_r11_.push_back({r, run(min_r10, r).y}); // y-gate: SUC at its z-safe point
                }
            }
        }
    }

    // Timing-coupled BIST (SUC dt_BC delivery): the consumer must receive BC groups in AGU
    // order through the per-cycle fabric+FIFO; the delay-fault self-test proves a too-short
    // delay is caught (consume-before-land -> detected). See cyc.cpp / docs.
    if (ports_[7].enabled) {
        SucBistResult b = cyc_suc_bist(ports_[7], cfg_.seqLen, cfg_.dInner);
        bool bist_ok    = b.clean_complete && b.clean_poison_free && b.fault_detected;
        layout_pass_    = layout_pass_ && bist_ok;
        std::fprintf(stderr,
                     "BIST (SUC dt_BC delivery, %ld groups): complete=%s  delay-respect=%s "
                     "(%d/%ld un-landed reads)  too-short-delay caught=%s (%d un-landed reads). %s\n",
                     b.groups, b.clean_complete ? "PASS" : "FAIL", b.clean_poison_free ? "PASS" : "FAIL",
                     b.clean_poison_count, b.groups, b.fault_detected ? "YES" : "NO", b.fault_poison_count,
                     bist_ok ? "BIST PASS" : "BIST FAIL");
        if (b.n_idx) {
            std::fprintf(stderr, "  %s un-landed at BC-group indices:",
                         b.clean_poison_count ? "[BUG] clean run" : "[fault self-test]");
            for (int i = 0; i < b.n_idx; i++) std::fprintf(stderr, " %d", b.first_idx[i]);
            std::fprintf(stderr, "%s\n", b.fault_poison_count > b.n_idx ? " ..." : "");
        }
    }
}

// Step the single engine (MEMSIM_ENGINE) up to `t`. Driven by the interp before every SNAX read and
// by the scheduler's blocked-poll path. The engine consumes the DMAs enqueued live by dma_submit, so
// DMA<->compute contention (and its FIFO-slack hiding) emerges. On completion accel_end_/perf_ latch.
void SimWorld::advance_to(uint64_t t) {
    if (!engine_active_ || t <= accel_start_) return;
    uint64_t target_rel = t - accel_start_;
    while (engine_.active() && engine_.cyc() < target_rel) engine_.step();
    if (!engine_.active()) {
        engine_active_ = false;
        const auto& r = engine_.result();
        accel_end_ = accel_start_ + r.busy;
        // SIMD (no GEMM subcore enabled) costs wall-clock (accel_end_) but does NOT tick the MambaCore
        // perf counter, so perf_ stays 0 for a SIMD pass; GEMM/SUC modes report busy.
        bool is_simd = !(((cfg_.mode >> 19) & 1) || ((cfg_.mode >> 18) & 1) || ((cfg_.mode >> 17) & 1));
        perf_      = is_simd ? 0u : (uint32_t)r.busy;
        // Stage-activity bars for the timeline plot: the engine reports the REAL per-stage windows
        // (osc_end / suc_start..suc_end / isc_start..isc_end / sw_start..sw_end), relative to
        // accel_start_. `ideal` = conflict-free MAC-group count -> the plot's utilization line.
        if (trace_on_) {
            long M_i = cfg_.seqLen / 16, osN = (cfg_.dInner >= 24) ? cfg_.dInner / 24 : 0;
            bool en_os = (cfg_.mode >> 19) & 1, en_isc = (cfg_.mode >> 17) & 1;
            uint64_t osc_id = (uint64_t)M_i * osN * cfg_.dModel;
            uint64_t suc_id = (uint64_t)cfg_.seqLen * cfg_.dInner;
            uint64_t isc_id = (uint64_t)M_i * cfg_.dFinal * osN;
            if (phase2_) {  // osCore -> SUC -> isCore, real windows from the co-sim
                if (en_os) rec(TR_OSCORE, accel_start_, accel_start_ + r.osc_end, osc_id);
                rec(TR_SUC, accel_start_ + r.suc_start, accel_start_ + r.suc_end, suc_id);
                if (en_isc) rec(TR_ISCORE, accel_start_ + r.isc_start, accel_start_ + r.isc_end, isc_id);
                if (r.sw_end > r.sw_start) rec(TR_SWITCHCORE, accel_start_ + r.sw_start, accel_start_ + r.sw_end, suc_id);
            } else {  // P1 / IS_OSGEMM: osCore/conv/isCore co-active across the invocation window
                if (en_os) rec(TR_OSCORE, accel_start_, accel_end_, osc_id);
                if (((cfg_.mode >> 8) & 0x3) == 1) rec(TR_SWITCHCORE, accel_start_, accel_end_, suc_id / 4);
                if (en_isc) rec(TR_ISCORE, accel_start_, accel_end_, isc_id);
            }
            // TCDM streamer-bandwidth demand over the real invocation window (run_invocation defers this
            // for the engine path since accel_end_ was a sentinel there).
            uint64_t tcdm_words = 0;
            for (int p = 0; p < N_PORTS; p++) {
                if (!ports_[p].enabled) continue;
                uint64_t steps = 1;
                for (int d = 0; d < 4; d++) steps *= ports_[p].t_bound[d] ? (uint64_t)ports_[p].t_bound[d] : 1;
                tcdm_words += steps * PORT_NCH[p];
            }
            rec(TR_TCDM, accel_start_, accel_end_, tcdm_words);
        }
        // Splice this invocation's per-cycle FIFO trace into the run-wide trace at its absolute cycle.
        append_fifo(engine_.result().fifo, accel_start_);
        if (std::getenv("MEMSIM_ENGDBG")) {
            const auto& rr = engine_.result();
            std::fprintf(stderr, "  [ENGDONE] busy=%llu osc_end=%u suc(%u..%u) isc(%u..%u) | suc_span=%d isc_tail=%d\n",
                         (unsigned long long)rr.busy, rr.osc_end, rr.suc_start, rr.suc_end, rr.isc_start, rr.isc_end,
                         (int)rr.suc_end - (int)rr.suc_start, (int)rr.isc_end - (int)rr.suc_end);
        }
        if (Fabric::h_on) Fabric::hist_dump(phase2_ ? "P2-tile" : "engine");
    }
}

uint64_t SimWorld::next_event_cycle() const {
    uint64_t e = UINT64_MAX;
    // While the engine runs, step it forward a bounded amount so a blocked poll re-checks the live
    // gauge (and the peer hart can interleave its DMAs) rather than jumping straight to completion.
    if (engine_active_) return accel_start_ + engine_.cyc() + 1;
    if (accel_end_ > accel_start_) e = accel_end_;  // accelerator completion
    if (dma_busy_until_ && dma_busy_until_ < e) e = dma_busy_until_;
    return e;
}
