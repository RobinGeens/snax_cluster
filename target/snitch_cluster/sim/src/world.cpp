// Copyright 2026 KU Leuven. memsim — SimWorld implementation.
#include "world.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <unordered_set>
#include <vector>

#include "cyc.hpp"
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
            break;  // functional no-op (pacing only)
        case SNAX_DELAYED_START_R11:
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
    switch (csr) {
        case SNAX_STREAMER_BUSY:
            is_poll = true;
            return at < accel_end_ ? 1 : 0;
        case SNAX_SIMBACORE_BUSY:
            is_poll = true;
            return at < accel_end_ ? 1 : 0;
        case SNAX_STREAMER_PERF:
            return 0;
        case SNAX_SIMBACORE_PERF:
            return perf_;
        case SNAX_R10_GAUGE:
            is_poll = true;
            return gauge_at(at, accel_start_, t_oscore_done_, g_r10_total_);
        case SNAX_R11_GAUGE:
            is_poll = true;
            return gauge_at(at, t_oscore_done_, t_suc_done_, g_r11_total_);
        case SNAX_ISCORE_TILE_CNT:
            is_poll = true;
            // P2: isCore runs last (after SUC); P1: overlaps osCore from start.
            return gauge_at(at, phase2_ ? t_suc_done_ : accel_start_, accel_end_, g_iscore_total_);
        default:
            if (csr >= SNAX_STREAMER_CFG_LO && csr <= SNAX_STREAMER_CFG_HI) return raw_[csr - SNAX_STREAMER_CFG_LO];
            return 0;
    }
}

void SimWorld::run_invocation(uint64_t at) {
    decode_ports();
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
        // SIMD pass (SimdCore): no GEMM subcore runs; the core consumes one bank-group per
        // cycle, so the busy time = the gating streamer's temporal beat count, taken from the
        // captured AGU bounds (no GEMM/SUC formula applies; this is what the FFT uses).
        uint64_t beats = 0;
        for (int p = 0; p < N_PORTS; p++) {
            if (!ports_[p].enabled) continue;
            uint64_t b = 1;
            for (int d = 0; d < 4; d++) b *= ports_[p].t_bound[d] ? (uint64_t)ports_[p].t_bound[d] : 1;
            if (b > beats) beats = b;
        }
        t_oscore_done_ = t_suc_done_ = at;
        accel_end_                   = at + beats;
    } else if (en_suCore) {
        // Strict serialize (gauges at 100% — see docs). Per-stage pipeline fills:
        // osCore+261, SUC (per-cycle conflict via cyc_suc_duration), isCore+118.
        t_oscore_done_ = at + osc_dur_ + fill_osc_p2_;
        t_suc_done_    = t_oscore_done_ + suc_dur_;
        accel_end_     = t_suc_done_ + isc_dur_ + fill_isc_p2_;
        // ideal: osCore/isCore arrays are conflict-free (1 MAC-group/cyc) so ideal==dur (util~1);
        // the SUC's ideal is the conflict-free group count seqLen*dInner_tile, so its bank
        // conflict shows as util<1 (suc_dur_ carries the ~1.75x at bc_pad=0). Fill = handoff bubble.
        rec(TR_OSCORE, accel_start_, accel_start_ + osc_dur_, osc_dur_);
        rec(TR_SUC, t_oscore_done_, t_oscore_done_ + suc_dur_, (uint64_t)cfg_.seqLen * cfg_.dInner);
        rec(TR_ISCORE, t_suc_done_, t_suc_done_ + isc_dur_, isc_dur_);
    } else if (en_osCore && !en_isCore) {
        // OSGEMM: cycle-stepped osCore array (R0=A, R1=B) draining through W0. The busy-cycle
        // count and the post-compute output drain are produced by stepping the array + writer.
        // (R0/R1/W0 = the fixed osCore hardware ports; num_channel 2/4/1 from StreamParamGen.)
        Agu rd[2]  = {ports_[0], ports_[1]};
        int nch[2] = {2, 4}, nsp[2] = {1, 1};
        GemmResult g   = cyc_gemm(rd, nch, nsp, 2, ports_[14], 1, 1, (long)M_i * osN, (long)cfg_.dModel, 0);
        t_oscore_done_ = t_suc_done_ = at;
        accel_end_                   = at + g.end;
        // ideal = array MAC-group count; g.end adds the output drain -> util<1.
        rec(TR_OSCORE, at, accel_end_, (uint64_t)M_i * osN * cfg_.dModel);
    } else if (en_isCore && !en_osCore) {
        // ISGEMM: cycle-stepped isCore array draining through W3. GEMM input readers are
        // conflict-free (gran>=lanes), so the array is fed at 1/cycle; the W3 output drain is
        // produced by stepping. n_out_tiles = M_i*dFinal, K_i = dInner/24. (W3 = ports_[17], 4ch.)
        GemmResult g = cyc_gemm(nullptr, nullptr, nullptr, 0, ports_[17], 4, 1, (long)M_i * cfg_.dFinal, (long)K_i, 0);
        t_oscore_done_ = t_suc_done_ = at;
        accel_end_                   = at + g.end;
        rec(TR_ISCORE, at, accel_end_, (uint64_t)M_i * cfg_.dFinal * K_i);  // ideal = MACs; drain -> util<1
    } else {
        // both-core chained (PHASE1, IS_OSGEMM): osCore -> switchCore -> isCore stream as a
        // pipeline; globalState (the perf counter) spans until all three finish, so the busy is
        // the slowest stage + the pipeline fill. In PHASE1 the switchCore runs the depthwise
        // conv1d (m_switchCoreMode = Conv), which processes seqLen*dInner elements convUnroll
        // (=delaySU=4) at a time -> seqLen*dInner/4 cycles (SwitchCore.scala loop order). conv is
        // the bottleneck when dModel is small (osCore/isCore cheap); it must be in the max.
        int sw_mode       = (cfg_.mode >> 8) & 0x3;  // SimbaCoreCtrlBundle.m_switchCoreMode (1=Conv)
        uint64_t conv_dur = (sw_mode == 1) ? (uint64_t)cfg_.seqLen * cfg_.dInner / 4 : 0;
        uint64_t comp     = osc_dur_;
        if (conv_dur > comp) comp = conv_dur;
        if (isc_dur_ > comp) comp = isc_dur_;
        // PHASE1 (conv) fill scales with seqLen tiles; IS_OSGEMM (no conv) keeps its flat fill.
        uint64_t fill  = (sw_mode == 1) ? ((uint64_t)M_i * fill_p1_per_mtile_ + fill_p1_base_) : fill_is_osgemm_;
        t_oscore_done_ = at + osc_dur_;
        t_suc_done_    = t_oscore_done_;
        accel_end_     = at + comp + fill;
        // PHASE1/IS_OSGEMM: the stages pipeline, so they overlap from `at` for their own
        // durations (osCore, switchCore conv, isCore are separate rows). All three run at peak
        // here (conflict-free, no per-stage drain modeled), so ideal==dur (util~1); the pipeline
        // fill is shared lead-in/drain, not chargeable to one engine.
        rec(TR_OSCORE, at, at + osc_dur_, osc_dur_);
        if (sw_mode == 1) rec(TR_SWITCHCORE, at, at + conv_dur, conv_dur);
        rec(TR_ISCORE, at, at + isc_dur_, isc_dur_);
    }
    // The perf counter counts MambaCore busy (globalState != sIDLE, MambaCore.scala:159). The
    // SimdCore is outside the MambaCore, so a SIMD pass costs wall-clock (accel_end_) but does
    // not tick the counter -> perf_ = GEMM/SUC busy only, 0 for a SIMD-only invocation.
    perf_           = is_simd ? 0u : (uint32_t)(accel_end_ - accel_start_);
    g_r10_total_    = M_i * osN;
    g_r11_total_    = cfg_.seqLen * cfg_.dInner;
    g_iscore_total_ = M_i * cfg_.dFinal * K_i;

    // TCDM traffic for the bandwidth line: every enabled streamer port issues PORT_NCH narrow
    // (8B) accesses per temporal step, so its total accesses = (product of temporal bounds)*NCH.
    // Summed over all ports active this invocation and spread over [accel_start_, accel_end_], this
    // is the average TCDM word demand; plot_timeline.py divides by the 32-bank peak. (In P2 the
    // serialized stages are averaged together — a single invocation-average, not per-stage peaks.)
    if (trace_on_) {
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
    if (golden_z_) {
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

    // Safe-to-start sweep (R10 z-gate osCore->SUC, R11 y-gate SUC->isCore): a per-element,
    // reorder-aware commit-vs-read schedule. Releasing a reader at the upstream gauge=start_cnt
    // overlaps the producer; each output element commits at commit_cyc and is read at
    // reader_start+read_off. read < commit -> a stale read (the consumer pulls an element the
    // producer hasn't written = the subtle in-buffer corruption). The min start_cnt with zero
    // stale is the optimum; it is computed by stepping the schedule (tile-granular commit +
    // the small commit-pipe). See docs/dataflow/10_memsim.md.
    if (phase2_ && osc_dur_ && suc_dur_ && isc_dur_) {
        const long Mu = 16, Nu = 24, delaySU = 4;
        const long M = cfg_.seqLen, N = cfg_.dInner, Ne = M * N, M_i = M / Mu, osN = N / Nu;
        const long Ki = (M_i * osN != 0) ? (long)osc_dur_ / (M_i * osN) : (long)osc_dur_;
        // R10 z-gate: producer osCore (N_M_K tile commit), consumer SUC (SUCFormat read).
        auto z_stale = [&](long sc) {
            double rstart = (double)sc / (M_i * osN) * osc_dur_, rrate = (double)suc_dur_ / Ne;
            long stale = 0;
            for (long m = 0; m < M; m++)
                for (long n = 0; n < N; n++) {
                    long tile     = (n / Nu) * M_i + (m / Mu);  // N_M_K produce order
                    double commit = (double)(tile + 1) * Ki + s2s_lat_z_;
                    long beat     = (n / delaySU) * (M * delaySU) + m * delaySU + (n % delaySU);  // SUCFormat read
                    double read   = rstart + (double)beat * rrate;
                    if (read < commit) stale++;
                }
            return stale;
        };
        // R11 y-gate: producer SUC (SUCFormat ~1/cyc), consumer isCore (rate-difference dominated).
        auto y_stale = [&](long sc) {
            double rstart = (double)sc / Ne * suc_dur_, prate = (double)suc_dur_ / Ne, crate = (double)isc_dur_ / Ne;
            long stale = 0;
            for (long e = 0; e < Ne; e++) {
                double commit = (double)(e + 1) * prate + s2s_lat_y_;
                double read   = rstart + (double)e * crate;
                if (read < commit) stale++;
            }
            return stale;
        };
        // Smallest start_cnt with zero stale reads = the optimum. The stale count is monotone
        // non-increasing in start_cnt (releasing later only moves reads later), so binary-search
        // it; report the optimum and the stale count one tick earlier (the boundary is tight).
        auto run = [&](const char* lbl, long total, auto&& fn) -> long {
            long lo = 1, hi = total, opt = total;
            while (lo <= hi) {
                long mid = lo + (hi - lo) / 2;
                if (fn(mid) == 0) {
                    opt = mid;
                    hi  = mid - 1;
                } else
                    lo = mid + 1;
            }
            long below = opt > 1 ? opt - 1 : opt, s_below = opt > 1 ? fn(below) : 0;
            std::fprintf(stderr,
                         "safe-to-start %s: optimal start_cnt=%ld/%ld (start_cnt=%ld has %ld stale element(s))\n", lbl,
                         opt, total, below, s_below);
            return opt;
        };
        s2s_total_r10_ = M_i * osN;
        s2s_total_r11_ = Ne;
        s2s_opt_r10_   = run("R10 (z, osCore->SUC)", s2s_total_r10_, z_stale);
        s2s_opt_r11_   = run("R11 (y, SUC->isCore)", s2s_total_r11_, y_stale);
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

uint64_t SimWorld::next_event_cycle() const {
    uint64_t e = UINT64_MAX;
    if (accel_end_ > accel_start_) e = accel_end_;  // accelerator completion
    if (dma_busy_until_ && dma_busy_until_ < e) e = dma_busy_until_;
    return e;
}
