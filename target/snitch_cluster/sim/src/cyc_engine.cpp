// Copyright 2026 KU Leuven. memsim — the single cycle-exact accelerator engine (see cyc_engine.hpp).
#include "cyc_engine.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

// Compact arbiter token-ids (0..8): the Fabric packs (id<<4)|lane in 8 bits, so real port numbers
// 16/17 (W2/W3) would overflow and skew fairness — use small ids.
enum { T_R0 = 0, T_R1 = 1, T_W0 = 2, T_R7 = 3, T_R10 = 4, T_W2 = 5, T_R11 = 6, T_R12 = 7, T_W3 = 8,
       T_R2 = 9, T_R3 = 10, T_R5 = 11, T_R13 = 12 };
enum { P_R0 = 0, P_R1 = 1, P_R2 = 2, P_R3 = 3, P_R4 = 4, P_R5 = 5, P_R7 = 7, P_R10 = 10, P_R11 = 11,
       P_R12 = 12, P_R13 = 13, P_W0 = 14, P_W1 = 15, P_W2 = 16, P_W3 = 17 };

static long word_off(const int32_t ts[4], const int ti[4]) {
    int64_t off = (int64_t)ti[0] * ts[0] + (int64_t)ti[1] * ts[1] + (int64_t)ti[2] * ts[2] + (int64_t)ti[3] * ts[3];
    return (long)(off / 8);
}

// Snapshot the end-of-cycle FIFO fullness of every port active in THIS invocation (readers R0..R13 =
// 0..13, writers W0..W3 = 14..17). -1 marks a port absent from this invocation so the plot leaves a
// gap there. Only the ports actually wired into step()/step_p1() are filled — matching their gating —
// so a streamer left configured from a prior tile doesn't leak a stale line.
void AccelEngine::sample_fifo() {
    if (!rec_fifo) return;
    std::array<int16_t, 18> row;
    row.fill(-1);
    auto setR = [&](int port, const CycReader& r) { row[port] = (int16_t)r.fifo_occ; };
    // Writer fifo_occ is now bounded by the physical depth (the producer back-pressures when full), so
    // the raw value is the true number of elements in the FIFO; pinned at depth = output-bound.
    auto setW = [&](int port, const CycWriter& w) { row[port] = (int16_t)w.fifo_occ; };
    if (kind_ == 1) {  // P2: osCore -> SUC -> isCore
        if (en_os_) { setR(P_R0, r0_); setR(P_R1, r1_); setW(P_W0, w0_); }
        // R7 dt_BC is hand-stepped (per-lane), not a CycReader: its data-FIFO holds min(landed)-consumed.
        if (have_bc_ || consumed7_ > 0 || landed_[0] > 0) {
            long m = landed_[0];
            for (int l = 1; l < NCH7; l++) if (landed_[l] < m) m = landed_[l];
            row[P_R7] = (int16_t)(m - consumed7_);
        }
        setR(P_R10, r10_); setW(P_W2, w2_);
        if (sw_en_) { setR(P_R2, r2sw_); setR(P_R3, r3sw_); if (p_[P_R5].enabled) setR(P_R5, r5sw_); }
        if (en_isc_) { setR(P_R11, r11_); setR(P_R12, r12_); if (r13_en_) setR(P_R13, r13_); setW(P_W3, w3_); }
    } else if (kind_ == 3) {  // osgemm: standalone osCore array
        setR(P_R0, r0_); setR(P_R1, r1_); setW(P_W0, w0_);
    } else if (kind_ == 4) {  // isgemm: standalone isCore array (output drain only)
        setW(P_W3, w3_);
    } else if (kind_ == 5) {  // SIMD: any enabled reader/writer ports
        for (int p = 0; p < 14; p++) if (simd_r_en_[p]) setR(p, rd_[p]);
        for (int p = 14; p < 18; p++) if (simd_w_en_[p - 14]) setW(p, wr_[p - 14]);
    } else if (kind_ == 2) {  // P1 / IS_OSGEMM: osCore -> conv -> isCore
        setR(P_R0, r0_); setR(P_R1, r1_);
        if (w0p1_en_) setW(P_W0, w0_);
        if (w1_en_) setW(P_W1, w1_);
        if (p1sw_en_) { setR(P_R3, r3c_); setR(P_R4, r4c_); }
        setR(P_R12, r12_);
        if (r11p1_en_) setR(P_R11, r11_);
        if (r13_en_) setR(P_R13, r13_);
        setW(P_W3, w3_);
    }
    res_.fifo.push_back(row);
}

int AccelEngine::r7bank(long st, int lane) const {
    long t0 = st % r7eb_[0], rr = st / r7eb_[0], t1 = rr % r7eb_[1];
    rr /= r7eb_[1];
    long t2 = rr % r7eb_[2];
    rr /= r7eb_[2];
    long t3 = rr % r7eb_[3];
    int64_t toff = t0 * r7ts_[0] + t1 * r7ts_[1] + t2 * r7ts_[2] + t3 * r7ts_[3];
    int i = lane % 2, j = lane / 2;
    int64_t soff = (int64_t)i * p_[P_R7].s_stride[0] + (int64_t)j * p_[P_R7].s_stride[1];
    return (int)(((uint32_t)(p_[P_R7].base + toff + soff) >> 3) & 31);
}

void AccelEngine::configure(const Agu* ports, const SimbacoreCfg& cfg, long r10_start_cnt, long r11_start_cnt,
                            long sw_cyc) {
    for (int i = 0; i < NPORT; i++) p_[i] = ports[i];
    seqLen_ = cfg.seqLen; dInner_ = cfg.dInner; dModel_ = cfg.dModel; dFinal_ = cfg.dFinal;
    en_os_  = (cfg.mode >> 19) & 1;
    en_suc_ = (cfg.mode >> 18) & 1;
    en_isc_ = (cfg.mode >> 17) & 1;
    sw_mode_ = (cfg.mode >> 8) & 0x3;
    r10_cnt_ = r10_start_cnt; r11_cnt_ = r11_start_cnt;
    if (std::getenv("MEMSIM_ENGDBG") && ((cfg.mode >> 18) & 1)) {
        std::fprintf(stderr, "  [RELCNT] r10_cnt_=%ld r11_cnt_=%ld (app safe_to_start)\n", r10_cnt_, r11_cnt_);
        auto bk = [](uint64_t a){ return (int)((a>>3)&31); };
        const Agu& w = ports[P_R12];
        std::fprintf(stderr, "  [R12BANK] base=%08x s_stride0=%d  lane banks=%d,%d,%d,%d  (gran-4 RTL wants 4 distinct residues)\n",
                     (uint32_t)w.base, (int)w.s_stride[0],
                     bk(w.base+0*w.s_stride[0]), bk(w.base+1*w.s_stride[0]), bk(w.base+2*w.s_stride[0]), bk(w.base+3*w.s_stride[0]));
    }

    const long Mu = 16, Nu = 24;
    M_i_ = seqLen_ / Mu; osN_ = (dInner_ >= Nu) ? dInner_ / Nu : 0;
    running_ = false; kind_ = 0; is_simd_ = false;
    if (!en_os_ && !en_suc_ && !en_isc_) { configure_simd(); return; }  // SIMD: no GEMM subcore (kind_=5)
    if (M_i_ <= 0 || osN_ <= 0) return;
    if (!en_suc_) {                                  // P1 / IS_OSGEMM (osCore -> conv -> isCore)
        if (en_os_ && en_isc_) configure_p1();
        else if (en_os_) configure_os();             // osgemm: standalone osCore (kind_=3)
        else if (en_isc_) configure_is();            // isgemm: standalone isCore (kind_=4)
        return;
    }
    kind_ = 1;                                       // P2 (osCore -> SUC -> isCore)

    K_os_ = dModel_; K_is_ = osN_; iters_ = (long)seqLen_ * dInner_;
    n_tiles_os_ = M_i_ * osN_; n_tiles_is_ = M_i_ * dFinal_;

    r0_.configure(p_[P_R0], 2, 1, snax_streamer_depth(P_R0), T_R0);
    r1_.configure(p_[P_R1], 4, 1, snax_streamer_depth(P_R1), T_R1);
    w0_.configure(p_[P_W0], 1, 1, T_W0); w0_.depth = snax_streamer_depth(P_W0);
    wbpt_os_ = w0_.total_beats() / std::max(1L, n_tiles_os_); if (wbpt_os_ < 1) wbpt_os_ = 1;
    // osCore disabled (SUC-only): z is preloaded, so every z word is already "written" (no hazard).
    Wz_ = w0_.total_beats(); zwr_.assign(Wz_ + 1, en_os_ ? 0 : 1);

    r7_fifo_d_ = snax_streamer_depth(P_R7);  // R7 addr FIFO (outstanding) = fifo_depth[7]
    // RTL read-ahead = responser+dataBuffer = 2x, but modelling it as a FREE 8 over-hides the dt_BC
    // 2-bank conflict (R7 pre-fills during the long osc_lead) AND triggers a big osCore-speedup cross-
    // coupling artifact (-108k/P2). The faithful fix is read-ahead 8 BOUNDED by switchCore dt_BC
    // production (R7 can't read what switchCore hasn't written) — until that's modelled, use 4 (closer:
    // suc_span +6% & isCore tail preserved, vs free-8's -8% & broken tail).
    r7_data_d_ = r7_fifo_d_;                 // TODO: = 2*r7_fifo_d_ once R7 reads are gated on sw_dt_
    for (int i = 0; i < 4; i++) { r7ts_[i] = p_[P_R7].t_stride[i]; r7eb_[i] = p_[P_R7].t_bound[i] ? p_[P_R7].t_bound[i] : 1; }
    if (p_[P_R7].t_stride[0] == 0 && p_[P_R7].t_bound[0] > 1) r7eb_[0] = 1;
    for (int l = 0; l < NCH7; l++) { issued_[l] = landed_[l] = 0; pend7_[l] = 0; }
    agu7_ = consumed7_ = 0; k7_ = 0; have_bc_ = false;
    r10_.configure(p_[P_R10], 1, 1, snax_streamer_depth(P_R10), T_R10);
    w2_.configure(p_[P_W2], 1, 1, T_W2); w2_.depth = snax_streamer_depth(P_W2);
    out_su_ = y_in_word_ = 0; Wy_ = w2_.total_beats(); ywr_.assign(Wy_ + 1, 0);

    sw_cyc_ = sw_cyc;  // computed by the caller (has dtRankUnroll); matches run_invocation's sw_cyc
    sw_en_ = (sw_cyc_ > 0) && p_[P_R2].enabled;
    if (sw_en_) {
        r2sw_.configure(p_[P_R2], 1, 1, snax_streamer_depth(P_R2), T_R2);
        r3sw_.configure(p_[P_R3], 1, 1, snax_streamer_depth(P_R3), T_R3);
        if (p_[P_R5].enabled) r5sw_.configure(p_[P_R5], 1, 1, snax_streamer_depth(P_R5), T_R5);
    }
    sw_dt_ = sw_acc_ = sw_first_ = sw_last_ = 0;

    r11_.configure(p_[P_R11], 1, 1, snax_streamer_depth(P_R11), T_R11);
    r12_.configure(p_[P_R12], 4, 1, snax_streamer_depth(P_R12), T_R12);
    r13_en_ = p_[P_R13].enabled;  // isCore psum read-back — real TCDM contention
    if (r13_en_) r13_.configure(p_[P_R13], 4, 1, snax_streamer_depth(P_R13), T_R13);
    w3_.configure(p_[P_W3], 4, 1, T_W3); w3_.depth = snax_streamer_depth(P_W3);
    wbpt_is_ = w3_.total_beats() / std::max(1L, n_tiles_is_); if (wbpt_is_ < 1) wbpt_is_ = 1;
    as_os_ = tiles_os_ = as_is_ = tiles_is_ = 0; os_owed_ = is_owed_ = 0;
    serDesA_ = (long)Mu * 24 / 8; isc_aload_ = serDesA_; isc_since_a_ = 0;  // isCore A=y s2p fill (48 beats)

    if (std::getenv("MEMSIM_ENGDBG")) {
        auto sb = [](uint64_t b) { return (int)((b >> 6) & 3); };
        std::fprintf(stderr, "  [LAYOUT] R0(osc_in)=%08x sb%d  R1(w)=%08x sb%d  R7(dtBC)=%08x sb%d  "
                     "R10(z)=%08x sb%d  W0(z)=%08x sb%d  W2(y)=%08x sb%d\n",
                     (uint32_t)p_[P_R0].base, sb(p_[P_R0].base), (uint32_t)p_[P_R1].base, sb(p_[P_R1].base),
                     (uint32_t)p_[P_R7].base, sb(p_[P_R7].base), (uint32_t)p_[P_R10].base, sb(p_[P_R10].base),
                     (uint32_t)p_[P_W0].base, sb(p_[P_W0].base), (uint32_t)p_[P_W2].base, sb(p_[P_W2].base));
        std::fprintf(stderr, "  [LAYOUT] R11(y)=%08x  R12(isW)=%08x  R13(psum)=%08x  W3(out)=%08x  "
                     "R2(sw)=%08x  R3(sw)=%08x\n",
                     (uint32_t)p_[P_R11].base, (uint32_t)p_[P_R12].base, (uint32_t)p_[P_R13].base,
                     (uint32_t)p_[P_W3].base, (uint32_t)p_[P_R2].base, (uint32_t)p_[P_R3].base);
        std::fprintf(stderr, "  [ISCNT] M_i=%ld osN=%ld K_is=%ld dFinal=%u n_tiles_is=%ld | "
                     "R11 groups=%d R12 groups=%d R13 groups=%d W3 beats=%ld | as_is_total=%ld\n",
                     M_i_, osN_, K_is_, dFinal_, n_tiles_is_, r11_.total_read_groups(), r12_.total_read_groups(),
                     r13_en_ ? r13_.total_read_groups() : 0, w3_.total_beats(), n_tiles_is_ * K_is_);
    }
    cap_ = (uint64_t)(n_tiles_os_ * K_os_ + iters_ + n_tiles_is_ * K_is_) * 8 + 500000;
    suc_started_ = isc_started_ = false;
    cyc_ = 0; g_r10_ = g_r11_ = g_iscore_ = 0;
    // Reset all per-invocation state so a new tile starts clean (else the previous tile's release
    // latches / result / DMA queue leak in -> the consumer fires from cycle 0 and reads garbage).
    rel_r10_ = rel_r11_ = false;
    res_ = EngineResult{};
    dma_.clear();
    fabric_ = Fabric{};
    running_ = true;
}

// P1 / IS_OSGEMM: osCore (R0,R1 -> array -> W1 + on-chip stream) -> switchCore conv (R3,R4) ->
// isCore (R12,R13 -> array -> W3). The conv/isCore stream ON-CHIP; only the input/output buffers
// touch TCDM. All stages step on the one shared fabric, so the pipeline fill/drain + DMA contention
// emerge. Ported from cyc_phase1.
void AccelEngine::configure_p1() {
    kind_ = 2; has_conv_ = (sw_mode_ == 1);
    const long Mu = 16, Nu = 24;
    K_os_ = dModel_; K_is_ = osN_; iters_ = (long)seqLen_ * dInner_;  // iters_ = P (on-chip elements)
    n_tiles_os_ = M_i_ * osN_; n_tiles_is_ = M_i_ * dFinal_;
    if (dFinal_ <= 0) { running_ = false; return; }
    r0_.configure(p_[P_R0], 2, 1, snax_streamer_depth(P_R0), T_R0);
    r1_.configure(p_[P_R1], 4, 1, snax_streamer_depth(P_R1), T_R1);
    // osCore output writer: W0 when the osCore spills z to TCDM (IS_OSGEMM), else W1 (on-chip-style).
    w0p1_en_ = p_[P_W0].enabled;
    w1_en_ = p_[P_W1].enabled;
    if (w0p1_en_) { w0_.configure(p_[P_W0], 1, 1, T_W0); w0_.depth = snax_streamer_depth(P_W0); }
    if (w1_en_) { w1_.configure(p_[P_W1], 1, 1, T_W0); w1_.depth = snax_streamer_depth(P_W1); }
    long osw_beats = w0p1_en_ ? w0_.total_beats() : (w1_en_ ? w1_.total_beats() : 0);
    wbpt_os_ = (w0p1_en_ || w1_en_) ? osw_beats / std::max(1L, n_tiles_os_) : 0;
    p1sw_en_ = has_conv_ && p_[P_R3].enabled;
    if (p1sw_en_) { r3c_.configure(p_[P_R3], 1, 1, snax_streamer_depth(P_R3), T_R3);
                    r4c_.configure(p_[P_R4], 1, 1, snax_streamer_depth(P_R4), T_R5); }
    r12_.configure(p_[P_R12], 4, 1, snax_streamer_depth(P_R12), T_R12);
    r11p1_en_ = p_[P_R11].enabled;  // IS_OSGEMM isCore A
    if (r11p1_en_) r11_.configure(p_[P_R11], 1, 1, snax_streamer_depth(P_R11), T_R11);
    r13_en_ = p_[P_R13].enabled;
    // isCore psum read-back: a 4-lane reader (Mu=16 BF16 = 32 B = 4 banks), same as the P2 path and the
    // RTL (vsim TCDM ports 23-26). Was nch=1 -> under-modelled the dominant psum-RMW contention ~4x.
    if (r13_en_) r13_.configure(p_[P_R13], 4, 1, snax_streamer_depth(P_R13), T_R13);
    w3_.configure(p_[P_W3], 4, 1, T_W3);
    w3_.depth = snax_streamer_depth(P_W3);
    wbpt_is_ = w3_.total_beats() / std::max(1L, n_tiles_is_); if (wbpt_is_ < 1) wbpt_is_ = 1;
    as_os_ = tiles_os_ = as_is_ = tiles_is_ = 0; os_owed_ = is_owed_ = 0; osc_elem_ = sw_elem_ = 0;
    cap_ = (uint64_t)(n_tiles_os_ * K_os_ + iters_ + n_tiles_is_ * K_is_) * 8 + 500000;
    cyc_ = 0; g_r10_ = g_r11_ = g_iscore_ = 0;
    rel_r10_ = rel_r11_ = false; res_ = EngineResult{}; dma_.clear(); fabric_ = Fabric{};
    running_ = true;
}

bool AccelEngine::step_p1() {
    if (!running_) return false;
    if (cyc_ >= cap_) { running_ = false; res_.busy = cyc_; return false; }
    cyc_++;
    const long Mu = 16, Nu = 24, convUnroll = 4;
    r0_.land_reads(); if (r0_.done) r0_.reset();
    r1_.land_reads(); if (r1_.done) r1_.reset();
    if (p1sw_en_) { r3c_.land_reads(); if (r3c_.done) r3c_.reset(); r4c_.land_reads(); if (r4c_.done) r4c_.reset(); }
    r12_.land_reads(); if (r12_.done) r12_.reset();
    if (r11p1_en_) { r11_.land_reads(); if (r11_.done) r11_.reset(); }
    if (r13_en_) { r13_.land_reads(); if (r13_.done) r13_.reset(); }

    // osCore output back-pressure on its active TCDM writer (W0 spill or W1). With no writer the osCore
    // output streams purely on-chip (paced by osc_elem_), so there's no FIFO to bound. Drip 1/cycle and
    // hold a tile until its predecessor has drained, exactly as in P2 -> the writer FIFO stays <= depth.
    CycWriter* osw = w0p1_en_ ? &w0_ : (w1_en_ ? &w1_ : nullptr);
    if (osw && os_owed_ > 0 && osw->fifo_occ < osw->depth) { osw->push(1); os_owed_--; }
    if (tiles_os_ < n_tiles_os_ && r0_.fifo_occ > 0 && r1_.fifo_occ > 0
        && (!osw || (as_os_ + 1) % K_os_ != 0 || os_owed_ == 0)) {
        r0_.pop(); r1_.pop();
        if (++as_os_ % K_os_ == 0) {
            tiles_os_++; osc_elem_ += Mu * Nu; g_r10_++;
            if (osw) os_owed_ += wbpt_os_;
        }
    }
    if (p1sw_en_) {
        if (sw_elem_ < osc_elem_ && sw_elem_ < iters_ && r3c_.fifo_occ > 0 && r4c_.fifo_occ > 0) {
            r3c_.pop();
            if (sw_elem_ % Nu == 0 && r4c_.fifo_occ > 0) r4c_.pop();
            sw_elem_ += convUnroll;
        }
    } else sw_elem_ = osc_elem_;
    // isCore output W3: drip 1/cycle, hold a tile until its predecessor has drained (Array.scala:200
    // output-FIFO back-pressure) -> a full W3 stalls the isCore and W3 stays <= depth.
    if (is_owed_ > 0 && w3_.fifo_occ < w3_.depth) { w3_.push(1); is_owed_--; }
    bool stream_ahead = (long double)as_is_ * iters_ <= (long double)sw_elem_ * n_tiles_is_ * K_is_;
    if (tiles_is_ < n_tiles_is_ && stream_ahead && r12_.fifo_occ > 0 && (!r13_en_ || r13_.fifo_occ > 0)
        && ((as_is_ + 1) % K_is_ != 0 || is_owed_ == 0)) {
        r12_.pop();
        if (r11p1_en_ && r11_.fifo_occ > 0) r11_.pop();  // isCore A (paced to array, never gates)
        if (r13_en_) r13_.pop();
        if (++as_is_ % K_is_ == 0) { tiles_is_++; is_owed_ += wbpt_is_; g_iscore_++; }
    }
    if (tiles_is_ == n_tiles_is_ && w3_.written >= w3_.total_beats()) { res_.busy = cyc_; running_ = false; return false; }

    fabric_.begin_cycle(dma_.step(cyc_));
    r0_.propose(fabric_); r1_.propose(fabric_);
    if (w0p1_en_) w0_.propose(fabric_); else if (w1_en_) w1_.propose(fabric_);
    if (p1sw_en_) { r3c_.propose(fabric_); r4c_.propose(fabric_); }
    r12_.propose(fabric_); if (r11p1_en_) r11_.propose(fabric_); if (r13_en_) r13_.propose(fabric_); w3_.propose(fabric_);
    bool g[64];
    fabric_.arbitrate(g);
    int gi = 0;
    r0_.commit(fabric_, g, gi); r1_.commit(fabric_, g, gi);
    if (w0p1_en_) w0_.commit(fabric_, g, gi); else if (w1_en_) w1_.commit(fabric_, g, gi);
    if (p1sw_en_) { r3c_.commit(fabric_, g, gi); r4c_.commit(fabric_, g, gi); }
    r12_.commit(fabric_, g, gi); if (r11p1_en_) r11_.commit(fabric_, g, gi); if (r13_en_) r13_.commit(fabric_, g, gi); w3_.commit(fabric_, g, gi);
    sample_fifo();
    return running_;
}

// kind_=3: standalone osCore GEMM (osgemm). R0,R1 -> array(K=dModel) -> W0, stepped on the one shared
// fabric with the live DMA beat engine, so the async oscore-in ring's refill DMA contends by arbitration
// (replaces cyc_gemm + its scalar dma_cycles). GEMM readers are residue-pinned (conflict-free), so the
// only cross effect is DMA superbank preemption.
void AccelEngine::configure_os() {
    kind_ = 3;
    K_os_ = dModel_;
    n_tiles_os_ = M_i_ * osN_;
    r0_.configure(p_[P_R0], 2, 1, snax_streamer_depth(P_R0), T_R0);
    r1_.configure(p_[P_R1], 4, 1, snax_streamer_depth(P_R1), T_R1);
    w0_.configure(p_[P_W0], 1, 1, T_W0); w0_.depth = snax_streamer_depth(P_W0);
    Wz_ = w0_.total_beats();
    wbpt_os_ = Wz_ / std::max(1L, n_tiles_os_); if (wbpt_os_ < 1) wbpt_os_ = 1;
    as_os_ = tiles_os_ = 0; os_owed_ = 0;
    cap_ = (uint64_t)n_tiles_os_ * K_os_ * 8 + 500000;
    cyc_ = 0; g_r10_ = g_r11_ = g_iscore_ = 0;
    suc_started_ = isc_started_ = false; rel_r10_ = rel_r11_ = false;
    res_ = EngineResult{}; dma_.clear(); fabric_ = Fabric{};
    running_ = true;
}

bool AccelEngine::step_os() {
    if (!running_) return false;
    if (cyc_ >= cap_) { running_ = false; res_.busy = cyc_; return false; }
    cyc_++;
    r0_.land_reads(); if (r0_.done) r0_.reset();
    r1_.land_reads(); if (r1_.done) r1_.reset();
    // Drip one output group per cycle into W0's FIFO (backpressure: a full W0 stalls the array).
    if (os_owed_ > 0 && w0_.fifo_occ < w0_.depth) { w0_.push(1); os_owed_--; }
    if (tiles_os_ < n_tiles_os_ && r0_.fifo_occ > 0 && r1_.fifo_occ > 0
        && ((as_os_ + 1) % K_os_ != 0 || os_owed_ == 0)) {
        r0_.pop(); r1_.pop();
        if (++as_os_ % K_os_ == 0) { tiles_os_++; os_owed_ += wbpt_os_; g_r10_++;
            if (rec_fires) res_.r10_fire.push_back((uint32_t)cyc_); }
    }
    if (tiles_os_ == n_tiles_os_ && w0_.written >= Wz_) {
        res_.busy = cyc_; res_.osc_end = cyc_; running_ = false; sample_fifo(); return false;
    }
    fabric_.begin_cycle(dma_.step(cyc_));
    r0_.propose(fabric_); r1_.propose(fabric_); w0_.propose(fabric_);
    bool g[64]; fabric_.arbitrate(g);
    int gi = 0;
    r0_.commit(fabric_, g, gi); r1_.commit(fabric_, g, gi); w0_.commit(fabric_, g, gi);
    sample_fifo();
    return running_;
}

// kind_=4: standalone isCore GEMM (isgemm). Input readers are conflict-free (gran>=lanes), so the array
// is fed at 1/cycle; only the W3 output drain (and any concurrent psum-ring DMA on W3's superbank) is
// stepped on the shared fabric. n_out_tiles = M_i*dFinal, K_i = osN. (Mirrors cyc_gemm's 0-reader path.)
void AccelEngine::configure_is() {
    kind_ = 4;
    K_is_ = osN_;
    n_tiles_is_ = M_i_ * dFinal_;
    w3_.configure(p_[P_W3], 4, 1, T_W3); w3_.depth = snax_streamer_depth(P_W3);
    wbpt_is_ = w3_.total_beats() / std::max(1L, n_tiles_is_); if (wbpt_is_ < 1) wbpt_is_ = 1;
    as_is_ = tiles_is_ = 0; is_owed_ = 0;
    cap_ = (uint64_t)n_tiles_is_ * std::max(1L, K_is_) * 8 + 500000;
    cyc_ = 0; g_r10_ = g_r11_ = g_iscore_ = 0;
    suc_started_ = isc_started_ = false; rel_r10_ = rel_r11_ = false;
    res_ = EngineResult{}; dma_.clear(); fabric_ = Fabric{};
    running_ = (n_tiles_is_ > 0 && K_is_ > 0);
}

bool AccelEngine::step_is() {
    if (!running_) return false;
    if (cyc_ >= cap_) { running_ = false; res_.busy = cyc_; return false; }
    cyc_++;
    if (is_owed_ > 0 && w3_.fifo_occ < w3_.depth) { w3_.push(1); is_owed_--; }
    if (tiles_is_ < n_tiles_is_ && ((as_is_ + 1) % K_is_ != 0 || is_owed_ == 0)) {
        if (++as_is_ % K_is_ == 0) { tiles_is_++; is_owed_ += wbpt_is_; g_iscore_++; }
    }
    if (tiles_is_ == n_tiles_is_ && w3_.written >= w3_.total_beats()) {
        res_.busy = cyc_; res_.isc_end = cyc_; running_ = false; sample_fifo(); return false;
    }
    fabric_.begin_cycle(dma_.step(cyc_));
    w3_.propose(fabric_);
    bool g[64]; fabric_.arbitrate(g);
    int gi = 0; w3_.commit(fabric_, g, gi);
    sample_fifo();
    return running_;
}

// kind_=5: SIMD pass (SimdCore). The enabled reader ports feed the core and the enabled writer ports
// drain it, all on the one shared fabric with the live DMA beat engine so strided-gather bank conflicts
// AND any concurrent ring DMA emerge (replaces cyc_simd + its scalar dma_cycles). Readers/writers run to
// their own beat totals; the pass ends when all have completed. perf counter stays 0 (SIMD is outside the
// MambaCore counter) — set by advance_to.
static const int PORT_NCH_SIMD[18] = {2, 4, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 4, 4, 1, 1, 1, 4};
void AccelEngine::configure_simd() {
    kind_ = 5; is_simd_ = true;
    long maxbeats = 0; bool any = false;
    for (int p = 0; p < 14; p++) {
        simd_r_en_[p] = false; simd_r_beats_[p] = simd_r_pops_[p] = 0;
        if (!p_[p].enabled) continue;
        rd_[p].configure(p_[p], PORT_NCH_SIMD[p], 1, snax_streamer_depth(p), p);
        long b = 1; for (int d = 0; d < 4; d++) b *= p_[p].t_bound[d] ? p_[p].t_bound[d] : 1;
        simd_r_en_[p] = true; simd_r_beats_[p] = b; if (b > maxbeats) maxbeats = b; any = true;
    }
    for (int p = 14; p < 18; p++) {
        simd_w_en_[p - 14] = false;
        if (!p_[p].enabled) continue;
        wr_[p - 14].configure(p_[p], PORT_NCH_SIMD[p], 1, p); wr_[p - 14].depth = snax_streamer_depth(p);
        simd_w_en_[p - 14] = true;
        long b = wr_[p - 14].total_beats(); if (b > maxbeats) maxbeats = b; any = true;
    }
    cap_ = (uint64_t)maxbeats * 8 + 500000;
    cyc_ = 0; g_r10_ = g_r11_ = g_iscore_ = 0;
    res_ = EngineResult{}; dma_.clear(); fabric_ = Fabric{};
    running_ = any;
}

bool AccelEngine::step_simd() {
    if (!running_) return false;
    if (cyc_ >= cap_) { running_ = false; res_.busy = cyc_; return false; }
    cyc_++;
    for (int p = 0; p < 14; p++) {                                   // core consumes one beat per reader
        if (!simd_r_en_[p]) continue;
        rd_[p].land_reads();
        if (simd_r_pops_[p] < simd_r_beats_[p] && rd_[p].fifo_occ > 0) { rd_[p].pop(); simd_r_pops_[p]++; }
    }
    for (int p = 14; p < 18; p++) {                                  // core produces into each writer FIFO
        if (!simd_w_en_[p - 14]) continue;
        CycWriter& wp = wr_[p - 14];
        if (wp.written + wp.fifo_occ < wp.total_beats() && wp.fifo_occ < wp.depth) wp.push(1);
    }
    sample_fifo();
    bool done = true;
    for (int p = 0; p < 14; p++) if (simd_r_en_[p] && simd_r_pops_[p] < simd_r_beats_[p]) done = false;
    for (int p = 14; p < 18; p++) if (simd_w_en_[p - 14] && wr_[p - 14].written < wr_[p - 14].total_beats()) done = false;
    if (done) { res_.busy = cyc_; running_ = false; return false; }
    fabric_.begin_cycle(dma_.step(cyc_));
    for (int p = 0; p < 14; p++) if (simd_r_en_[p] && simd_r_pops_[p] < simd_r_beats_[p]) rd_[p].propose(fabric_);
    for (int p = 14; p < 18; p++) if (simd_w_en_[p - 14] && wr_[p - 14].written < wr_[p - 14].total_beats()) wr_[p - 14].propose(fabric_);
    bool g[64]; fabric_.arbitrate(g);
    int gi = 0;
    for (int p = 0; p < 14; p++) if (simd_r_en_[p] && simd_r_pops_[p] < simd_r_beats_[p]) rd_[p].commit(fabric_, g, gi);
    for (int p = 14; p < 18; p++) if (simd_w_en_[p - 14] && wr_[p - 14].written < wr_[p - 14].total_beats()) wr_[p - 14].commit(fabric_, g, gi);
    return running_;
}

bool AccelEngine::step() {
    if (kind_ == 2) return step_p1();
    if (kind_ == 3) return step_os();
    if (kind_ == 4) return step_is();
    if (kind_ == 5) return step_simd();
    if (!running_) return false;
    if (cyc_ >= cap_) { running_ = false; res_.busy = cyc_; return false; }
    cyc_++;
    if (std::getenv("MEMSIM_ENGDBG") && (cyc_ % 5000 == 0))
        std::fprintf(stderr, "  [ENGDBG] cyc=%llu g_r10=%ld g_r11=%ld tiles_os=%ld out_su=%ld tiles_is=%ld "
                     "w3=%ld/%ld sucrel=%d iscrel=%d\n", (unsigned long long)cyc_, g_r10_, g_r11_, tiles_os_,
                     out_su_, tiles_is_, w3_.written, w3_.total_beats(),
                     (int)(rel_r10_ || g_r10_ >= r10_cnt_), (int)(rel_r11_ || g_r11_ >= r11_cnt_));

    // ---- land last cycle's granted reads ----
    if (en_os_) { r0_.land_reads(); if (r0_.done) r0_.reset();
                  r1_.land_reads(); if (r1_.done) r1_.reset(); }
    for (int l = 0; l < NCH7; l++) { landed_[l] += pend7_[l]; pend7_[l] = 0; }
    r10_.land_reads();
    if (sw_en_) {
        r2sw_.land_reads(); if (r2sw_.done) r2sw_.reset();
        r3sw_.land_reads(); if (r3sw_.done) r3sw_.reset();
        if (r5sw_.active || r5sw_.done) { r5sw_.land_reads(); if (r5sw_.done) r5sw_.reset(); }
    }
    if (en_isc_) { r11_.land_reads(); if (r11_.done) r11_.reset();
                   r12_.land_reads(); if (r12_.done) r12_.reset();
                   if (r13_en_) r13_.land_reads(); }  // psum read-back is single-pass (no reset)

    // osCore disabled => z preloaded, release the SUC at once; isCore disabled => stays parked.
    bool suc_rel = !en_os_ || rel_r10_ || (g_r10_ >= r10_cnt_);
    bool isc_rel = en_isc_ && (rel_r11_ || (g_r11_ >= r11_cnt_));
    if (suc_rel && !suc_started_) { res_.suc_start = cyc_; suc_started_ = true; }
    if (isc_rel && !isc_started_) { res_.isc_start = cyc_; isc_started_ = true; }

    // ---- control: array/scan steps that pop inputs + push outputs ----
    // osCore output W0: drip one group per cycle into the FIFO while it has room (the writer commits one
    // per cycle), and don't retire a tile until its predecessor's output has fully entered the FIFO (the
    // array's output registers hold at most one tile). A full W0 thus stalls the array -> R0/R1, so the
    // FIFO never exceeds its depth and write-side contention back-pressures the core, as in the RTL.
    if (en_os_ && os_owed_ > 0 && w0_.fifo_occ < w0_.depth) { w0_.push(1); os_owed_--; }
    if (en_os_ && tiles_os_ < n_tiles_os_ && r0_.fifo_occ > 0 && r1_.fifo_occ > 0
        && ((as_os_ + 1) % K_os_ != 0 || os_owed_ == 0)) {
        r0_.pop(); r1_.pop();
        if (++as_os_ % K_os_ == 0) {
            tiles_os_++; os_owed_ += wbpt_os_; g_r10_++;
            if (rec_fires) res_.r10_fire.push_back((uint32_t)cyc_);
        }
    }
    bool room7 = true;
    for (int l = 0; l < NCH7; l++) if (agu7_ - issued_[l] >= r7_fifo_d_) room7 = false;
    if (room7) agu7_++;
    if (k7_ == 0 && !have_bc_) {
        long avail = landed_[0];
        for (int l = 1; l < NCH7; l++) if (landed_[l] < avail) avail = landed_[l];
        if (avail - consumed7_ >= RPR) { consumed7_ += RPR; have_bc_ = true; }
    }
    if (sw_en_ && sw_dt_ < iters_ && (sw_dt_ - out_su_) < SWBUF) {
        bool need_read = (sw_dt_ % 8) == 0;
        if (!need_read || (r2sw_.fifo_occ > 0 && r3sw_.fifo_occ > 0)) {
            sw_acc_ += iters_;
            if (sw_acc_ >= sw_cyc_) {
                sw_acc_ -= sw_cyc_;
                if (need_read) { r2sw_.pop(); r3sw_.pop(); }
                if (sw_dt_ == 0) sw_first_ = cyc_;
                sw_dt_++; sw_last_ = cyc_;
            }
        }
    }
    if (suc_rel && out_su_ < iters_ && have_bc_ && r10_.fifo_occ > 0 && (!sw_en_ || sw_dt_ > out_su_)) {
        out_su_++; g_r11_++;
        if (rec_fires) res_.r11_fire.push_back((uint32_t)cyc_);
        if (++y_in_word_ == 8) { y_in_word_ = 0; w2_.push(1); }
        if ((out_su_ % 8) == 0) r10_.pop();
        if (++k7_ == delaySU) { k7_ = 0; have_bc_ = false; }
    }
    // isCore output W3: same physical back-pressure as W0 (drip 1/cycle, hold a tile until its
    // predecessor has drained out of the array's output registers; a full W3 stalls the isCore).
    if (is_owed_ > 0 && w3_.fifo_occ < w3_.depth) { w3_.push(1); is_owed_--; }
    // isCore input-stationary K_M_N: each (k,m) A=y tile is serial-loaded over serDesA_ beats with the
    // array IDLE (s2p fill), then reused for dFinal compute steps each reading B (R12) + psum (R13 RMW).
    // The s2p idle is the VersaCore latency the array genuinely stalls on (address advances every
    // serDesA_+dFinal cyc) -> isCore trails the SUC by what the dataflow dictates.
    if (isc_rel && tiles_is_ < n_tiles_is_) {
        if (isc_aload_ > 0) {                                // A=y s2p fill: array idle this cycle
            isc_aload_--;
        } else if (r11_.fifo_occ > 0 && r12_.fifo_occ > 0 && ((as_is_ + 1) % K_is_ != 0 || is_owed_ == 0)) {
            // NOTE: a per-array-step `as_is_ < out_su_` gate (isCore downstream of SUC) was tried and
            // REVERTED — it forces 1:1 lockstep, adding sustained fabric contention that wrongly inflates
            // suc_span (22,034->24,561 vs verified-correct vsim 21,962). The correct isc_tail (~1,324)
            // needs a PER-A-TILE production gate (burst produced backlog, only the last A-tile waits past
            // suc_end), accounting for the K_M_N (k-outer) vs y-production (seq-major) transpose.
            r11_.pop(); r12_.pop();
            if (r13_en_ && r13_.fifo_occ > 0) r13_.pop();    // psum read-back, paced to array (never gates)
            if (++as_is_ % K_is_ == 0) { tiles_is_++; is_owed_ += wbpt_is_; g_iscore_++; }
            if (++isc_since_a_ == (long)dFinal_) { isc_since_a_ = 0; isc_aload_ = serDesA_; }  // next A-tile s2p
        }
    }

    if (res_.osc_end == 0 && tiles_os_ == n_tiles_os_ && w0_.written >= Wz_) res_.osc_end = cyc_;
    if (res_.suc_end == 0 && out_su_ >= iters_) res_.suc_end = cyc_;
    // isCore present => end when W3 drains; SUC-only (isCore disabled) => end when the SUC's W2 drains.
    bool fin = en_isc_ ? (tiles_is_ == n_tiles_is_ && w3_.written >= w3_.total_beats())
                       : (out_su_ >= iters_ && w2_.written >= w2_.total_beats());
    if (fin) {
        res_.busy = cyc_; res_.isc_end = cyc_;
        res_.sw_start = sw_first_; res_.sw_end = sw_last_;
        if (res_.osc_end == 0) res_.osc_end = cyc_;
        if (res_.suc_end == 0) res_.suc_end = cyc_;
        running_ = false;
        return false;
    }

    // ---- propose ALL active ports to the ONE shared fabric (DMA beat engine stepped in-loop) ----
    fabric_.begin_cycle(dma_.step(cyc_));
    if (en_os_) { r0_.propose(fabric_); r1_.propose(fabric_); w0_.propose(fabric_); }
    for (int l = 0; l < NCH7; l++)
        if (issued_[l] < agu7_ && issued_[l] - consumed7_ < r7_data_d_)
            fabric_.post(r7bank(issued_[l], l), (T_R7 << 4) | l, (landed_[l] - consumed7_ <= 1) ? 1 : 0);
    if (suc_rel) r10_.propose(fabric_);
    w2_.propose(fabric_);
    if (sw_en_) { r2sw_.propose(fabric_); r3sw_.propose(fabric_); if (r5sw_.active) r5sw_.propose(fabric_); }
    if (isc_rel) { r11_.propose(fabric_); r12_.propose(fabric_); if (r13_en_) r13_.propose(fabric_); }
    if (en_isc_) w3_.propose(fabric_);

    bool g[64];
    fabric_.arbitrate(g);
    int gi = 0;
    if (en_os_) {
        r0_.commit(fabric_, g, gi);
        r1_.commit(fabric_, g, gi);
        long off = word_off(w0_.ts, w0_.ti); long pw = w0_.written; w0_.commit(fabric_, g, gi);
        if (w0_.written > pw && off >= 0 && off < Wz_) zwr_[off] = 1;
    }
    for (int l = 0; l < NCH7; l++)
        if (issued_[l] < agu7_ && issued_[l] - consumed7_ < r7_data_d_)
            if (g[gi++]) { issued_[l]++; pend7_[l] = 1; }
    if (suc_rel) { long off = word_off(r10_.ts, r10_.ti); long pp = r10_.pending_push; r10_.commit(fabric_, g, gi);
                   if (r10_.pending_push > pp && off >= 0 && off < Wz_ && !zwr_[off]) res_.stale_z++; }
    { long off = word_off(w2_.ts, w2_.ti); long pw = w2_.written; w2_.commit(fabric_, g, gi);
      if (w2_.written > pw && off >= 0 && off < Wy_) ywr_[off] = 1; }
    if (sw_en_) { r2sw_.commit(fabric_, g, gi); r3sw_.commit(fabric_, g, gi); if (r5sw_.active) r5sw_.commit(fabric_, g, gi); }
    if (isc_rel) { long off = word_off(r11_.ts, r11_.ti); long pp = r11_.pending_push; r11_.commit(fabric_, g, gi);
                   if (r11_.pending_push > pp && off >= 0 && off < Wy_ && !ywr_[off]) res_.stale_y++;
                   r12_.commit(fabric_, g, gi); if (r13_en_) r13_.commit(fabric_, g, gi); }
    if (en_isc_) w3_.commit(fabric_, g, gi);
    sample_fifo();
    return running_;
}
