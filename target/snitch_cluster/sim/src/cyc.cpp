// Copyright 2026 KU Leuven. memsim — per-cycle fabric + reader/writer streamers.
#include "cyc.hpp"

#include <cstdio>
#include <cstdlib>

// --- per-bank contention histogram (MEMSIM_BANKHIST), debug-only ---
bool Fabric::h_on = false;
long Fabric::h_req[Fabric::NB] = {0};
long Fabric::h_conf[Fabric::NB] = {0};
long Fabric::h_portbank[18][Fabric::NB] = {{0}};
long Fabric::h_portgrant[18] = {0};
long Fabric::h_pair[18][18] = {{0}};
void Fabric::hist_reset() {
    for (int b = 0; b < NB; b++) { h_req[b] = h_conf[b] = 0; for (int p = 0; p < 18; p++) h_portbank[p][b] = 0; }
    for (int p = 0; p < 18; p++) { h_portgrant[p] = 0; for (int q = 0; q < 18; q++) h_pair[p][q] = 0; }
}
void Fabric::hist_dump(const char* tag) {
    if (!h_on) return;
    // fabric tokens are the cyc_engine ids (cyc_engine.cpp enum T_*): osCore=R0(0),R1(1),W0(2);
    // SUC=R7/dtBC(3),R10/z(4),W2/y(5); isCore=R11(6),R12(7),W3(8),R13(12); switchCore=R2(9),R3(10),R5(11).
    const int osP[] = {0, 1, 2}, sucP[] = {3, 4, 5}, isP[] = {6, 7, 8, 12};
    long tot_req = 0, tot_conf = 0;
    fprintf(stderr, "[BANKHIST %s] bank: req conf | osCore SUC isCore (which streams hit each bank)\n", tag);
    for (int b = 0; b < NB; b++) {
        if (h_req[b] == 0) continue;
        long os = 0, su = 0, is = 0;
        for (int p : osP) os += h_portbank[p][b];
        for (int p : sucP) su += h_portbank[p][b];
        for (int p : isP) is += h_portbank[p][b];
        tot_req += h_req[b];
        tot_conf += h_conf[b];
        fprintf(stderr, "  b%02d: %8ld %8ld | os=%-8ld suc=%-8ld is=%-8ld\n", b, h_req[b], h_conf[b], os, su, is);
    }
    long pt[18] = {0};
    for (int p = 0; p < 18; p++) for (int b = 0; b < NB; b++) pt[p] += h_portbank[p][b];
    fprintf(stderr, "[BANKHIST %s] per-port lane-reqs: R0=%ld R1=%ld W0=%ld R7dtBC=%ld R10z=%ld W2y=%ld R11=%ld R12=%ld W3=%ld R13=%ld\n",
            tag, pt[0], pt[1], pt[2], pt[3], pt[4], pt[5], pt[6], pt[7], pt[8], pt[12]);
    // per-port stall = requests - grants (cycles a lane wanted a bank but lost arbitration) -> compare
    // to the RTL [TCDM_STALL] per-port counters from sparse_interconnect_wrapper.sv.
    const char* nm[18] = {"R0","R1","W0","R7dtBC","R10z","W2y","R11","R12","W3","R2","R3","R5","R13",
                          "?13","?14","?15","?16","?17"};
    fprintf(stderr, "[BANKHIST %s] per-port req/grant/stall (engine):\n", tag);
    for (int p = 0; p < 18; p++) {
        if (pt[p] == 0 && h_portgrant[p] == 0) continue;
        fprintf(stderr, "  %-7s req=%-9ld grant=%-9ld stall=%-9ld (%.0f%%)\n", nm[p], pt[p], h_portgrant[p],
                pt[p] - h_portgrant[p], pt[p] ? 100.0 * (pt[p] - h_portgrant[p]) / pt[p] : 0.0);
    }
    fprintf(stderr, "[BANKHIST %s] total req=%ld conflicts=%ld (%.1f%% of requests lost arbitration)\n",
            tag, tot_req, tot_conf, tot_req ? 100.0 * tot_conf / tot_req : 0.0);
    // who-loses-to-whom: for each port that lost any arbitration, list the winners it lost to.
    fprintf(stderr, "[BANKHIST %s] loser -> winners (times `loser` lost a bank to `winner`):\n", tag);
    for (int lp = 0; lp < 18; lp++) {
        long lt = 0; for (int wp = 0; wp < 18; wp++) lt += h_pair[lp][wp];
        if (lt == 0) continue;
        fprintf(stderr, "  %-7s lost %-8ld to:", nm[lp], lt);
        for (int wp = 0; wp < 18; wp++) if (h_pair[lp][wp] > 0) fprintf(stderr, " %s=%ld", nm[wp], h_pair[lp][wp]);
        fprintf(stderr, "\n");
    }
}

void CycReader::configure(const Agu& a, int nch, int nsp, int fd, int prt) {
    base = a.base;
    for (int i = 0; i < 4; i++) ts[i] = a.t_stride[i];
    num_channel = nch;
    n_spatial = nsp;
    data_depth = fd;       // response-side data FIFO depth (read-ahead vs consumer)
    addr_depth = fd;       // request-side address FIFO == data FIFO depth in the RTL (StreamParamGen)
    port = prt;
    remap = a.remap;
    sstride[0] = a.s_stride[0];
    sstride[1] = a.s_stride[1];
    if (nsp == 2) { sbound[0] = 2; sbound[1] = 2; }
    else { sbound[0] = nch; sbound[1] = 1; }
    // stride-0 INNERMOST temporal dim = reuse (read once, replay; no TCDM re-read)
    reuse = (a.t_stride[0] == 0 && a.t_bound[0] > 1) ? a.t_bound[0] : 1;
    eb[0] = (reuse > 1) ? 1 : (a.t_bound[0] ? a.t_bound[0] : 1);
    for (int i = 1; i < 4; i++) eb[i] = a.t_bound[i] ? a.t_bound[i] : 1;
    reset();
}

// Byte address of `lane` at temporal group index `group` (cyclic over the AGU extent).
uint64_t CycReader::lane_addr(long group, int lane) const {
    long r = group % (total() ? total() : 1), ti0[4];
    for (int d = 0; d < 4; d++) { ti0[d] = r % eb[d]; r /= eb[d]; }
    int64_t toff = ti0[0] * ts[0] + ti0[1] * ts[1] + ti0[2] * ts[2] + ti0[3] * ts[3];
    int64_t soff;
    if (n_spatial == 2) { int i = lane % sbound[0], j = lane / sbound[0]; soff = (int64_t)i * sstride[0] + (int64_t)j * sstride[1]; }
    else soff = (int64_t)lane * sstride[0];
    return base + toff + soff;
}

// Each lane proposes its NEXT un-issued group's address, independently: bounded by the data FIFO
// (don't read more than data_depth ahead of the consumer) and the address FIFO (at most addr_depth
// requests outstanding/unlanded). A blocked lane simply re-proposes; others run ahead.
void CycReader::propose(Fabric& f) {
    if (!active) return;
    int prio = (min_landed() - consumed <= 1) ? 1 : 0;   // near-empty -> urgent (ComplexQueue QoS)
    for (int l = 0; l < num_channel; l++) {
        long g = lane_issued[l];
        if (g - consumed >= data_depth) continue;        // data FIFO full
        if (g - lane_landed[l] >= addr_depth) continue;  // address FIFO full (outstanding cap)
        uint32_t a = agu_swz((uint32_t)lane_addr(g, l), remap);
        f.post((a >> 3) & 31, (port << 4) | l, prio);
    }
}

void CycReader::commit(Fabric& f, const bool granted[64], int& grant_idx) {
    pending_push = 0;
    if (!active) return;
    for (int l = 0; l < num_channel; l++) {
        long g = lane_issued[l];
        if (g - consumed >= data_depth) continue;
        if (g - lane_landed[l] >= addr_depth) continue;
        if (granted[grant_idx++]) {                      // this lane's read accepted by the bank
            if (l == 0) set_ti(g);                       // expose the granted group (stale check)
            lane_issued[l]++; pend[l]++; pending_push++;
        }
    }
}

bool CycReader::pop() {
    if (reuse_left > 0) { reuse_left--; return true; }   // replay (stride-0 dim0 reuse)
    if (min_landed() > consumed) {
        consumed++;
        if (reuse > 1) reuse_left = reuse - 1;
        long m = min_landed();
        fifo_occ = (reuse > 1) ? (m > consumed ? 1 : 0) : (int)(m - consumed);
        return true;
    }
    return false;
}

void CycWriter::configure(const Agu& a, int nch, int nsp, int prt) {
    base = a.base;
    for (int i = 0; i < 4; i++) { ts[i] = a.t_stride[i]; eb[i] = a.t_bound[i] ? a.t_bound[i] : 1; }
    num_channel = nch; n_spatial = nsp; port = prt; remap = a.remap;
    sstride[0] = a.s_stride[0]; sstride[1] = a.s_stride[1];
    if (nsp == 2) { sbound[0] = 2; sbound[1] = 2; } else { sbound[0] = nch; sbound[1] = 1; }
    for (int i = 0; i < 4; i++) ti[i] = 0;
    for (int l = 0; l < 8; l++) lane_done[l] = false;
    fifo_occ = 0; written = 0; done = false;
}

uint64_t CycWriter::lane_addr(int lane) const {
    int64_t toff = (int64_t)ti[0] * ts[0] + (int64_t)ti[1] * ts[1] +
                   (int64_t)ti[2] * ts[2] + (int64_t)ti[3] * ts[3];
    int64_t soff;
    if (n_spatial == 2) { int i = lane % sbound[0], j = lane / sbound[0];
                          soff = (int64_t)i * sstride[0] + (int64_t)j * sstride[1]; }
    else soff = (int64_t)lane * sstride[0];
    return base + toff + soff;
}

void CycWriter::propose(Fabric& f) {
    if (done || fifo_occ <= 0) return;
    // Buffer-urgency QoS (ComplexQueue.scala:67): a writer whose input FIFO is near-full is about to
    // stall the producing core, so it asserts TCDM priority to drain.
    int prio = (fifo_occ >= depth - 1) ? 1 : 0;
    for (int l = 0; l < num_channel; l++)
        if (!lane_done[l]) { uint32_t a = agu_swz((uint32_t)lane_addr(l), remap); f.post((a >> 3) & 31, (port << 4) | l, prio); }
}

void CycWriter::commit(Fabric& f, const bool granted[64], int& grant_idx) {
    if (done || fifo_occ <= 0) return;
    int all = 1;
    for (int l = 0; l < num_channel; l++) {
        if (lane_done[l]) continue;
        if (granted[grant_idx++]) lane_done[l] = true; else all = 0;
    }
    if (all) {                                   // whole output group written to TCDM
        written++; fifo_occ--;
        for (int l = 0; l < 8; l++) lane_done[l] = false;
        int d = 0;
        for (; d < 4; d++) { if (++ti[d] < eb[d]) break; ti[d] = 0; }
    }
}

uint64_t cyc_suc_duration(const Agu& r7_agu, int seqLen, int dInner_tile, uint32_t dma_mask) {
    // Per-cycle SUC: the R7 dt_BC reader is not group-synchronous -- each lane's DataRequestor
    // issues its head address independently and retries on conflict while lanes on free banks
    // run ahead (bounded by FIFO depth), so the bank-conflict magnitude (~1.75x at pad0/2 banks,
    // 1.0x at pad4/4 banks) comes from the per-cycle arbitration. See target/snitch_cluster/sim/docs/memsim.md.
    const int NCH = 4, delaySU = 4, RPR = 4;   // R7 group = 4 lanes; RPR groups (B+C) per delaySU
    // RTL FIFO depths (Reader.scala/StreamParamGen): request-side addr FIFO = fifo_depth[7]; response-
    // side read-ahead = responser FIFO + dataBuffer = 2x (the 2-refresh cross-overlap). Do not change
    // the 2x without RTL re-check.
    const int ADDR_D = snax_streamer_depth(7), DATA_D = 2 * ADDR_D;
    uint64_t iters = (uint64_t)seqLen * (uint64_t)dInner_tile;
    if (iters == 0) return 0;
    int32_t ts[4];
    int eb[4];
    for (int i = 0; i < 4; i++) { ts[i] = r7_agu.t_stride[i]; eb[i] = r7_agu.t_bound[i] ? r7_agu.t_bound[i] : 1; }
    if (r7_agu.t_stride[0] == 0 && r7_agu.t_bound[0] > 1) eb[0] = 1;   // stride-0 dim0 = reuse
    auto step_bank = [&](long s, int lane) -> int {                    // bank for lane at temporal step s
        long t0 = s % eb[0]; long r = s / eb[0];
        long t1 = r % eb[1]; r /= eb[1];
        long t2 = r % eb[2]; r /= eb[2];
        long t3 = r % eb[3];
        int64_t toff = t0 * ts[0] + t1 * ts[1] + t2 * ts[2] + t3 * ts[3];
        int i = lane % 2, j = lane / 2;
        int64_t soff = (int64_t)i * r7_agu.s_stride[0] + (int64_t)j * r7_agu.s_stride[1];
        return (int)((agu_swz((uint32_t)(r7_agu.base + toff + soff), r7_agu.remap) >> 3) & 31);
    };
    long issued[NCH] = {0}, landed[NCH] = {0};   // steps each lane has issued (granted) / landed (+1cc)
    int  pend[NCH] = {0};                         // granted this cycle, lands next
    long agu = 0, consumed = 0;                   // groups generated / combined groups consumed
    Fabric f;
    uint64_t cyc = 0, out = 0;
    int k = 0; bool have_bc = false;
    while (out < iters && cyc < 1000000000ULL) {
        cyc++;
        for (int l = 0; l < NCH; l++) { landed[l] += pend[l]; pend[l] = 0; }   // land last cycle's grants
        bool room = true;                                                       // AGU fills group-sync
        for (int l = 0; l < NCH; l++) if (agu - issued[l] >= ADDR_D) room = false;
        if (room) agu++;
        if (k == 0 && !have_bc) {                                               // refresh boundary: consume RPR
            long avail = landed[0];
            for (int l = 1; l < NCH; l++) if (landed[l] < avail) avail = landed[l];
            if (avail - consumed >= RPR) { consumed += RPR; have_bc = true; }
        }
        f.begin_cycle(dma_mask);
        for (int l = 0; l < NCH; l++)                                           // each lane issues independently
            if (issued[l] < agu && issued[l] - consumed < DATA_D)               // addr available + data-FIFO room
                f.post(step_bank(issued[l], l), (7 << 4) | l);
        bool g[64]; f.arbitrate(g);
        int gi = 0;
        for (int l = 0; l < NCH; l++)
            if (issued[l] < agu && issued[l] - consumed < DATA_D)
                if (g[gi++]) { issued[l]++; pend[l] = 1; }                      // granted -> lands next cycle
        if (have_bc) { out++; if (++k == delaySU) { k = 0; have_bc = false; } } // 1 output/cyc when fed
    }
    return cyc;
}

// One BIST pass over the per-cycle SUC BC delivery: the consumer pops a full refresh
// (RPR groups) only once they have landed (+1cc after grant); a pop on the empty ring is
// a poison (un-landed) read. With `inject`, one refresh fires a cycle too early (delay
// modelled too short) so its pop hits poison -- proving a wrong delay is detected.
struct BistPass {
    long groups = 0;
    bool completed = false;
    int poison_count = 0;
    int n_idx = 0;
    int idx[8] = {0};          // expected BC-group (production) indices that read poison
};

static void suc_bist_pass(const Agu& agu, int seqLen, int dInner_tile, bool inject, BistPass& out) {
    const int delaySU = 4, RPR = 4;          // 4 R7 groups per BC refresh (B+C, Vec N)
    long iters = (long)seqLen * dInner_tile;
    CycReader r;
    r.configure(agu, 4, 2, 4, 7);
    r.track_idx = true;
    Fabric f;
    long iter = 0, expect = 0;
    int k = 0;
    bool have_bc = false, fault_done = false;
    uint64_t cyc = 0;
    // Pop a full refresh; any pop on an empty ring is a poison (un-landed) read,
    // recorded by the BC-group index (expect+p) it should have delivered.
    auto consume = [&]() {
        for (int p = 0; p < RPR; p++) {
            int idx;
            if (!r.pop_idx(idx)) {
                out.poison_count++;
                if (out.n_idx < 8) out.idx[out.n_idx++] = (int)(expect + p);
            } else {
                r.pop();  // advance the reader's consumer cursor so it keeps issuing (data-FIFO frees)
            }
        }
        expect += RPR;
        have_bc = true;
    };
    while (iter < iters && cyc < 2000000000ULL) {
        cyc++;
        // FAULT (delay one cycle too short): a consumer that treats THIS cycle's
        // not-yet-landed arrivals (pending_push) as already-ready fires before the
        // +1cc response lands -> its last pop hits an empty ring (poison). Checked
        // BEFORE land_reads, so it reads the pre-land state.
        if (inject && !fault_done && k == 0 && !have_bc &&
            r.ir_cnt < RPR && r.ir_cnt + r.pending_push >= RPR) {
            fault_done = true;
            consume();
        }
        r.land_reads();
        if (r.done) r.reset();
        if (k == 0 && !have_bc && r.ir_cnt >= RPR) consume();   // real run: needs RPR landed
        f.begin_cycle(0);
        r.propose(f);
        bool g[64];
        f.arbitrate(g);
        int gi = 0;
        r.commit(f, g, gi);
        if (have_bc) { iter++; if (++k == delaySU) { k = 0; have_bc = false; } }
    }
    out.groups = expect;
    out.completed = (iter == iters);
}

SucBistResult cyc_suc_bist(const Agu& r7_agu, int seqLen, int dInner_tile) {
    BistPass clean, fault;
    suc_bist_pass(r7_agu, seqLen, dInner_tile, false, clean);
    suc_bist_pass(r7_agu, seqLen, dInner_tile, true, fault);
    SucBistResult res{};
    res.clean_complete = clean.completed;
    res.clean_poison_free = (clean.poison_count == 0);
    res.fault_detected = (fault.poison_count > 0);
    res.groups = clean.groups;
    res.clean_poison_count = clean.poison_count;
    res.fault_poison_count = fault.poison_count;
    // Report the offending indices: the clean run's poison (a real bug) if any, else
    // the fault run's (proof the check localises which BC group goes wrong).
    const BistPass& src = clean.poison_count ? clean : fault;
    res.n_idx = src.n_idx;
    for (int i = 0; i < src.n_idx; i++) res.first_idx[i] = src.idx[i];
    return res;
}

uint64_t cyc_reader_test(const Agu& agu, int nch, int nsp, int fifo_depth,
                         int consume_period, int n_beats, uint32_t dma_mask) {
    CycReader r;
    r.configure(agu, nch, nsp, fifo_depth, 7);
    int total = r.total_read_groups();
    if (n_beats > total) n_beats = total;
    Fabric f;
    uint64_t cyc = 0;
    int consumed = 0, since_pop = 0;
    while (consumed < n_beats && cyc < 100000000ULL) {
        cyc++;
        r.land_reads();
        f.begin_cycle(dma_mask);
        r.propose(f);
        bool granted[64];
        f.arbitrate(granted);
        int gi = 0;
        r.commit(f, granted, gi);
        if (++since_pop >= consume_period) {
            if (r.pop()) { consumed++; since_pop = 0; }
            // else: consumer stalls (FIFO empty) -> since_pop stays high, retries next cycle
        }
    }
    return cyc;
}
