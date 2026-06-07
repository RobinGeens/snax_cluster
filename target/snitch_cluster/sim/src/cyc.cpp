// Copyright 2026 KU Leuven. memsim — per-cycle fabric + reader/writer streamers.
#include "cyc.hpp"

void CycReader::configure(const Agu& a, int nch, int nsp, int fd, int prt) {
    base = a.base;
    for (int i = 0; i < 4; i++) ts[i] = a.t_stride[i];
    num_channel = nch;
    n_spatial = nsp;
    fifo_depth = fd;
    port = prt;
    sstride[0] = a.s_stride[0];
    sstride[1] = a.s_stride[1];
    if (nsp == 2) { sbound[0] = 2; sbound[1] = 2; }
    else { sbound[0] = nch; sbound[1] = 1; }
    // stride-0 INNERMOST temporal dim = reuse (read once, replay; no TCDM re-read)
    reuse = (a.t_stride[0] == 0 && a.t_bound[0] > 1) ? a.t_bound[0] : 1;
    eb[0] = (reuse > 1) ? 1 : (a.t_bound[0] ? a.t_bound[0] : 1);
    for (int i = 1; i < 4; i++) eb[i] = a.t_bound[i] ? a.t_bound[i] : 1;
    for (int i = 0; i < 4; i++) ti[i] = 0;
    for (int l = 0; l < 8; l++) lane_done[l] = false;
    active = true; done = false;
    fifo_occ = outstanding = pending_push = reuse_left = 0;
}

uint64_t CycReader::lane_addr(int lane) const {
    int64_t toff = (int64_t)ti[0] * ts[0] + (int64_t)ti[1] * ts[1] +
                   (int64_t)ti[2] * ts[2] + (int64_t)ti[3] * ts[3];
    int64_t soff;
    if (n_spatial == 2) {
        int i = lane % sbound[0], j = lane / sbound[0];
        soff = (int64_t)i * sstride[0] + (int64_t)j * sstride[1];
    } else {
        soff = (int64_t)lane * sstride[0];
    }
    return base + toff + soff;
}

void CycReader::propose(Fabric& f) {
    if (!active || done) return;
    // Cap on LANDED data only (data FIFO). The address buffer (deeper) lets the
    // reader issue reads ahead while earlier ones are still in flight, so the
    // +1cc TCDM response latency is hidden — matching the real 2-buffer streamer.
    if (fifo_occ >= fifo_depth) return;  // data FIFO full: stop reading ahead
    for (int l = 0; l < num_channel; l++)
        if (!lane_done[l]) {
            uint32_t a = (uint32_t)lane_addr(l);
            f.post((a >> 3) & 31, (port << 4) | l);
        }
}

void CycReader::commit(Fabric& f, const bool granted[64], int& grant_idx) {
    if (!active || done) return;
    if (fifo_occ >= fifo_depth) return;  // mirror propose's gate (landed-data cap)
    int all = 1;
    for (int l = 0; l < num_channel; l++) {
        if (lane_done[l]) continue;
        if (granted[grant_idx++]) lane_done[l] = true;   // granted this cycle
        else all = 0;                                    // lost arbitration -> retry next cycle
    }
    if (all) {                       // whole group delivered this cycle
        outstanding++;               // lands in FIFO next cycle (+1cc response)
        pending_push++;
        if (track_idx) pend_idx = prod_idx++;   // BIST: tag this group's order
        for (int l = 0; l < 8; l++) lane_done[l] = false;
        // advance the temporal AGU (inner -> outer)
        int d = 0;
        for (; d < 4; d++) { if (++ti[d] < eb[d]) break; ti[d] = 0; }
        if (d == 4) { done = true; active = false; }
    }
}

bool CycReader::pop() {
    if (fifo_occ > 0) { fifo_occ--; return true; }
    return false;
}

void CycWriter::configure(const Agu& a, int nch, int nsp, int prt) {
    base = a.base;
    for (int i = 0; i < 4; i++) { ts[i] = a.t_stride[i]; eb[i] = a.t_bound[i] ? a.t_bound[i] : 1; }
    num_channel = nch; n_spatial = nsp; port = prt;
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
    for (int l = 0; l < num_channel; l++)
        if (!lane_done[l]) { uint32_t a = (uint32_t)lane_addr(l); f.post((a >> 3) & 31, (port << 4) | l); }
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

GemmResult cyc_gemm(const Agu* in_readers, const int* rd_nch, const int* rd_nsp, int n_readers,
                    const Agu& out_writer, int w_nch, int w_nsp,
                    long n_out_tiles, long K_i, uint32_t dma_mask) {
    if (n_out_tiles <= 0 || K_i <= 0) return {0, 0};
    CycReader r[4];
    for (int i = 0; i < n_readers && i < 4; i++) r[i].configure(in_readers[i], rd_nch[i], rd_nsp[i], 4, i);
    CycWriter w; w.configure(out_writer, w_nch, w_nsp, 8);
    long wbpt = w.total_beats() / n_out_tiles;            // writer groups per output tile
    if (wbpt < 1) wbpt = 1;
    // Sparse interconnect: each port is residue-pinned, so ports contend only within their
    // own lanes (own Fabric), not across each other; the only shared cross-port effect is DMA
    // superbank preemption (dma_mask). See docs/dataflow/10_memsim.md (sparse interconnect).
    Fabric fr[4], fw;
    uint64_t cyc = 0, busy = 0;
    long as = 0, tiles = 0;
    uint64_t cap = (uint64_t)n_out_tiles * K_i * 8 + 100000;   // safety: ~8x ideal, never spin
    while (cyc < cap) {
        cyc++;
        for (int i = 0; i < n_readers; i++) { r[i].land_reads(); if (r[i].done) r[i].reset(); }
        // Array step: consume one group from each input reader (stall only if a reader's FIFO
        // is empty from a within-port conflict); every K_i steps, emit one output tile.
        if (tiles < n_out_tiles) {
            bool fed = true;
            for (int i = 0; i < n_readers; i++) if (r[i].fifo_occ <= 0) fed = false;
            if (fed) {
                for (int i = 0; i < n_readers; i++) r[i].pop();
                if (++as % K_i == 0) { tiles++; w.push(wbpt); }
            }
        }
        // MambaCore busy (perf counter) ends when the subcore is DONE = its output (out_d
        // -> W0) has all fired, i.e. the writer has drained every group.
        if (tiles == n_out_tiles && w.written >= w.total_beats()) { busy = cyc; break; }
        for (int i = 0; i < n_readers; i++) {            // each port arbitrates within itself
            fr[i].begin_cycle(dma_mask);
            r[i].propose(fr[i]);
            bool g[64]; fr[i].arbitrate(g);
            int gi = 0; r[i].commit(fr[i], g, gi);
        }
        fw.begin_cycle(dma_mask);
        w.propose(fw);
        bool gw[64]; fw.arbitrate(gw);
        int giw = 0; w.commit(fw, gw, giw);
    }
    return {busy, busy};
}

uint64_t cyc_suc_duration(const Agu& r7_agu, int seqLen, int dInner_tile, uint32_t dma_mask) {
    // Per-cycle SUC: the R7 dt_BC reader is not group-synchronous -- each lane's DataRequestor
    // issues its head address independently and retries on conflict while lanes on free banks
    // run ahead (bounded by FIFO depth), so the bank-conflict magnitude (~1.75x at pad0/2 banks,
    // 1.0x at pad4/4 banks) comes from the per-cycle arbitration. See docs/dataflow/10_memsim.md.
    const int NCH = 4, delaySU = 4, RPR = 4;   // R7 group = 4 lanes; RPR groups (B+C) per delaySU
    // RTL FIFO depths (Reader.scala/StreamParamGen): request-side addr FIFO = 4; response-side
    // read-ahead = responser FIFO (4) + dataBuffer (4) = 8 (the 2-refresh cross-overlap). Do
    // not change without RTL re-check.
    const int ADDR_D = 4, DATA_D = 8;
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
        return (int)(((uint32_t)(r7_agu.base + toff + soff) >> 3) & 31);
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
            }
        }
        r.fifo_occ = r.ir_cnt;                                  // single ledger: mirror ring
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
