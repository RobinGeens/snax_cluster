// Copyright 2026 KU Leuven. memsim — dual-hart co-simulation scheduler.
#include "sched.hpp"

#include <cstdio>

#include "snax_csr.hpp"  // HW_BARRIER_RELEASE_COST

namespace {
bool runnable(const Hart& h) { return h.state == HartState::Run; }
}  // namespace

void Scheduler::run() {
    Hart& h0 = m.harts[0];
    Hart& h1 = m.harts[1];
    uint64_t steps = 0;

    auto all_done = [&]() {
        return (h0.state == HartState::Halted || h0.state == HartState::Parked) &&
               (h1.state == HartState::Halted || h1.state == HartState::Parked);
    };
    auto both_at_barrier = [&]() {
        return h0.state == HartState::AtBarrier && h1.state == HartState::AtBarrier;
    };
    auto release_barrier = [&]() {
        // Sync to the slowest hart + the barrier rendezvous/release latency
        // (arrival latch + csr_stall release, snitch_barrier.sv).
        uint64_t t = (h0.cycle > h1.cycle ? h0.cycle : h1.cycle) + HW_BARRIER_RELEASE_COST;
        h0.cycle = h1.cycle = t;
        h0.state = h1.state = HartState::Run;
    };

    while (!m.exited && steps < max_insns) {
        // Choose the runnable hart with the smaller cycle (ties: h0 first).
        Hart* h = nullptr;
        if (runnable(h0) && runnable(h1))
            h = (h0.cycle <= h1.cycle) ? &h0 : &h1;
        else if (runnable(h0))
            h = &h0;
        else if (runnable(h1))
            h = &h1;

        if (!h) {
            if (both_at_barrier()) { release_barrier(); continue; }
            if (all_done()) break;
            // One at barrier, the other halted/parked, or a stuck poll: advance
            // the world to its next event so blocked polls can make progress.
            uint64_t t = m.world->next_event_cycle();
            if (t == UINT64_MAX) {
                // Nothing left to change any predicate -> genuine deadlock.
                // (Exception: a lone AtBarrier opposite a Halted hart that has
                //  already signalled exit is handled by m.exited above.)
                deadlocked = true;
                std::fprintf(stderr,
                    "memsim: DEADLOCK. h0{pc=0x%08x st=%d cyc=%llu} "
                    "h1{pc=0x%08x st=%d cyc=%llu}\n",
                    h0.pc, (int)h0.state, (unsigned long long)h0.cycle,
                    h1.pc, (int)h1.state, (unsigned long long)h1.cycle);
                break;
            }
            m.world->advance_to(t);
            // Re-evaluate blocked polls by turning them back to Run; the poll
            // loop re-reads world state and either exits or re-blocks.
            if (h0.state == HartState::BlockedPoll) { h0.state = HartState::Run; h0.cycle = t > h0.cycle ? t : h0.cycle; }
            if (h1.state == HartState::BlockedPoll) { h1.state = HartState::Run; h1.cycle = t > h1.cycle ? t : h1.cycle; }
            continue;
        }

        Step s = interp_step(m, *h, dma[h->id]);
        steps++;
        switch (s) {
            case Step::Barrier:
                if (both_at_barrier()) release_barrier();
                break;
            case Step::Halt:
                m.exited = true;
                break;
            case Step::Wfi:
                h->state = HartState::Parked;
                break;
            case Step::PollRead:
                // Heuristic: if a hart has spun on the same backward poll many
                // times with no world change, block it so the world can advance.
                if (h->poll_spins > 64) {
                    h->state = HartState::BlockedPoll;
                    h->poll_spins = 0;
                }
                break;
            case Step::Normal:
                break;
        }
    }

    if (steps >= max_insns)
        std::fprintf(stderr, "memsim: hit instruction cap (%llu)\n",
                     (unsigned long long)max_insns);
}
