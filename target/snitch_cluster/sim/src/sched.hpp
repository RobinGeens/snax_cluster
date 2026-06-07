// Copyright 2026 KU Leuven. memsim — dual-hart co-simulation scheduler.
#pragma once
#include "interp.hpp"
#include "machine.hpp"

struct Scheduler {
    Machine& m;
    DmaStage dma[2];
    uint64_t max_insns = 2'000'000'000ull;  // safety cap
    bool deadlocked = false;

    explicit Scheduler(Machine& mac) : m(mac) {}

    // Run until both harts halt (exit) or a deadlock/cap is hit.
    void run();
};
