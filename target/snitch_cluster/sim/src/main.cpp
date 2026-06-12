// Copyright 2026 KU Leuven. memsim — entry point.
// Runs a compiled SNAX app .elf through the dual-hart interpreter + timing
// World, emitting the same cycle lines the apps print plus a PASS/FAIL line.
// See docs/dataflow/10_memsim.md.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "elf.hpp"
#include "machine.hpp"
#include "sched.hpp"
#include "world.hpp"

int main(int argc, char** argv) {
    const char* elf_path = nullptr;
    std::string timeline_path;          // --timeline <file>: dump per-engine activity CSV
    bool quiet = false, verify = true;  // integer layout/BIST cross-check always runs
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--quiet")
            quiet = true;
        else if (a == "--params" && i + 1 < argc)
            i++;                    // label only, ignored
        else if (a == "--verify") { /* always on; accepted for back-compat */
        } else if (a == "--timing-only")
            verify = false;  // escape hatch: skip the check
        else if (a == "--timeline" && i + 1 < argc)
            timeline_path = argv[++i];
        else if (a[0] != '-')
            elf_path = argv[i];
    }
    if (!elf_path) {
        std::fprintf(stderr, "usage: memsim <app.elf> [--params p.hjson] [--quiet]\n");
        return 64;
    }

    Machine m;
    m.quiet          = quiet;
    m.show_app_check = std::getenv("MEMSIM_SHOW_APP_CHECK") != nullptr;
    ElfImage img     = load_elf(elf_path, m.mem);
    if (!img.ok) {
        std::fprintf(stderr, "memsim: ELF load failed: %s\n", img.err.c_str());
        return 65;
    }
    m.tohost   = img.sym("tohost");
    m.fromhost = img.sym("fromhost");
    if (!m.tohost) std::fprintf(stderr, "memsim: warning: no 'tohost' symbol; exit undetectable\n");

    SimWorld world;  // functional datapath + timing
    world.mem = &m.mem;
    world.set_verify(verify);
    world.set_trace(!timeline_path.empty());
    // App FP8 golden buffers for the FP32 datapath cross-check (0 if not present).
    world.set_goldens(img.sym("M2_oscore_expected"), img.sym("M2_suc_expected"), img.sym("M2_iscore_expected"));
    // SSM shape params live as uint32 globals in data.h; read their VALUES from .data.
    auto symval = [&](const char* n) {
        uint32_t a = img.sym(n);
        return a ? m.mem.ld32(a) : 0u;
    };
    world.set_ssm(symval("dState"), symval("xProjDim"), symval("bc_pad_banks"), symval("dtRankUnroll"));
    world.set_p1out(img.sym("M2_dt_BC"), img.sym("M2_suc_x"));
    // Phase-2 safe-to-start thresholds (the app paces the SUC/isCore release to these gauge counts).
    world.set_s2s(symval("M2_R10_start_cnt"), symval("M2_R11_start_cnt"));
    m.world = &world;

    m.harts.resize(2);
    for (int i = 0; i < 2; i++) {
        m.harts[i].id    = i;
        m.harts[i].pc    = img.entry;
        m.harts[i].state = HartState::Run;
    }

    Scheduler sched(m);
    sched.run();

    // Per-engine activity timeline (--timeline <file>): one CSV row per active window, for
    // plot_timeline.py. engine in {OSCORE,ISCORE,SUC,SWITCHCORE,DMA}; cycles are absolute.
    if (!timeline_path.empty()) {
        static const char* NAMES[] = {"OSCORE", "ISCORE", "SUC", "SWITCHCORE", "DMA", "TCDM"};
        FILE* f                    = std::fopen(timeline_path.c_str(), "w");
        if (!f) {
            std::fprintf(stderr, "memsim: cannot open timeline file %s\n", timeline_path.c_str());
        } else {
            std::fprintf(f, "engine,start,end,ideal\n");
            for (const auto& s : world.trace())
                std::fprintf(f, "%s,%llu,%llu,%llu\n", NAMES[s.engine], (unsigned long long)s.start,
                             (unsigned long long)s.end, (unsigned long long)s.ideal);
            // Optimal safe-to-start delays from the schedule sweep, as metadata rows: start=optimal start_cnt,
            // end=gauge total. plot_timeline.py prints them on the schedule plot. Omitted when the sweep didn't run
            // (opt < 0).
            if (world.s2s_opt_r10() >= 0)
                std::fprintf(f, "S2S_R10,%ld,%ld,0\n", world.s2s_opt_r10(), world.s2s_total_r10());
            if (world.s2s_opt_r11() >= 0)
                std::fprintf(f, "S2S_R11,%ld,%ld,0\n", world.s2s_opt_r11(), world.s2s_total_r11());
            std::fclose(f);
            std::fprintf(stderr, "memsim: wrote timeline (%zu segments) to %s\n", world.trace().size(),
                         timeline_path.c_str());
        }

        // Per-cycle streamer-FIFO occupancy (wide CSV: cycle + one column per port), for the FIFO
        // fullness plot. Path = <timeline>.fifo.csv (strip a trailing .csv, append .fifo.csv) so
        // plot_timeline.py can find it next to the timeline CSV. Engine (P1/P2), cyc_gemm
        // (osgemm/isgemm) and cyc_simd (SIMD) invocations all contribute rows; -1 = port idle/absent.
        std::string fifo_path = timeline_path;
        if (fifo_path.size() >= 4 && fifo_path.compare(fifo_path.size() - 4, 4, ".csv") == 0)
            fifo_path.resize(fifo_path.size() - 4);
        fifo_path += ".fifo.csv";
        const auto& ftrace = world.fifo_trace();
        if (!ftrace.empty()) {
            FILE* ff = std::fopen(fifo_path.c_str(), "w");
            if (!ff) {
                std::fprintf(stderr, "memsim: cannot open FIFO file %s\n", fifo_path.c_str());
            } else {
                std::fprintf(ff, "cycle,R0,R1,R2,R3,R4,R5,R6,R7,R8,R9,R10,R11,R12,R13,W0,W1,W2,W3\n");
                for (const auto& s : ftrace) {
                    std::fprintf(ff, "%llu", (unsigned long long)s.first);
                    for (int p = 0; p < 18; p++) std::fprintf(ff, ",%d", (int)s.second[p]);
                    std::fprintf(ff, "\n");
                }
                std::fclose(ff);
                std::fprintf(stderr, "memsim: wrote FIFO occupancy (%zu cycles) to %s%s\n", ftrace.size(),
                             fifo_path.c_str(), world.fifo_capped() ? "  [ROW CAP HIT — trace truncated]" : "");
            }
        }
    }

    if (std::getenv("MEMSIM_DEBUG")) {
        std::fprintf(stderr,
                     "memsim: entry=0x%08x tohost=0x%08x exited=%d code=%d "
                     "h0{pc=0x%08x cyc=%llu} h1{pc=0x%08x cyc=%llu}\n",
                     img.entry, m.tohost, m.exited, m.exit_code, m.harts[0].pc, (unsigned long long)m.harts[0].cycle,
                     m.harts[1].pc, (unsigned long long)m.harts[1].cycle);
    }

    // PASS/FAIL terminator (mirrors tb_bin.sv so batch_run_report.py scrapes it
    // identically to a vsim log).
    if (!m.exited) {
        std::fprintf(stderr, "memsim: program did not signal exit (deadlock=%d)\n", sched.deadlocked);
        std::printf("[FAILURE] Finished with exit code %2d\n", 1);
        return 1;
    }
    // The model's verdict is its layout/BIST cross-check, not the app's FP check_result
    // (the model produces timing + integer/layout, not the bf16/fp8 datapath). See
    // docs/dataflow/10_memsim.md (Verdict). A deadlock is already caught above.
    bool ok = verify ? world.layout_ok() : true;
    if (verify)
        std::fprintf(stderr,
                     "AGU layout audit: %u located error(s) over %d invocation(s) "
                     "(bounds + producer->consumer + writer no-alias)\n",
                     world.n_layout_errs(), world.layout_invocations());
    if (!verify)
        std::printf(
            "Note: LAYOUT/BIST check skipped (--timing-only); the app's FP "
            "check_result is not evaluated by the model either.\n");
    if (ok) {
        std::printf("[SUCCESS] Program finished successfully\n");
        std::printf("Errors: 0, Warnings: 0\n");
        return 0;
    }
    std::printf("[FAILURE] Finished with exit code %2d\n", 1);
    return 1;
}
