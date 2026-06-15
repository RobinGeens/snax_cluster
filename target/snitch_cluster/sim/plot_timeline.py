#!/usr/bin/env python3
"""Plot a per-engine activity timeline from the memsim cycle-accurate model.

Usage:
  plot_timeline.py <app.elf> [-o out.png]      # run memsim, then plot
  plot_timeline.py --csv timeline.csv [-o out.png]

Draws one row per hardware engine (OSCORE, ISCORE, SUC, switchCore, DMA) with the
cycle count on the x-axis, shading each cycle window the engine is active — so you can
see *when* and *how long* each module runs on the schedule.

Each row is annotated with its average hardware utilization: of the cycles the module
was active, how much real work it did. Utilization = ideal / actual, where `ideal` is
the shortest possible cycle count for that block's compute (its MAC-group count at peak,
conflict- and drain-free) and `actual` is the time the block actually took. A conflict-
free array is ~100%; the SUC's bank conflict pulls it down (~57% at bc_pad=0); a GEMM's
output drain and the DMA first-beat latency also drop it below 100%.

If memsim also wrote the per-cycle FIFO CSV next to the timeline (`<name>.fifo.csv`, one
row per cycle, one column per streamer port), a second subplot is added to the SAME figure,
sharing the cycle x-axis: the streamer-FIFO fullness over time (y = elements in the FIFO),
one coloured line per FIFO, readers solid and writers dotted. Living in timeline.png means it
shows up wherever the timeline does (e.g. the batch-run report) with no extra wiring.

The windows + ideal counts come from `memsim --timeline <csv>` (see src/world.cpp); the
timing model itself is documented in docs/dataflow/10_memsim.md.
"""
import argparse
import csv
import os
import subprocess
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Streamer ports in CSV column order: 14 readers R0..R13 then 4 writers W0..W3.
PORT_NAMES = [f"R{i}" for i in range(14)] + [f"W{i}" for i in range(4)]

# Row order + colour, top to bottom.
ENGINES = [
    ("OSCORE", "#1f77b4"),
    ("SUC", "#2ca02c"),
    ("ISCORE", "#ff7f0e"),
    ("SWITCHCORE", "#9467bd"),
    ("DMA", "#7f7f7f"),
]

# TCDM peak = 32 banks x 1 narrow (8-byte) access per cycle. The bandwidth line below the
# engine rows shows the combined streamer + DMA word demand as a fraction of this.
TCDM_PEAK_WORDS_PER_CYC = 32


def split_label(raw):
    """Turn a timeline filename into (app, params_str) for the title.

    Batch filenames look like `<app>__<tag>.timeline.csv`, where <tag> crams every
    override together with underscores (`bc_pad_banks0_dModel384_...`). For a clean
    title we keep the app name and, lacking the real key/value split, hand back the
    raw tag (callers that know the overrides pass them explicitly via --params)."""
    base = raw
    for suf in (".timeline.csv", ".timeline.png", ".timeline", ".csv", ".elf"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    if "__" in base:
        app, tag = base.split("__", 1)
        return app, tag
    return base, ""


def memsim_bin():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.environ.get("MEMSIM_BIN", os.path.join(here, "..", "bin", "snitch_cluster.memsim"))


def run_memsim(elf, csv_path):
    bin_ = memsim_bin()
    if not os.path.exists(bin_):
        sys.exit(f"memsim binary not found: {bin_} (build with `make -C sim`)")
    r = subprocess.run([bin_, elf, "--timeline", csv_path], capture_output=True, text=True)
    if not os.path.exists(csv_path):
        sys.stderr.write(r.stderr)
        sys.exit(f"memsim did not produce a timeline (exit {r.returncode})")


def load_segments(csv_path):
    # per engine: list of (start, end, ideal); plus TCDM traffic: list of (start, end, words);
    # plus optimal safe-to-start delays (S2S_R10/S2S_R11 metadata rows: start=optimal, end=total).
    segs = {name: [] for name, _ in ENGINES}
    tcdm = []
    s2s = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            eng = row["engine"]
            rec = (int(row["start"]), int(row["end"]), int(row["ideal"]))
            if eng == "TCDM":
                tcdm.append(rec)
            elif eng in ("S2S_R10", "S2S_R11"):
                s2s[eng] = (int(row["start"]), int(row["end"]))  # (optimal, total)
            elif eng in segs:
                segs[eng].append(rec)
    return segs, tcdm, s2s


def bandwidth_curve(tcdm_segs, peak):
    """Piecewise-constant TCDM bandwidth % over time.

    Each segment spreads its `words` uniformly over [start,end] (rate = words/(end-start));
    overlapping segments (e.g. DMA during compute) sum. Returns (xs, ys, avg_pct) where
    (xs,ys) are step-line points in % and avg_pct is total words / total span / peak.
    """
    if not tcdm_segs:
        return [], [], 0.0
    bounds = sorted({s for s, _, _ in tcdm_segs} | {e for _, e, _ in tcdm_segs})
    xs, ys = [], []
    for a, b in zip(bounds, bounds[1:]):
        if b <= a:
            continue
        rate = sum(w / (e - s) for s, e, w in tcdm_segs if s <= a and e >= b)
        pct = 100.0 * rate / peak
        xs += [a, b]
        ys += [pct, pct]
    span = bounds[-1] - bounds[0] or 1
    avg = 100.0 * sum(w for _, _, w in tcdm_segs) / span / peak
    return xs, ys, avg


def summarize(intervals):
    """Merge overlapping bars (for drawing) and total actual/ideal cycles (for utilization).

    Returns (merged_bars, active_cycles, ideal_cycles). Each block keeps its own ideal floor,
    so utilization = ideal_cycles / active_cycles is the average over the engine's active time.
    """
    if not intervals:
        return [], 0, 0
    active = sum(e - s for s, e, _ in intervals)  # blocks per engine are disjoint in time
    ideal = sum(i for _, _, i in intervals)
    bars = [list(iv[:2]) for iv in sorted(intervals)]
    out = [bars[0]]
    for s, e in bars[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out, active, ideal


def fifo_csv_path(timeline_csv):
    """Per-cycle FIFO CSV that memsim writes next to the timeline CSV (main.cpp uses the
    same rule: strip a trailing `.csv`, append `.fifo.csv`)."""
    base = timeline_csv[:-4] if timeline_csv.endswith(".csv") else timeline_csv
    return base + ".fifo.csv"


# A run of tens/hundreds of thousands of cycles drawn at ~1 px/cycle turns the per-cycle FIFO
# push/pop sawtooth into a solid colour band. Block-average the occupancy into this many bins
# across the run so the lines stay readable; the bin width (= smoothing horizon) therefore scales
# with total runtime. Small runs (span <= this) keep bin width 1 = the exact per-cycle trace.
FIFO_TARGET_BINS = 500


def load_fifo(path, target_bins=FIFO_TARGET_BINS):
    """Read the wide FIFO CSV (`cycle,R0..R13,W0..W3`) into (x, {port: y}, bin_width).

    -1 (port idle/absent that cycle) is dropped from the average so a line shows only while its port
    is live. For long runs the per-cycle occupancy is averaged into fixed cycle-width bins (bin_width =
    max(1, span/target_bins)) — the mean fullness per bin — so the rapid sawtooth becomes a readable
    curve; a bin with no live samples for a port is NaN so its line breaks across idle stretches. When
    the run is short enough that bin_width == 1 this is the exact per-cycle trace (with gap breaks).
    Only ports that are ever active are returned."""
    with open(path) as f:
        rd = csv.reader(f)
        ports = next(rd)[1:]
        rows = list(rd)
    if not rows:
        return None, {}, 1
    data = np.array(rows, dtype=np.int64)
    cyc, vals = data[:, 0], data[:, 1:].astype(float)
    vals[vals < 0] = np.nan  # port idle that cycle -> excluded from the mean
    active = [i for i in range(vals.shape[1]) if np.isfinite(vals[:, i]).any()]
    if not active:
        return None, {}, 1
    cyc0, cyc1 = int(cyc[0]), int(cyc[-1])
    span = cyc1 - cyc0
    bw = max(1, span // target_bins)
    if bw <= 1:  # short run: exact per-cycle trace, NaN-break the idle gaps so lines don't bridge them
        x = cyc.astype(float)
        ins = np.where(np.diff(cyc) > 1)[0] + 1
        series = {}
        for i in active:
            col = vals[:, i]
            series[ports[i]] = np.insert(col, ins, np.nan) if ins.size else col
        if ins.size:
            x = np.insert(x, ins, np.nan)
        return x, series, 1
    # long run: mean occupancy per cycle-bin (np.bincount sum/count, NaN where a port has no samples)
    b = ((cyc - cyc0) // bw).astype(np.int64)
    nbins = int(b[-1]) + 1
    xs = cyc0 + (np.arange(nbins) + 0.5) * bw
    series = {}
    for i in active:
        col = vals[:, i]
        m = np.isfinite(col)
        cnt = np.bincount(b[m], minlength=nbins).astype(float)
        tot = np.bincount(b[m], weights=col[m], minlength=nbins)
        series[ports[i]] = np.where(cnt > 0, tot / np.where(cnt > 0, cnt, 1), np.nan)
    return xs, series, bw


def draw_fifo(ax, path):
    """Draw the streamer-FIFO occupancy onto `ax` (the timeline's lower subplot, sharing the cycle
    x-axis): y = elements in the FIFO, one thin coloured line per FIFO (port), readers solid and writers
    dotted. Both are the physical occupancy (0..depth): readers = landed-not-consumed data-FIFO entries;
    writers = produced-not-drained output entries (the model back-pressures the core when the writer FIFO
    is full, so it never exceeds depth). For long runs the per-cycle trace is averaged into cycle-bins
    (see load_fifo) to stay legible. Returns the bin width (>=1) if drawn, else 0."""
    x, series, bw = load_fifo(path)
    if x is None or not series:
        return 0
    cmap = plt.get_cmap("tab20")
    for name, y in series.items():
        idx = PORT_NAMES.index(name)
        writer = name.startswith("W")
        ax.plot(x, y, color=cmap(idx % 20), linestyle=":" if writer else "-",
                lw=1.0 if writer else 0.8, label=name)
    ax.set_ylabel("elements in FIFO" + (f"\n(mean / {bw} cc)" if bw > 1 else ""))
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=7, ncol=3, framealpha=0.9,
              title="solid = reader   ·   dotted = writer", title_fontsize=7)
    return bw


def s2s_csv_path(timeline_csv):
    """Safe-to-start sweep CSV memsim writes next to the timeline CSV (main.cpp: strip a trailing
    `.csv`, append `.s2s.csv`)."""
    base = timeline_csv[:-4] if timeline_csv.endswith(".csv") else timeline_csv
    return base + ".s2s.csv"


def load_s2s_sweep(path):
    """Read the S2S sweep CSV (`gate,value,stale,optimal,total`) into ({R10:[(v,stale)]}, opt, tot).

    Two gates: R10 (z, osCore->SUC) and R11 (y, SUC->isCore). Each row is one gate value with the
    predicted read-before-write count at that release point. optimal/total are repeated per row."""
    curves = {"R10": [], "R11": []}
    opt, tot = {}, {}
    with open(path) as f:
        for row in csv.DictReader(f):
            g = row["gate"]
            curves[g].append((int(row["value"]), int(row["stale"])))
            opt[g] = int(row["optimal"])
            tot[g] = int(row["total"])
    return curves, opt, tot


# R10 (z-gate) and R11 (y-gate) live on the same y-axis (predicted stale reads) but on two very
# differently scaled x-axes (R10 ~ tens of osCore tiles; R11 ~ thousands of y-elements). R11 is the
# primary (bottom) axis, R10 a twinned top axis.
S2S_C10, S2S_C11 = "#ff7f0e", "#1f4e79"


def draw_s2s(ax, curves, opt, tot):
    """Draw the safe-to-start hazard curves onto `ax`: y = predicted read-before-write count, x =
    release gate. R11 (y, SUC->isCore) on the bottom axis, R10 (z, osCore->SUC) on a twinned top axis
    (different scale, same y). Optimal release for each gate marked with a dashed vline. Returns True
    if anything was drawn."""
    r11 = curves.get("R11") or []
    r10 = curves.get("R10") or []
    if not r11 and not r10:
        return False
    handles = []
    if r11:
        x, y = zip(*r11)
        (h,) = ax.plot(x, y, "-", color=S2S_C11, lw=1.8, label="R11 (y): SUC→isCore")
        handles.append(h)
        ax.fill_between(x, y, alpha=0.10, color=S2S_C11)
        if "R11" in opt:
            ax.axvline(opt["R11"], color=S2S_C11, ls="--", lw=1.3)
            ax.text(opt["R11"], ax.get_ylim()[1], f" opt R11={opt['R11']}/{tot.get('R11','?')}",
                    color=S2S_C11, fontsize=7.5, va="top", ha="left")
        ax.set_xlim(0, tot.get("R11", x[-1]))
        ax.set_xlabel("R11 (y) gate — SUC y-elements produced before isCore release", color=S2S_C11, fontsize=9)
        ax.tick_params(axis="x", colors=S2S_C11)
    ax.set_ylabel("predicted stale reads\n(read-before-write)", fontsize=9)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle=":", alpha=0.4)
    if r10:
        axt = ax.twiny()
        x, y = zip(*r10)
        (h,) = axt.plot(x, y, "-", color=S2S_C10, lw=1.8, label="R10 (z): osCore→SUC")
        handles.append(h)
        axt.fill_between(x, y, alpha=0.10, color=S2S_C10)
        if "R10" in opt:
            axt.axvline(opt["R10"], color=S2S_C10, ls="--", lw=1.3)
            axt.text(opt["R10"], axt.get_ylim()[1], f"opt R10={opt['R10']}/{tot.get('R10','?')} ",
                     color=S2S_C10, fontsize=7.5, va="top", ha="right")
        axt.set_xlim(0, tot.get("R10", x[-1]))
        axt.set_xlabel("R10 (z) gate — osCore z-tiles produced before SUC release", color=S2S_C10, fontsize=9)
        axt.tick_params(axis="x", colors=S2S_C10)
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9,
              title="safe-to-start sweep (per P2 tile)", title_fontsize=8)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("elf", nargs="?", help="app .elf to run through memsim")
    ap.add_argument("--csv", help="use an existing timeline CSV instead of running memsim")
    ap.add_argument("-o", "--out", help="output PNG (default: <elf|csv>_timeline.png)")
    ap.add_argument("--name", help="app name for the title (default: parsed from the filename)")
    ap.add_argument("--params", help="pre-formatted params line for the title (e.g. 'dModel=384  seqLen=192')")
    args = ap.parse_args()

    if args.csv:
        csv_path = args.csv
        label = os.path.basename(csv_path)
        out = args.out or os.path.splitext(csv_path)[0] + "_timeline.png"
    elif args.elf:
        label = os.path.basename(args.elf)
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        csv_path = tmp.name
        tmp.close()
        run_memsim(args.elf, csv_path)
        out = args.out or os.path.splitext(args.elf)[0] + "_timeline.png"
    else:
        ap.error("provide an .elf or --csv")

    segs, tcdm, s2s = load_segments(csv_path)
    summ = {name: summarize(segs[name]) for name, _ in ENGINES}

    starts = [s for name, _ in ENGINES for s, _, _ in segs[name]] + [s for s, _, _ in tcdm]
    ends = [e for name, _ in ENGINES for _, e, _ in segs[name]] + [e for _, e, _ in tcdm]
    if not starts:
        sys.exit("timeline is empty (no engine activity recorded)")
    t0, t1 = min(starts), max(ends)
    span = t1 - t0 or 1

    # The TCDM bandwidth gets its own row below DMA: a 0-100% line drawn inside a band of
    # height BW_H, with reference gridlines at 0/50/100%.
    BW_BASE, BW_H = -1.8, 1.0
    # If memsim emitted the per-cycle FIFO CSV next to the timeline, draw the streamer-FIFO occupancy
    # as a second subplot sharing the cycle x-axis -> it lands inside the same timeline.png the batch
    # report already links, so no separate figure / report column is needed.
    fifo_path = fifo_csv_path(csv_path)
    have_fifo = os.path.exists(fifo_path)
    # Safe-to-start sweep subplot (independent gate x-axes, NOT the cycle axis): added as the bottom
    # row, full width, whenever memsim emitted the .s2s.csv.
    s2s_path = s2s_csv_path(csv_path)
    s2s_curves, s2s_opt, s2s_tot = load_s2s_sweep(s2s_path) if os.path.exists(s2s_path) else ({}, {}, {})
    have_s2s = bool(s2s_curves.get("R10") or s2s_curves.get("R11"))

    heights = [3.6] + ([3.2] if have_fifo else []) + ([3.0] if have_s2s else [])
    fig = plt.figure(figsize=(13, sum(heights) + 0.4))
    gs = fig.add_gridspec(len(heights), 1, height_ratios=heights, hspace=0.42)
    ax = fig.add_subplot(gs[0, 0])
    row = 1
    ax_fifo = fig.add_subplot(gs[row, 0], sharex=ax) if have_fifo else None
    if have_fifo:
        row += 1
    ax_s2s = fig.add_subplot(gs[row, 0]) if have_s2s else None  # own gate axes, not shared with cycle x
    yticks, ylabels = [], []
    for i, (name, color) in enumerate(ENGINES):
        y = len(ENGINES) - 1 - i  # OSCORE on top
        bars, active, ideal = summ[name]
        ax.broken_barh([(s, e - s) for s, e in bars], (y - 0.4, 0.8), facecolors=color, edgecolor="none")
        if active:
            util = 100.0 * ideal / active
            ax.text(
                t1 + 0.01 * span,
                y,
                f"util {util:5.1f}%  (active {active} cc)",
                va="center",
                ha="left",
                fontsize=9,
                family="monospace",
            )
        else:
            ax.text(
                t1 + 0.01 * span,
                y,
                "    —   (inactive)",
                va="center",
                ha="left",
                fontsize=9,
                family="monospace",
                color="0.5",
            )
        yticks.append(y)
        ylabels.append(name)

    # TCDM bandwidth line.
    xs, ys, bw_avg = bandwidth_curve(tcdm, TCDM_PEAK_WORDS_PER_CYC)
    for frac in (0.0, 0.5, 1.0):  # 0/50/100% reference lines + ticks
        yref = BW_BASE + frac * BW_H
        ax.plot([t0, t1], [yref, yref], color="0.8", lw=0.6, zorder=1)
        ax.text(
            t0 - 0.008 * span, yref, f"{int(frac * 100)}%", va="center", ha="right", fontsize=7, color="0.55", zorder=4
        )
    if xs:
        yline = [BW_BASE + min(p, 100.0) / 100.0 * BW_H for p in ys]
        ax.plot(xs, yline, color="#d62728", lw=1.1, zorder=3, solid_joinstyle="miter")
    ax.text(
        t1 + 0.01 * span,
        BW_BASE + 0.5 * BW_H,
        f"avg {bw_avg:5.1f}%",
        va="center",
        ha="left",
        fontsize=9,
        family="monospace",
        color="#d62728",
    )
    yticks.append(BW_BASE + 0.5 * BW_H)
    ylabels.append("TCDM BW")

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_ylim(BW_BASE - 0.25, len(ENGINES) - 0.3)
    ax.set_xlim(t0 - 0.05 * span, t1 + 0.30 * span)  # left margin for the TCDM BW % ticks
    (ax_fifo or ax).set_xlabel("cycle (cc)")  # x label on the lower subplot (FIFO) when present
    if ax_fifo is not None:
        ax.tick_params(labelbottom=False)  # x ticks live on the shared lower (FIFO) subplot only
    parsed_app, parsed_params = split_label(label)
    app = args.name or parsed_app
    params_line = args.params or parsed_params
    title = f"memsim timeline — {app}   ·   total runtime {span} cc"
    if params_line:
        title += f"\n{params_line}"
    if s2s:
        title += "\noptimal S2S: " + "   ".join(
            f"{gate} = {opt}/{tot}"
            for key, gate in (("S2S_R10", "R10"), ("S2S_R11", "R11"))
            if key in s2s
            for opt, tot in (s2s[key],)
        )
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    bw = draw_fifo(ax_fifo, fifo_path) if ax_fifo is not None else 0
    if bw:
        note = "FIFO: elements physically in each FIFO (0..depth); a streamer near its depth is stalling the core"
        if bw > 1:
            note += f"  ·  per-cycle occupancy averaged over {bw}-cycle bins for legibility"
        fig.text(0.01, 0.005, note, fontsize=7, color="0.45")

    if ax_s2s is not None:
        draw_s2s(ax_s2s, s2s_curves, s2s_opt, s2s_tot)

    # gridspec hspace is set explicitly (twiny top labels need the room); keep it rather than letting
    # tight_layout recompute the row spacing — just reserve the bottom strip for the FIFO note.
    fig.subplots_adjust(left=0.07, right=0.985, top=0.93, bottom=0.06 if (bw or ax_s2s is not None) else 0.04)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")

    for name, _ in ENGINES:
        _, active, ideal = summ[name]
        if active:
            print(f"  {name:11s} util {100.0 * ideal / active:5.1f}%  " f"(active {active} cc, ideal {ideal} cc)")
        else:
            print(f"  {name:11s}   —   (inactive)")
    print(f"  {'TCDM BW':11s} avg  {bw_avg:5.1f}%  of {TCDM_PEAK_WORDS_PER_CYC} words/cyc peak")
    for key, gate in (("S2S_R10", "R10 (z)"), ("S2S_R11", "R11 (y)")):
        if key in s2s:
            opt, tot = s2s[key]
            print(f"  optimal S2S {gate:8s} {opt}/{tot}")


if __name__ == "__main__":
    main()
