#!/usr/bin/env python3
"""Score every evaluated depth vector (points.json + any *_trace.json from later sweep
runs) under the weights in weights.json and write ../fifo_sweep_summary_batch_pareto.txt:
the model Pareto front plus always-included reference rows (current config, the
vsim-validated config, salvage variants). Evaluations come from eval_cache.json when
present; missing ones re-run memsim (fast, no license). Run build_rows.py first if
rows.json / weights.json are missing."""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep_lib import Evaluator, Objective, load_rows, cost, CURRENT, SCRATCH

OUT = os.path.join(os.path.dirname(SCRATCH), "fifo_sweep_summary_batch_pareto.txt")
# phases always shown in the summary, with their note label
REF_NOTES = {"current": "current config", "trio-vsim": "vsim-validated (recommended)",
             "trio-r7-5": "vsim base + R7=5 (model)", "trio-r7-6": "vsim base + R7=6 (model)",
             "salvage-strict": "zero-risk cuts only", "salvage-r7": "zero-risk + R7=5 (model)"}


def main():
    rows = load_rows()
    weights = json.load(open(os.path.join(SCRATCH, "weights.json")))
    points = json.load(open(os.path.join(SCRATCH, "points.json")))
    seen = {tuple(p["depths"]) for p in points}
    for f in glob.glob(os.path.join(SCRATCH, "*_trace.json")):   # later sweep runs
        for p in json.load(open(f)):
            if tuple(p["depths"]) not in seen:
                seen.add(tuple(p["depths"]))
                points.append({"depths": p["depths"], "phase": p.get("phase", "")})
    if tuple(CURRENT) not in seen:
        points.append({"depths": list(CURRENT), "phase": "current"})
    for p in points:
        if tuple(p["depths"]) == tuple(CURRENT):
            p["phase"] = "current"

    ev = Evaluator([r for r in rows if r["swept"]])
    oat_path = os.path.join(SCRATCH, "oat.json")
    if os.path.exists(oat_path):   # usage map only speeds up caching; optional
        oat = json.load(open(oat_path))
        ev.set_usage({j: set(ps) for j, ps in oat["usage"].items()}, CURRENT)
    allrows = [r for r in rows if not r["excluded"] and r["vsim_total"] is not None]
    obj = Objective(allrows, weights, ev)

    for p in points:
        p["cost"] = cost(p["depths"])
        p["total"] = obj.total(tuple(p["depths"]))
    ev.save()

    cur_t = next(p["total"] for p in points if p["phase"] == "current")
    # model Pareto front (min total per cost, ascending)
    front, best = [], float("inf")
    for p in sorted((q for q in points if q["total"] is not None),
                    key=lambda q: (q["cost"], q["total"])):
        if p["total"] < best - 1e-9:
            front.append(p)
            best = p["total"]
    shown = {id(p) for p in front}
    for p in points:
        if p["phase"] in REF_NOTES and id(p) not in shown and p["total"] is not None:
            front.append(p)
    front.sort(key=lambda p: (p["cost"], p["total"]))

    lines = []
    hdr = (f"{'reader_fifo_depth':<29} {'writer_fifo_depth':<18} {'cost':>5} "
           f"{'batch_Mcc':>10} {'d_vs_cur':>9}  note")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for p in front:
        r = ",".join(str(x) for x in p["depths"][:14])
        w = ",".join(str(x) for x in p["depths"][14:])
        d = (p["total"] - cur_t) / cur_t * 100
        note = REF_NOTES.get(p["phase"], "")
        if not note and p["phase"].startswith("may"):
            note = "May main-full-optimal"
        # thinning R0/R1/W0 below current is where vsim showed the model is optimistic
        if not note and any(p["depths"][i] < CURRENT[i] for i in (0, 1, 14)):
            note = "model-optimistic (thin R0/R1/W0 unvalidated)"
        lines.append(f"{r:<29} {w:<18} {p['cost']:>5} {p['total']/1e6:>10.2f} "
                     f"{d:>+8.2f}%  {note}")
    txt = "\n".join(lines) + "\n"
    open(OUT, "w").write(txt)
    print(txt)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
