#!/usr/bin/env python3
"""FIFO-depth sweep driver.

Subcommands:
  parity   -- fresh memsim vs vsim on every row (model quality table)
  oat      -- one-at-a-time screen: per port, +1 and drop-to-1 vs CURRENT.
              Produces oat.json: per-port weighted gradients + per-row usage map.
  greedy   -- frontier: greedy-add from all-1 and greedy-remove from CURRENT,
              union-Pareto, writes frontier.json + step traces.
  refine   -- +1/-1 pairwise swap refinement at selected frontier budgets.
  report   -- final Pareto table (both weight scenarios).
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep_lib import (Evaluator, Objective, load_rows, cost, run_memsim,
                       CURRENT, NCH, PORT_NAMES, ELF_CACHE, SCRATCH)

WEIGHTS = json.load(open(os.path.join(SCRATCH, "weights.json")))

def swept_rows(rows):
    return [r for r in rows if r["swept"]]

# ---------------------------------------------------------------- oat
def cmd_oat():
    rows = load_rows()
    ev = Evaluator(swept_rows(rows))
    obj = Objective(swept_rows(rows) + [r for r in rows if not r["swept"] and not r["excluded"]
                                        and r["vsim_total"] is not None],
                    WEIGHTS, ev)
    base_total = obj.total(CURRENT)
    print(f"baseline weighted total: {base_total/1e6:.3f} Mcc  cost={cost(CURRENT)}")
    usage = {r["jid"]: [] for r in obj.swept}          # ports each row responds to
    grads = {}
    for p in range(18):
        for delta, tag in ((+1, "up"), (None, "drop1")):
            d = list(CURRENT)
            d[p] = d[p] + 1 if tag == "up" else 1
            if tuple(d) == CURRENT:
                grads[f"{PORT_NAMES[p]}:{tag}"] = 0.0
                continue
            deltas = obj.per_row_delta(tuple(d))
            tot = 0.0
            for r in obj.swept:
                j = r["jid"]
                dv = deltas[j]
                if dv is None:                          # diverged => huge
                    tot = float("inf"); break
                tot += obj.w[j] * dv
                if dv != 0 and p not in usage[j]:
                    usage[j].append(p)
            grads[f"{PORT_NAMES[p]}:{tag}"] = tot
            print(f"  {PORT_NAMES[p]:>4} {tag:>5}: {tot/1e3:+12.1f} kcc "
                  f"(cost {'+' if tag=='up' else '->'}{NCH[p] if tag=='up' else d[p]*NCH[p]})",
                  flush=True)
    ev.save()
    json.dump({"base_total": base_total, "grads": grads, "usage": usage},
              open(os.path.join(SCRATCH, "oat.json"), "w"), indent=1)
    print("wrote oat.json")

# ---------------------------------------------------------------- greedy
def active_ports(oat):
    """Ports worth touching: any up-gain or any drop-to-1 loss."""
    act = set()
    for p in range(18):
        up = oat["grads"].get(f"{PORT_NAMES[p]}:up", 0.0)
        dr = oat["grads"].get(f"{PORT_NAMES[p]}:drop1", 0.0)
        if up < 0 or dr != 0.0:
            act.add(p)
    return sorted(act)

def cmd_greedy():
    rows = load_rows()
    oat = json.load(open(os.path.join(SCRATCH, "oat.json")))
    ev = Evaluator(swept_rows(rows))
    ev.set_usage({j: set(ps) for j, ps in oat["usage"].items()}, CURRENT)
    allrows = swept_rows(rows) + [r for r in rows if not r["swept"] and not r["excluded"]
                                  and r["vsim_total"] is not None]
    obj = Objective(allrows, WEIGHTS, ev)
    act = active_ports(oat)
    print("active ports:", [PORT_NAMES[p] for p in act])
    MAXD = 9
    trace = []

    def record(d, tot, phase):
        trace.append({"depths": list(d), "cost": cost(d), "total": tot, "phase": phase})

    # -- greedy-add from the functional floor. R7 is pinned >= 4: the engine models the R7
    # data read-ahead as 1x fifo_depth (documented compromise), so the 4-group BC refresh
    # deadlocks below 4 -- the model cannot honestly evaluate R7 in {1,2,3}.
    R7_MIN = 4
    d = [1] * 18
    for p, v in ((2, 2), (7, R7_MIN), (12, 2), (13, 2), (17, 2)):   # R2, R7, R12, R13, W3
        d[p] = v
    tot = obj.total(tuple(d))
    record(d, tot, "add")
    print(f"[add] start cost={cost(d)} total={'inf' if tot is None else round((tot or 0)/1e6,2)}")
    flat_run = 0
    while True:
        best = None
        for p in act:
            if d[p] >= MAXD: continue
            cand = list(d); cand[p] += 1
            t = obj.total(tuple(cand))
            if t is None: continue
            gain = (tot - t) if tot is not None else float("inf") if t is not None else 0
            per_cost = gain / NCH[p]
            if best is None or per_cost > best[0]:
                best = (per_cost, p, t)
        if best is None: break
        per_cost, p, t = best
        if tot is not None and per_cost <= 0 and cost(d) > cost(CURRENT) + 30:
            break                                       # flat top reached past current budget
        d[p] += 1; tot = t
        record(d, tot, "add")
        print(f"[add] +{PORT_NAMES[p]} -> cost={cost(d)} total={round(tot/1e6,3)} Mcc "
              f"(gain/cost {per_cost:.0f})", flush=True)
        ev.save()
        # The add path serves the LOW-cost frontier only; the remove path (from the known-good
        # CURRENT joint operating point) owns the mid/high-cost region. Stop crawling once the
        # marginal gain collapses -- coordinate ascent can't climb the R12/R13/W3 joint step.
        flat = tot is not None and per_cost < 15000
        flat_run = flat_run + 1 if flat else 0
        if flat_run >= 3 or cost(d) >= 210: break

    # -- greedy-remove from CURRENT ---------------------------------------
    d = list(CURRENT)
    tot = obj.total(tuple(d))
    record(d, tot, "remove")
    print(f"[rm] start cost={cost(d)} total={round(tot/1e6,3)}")
    while True:
        best = None
        for p in range(18):
            if d[p] <= (R7_MIN if p == 7 else 1): continue
            cand = list(d); cand[p] -= 1
            t = obj.total(tuple(cand))
            if t is None: continue
            loss = t - tot
            per_cost = loss / NCH[p]                    # cycles lost per cost unit saved
            if best is None or per_cost < best[0]:
                best = (per_cost, p, t)
        if best is None: break
        per_cost, p, t = best
        d[p] -= 1; tot = t
        record(d, tot, "remove")
        print(f"[rm] -{PORT_NAMES[p]} -> cost={cost(d)} total={round(tot/1e6,3)} Mcc "
              f"(loss/cost {per_cost:.0f})", flush=True)
        ev.save()
        if cost(d) <= 34: break

    ev.save()
    json.dump(trace, open(os.path.join(SCRATCH, "greedy_trace.json"), "w"), indent=1)
    print("wrote greedy_trace.json")

# ---------------------------------------------------------------- refine
def pareto(points):
    """points: list of dicts with cost/total. Keep non-dominated (min cost, min total)."""
    pts = sorted((p for p in points if p["total"] is not None),
                 key=lambda x: (x["cost"], x["total"]))
    out, best = [], float("inf")
    for p in pts:
        if p["total"] < best - 1e-9:
            out.append(p); best = p["total"]
    return out

def cmd_refine():
    rows = load_rows()
    oat = json.load(open(os.path.join(SCRATCH, "oat.json")))
    trace = json.load(open(os.path.join(SCRATCH, "greedy_trace.json")))
    ev = Evaluator(swept_rows(rows))
    ev.set_usage({j: set(ps) for j, ps in oat["usage"].items()}, CURRENT)
    allrows = swept_rows(rows) + [r for r in rows if not r["swept"] and not r["excluded"]
                                  and r["vsim_total"] is not None]
    obj = Objective(allrows, WEIGHTS, ev)
    act = active_ports(oat)
    front = pareto(trace)
    # ~6 budgets: spread over the frontier + the region just below CURRENT cost
    budgets = sorted({p["cost"] for p in front})
    cur_c = cost(CURRENT)
    lo, hi = budgets[0], budgets[-1]
    sel = sorted({lo, lo + (hi - lo) // 4, lo + (hi - lo) // 2, lo + 3 * (hi - lo) // 4, hi,
                  max(b for b in budgets if b <= cur_c)})
    sel = [min(budgets, key=lambda b, s=s: abs(b - s)) for s in sel]
    sel = sorted(set(sel))
    refined = list(trace)
    for b in sel:
        cands = [p for p in front if p["cost"] <= b]
        if not cands: continue
        start = max(cands, key=lambda p: p["cost"])
        d, tot = list(start["depths"]), start["total"]
        improved, rounds = True, 0
        while improved and rounds < 6:
            improved = False; rounds += 1
            for pi in act:
                for qi in range(18):
                    if pi == qi or d[qi] <= (4 if qi == 7 else 1) or d[pi] >= 9: continue
                    cand = list(d); cand[pi] += 1; cand[qi] -= 1
                    if cost(cand) > b: continue
                    t = obj.total(tuple(cand))
                    if t is not None and t < tot - 1:
                        d, tot = cand, t; improved = True
            ev.save()
        refined.append({"depths": d, "cost": cost(d), "total": tot, "phase": f"refine@{b}"})
        print(f"[refine b={b}] cost={cost(d)} total={round(tot/1e6,3)} Mcc", flush=True)
    ev.save()
    json.dump(refined, open(os.path.join(SCRATCH, "refined_trace.json"), "w"), indent=1)
    print("wrote refined_trace.json")

if __name__ == "__main__":
    {"oat": cmd_oat, "greedy": cmd_greedy, "refine": cmd_refine}[sys.argv[1]]()
