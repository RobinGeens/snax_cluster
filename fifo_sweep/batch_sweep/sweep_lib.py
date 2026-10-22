#!/usr/bin/env python3
"""FIFO-depth sweep library.

Rows      = current batch_run_config jobs, deduped (identical params; alternative
            implementations of the same (name, model) workload keep the vsim-cheapest).
Weights   = weights.json: per-row mult (from config comments) x n_layer(stage,model)
            x model_mix. All editable; two scenarios (default / n_layer=1) supported.
Objective = sum_r w_r * [vsim_total_r(d0) + model_sc_r(d) - model_sc_r(d0)]
            (depth deltas measured on the SimbaCore counter where memsim is accurate,
            overlaid on the vsim wall-clock baseline; SW overhead is depth-independent).
Cost      = sum_p depth_p * nch_p  (the established fifo_sweep convention).
"""
import json, os, re, subprocess
from concurrent.futures import ThreadPoolExecutor

ROOT = "/esat/micas-lapserv11/users/rgeens/snax_cluster"
CLUSTER = os.path.join(ROOT, "target", "snitch_cluster")
MEMSIM = os.path.join(CLUSTER, "bin", "snitch_cluster.memsim")
ELF_CACHE = os.path.join(ROOT, "batch_run_out", ".elf_cache")
SCRATCH = os.path.dirname(os.path.abspath(__file__))  # sweep home (repo: fifo_sweep/batch_sweep)
CACHE_PATH = os.path.join(SCRATCH, "eval_cache.json")

NCH = [2, 4, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 4, 4, 1, 1, 1, 4]  # R0..R13, W0..W3
PORT_NAMES = [f"R{i}" for i in range(14)] + [f"W{i}" for i in range(4)]
CURRENT = (8, 3, 8, 1, 6, 3, 2, 4, 1, 6, 7, 2, 6, 4, 4, 8, 3, 4)

def cost(d):
    return sum(x * w for x, w in zip(d, NCH))

# ---------------------------------------------------------------- rows
def load_rows():
    rows = json.load(open(os.path.join(SCRATCH, "rows.json")))
    for r in rows:
        if r["swept"] and not os.path.exists(os.path.join(ELF_CACHE, r["jid"] + ".elf")):
            print(f"load_rows: {r['jid']} elf missing -> treated as constant row")
            r["swept"] = False
    return rows

def run_memsim(elf, depths, timeout=120):
    env = dict(os.environ)
    env["MEMSIM_STREAMER_DEPTHS"] = ",".join(str(x) for x in depths)
    try:
        p = subprocess.run([MEMSIM, elf, "--timing-only"], cwd=CLUSTER, env=env,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None  # treat as divergent config
    txt = p.stdout + p.stderr
    sc = [int(m.group(1)) for m in re.finditer(r"Simbacore elapsed time:\s+(\d+)\s+cycles", txt)]
    if not sc or "[SUCCESS]" not in txt and "[FAILURE]" not in txt:
        return sc[-1] if sc else None
    return sc[-1]

class Evaluator:
    """model_sc per (jid, depth-vector), disk-cached, parallel."""
    def __init__(self, rows, workers=32):
        self.rows = {r["jid"]: r for r in rows}
        self.workers = workers
        # cache is only valid for the memsim binary it was produced with
        import hashlib
        self.binver = hashlib.md5(open(MEMSIM, "rb").read()).hexdigest()[:12]
        self.cache = {}
        if os.path.exists(CACHE_PATH):
            d = json.load(open(CACHE_PATH))
            if d.get("binver") == self.binver:
                self.cache = d.get("entries", {})
            else:
                print(f"eval cache: binary changed ({d.get('binver')} -> {self.binver}), purged")
        self.dirty = 0
        self.usage = None       # jid -> set(ports the row's sc depends on)
        self.mask_base = None   # depths substituted for masked (irrelevant) ports

    def set_usage(self, usage, mask_base):
        """Enable per-row masked cache keys: ports outside a row's usage set are
        canonicalized to mask_base, so vectors differing only there share entries."""
        self.usage = usage
        self.mask_base = tuple(mask_base)

    def key(self, jid, depths):
        if self.usage is not None and jid in self.usage:
            u = self.usage[jid]
            depths = tuple(d if i in u else self.mask_base[i] for i, d in enumerate(depths))
            return jid + "|m|" + ",".join(str(x) for x in depths)
        return jid + "|" + ",".join(str(x) for x in depths)

    def eval_rows(self, jids, depths):
        """Return {jid: model_sc or None}, running what's missing in parallel."""
        depths = tuple(depths)
        todo = [j for j in jids if self.key(j, depths) not in self.cache]
        if todo:
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                futs = {j: ex.submit(run_memsim,
                                     os.path.join(ELF_CACHE, self.rows[j]["jid"] + ".elf"),
                                     depths)
                        for j in todo}
                for j, f in futs.items():
                    self.cache[self.key(j, depths)] = f.result()
                    self.dirty += 1
            if self.dirty >= 50:
                self.save()
        return {j: self.cache[self.key(j, depths)] for j in jids}

    def save(self):
        json.dump({"binver": self.binver, "entries": self.cache}, open(CACHE_PATH, "w"))
        self.dirty = 0

# ---------------------------------------------------------------- objective
class Objective:
    def __init__(self, rows, weights, ev, scenario="default"):
        self.ev = ev
        self.scenario = scenario
        self.swept = [r for r in rows if r["swept"]]
        self.const = [r for r in rows if not r["swept"]]
        self.w = {}
        for r in rows:
            wr = weights["rows"][r["jid"]]
            nl = 1 if scenario == "flat" else wr["n_layer"]
            self.w[r["jid"]] = wr["mult"] * nl * wr["model_mix"]
        # baseline model_sc at CURRENT depths (offset calibration anchor)
        base = ev.eval_rows([r["jid"] for r in self.swept], CURRENT)
        self.base_sc = base
        self.const_cc = sum(self.w[r["jid"]] * r["vsim_total"] for r in self.const)
        self.swept_base_cc = sum(self.w[r["jid"]] * r["vsim_total"] for r in self.swept)

    def total(self, depths):
        """Weighted batch cycles; None if any swept row diverged/timed out."""
        sc = self.ev.eval_rows([r["jid"] for r in self.swept], depths)
        tot = self.const_cc
        for r in self.swept:
            j = r["jid"]
            if sc[j] is None or self.base_sc[j] is None:
                return None
            tot += self.w[j] * (r["vsim_total"] + sc[j] - self.base_sc[j])
        return tot

    def per_row_delta(self, depths):
        sc = self.ev.eval_rows([r["jid"] for r in self.swept], depths)
        out = {}
        for r in self.swept:
            j = r["jid"]
            out[j] = None if (sc[j] is None or self.base_sc[j] is None) else sc[j] - self.base_sc[j]
        return out
