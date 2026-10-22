#!/usr/bin/env python3
"""Build rows.json (eval rows: jid, elf, vsim baselines, swept flag) and weights.json
(per-row mult / n_layer / model_mix, all user-editable) from the current batch config
+ report.json. Multipliers transcribed from batch_run_config.hjson's // xN comments."""
import json, os, sys
sys.path.insert(0, "/esat/micas-lapserv11/users/rgeens/snax_cluster/scripts")
import hjson
from batch_run import make_tag

ROOT = "/esat/micas-lapserv11/users/rgeens/snax_cluster"
CLUSTER = os.path.join(ROOT, "target", "snitch_cluster")
ELF_CACHE = os.path.join(ROOT, "batch_run_out", ".elf_cache")
SCRATCH = os.path.dirname(os.path.abspath(__file__))

# (app, name, model) -> (multiplier, n_layer), transcribed from the data excel (true weights).
# multiplier = stage_L / run_L; the CS deployment is assembled from L-scaled kernel runs, so
# Simba-S rows, fused-alternative apps (main-tiled*, p2-carry), einfft-A-CS, fft A/B and
# isgemm/dgcf "fft D" rows carry no weight. fft C folds three stages: 96*6 + 48*12 + 24*36.
MULT_NL = {
    ("P1-tiled-D", "P1 A", "Simba-L IN"): (83.6, 3),
    ("suc-carry", "SUC A", "Simba-L CS"): (2.0, 3),
    ("double-gemm-conflict-free", "P2 in proj A", "Simba-L IN"): (83.6, 3),
    ("double-gemm-conflict-free", "P2 out proj A", "Simba-L IN"): (163.8, 3),
    ("P1-tiled-D", "P1 B", "Simba-L IN"): (20.9, 6),
    ("suc-carry", "SUC B", "Simba-L CS"): (1, 6),
    ("double-gemm-conflict-free", "P2 in proj B", "Simba-L CS"): (16, 6),
    ("double-gemm-conflict-free", "P2 out proj B", "Simba-L CS"): (32, 6),
    ("P1-tiled-D", "P1 C", "Simba-L CS"): (8, 18),
    ("suc-carry", "SUC C", "Simba-L CS"): (1, 18),
    ("double-gemm-conflict-free", "P2 in proj C", "Simba-L CS"): (16, 18),
    ("double-gemm-conflict-free", "P2 out proj C", "Simba-L CS"): (32, 18),
    ("P1-tiled-D", "P1 D", "Simba-L CS"): (4, 3),
    ("suc-carry", "SUC D", "Simba-L CS"): (1, 3),
    ("double-gemm-conflict-free", "P2 in proj D", "Simba-L CS"): (16, 3),
    ("double-gemm-conflict-free", "P2 out proj D", "Simba-L CS"): (8, 3),
    ("fft-3way-tiled", "fft C", "Simba-L CS"): (2016, 1),
    ("fft-tiled", "fft D", "Simba-L CS"): (8, 6),
    ("einfft-tiled", "einfft A", "Simba-L IN"): (83.6, 3),
    ("einfft-double-conflictfree", "einfft B", "Simba-L IN"): (85.3, 6),
    ("einfft-double-conflictfree", "einfft C", "Simba-L IN"): (42.7, 18),
    ("einfft-double-conflictfree", "einfft D", "Simba-L IN"): (10.5, 3),
    ("double-gemm-conflict-free", "downsample A", "Simba-L IN"): (85.33333, 1),
    ("double-gemm-conflict-free", "downsample B", "Simba-L IN"): (42.66667, 1),
    ("conv-downsample", "downsample C", "Simba-L IN"): (32, 1),
    ("double-gemm-conflict-free", "segformer A", "Simba-L CS"): (256, 1),
    ("double-gemm-conflict-free", "segformer B", "Simba-L CS"): (64, 1),
    ("double-gemm-conflict-free", "segformer C", "Simba-L CS"): (32, 1),
    ("double-gemm-conflict-free", "segformer D", "Simba-L CS"): (8, 1),
    ("double-gemm-conflict-free", "segformer E", "Simba-L CS"): (1024, 1),
    ("rmsnorm-tiled", "rms A", "Simba-L IN"): (20.9, 6),
    ("rmsnorm-tiled", "rms B", "Simba-L IN"): (20.9, 12),
    ("rmsnorm-tiled", "rms C", "Simba-L IN"): (21.3, 36),
    ("rmsnorm-tiled", "rms D", "Simba-L IN"): (21.3, 6),
    ("batchnorm", "batchnorm", "Simba-L CS"): (1192, 1),
}

NOT_SWEPT_APPS = {"rmsnorm-tiled", "batchnorm"}  # SIMD-launch bound; memsim sc == 0

cfg = hjson.load(open(os.path.join(ROOT, "batch_run_config.hjson")))
report = json.load(open(os.path.join(ROOT, "report.json")))["jobs"]

jobs, seen = [], set()
for entry in cfg["runs"]:
    app = entry["app"]
    bpath = os.path.join(CLUSTER, "sw", "apps", app, "data", "params_in.hjson")
    base = hjson.load(open(bpath)) if os.path.exists(bpath) else {}
    for overrides in entry.get("params") or [{}]:
        o = dict(overrides)
        o.pop("force", None); name = o.pop("name", None); model = o.pop("model", None)
        tag = make_tag(o)
        jid = f"{app}__{tag}"
        n = 2
        while jid in seen:
            jid = f"{app}__{tag}__{n}"; n += 1
        seen.add(jid)
        jobs.append({"jid": jid, "app": app, "name": name, "model": model,
                     "params": {**base, **o}})

rows, weights = [], {}
missing_vsim, missing_elf = [], []
for j in jobs:
    key = (j["app"], j["name"], j["model"])
    mult, nl = MULT_NL.get(key, (0, 1))
    rep = report.get(j["jid"], {})
    vs = rep.get("simbacore"); vt = rep.get("total")
    vs = int(vs) if vs not in (None, "", "None") else None
    vt = int(vt) if vt not in (None, "", "None") else None
    elf = os.path.join(ELF_CACHE, j["jid"] + ".elf")
    if vt is None: missing_vsim.append(j["jid"])
    if not os.path.exists(elf): missing_elf.append(j["jid"])
    n_layer = nl
    swept = (j["app"] not in NOT_SWEPT_APPS) and mult > 0 and vt is not None
    rows.append({"jid": j["jid"], "app": j["app"], "name": j["name"], "model": j["model"],
                 "vsim_sc": vs, "vsim_total": vt, "swept": swept,
                 "excluded": mult == 0})
    weights[j["jid"]] = {"name": j["name"], "model": j["model"],
                         "mult": mult, "n_layer": n_layer, "model_mix": 1.0}

json.dump(rows, open(os.path.join(SCRATCH, "rows.json"), "w"), indent=1)
json.dump({"n_layer_defaults": "excel", "rows": weights},
          open(os.path.join(SCRATCH, "weights.json"), "w"), indent=1)
print(f"{len(rows)} rows ({sum(r['swept'] for r in rows)} swept, "
      f"{sum(r['excluded'] for r in rows)} excluded)")
print("missing vsim:", missing_vsim)
print("missing elf:", missing_elf)

# weighted totals per app (default scenario) for the report
tot = 0.0
per_app = {}
for r in rows:
    if r["vsim_total"] is None: continue
    w = weights[r["jid"]]
    cc = w["mult"] * w["n_layer"] * w["model_mix"] * r["vsim_total"]
    tot += cc
    per_app[r["app"]] = per_app.get(r["app"], 0) + cc
print(f"\nWeighted batch total (vsim baseline, default n_layer): {tot/1e6:.1f} Mcc")
for a, cc in sorted(per_app.items(), key=lambda x: -x[1]):
    print(f"  {a:<28} {cc/1e6:9.1f} Mcc  {cc/tot*100:5.1f}%")
