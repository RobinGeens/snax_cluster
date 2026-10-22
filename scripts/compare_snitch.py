#!/usr/bin/env python3
"""Compare Snitch (total) latencies between two batch-run report JSONs.

Usage: compare_snitch.py [current.json] [baseline.json]   (defaults: report.json report_bak.json)
Joins jobs by their key; prints per-job baseline vs current Snitch cycles and the speedup
(baseline/current, >1 = current is faster). Also lists jobs present in only one report.

It additionally prints a "replacement" section that, within the current report, compares each
old app against the app that replaces it (see REPLACEMENTS), matched on workload identity
(model, seqLen, dModel, dtRank) rather than the job key (which differs because the app name and
tiling params differ between implementations).
"""
import json
import sys

# old app -> replacement app. Extend as new implementations supersede old ones.
REPLACEMENTS = {
    "suc-async": "suc-carry",
    "suc-async-dt": "suc-carry",
    "P2-async-OS-no-IS": "p2-carry",
}


def load(path):
    d = json.load(open(path))
    return d.get("jobs", d)


def wkey(job):
    """Workload identity shared by an app and its replacement (implementation-independent)."""
    p = job.get("params", {})
    return (job.get("model"), p.get("seqLen"), p.get("dModel"), p.get("dtRank"))


def snitch(job):
    v = job.get("total")
    if v in (None, "", "None"):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def label(job):
    return f'{job.get("app","?"):<16} {str(job.get("name","")):<14} {str(job.get("model","")):<11}'


def main():
    cur_path = sys.argv[1] if len(sys.argv) > 1 else "report.json"
    bak_path = sys.argv[2] if len(sys.argv) > 2 else "report_bak.json"
    cur, bak = load(cur_path), load(bak_path)

    rows, only_cur, only_bak = [], [], []
    for k in sorted(set(cur) | set(bak)):
        cj, bj = cur.get(k), bak.get(k)
        if cj and bj:
            rows.append((label(cj), snitch(bj), snitch(cj)))
        elif cj:
            only_cur.append((label(cj), snitch(cj)))
        else:
            only_bak.append((label(bj), snitch(bj)))

    print(f"# Snitch (total) cycles — current={cur_path}  baseline={bak_path}\n")
    print(f'{"app / name / model":<44} {"baseline":>12} {"current":>12} {"speedup":>9}')
    print("-" * 80)
    geo, n = 1.0, 0
    for lab, b, c in sorted(rows, key=lambda r: r[0]):
        if b and c:
            sp = b / c
            geo *= sp
            n += 1
            print(f"{lab:<44} {b:>12,} {c:>12,} {sp:>8.3f}x")
        else:
            print(f'{lab:<44} {(f"{b:,}" if b else "-"):>12} {(f"{c:,}" if c else "-"):>12} {"-":>9}')
    if n:
        print("-" * 80)
        print(f'{"geomean speedup (matched, both done)":<44} {"":>12} {"":>12} {geo**(1/n):>8.3f}x')

    for title, lst in [("only in current", only_cur), ("only in baseline", only_bak)]:
        if lst:
            print(f"\n# {title}")
            for lab, v in sorted(lst, key=lambda x: x[0]):
                print(f'{lab:<44} {(f"{v:,}" if v else "-"):>12}')

    replacement_section(cur, bak, cur_path, bak_path)


def index(rep):
    idx = {}
    for job in rep.values():
        idx.setdefault((job["app"], wkey(job)), []).append(job)
    return idx


def best(jobs):
    """(best valid Snitch total, its job) among jobs, or (None, first job / None)."""
    vals = [(snitch(j), j) for j in (jobs or [])]
    ok = [v for v in vals if v[0] is not None]
    if ok:
        return min(ok, key=lambda v: v[0])
    return (None, jobs[0] if jobs else None)


def replacement_section(cur, bak, cur_path, bak_path):
    cur_idx, bak_idx = index(cur), index(bak)

    rows, used_fallback = [], False
    for (old_app, k), old_jobs in cur_idx.items():
        new_app = REPLACEMENTS.get(old_app)
        if new_app is None:
            continue
        o, oj = best(old_jobs)
        src = "cur"
        if o is None:  # old execution time missing in current report -> fall back to baseline
            o, boj = best(bak_idx.get((old_app, k)))
            if o is not None:
                oj, src, used_fallback = boj, "bak", True
        c, _ = best(cur_idx.get((new_app, k)))
        rows.append((old_app, new_app, oj, o, src, c))

    if not rows:
        return

    print(f"\n# Replacement comparison (old app -> new app; old time from {cur_path}, '*' = fell back to {bak_path})")
    print(f'{"old -> new":<30} {"name / model":<26} {"old":>13} {"new":>12} {"speedup":>9}')
    print("-" * 95)
    geo, n = 1.0, 0
    for old_app, new_app, oj, o, src, c in sorted(rows, key=lambda r: (r[0], label(r[2]))):
        pair = f"{old_app} -> {new_app}"
        nm = f'{str(oj.get("name","")):<14} {str(oj.get("model","")):<11}'
        ostr = (f"{o:,}*" if src == "bak" else f"{o:,}") if o else "-"
        if o and c:
            sp = o / c
            geo *= sp
            n += 1
            print(f"{pair:<30} {nm:<26} {ostr:>13} {c:>12,} {sp:>8.3f}x")
        else:
            print(f'{pair:<30} {nm:<26} {ostr:>13} {(f"{c:,}" if c else "-"):>12} {"-":>9}')
    if n:
        print("-" * 95)
        print(f'{"geomean speedup (matched, both done)":<57} {"":>13} {"":>12} {geo**(1/n):>8.3f}x')
    if used_fallback:
        print(f"* old execution time taken from baseline {bak_path} (absent/incomplete in {cur_path})")


if __name__ == "__main__":
    main()
