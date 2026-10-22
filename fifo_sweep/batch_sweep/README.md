# Batch-weighted streamer-FIFO depth sweep

Produces `../fifo_sweep_summary_batch_pareto.txt`: the cost-vs-weighted-batch-latency
Pareto front over per-port FIFO depths (cost = Σ depth × channels), plus always-included
reference rows (current config, the vsim-validated recommendation, salvage variants).
Evaluator is memsim with the `MEMSIM_STREAMER_DEPTHS` override; deltas are measured on the
SimbaCore counter and overlaid on the vsim wall-clock baselines from `report.json`.

## Files

Tracked:
- `sweep_lib.py` — cached parallel memsim evaluator + weighted objective.
- `sweep_driver.py` — `oat` (per-port ±1 screen), `greedy` (frontier), `refine` (swaps).
- `build_rows.py` — regenerates `rows.json`/`weights.json` from `batch_run_config.hjson`,
  `report.json` and the excel weight table in its `MULT_NL` dict.
- `make_report.py` — scores every vector in `points.json` (+ any new `*_trace.json`) and
  writes the summary.
- `points.json` — the evaluated depth vectors (the only non-regenerable data).

Gitignored working data: `rows.json`/`weights.json` (from `build_rows.py`),
`eval_cache.json` (memsim results, keyed on the memsim binary hash — missing entries
re-run automatically, no license needed), `oat.json` (caching-speedup usage map).

## Re-score, e.g. with different weights

```bash
python3 build_rows.py     # after editing MULT_NL, or on a fresh checkout
python3 make_report.py
```

## Model quality and validation

memsim-vs-vsim SimbaCore drift at current depths: scans −0.1…−2.2 % (SUC C/D −7…−11 %,
offset-calibrated), p2-carry +1.2 %, dgcf +8.5 %, einfft +1…+5 %, fft −3 %, P1 +4 %.

RTL spot-checks (full rebuild + vsim, same ELFs as the baselines):
- cost-123 `8,3,3,1,1,1,1,4,1,1,4,2,6,4 | 4,1,3,4`: cycle-identical on einfft-dcf-B (22287),
  dgcf-P2-in-D (103258) and fft-3way-C (42502); main-D +0.13 %. ≈3.7k FFs saved
  (reader ≈147 FF/cost-point, writer ≈83).
- cost-104 (thin R0=2/R1=2/W0=1): FALSIFIED — einfft-dcf +75 %, dgcf +5.2 % where the
  model predicts ~0 (concurrent dual-GEMM + DMA is model-blind on those ports).

## Caveats

Rows noted `model-optimistic` thin R0/R1/W0 below current — do not adopt without a vsim
check. R7 < 4 is not evaluable (model refresh-window limitation). Re-run the safe-to-start
sweep for whichever config is adopted.
