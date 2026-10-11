"""Aggregate exp8 classifier runs (and cold-start files) into exp8_summary.json.

Per run (one directory from exp8_classifier_repeat.sh):
  ms/row     engine_ms / batch_size of each batch in metrics.csv; mean, p95, p99
  capacity   rows served / sum of engine_ms (inference-only rows/s)
  RSS        max VmRSS of the classifier process while batches were arriving
  power      mean tegrastats VDD_IN over that window (total board, not idle-subtracted)
Across the repeats of a backend: mean and sample SD of each per-run figure.

    python lncs/code/experiments/exp8_summary.py --series DIR [DIR ...] --cold A.json [B.json ...] --out exp8_summary.json
"""

import csv
import glob
import json
import os
import re
import statistics
import sys
from datetime import datetime

import numpy as np

TEGRA = re.compile(r"^(\d\d-\d\d-\d{4} \d\d:\d\d:\d\d) .*VDD_IN (\d+)mW")


def run_figures(d):
    m = list(csv.DictReader(open(os.path.join(d, "metrics.csv"))))
    bs = np.array([int(r["batch_size"]) for r in m])
    ms = np.array([float(r["engine_ms"]) for r in m])
    per_row = ms / bs
    mon = list(csv.DictReader(open(os.path.join(d, "monitor.csv"))))
    t = np.array([float(r["epoch"]) for r in mon])
    n = np.array([int(r["batches"]) for r in mon])
    rss = np.array([int(r["vmrss_kb"]) for r in mon]) / 1024.0
    grew = np.flatnonzero(np.diff(n) > 0) + 1
    t0, t1 = t[grew[0] - 1], t[grew[-1]]
    active = (t >= t0) & (t <= t1)
    pw = []
    for line in open(os.path.join(d, "tegrastats.log")):
        g = TEGRA.match(line)
        if g:
            ts = datetime.strptime(g.group(1), "%m-%d-%Y %H:%M:%S").timestamp()
            if np.floor(t0) <= ts <= np.ceil(t1):
                pw.append(int(g.group(2)) / 1000.0)
    return {
        "batches": int(len(m)),
        "rows": int(bs.sum()),
        "full_batch_share": round(float((bs == 20).mean()), 3),
        "mean_ms_row": float(per_row.mean()),
        "p95_ms_row": float(np.percentile(per_row, 95)),
        "p99_ms_row": float(np.percentile(per_row, 99)),
        "capacity_rows_s": float(bs.sum() / (ms.sum() / 1000.0)),
        "rss_mb": float(rss[active].max()),
        "power_w": float(np.mean(pw)),
        "power_samples": len(pw),
    }


def agg(values, nd):
    return {"mean": round(statistics.mean(values), nd),
            "sd": round(statistics.stdev(values), nd) if len(values) > 1 else 0.0}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", nargs="+", required=True, help="directories of <engine>_repN runs")
    ap.add_argument("--cold", nargs="*", default=[], help="exp8_cold_start.py JSON files to pool")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = {"classifier": {}, "cold_start": {}}
    keys = {"mean_ms_row": 4, "p95_ms_row": 4, "p99_ms_row": 4, "capacity_rows_s": 0,
            "rss_mb": 1, "power_w": 2}
    for engine in ("onnx", "numpy", "numpylockstep"):
        runs = sorted(d for series in args.series for d in glob.glob(os.path.join(series, f"{engine}_rep*")))
        if not runs:
            continue
        figs = [run_figures(d) for d in runs]
        out["classifier"][engine] = {
            "repeats": len(figs),
            "rows_per_run": [f["rows"] for f in figs],
            "full_batch_share": [f["full_batch_share"] for f in figs],
            **{k: agg([f[k] for f in figs], nd) for k, nd in keys.items()},
            "runs": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items()} for f in figs],
        }
    for path in args.cold:
        out["cold_start"][os.path.basename(path)] = json.load(open(path))["summary"]
    if args.cold:
        pooled = {}
        for engine in ("onnx", "numpy"):
            pooled[engine] = {k: agg([r[k] for path in args.cold for r in json.load(open(path))["runs"][engine]], 1)
                              for k in ("load_ms", "first_ms", "to_first_verdict_ms")}
        out["cold_start"]["pooled"] = pooled
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    for engine, c in out["classifier"].items():
        print(engine, {k: c[k] for k in keys}, "rows/run", c["rows_per_run"])
    if args.cold:
        print("cold pooled", json.dumps(out["cold_start"]["pooled"]))


if __name__ == "__main__":
    main()
