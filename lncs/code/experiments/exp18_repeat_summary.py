"""Aggregate exp18_board_repeats.sh output: mean, SD, median and range per configuration.

    python lncs/code/experiments/exp18_repeat_summary.py <out-dir> [summary.json]

Reads <out-dir>/exp18_e2e.json (one entry per end-to-end run, named
<transport>_rate<r>_rep<n>) and <out-dir>/gate/gate_<transport>_rep<n>.json
(one exp6 summary per gate run). Writes <out-dir>/exp18_summary.json unless a
second argument names the file, and prints one line per configuration.
"""

import glob
import json
import os
import re
import sys

import numpy as np


def stats(values):
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return None
    return {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}


def group(names_and_items):
    out = {}
    for name, item in names_and_items:
        out.setdefault(re.sub(r"_rep\d+$", "", name), []).append(item)
    return out


def main():
    d = sys.argv[1]
    path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(d, "exp18_summary.json")
    report = {"e2e": {}, "gate": {}}

    e2e_file = os.path.join(d, "exp18_e2e.json")
    if os.path.exists(e2e_file):
        runs = json.load(open(e2e_file))
        for cfg, items in group((r["run"], r) for r in runs).items():
            entry = {"runs": [r["run"] for r in items],
                     "verdicts": stats(r["verdicts"] for r in items),
                     "producer_rows_s": stats(np.mean(r["producer_rows_s"]) for r in items)}
            for stage in ("gate", "classify", "e2e"):
                for q in ("p50_ms", "p95_ms"):
                    entry[f"{stage}_{q}"] = stats(r[stage][q] for r in items)
            report["e2e"][cfg] = entry
            e = entry
            print(f"e2e  {cfg:16s} n={e['e2e_p50_ms']['n']}  end-to-end p50 {e['e2e_p50_ms']['mean']:.0f}+-{e['e2e_p50_ms']['sd']:.0f} ms"
                  f"  p95 {e['e2e_p95_ms']['mean']:.0f}+-{e['e2e_p95_ms']['sd']:.0f} ms"
                  f"  (median p50 {e['e2e_p50_ms']['median']:.0f}, range {e['e2e_p50_ms']['min']:.0f}-{e['e2e_p50_ms']['max']:.0f})")

    gate_files = sorted(glob.glob(os.path.join(d, "gate", "gate_*_rep*.json")))
    pairs = [(os.path.basename(f)[:-5], json.load(open(f))) for f in gate_files]
    keys = ("records", "consumed_rows_s", "inference_throughput_rows_s", "inference_p95_ms_row", "rss_idle_mb",
            "rss_active_max_mb", "power_active_vdd_in_w", "power_idle_vdd_in_w", "forward_rate")
    for cfg, items in group(pairs).items():
        entry = {k: stats(r.get(k) for r in items) for k in keys}
        entry["runs"] = len(items)
        report["gate"][cfg] = entry
        g = entry
        print(f"gate {cfg:16s} n={len(items)}  {g['inference_throughput_rows_s']['mean']:.0f}+-{g['inference_throughput_rows_s']['sd']:.0f} rows/s"
              f"  p95 {g['inference_p95_ms_row']['mean']:.4f}+-{g['inference_p95_ms_row']['sd']:.4f} ms"
              f"  RSS {g['rss_active_max_mb']['mean']:.1f}+-{g['rss_active_max_mb']['sd']:.1f} MB"
              f"  power {g['power_active_vdd_in_w']['mean']:.2f}+-{g['power_active_vdd_in_w']['sd']:.2f} W"
              f" (idle {g['power_idle_vdd_in_w']['mean']:.2f})")

    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
