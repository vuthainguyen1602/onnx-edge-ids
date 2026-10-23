"""Summarise exp7_e2e_run.sh output dirs into exp7_e2e.json.

Each board's timestamps are moved onto the coordinator's clock with the mean of
its before/after offsets (exp7_clock_offset.py). Per forwarded flow:
  gate     = forwarded_at - sent_at   (transport hop + gate batching + gate inference)
  classify = verdict_at - forwarded_at (transport hop + classifier batching + inference)
  e2e      = verdict_at - sent_at
Only flows the gate forwards reach a verdict, so these are their latencies.

    python lncs/code/experiments/exp7_e2e_summary.py out.json run_dir [run_dir ...]
"""

import csv
import json
import os
import re
import sys

import numpy as np


def offset(d, board):
    a = json.load(open(os.path.join(d, f"offset_{board}_before.json")))
    b = json.load(open(os.path.join(d, f"offset_{board}_after.json")))
    return (a["offset_s"] + b["offset_s"]) / 2, abs(a["offset_s"] - b["offset_s"]), max(
        a["error_bound_s"], b["error_bound_s"]), a["rtt_median_ms"], b["rtt_median_ms"]


def pct(x):
    x = np.asarray(x) * 1000.0
    return {"p50_ms": round(float(np.percentile(x, 50)), 1),
            "p95_ms": round(float(np.percentile(x, 95)), 1),
            "max_ms": round(float(x.max()), 1)}


def summarise(d):
    o1, drift1, err1, rtt1a, rtt1b = offset(d, "j1")
    o2, drift2, err2, rtt2a, rtt2b = offset(d, "j2")
    sent, fwd, ver = [], [], []
    for r in csv.DictReader(open(os.path.join(d, "latency.csv"))):
        sent.append(float(r["sent_at"]))
        fwd.append(float(r["forwarded_at"]) - o1)
        ver.append(float(r["verdict_at"]) - o2)
    sent, fwd, ver = map(np.array, (sent, fwd, ver))
    run = open(os.path.join(d, "run.txt")).read()
    rate = re.search(r"rate=(\S+)", run).group(1)
    sent_rates = [float(m) for m in re.findall(r"\(([\d.]+) rows/s\)", run)]
    gate_lines = [l for l in open(os.path.join(d, "gate.log")) if l.startswith("[") and "gate=" in l]
    consumed = int(gate_lines[-1].split("]")[0][1:].replace(",", "")) if gate_lines else None
    return {
        "run": os.path.basename(d.rstrip("/")),
        "offered_rate": "max" if rate in ("0", "0.0") else float(rate),
        "producer_rows_s": sent_rates,
        "gate_consumed": consumed,
        "verdicts": int(sent.size),
        "gate": pct(fwd - sent),
        "classify": pct(ver - fwd),
        "e2e": pct(ver - sent),
        "clock": {"j1_offset_s": round(o1, 6), "j1_drift_s": round(drift1, 6), "j1_error_s": err1,
                  "j2_offset_s": round(o2, 6), "j2_drift_s": round(drift2, 6), "j2_error_s": err2,
                  "rtt_median_ms": [rtt1a, rtt1b, rtt2a, rtt2b]},
    }


def main():
    out, dirs = sys.argv[1], sys.argv[2:]
    res = [summarise(d) for d in dirs]
    print(json.dumps(res, indent=2))
    with open(out, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
