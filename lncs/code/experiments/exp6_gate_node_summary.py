"""Summarise an exp6_gate_node_run.sh directory into exp6_gate_node.json.

Active window: from the first to the last monitor sample at which the gate's
logged batch count grew. Idle window: the samples before the first batch.
Throughput and p95 are inference-only (the gate's own per-batch timer), as in
the engine table; RSS is the gate process's VmRSS; power is tegrastats VDD_IN
(total board input), not idle-subtracted.

    python lncs/code/experiments/exp6_gate_node_summary.py lncs/code/results_jetson/exp6_gate_node
"""

import csv
import json
import os
import re
import sys
from datetime import datetime

import numpy as np

BATCH = re.compile(r"^\[([\d,]+)\] gate=([\d.]+)ms forwarded=([\d,]+)/")
TEGRA = re.compile(r"^(\d\d-\d\d-\d{4} \d\d:\d\d:\d\d) .*VDD_IN (\d+)mW")


def main():
    d = sys.argv[1]
    rows = list(csv.DictReader(open(os.path.join(d, "monitor.csv"))))
    t = np.array([float(r["epoch"]) for r in rows])
    n = np.array([int(r["batches"]) for r in rows])
    rss = np.array([int(r["vmrss_kb"]) for r in rows]) / 1024.0
    hwm = np.array([int(r["vmhwm_kb"]) for r in rows]) / 1024.0
    grew = np.flatnonzero(np.diff(n) > 0) + 1
    t0, t1 = t[grew[0] - 1], t[grew[-1]]
    idle = t < t0
    active = (t >= t0) & (t <= t1)

    batches = []
    for line in open(os.path.join(d, "gate.log")):
        m = BATCH.match(line)
        if m:
            batches.append((int(m.group(1).replace(",", "")), float(m.group(2)), int(m.group(3).replace(",", ""))))
    sizes = np.diff([0] + [b[0] for b in batches])
    ms = np.array([b[1] for b in batches])
    full = sizes == 100
    per_row = ms[full] / 100.0

    power = []
    for line in open(os.path.join(d, "tegrastats.log")):
        m = TEGRA.match(line)
        if m:
            ts = datetime.strptime(m.group(1), "%m-%d-%Y %H:%M:%S").timestamp()
            power.append((ts, int(m.group(2)) / 1000.0))
    pt = np.array([p[0] for p in power])
    pw = np.array([p[1] for p in power])
    # tegrastats stamps whole seconds; keep samples inside the window, widened by 1 s.
    act_w = pw[(pt >= np.floor(t0)) & (pt <= np.ceil(t1))]
    idle_w = pw[pt < np.floor(t0) - 1]

    out = {
        "records": int(batches[-1][0]),
        "batches": len(batches),
        "batch_size": 100,
        "forward_rate": round(batches[-1][2] / batches[-1][0], 4),
        "active_seconds": round(float(t1 - t0), 1),
        "consumed_rows_s": round(batches[-1][0] / float(t1 - t0), 1),
        "inference_throughput_rows_s": round(float(sizes.sum() / (ms.sum() / 1000.0)), 1),
        "inference_p95_ms_row": round(float(np.percentile(per_row, 95)), 4),
        "rss_idle_mb": round(float(rss[idle].max()), 1),
        "rss_active_max_mb": round(float(rss[active].max()), 1),
        "rss_peak_hwm_mb": round(float(hwm.max()), 1),
        "power_active_vdd_in_w": round(float(act_w.mean()), 3),
        "power_active_samples": int(act_w.size),
        "power_idle_vdd_in_w": round(float(idle_w.mean()), 3),
        "power_idle_samples": int(idle_w.size),
    }
    print(json.dumps(out, indent=2))
    with open(os.path.join(os.path.dirname(d.rstrip("/")), "exp6_gate_node.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
