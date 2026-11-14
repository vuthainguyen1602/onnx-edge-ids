"""Per-inference energy of the classifier backends, idle-subtracted (run ON a Jetson board).

The paper's power column is total-board tegrastats VDD_IN while a node serves a
replay at a modest rate, without an idle baseline from the same session, so it
cannot be turned into energy per inference. This measures it directly, in one
session and one process:

    idle -> engine A at saturation -> idle -> engine B at saturation -> ... -> idle

Each window runs its own tegrastats (--interval 200 ms) and reports mean
VDD_IN. During an engine window the process calls predict_batch back to back
on the replay rows for --seconds, so the CPU is saturated; energy per row is

    (P_engine - P_idle) * seconds / rows_served,

with P_idle the mean of the idle windows on either side of that engine window.
Network, transport and batching are excluded: this is inference energy only.

    python lncs/code/experiments/exp20_classifier_energy.py --csv replay_cicids2017.csv \\
        --engines onnx,numpy --batch 20 --seconds 60 --idle 60 --out exp20_classifier_energy.json
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from onnx_edge_ids import FeatureMatrixBuilder, create_inference_engine  # noqa: E402

VDD_IN = re.compile(r"VDD_IN (\d+)mW")
MODELS = {"onnx": ROOT / "artifacts" / "ids_rf.onnx", "numpy": ROOT / "artifacts" / "ids_rf_numpy.npz"}


class Tegrastats:
    """One tegrastats process per measurement window."""

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.f = open(self.path, "w")
        self.p = subprocess.Popen(["tegrastats", "--interval", "200"], stdout=self.f, stderr=subprocess.DEVNULL)
        time.sleep(0.5)
        return self

    def __exit__(self, *exc):
        self.p.terminate()
        self.p.wait()
        self.f.close()

    def watts(self):
        mw = [int(m.group(1)) for line in open(self.path) for m in [VDD_IN.search(line)] if m]
        return {"mean_w": round(statistics.fmean(mw) / 1000.0, 4), "samples": len(mw)}


def idle(seconds, path):
    with Tegrastats(path) as t:
        time.sleep(seconds)
    return t.watts()


def saturate(engine, matrix, batch, seconds, path):
    rows = 0
    calls = 0
    with Tegrastats(path) as t:
        start = time.perf_counter()
        i = 0
        while time.perf_counter() - start < seconds:
            chunk = matrix[i:i + batch]
            if len(chunk) < batch:
                i = 0
                chunk = matrix[:batch]
            engine.predict_batch(chunk)
            rows += len(chunk)
            calls += 1
            i += batch
        elapsed = time.perf_counter() - start
    return t.watts(), rows, calls, elapsed


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", required=True, help="replay CSV (the thesis replay the board figures use)")
    ap.add_argument("--features", default=str(ROOT / "artifacts" / "feature_columns.json"))
    ap.add_argument("--engines", default="onnx,numpy")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=60.0, help="saturated window per engine")
    ap.add_argument("--idle", type=float, default=60.0, help="idle window before, between and after engines")
    ap.add_argument("--repeats", type=int, default=3, help="idle/engine cycles per engine")
    ap.add_argument("--workdir", default="exp20_tegrastats")
    ap.add_argument("--out", default="exp20_classifier_energy.json")
    args = ap.parse_args()

    import csv
    with open(args.csv, encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    matrix = FeatureMatrixBuilder(args.features).build(records)
    engines = {name: create_inference_engine(name, str(MODELS[name])) for name in args.engines.split(",")}
    for e in engines.values():  # warm up outside the measured windows
        e.predict_batch(matrix[:args.batch])

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    report = {"rows_in_replay": int(len(matrix)), "batch": args.batch, "seconds": args.seconds,
              "idle_seconds": args.idle, "repeats": args.repeats, "windows": [], "engines": {}}
    k = 0
    idles = [idle(args.idle, work / f"{k:02d}_idle.log")]
    report["windows"].append({"kind": "idle", **idles[-1]})
    for rep in range(args.repeats):
        for name, engine in engines.items():
            k += 1
            w, rows, calls, elapsed = saturate(engine, matrix, args.batch, args.seconds, work / f"{k:02d}_{name}.log")
            k += 1
            idles.append(idle(args.idle, work / f"{k:02d}_idle.log"))
            p_idle = (idles[-2]["mean_w"] + idles[-1]["mean_w"]) / 2.0
            entry = {"kind": name, "repeat": rep + 1, **w, "rows": rows, "calls": calls, "elapsed_s": round(elapsed, 2),
                     "rows_s": round(rows / elapsed, 1), "idle_w": round(p_idle, 4),
                     "delta_w": round(w["mean_w"] - p_idle, 4),
                     "energy_mj_per_row": round((w["mean_w"] - p_idle) * elapsed / rows * 1000.0, 5)}
            report["windows"].append(entry)
            report["windows"].append({"kind": "idle", **idles[-1]})
            print(json.dumps(entry), flush=True)

    for name in engines:
        runs = [w for w in report["windows"] if w["kind"] == name]
        def agg(key):
            v = [r[key] for r in runs]
            return {"mean": statistics.fmean(v), "sd": statistics.stdev(v) if len(v) > 1 else 0.0}
        report["engines"][name] = {k2: agg(k2) for k2 in ("rows_s", "mean_w", "idle_w", "delta_w", "energy_mj_per_row")}
        e = report["engines"][name]
        print(f"{name:6s} {e['rows_s']['mean']:.0f} rows/s  {e['mean_w']['mean']:.2f} W (idle {e['idle_w']['mean']:.2f})"
              f"  +{e['delta_w']['mean']:.2f}+-{e['delta_w']['sd']:.2f} W"
              f"  {e['energy_mj_per_row']['mean']:.4f}+-{e['energy_mj_per_row']['sd']:.4f} mJ/row", flush=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
