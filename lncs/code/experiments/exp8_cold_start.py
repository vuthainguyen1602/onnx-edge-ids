"""Cold start per backend, page cache dropped before every repeat (run ON a board).

For each repeat: sync and drop the page cache (needs passwordless sudo), then
launch a fresh Python process that imports the package, builds the engine
(model load) and predicts one row (first prediction). "To first verdict" is the
wall time from launching that process to the first prediction returning, so it
includes interpreter start-up and imports. The row is all zeros in the deployed
feature order; its value does not change the work a forest traversal does by
more than path length.

    python lncs/code/experiments/exp8_cold_start.py --reps 5 --out cold_start_<board>.json
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODELS = {"onnx": "artifacts/ids_rf.onnx", "numpy": "artifacts/ids_rf_numpy.npz"}

CHILD = r"""
import json, sys, time
sys.path.insert(0, {src!r})
t_start = time.time()
import numpy as np
from onnx_edge_ids.inference_engine import create_inference_engine
features = json.load(open({features!r}))
t0 = time.perf_counter()
engine = create_inference_engine({engine!r}, {model!r})
load_ms = (time.perf_counter() - t0) * 1000.0
x = np.zeros((1, len(features)), dtype=np.float32)
t1 = time.perf_counter()
engine.predict_batch(x)
first_ms = (time.perf_counter() - t1) * 1000.0
print(json.dumps({{"t_start": t_start, "t_verdict": time.time(), "load_ms": load_ms, "first_ms": first_ms}}))
"""


def drop_caches():
    subprocess.run(["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"], check=True)
    time.sleep(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runs = {e: [] for e in MODELS}
    for rep in range(args.reps):
        for engine, model in MODELS.items():
            code = CHILD.format(src=os.path.join(REPO, "src"),
                                features=os.path.join(REPO, "artifacts", "feature_columns.json"),
                                engine=engine, model=os.path.join(REPO, model))
            drop_caches()
            launched = time.time()
            out = subprocess.run([args.python, "-c", code], capture_output=True, text=True, check=True)
            r = json.loads(out.stdout.strip().splitlines()[-1])
            runs[engine].append({"load_ms": round(r["load_ms"], 1),
                                 "first_ms": round(r["first_ms"], 1),
                                 "to_first_verdict_ms": round((r["t_verdict"] - launched) * 1000.0, 1)})
            print(engine, rep + 1, runs[engine][-1], flush=True)

    summary = {}
    for engine, rs in runs.items():
        summary[engine] = {}
        for k in ("load_ms", "first_ms", "to_first_verdict_ms"):
            v = [r[k] for r in rs]
            summary[engine][k] = {"mean": round(statistics.mean(v), 1),
                                  "sd": round(statistics.stdev(v), 1) if len(v) > 1 else 0.0}
    res = {"machine": platform.node(), "reps": args.reps, "summary": summary, "runs": runs}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
