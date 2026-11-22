"""Latency, throughput, energy, memory and CPU of each classifier backend across batch sizes (run ON a Jetson).

exp8 measured the backends at one batch size (20) and exp20 measured energy at that
batch size only; exp1/exp23 swept batch sizes for throughput alone. This sweeps
batch sizes 1 to 2,000 for three backends, all on the deployed forest:

  * onnx     ONNX Runtime (CPU provider, default thread pool), as served;
  * numpy    the shipped NumPy engine (block-lockstep traversal);
  * pertree  the per-tree NumPy traversal of earlier releases (research/code/edgeids/forest.py).

Each (backend, batch, repeat) window runs in its own child process, so peak memory is
that of a process serving one backend. The child loads the model, warms up and reports
READY; only then does the parent start tegrastats and let the child call predict back
to back for --seconds on the replay rows. Idle windows (tegrastats with nothing running)
separate the measured windows. Per window:

  * per-call latency p50 / p95 / p99 and rows/s;
  * CPU utilization: process CPU time over wall time (1.0 = one core busy);
  * peak resident memory of the child (ru_maxrss);
  * mean VDD_IN, and energy per row idle-subtracted as in exp20:
    (P_window - P_idle) * seconds / rows, P_idle the mean of the idle windows either side.

It also times the release gate on the board: building the crossing-pair probes from the
ONNX graph and running them through both runtimes (test-generation overhead).

    python exp46_board_batch_sweep.py --csv replay.csv --out exp46_board_batch_sweep.json
"""

import argparse
import json
import os
import re
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "code"))

VDD_IN = re.compile(r"VDD_IN (\d+)mW")
ONNX = ROOT / "artifacts" / "ids_rf.onnx"
NPZ = ROOT / "artifacts" / "ids_rf_numpy.npz"


def load_rows(csv_path):
    import csv
    from onnx_edge_ids import FeatureMatrixBuilder
    with open(csv_path, encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    return FeatureMatrixBuilder(str(ROOT / "artifacts" / "feature_columns.json")).build(records)


def make_predict(backend):
    if backend == "pertree":
        from edgeids.forest import load_forest, predict_per_tree
        forest = load_forest(str(NPZ))
        return lambda x: predict_per_tree(forest, x)
    from onnx_edge_ids import create_inference_engine
    engine = create_inference_engine(backend, str(ONNX if backend == "onnx" else NPZ))
    return engine.predict_batch


def vm(key):
    """VmRSS / VmHWM of this process in MB (Linux /proc)."""
    for line in open("/proc/self/status"):
        if line.startswith(key + ":"):
            return int(line.split()[1]) / 1024.0
    return None


def child(args):
    """One measured window: load, warm up, READY, wait for GO, run, print the result as JSON."""
    import numpy as np
    from onnx_edge_ids import FeatureMatrixBuilder  # noqa: F401  (imported before the baseline, as every backend needs it)
    b = args.batch
    rss_base = vm("VmRSS")
    # the backend's own footprint (runtime library, model, buffers for this batch), before any replay data
    predict = make_predict(args.backend)
    n_features = len(json.load(open(ROOT / "artifacts" / "feature_columns.json")))
    for _ in range(3):
        predict(np.zeros((b, n_features), dtype=np.float32))
    model_rss = vm("VmRSS") - rss_base
    model_hwm = vm("VmHWM") - rss_base
    x = load_rows(args.csv)
    rss_data = vm("VmRSS")
    for _ in range(3):
        predict(x[:b])
    print("READY", flush=True)
    sys.stdin.readline()
    lat, rows, i = [], 0, 0
    t_cpu0 = os.times()
    start = time.perf_counter()
    while time.perf_counter() - start < args.seconds:
        chunk = x[i:i + b]
        if len(chunk) < b:
            i, chunk = 0, x[:b]
        t0 = time.perf_counter()
        predict(chunk)
        lat.append(time.perf_counter() - t0)
        rows += len(chunk)
        i += b
    elapsed = time.perf_counter() - start
    t_cpu1 = os.times()
    cpu = (t_cpu1.user - t_cpu0.user) + (t_cpu1.system - t_cpu0.system)
    lat_ms = np.array(lat) * 1000.0
    print(json.dumps({
        "rows": rows, "calls": len(lat), "elapsed_s": round(elapsed, 3), "rows_s": round(rows / elapsed, 1),
        "latency_ms": {"p50": float(np.percentile(lat_ms, 50)), "p95": float(np.percentile(lat_ms, 95)),
                       "p99": float(np.percentile(lat_ms, 99)), "mean": float(lat_ms.mean())},
        "cpu_cores": round(cpu / elapsed, 3),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
        # memory the backend adds on top of the loaded replay rows: after loading and warm-up, then at the end
        "model_rss_mb": round(model_rss, 1), "model_peak_mb": round(model_hwm, 1),
        "serving_growth_mb": round(vm("VmRSS") - rss_data, 1)}), flush=True)


class Tegrastats:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.f = open(self.path, "w")
        self.p = subprocess.Popen(["tegrastats", "--interval", "200"], stdout=self.f, stderr=subprocess.DEVNULL)
        time.sleep(0.5)
        return self

    def __exit__(self, *exc):
        self.p.terminate(); self.p.wait(); self.f.close()

    def watts(self):
        mw = [int(m.group(1)) for line in open(self.path) for m in [VDD_IN.search(line)] if m]
        return round(statistics.fmean(mw) / 1000.0, 4) if mw else None


def idle(seconds, path):
    with Tegrastats(path) as t:
        time.sleep(seconds)
    return t.watts()


def window(args, backend, batch, path):
    p = subprocess.Popen([sys.executable, __file__, "--child", "--csv", args.csv, "--backend", backend,
                          "--batch", str(batch), "--seconds", str(args.seconds)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    while p.stdout.readline().strip() != "READY":
        pass
    with Tegrastats(path) as t:
        p.stdin.write("GO\n"); p.stdin.flush()
        out = p.stdout.readline()
        p.wait()
    r = json.loads(out)
    r["mean_w"] = t.watts()
    return r


def gate_cost(x_sample):
    """Build the crossing-pair probes from the graph and run them through both runtimes, timed."""
    import numpy as np
    from onnx_edge_ids import create_inference_engine
    from onnx_edge_ids.boundary_probe import CROSSING_VARIANTS, build_probes
    t0 = time.perf_counter()
    pr = build_probes(str(ONNX), CROSSING_VARIANTS)
    t1 = time.perf_counter()
    ort = create_inference_engine("onnx", str(ONNX))
    npe = create_inference_engine("numpy", str(NPZ))
    out = {}
    for name, e in (("onnx", ort), ("numpy", npe)):
        s = time.perf_counter()
        for i in range(0, len(pr.rows), 2000):
            e.predict_batch(pr.rows[i:i + 2000])
        out[name] = time.perf_counter() - s
    s = time.perf_counter()
    ort.predict_batch(x_sample); npe.predict_batch(x_sample)
    replay = time.perf_counter() - s
    return {"probes": len(pr), "build_s": round(t1 - t0, 2), "run_onnx_s": round(out["onnx"], 2),
            "run_numpy_s": round(out["numpy"], 2), "replay_2000_both_s": round(replay, 3),
            "total_s": round(t1 - t0 + out["onnx"] + out["numpy"] + replay, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--backends", default="onnx,numpy,pertree")
    ap.add_argument("--batches", default="1,8,20,100,500,2000")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--idle", type=float, default=15.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--workdir", default="exp46_tegrastats")
    ap.add_argument("--out", default="exp46_board_batch_sweep.json")
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--backend")
    ap.add_argument("--batch", type=int)
    args = ap.parse_args()
    if args.child:
        return child(args)

    import platform
    import onnxruntime
    import numpy as np
    work = Path(args.workdir); work.mkdir(parents=True, exist_ok=True)
    nv = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True).stdout.splitlines()[:1]
    report = {"machine": platform.node(), "power_mode": nv, "numpy": np.__version__, "onnxruntime": onnxruntime.__version__,
              "seconds": args.seconds, "idle_seconds": args.idle, "repeats": args.repeats, "windows": [], "summary": {}}
    x = load_rows(args.csv)
    report["gate"] = [gate_cost(x[:2000]) for _ in range(3)]
    print("gate", report["gate"], flush=True)

    k = 0
    last_idle = idle(args.idle, work / f"{k:03d}_idle.log")
    for rep in range(args.repeats):
        for backend in args.backends.split(","):
            for batch in [int(b) for b in args.batches.split(",")]:
                k += 1
                r = window(args, backend, batch, work / f"{k:03d}_{backend}_{batch}.log")
                k += 1
                next_idle = idle(args.idle, work / f"{k:03d}_idle.log")
                p_idle = (last_idle + next_idle) / 2.0
                r.update({"backend": backend, "batch": batch, "repeat": rep + 1, "idle_w": round(p_idle, 4),
                          "delta_w": round(r["mean_w"] - p_idle, 4),
                          "energy_mj_per_row": round((r["mean_w"] - p_idle) * r["elapsed_s"] / r["rows"] * 1000.0, 5)})
                last_idle = next_idle
                report["windows"].append(r)
                print(json.dumps({k2: r[k2] for k2 in ("backend", "batch", "repeat", "rows_s", "cpu_cores", "model_rss_mb", "model_peak_mb",
                                                       "delta_w", "energy_mj_per_row")} | {"p99": r["latency_ms"]["p99"]}),
                      flush=True)
                with open(args.out, "w") as f:
                    json.dump(report, f, indent=1)

    for backend in args.backends.split(","):
        for batch in [int(b) for b in args.batches.split(",")]:
            ws = [w for w in report["windows"] if w["backend"] == backend and w["batch"] == batch]

            def ms(get):
                v = [get(w) for w in ws]
                return {"mean": round(statistics.fmean(v), 5), "sd": round(statistics.stdev(v), 5) if len(v) > 1 else 0.0}
            report["summary"][f"{backend}@{batch}"] = {
                "rows_s": ms(lambda w: w["rows_s"]), "p50_ms": ms(lambda w: w["latency_ms"]["p50"]),
                "p95_ms": ms(lambda w: w["latency_ms"]["p95"]), "p99_ms": ms(lambda w: w["latency_ms"]["p99"]),
                "cpu_cores": ms(lambda w: w["cpu_cores"]), "peak_rss_mb": ms(lambda w: w["peak_rss_mb"]),
                "model_rss_mb": ms(lambda w: w["model_rss_mb"]), "model_peak_mb": ms(lambda w: w["model_peak_mb"]),
                "delta_w": ms(lambda w: w["delta_w"]), "energy_mj_per_row": ms(lambda w: w["energy_mj_per_row"])}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
