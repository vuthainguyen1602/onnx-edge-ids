"""Cost of the shipped gate's probe check on all thirteen models, crossing pair vs. three-probe set.

From v0.7.0 the gate's default probe set is the crossing pair (two probes per
node, located by bisection); up to v0.6.0 it was the three-probe set. exp19
timed the three-probe set with the experiment code's traversal. This times the
shipped toolkit, as the gate runs it: build the data-free probes from the ONNX
graph (onnx_edge_ids.boundary_probe), run them through ONNX Runtime and through
the shipped NumPy engine serving the exported artifact (batches of 2,000, the
gate's probe batch), and check that the engine agrees. Best of three runs.

    python exp32_gate_timing.py
"""

import json
import os
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np

from edgeids.data import ARTIFACTS, RESULTS
from edgeids.models import MODEL_KEYS

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
from onnx_edge_ids import create_inference_engine  # noqa: E402
from onnx_edge_ids.boundary_probe import CROSSING_VARIANTS, VARIANTS, build_probes  # noqa: E402

TOL = 1e-5
REPEATS = 3


def run(engine, x, batch):
    p, c = [], []
    for i in range(0, len(x), batch):
        r = engine.predict_batch(x[i:i + batch])
        p.append(r.predictions); c.append(r.confidences)
    return np.concatenate(p), np.concatenate(c)


def timed(fn):
    best, out = None, None
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return out, round(best, 2)


def main():
    jobs = [("CICIDS2017", "deployed_rf", os.path.join(ARTIFACTS, "ids_rf.onnx"))]
    for ds, d in (("CICIDS2017", "exp5_models"), ("UNSW-NB15", os.path.join("exp10_models", "unsw_nb15")),
                  ("ToN-IoT", os.path.join("exp10_models", "ton_iot"))):
        jobs += [(ds, k, os.path.join(RESULTS, d, f"{k}.onnx")) for k in MODEL_KEYS]
    tmp = tempfile.mkdtemp()
    import onnxruntime
    report = {"machine": f"{platform.machine()} {platform.system()}", "numpy": np.__version__,
              "onnxruntime": onnxruntime.__version__, "repeats": REPEATS, "models": {}}
    for ds, key, path in jobs:
        npz = os.path.join(tmp, f"{ds}_{key}.npz")
        subprocess.run([sys.executable, os.path.join(REPO, "scripts", "export_numpy.py"), "--model", path,
                        "--out", npz], check=True, capture_output=True)
        ort, eng = create_inference_engine("onnx", path), create_inference_engine("numpy", npz)
        entry = {}
        for name, variants in (("crossing", CROSSING_VARIANTS), ("three", VARIANTS)):
            probes, t_build = timed(lambda: build_probes(path, variants))
            (lo, co), t_ort = timed(lambda: run(ort, probes.rows, 20000))
            (ln, cn), t_np = timed(lambda: run(eng, probes.rows, 2000))
            entry[name] = {"probes": int(len(probes)),
                           "seconds": {"build": t_build, "onnxruntime": t_ort, "numpy_engine": t_np,
                                       "total": round(t_build + t_ort + t_np, 2)},
                           "label_flips": int((lo != ln).sum()),
                           "max_confidence_delta": float(np.abs(co - cn).max())}
        report["models"][f"{ds}/{key}"] = entry
        print(ds, key, {k: (v["probes"], v["seconds"]["total"], v["label_flips"]) for k, v in entry.items()}, flush=True)
    out = os.path.join(RESULTS, "exp32_gate_timing.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
