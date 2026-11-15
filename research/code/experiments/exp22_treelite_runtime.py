"""Treelite as a third runtime of the same ONNX graphs: replay and data-free probes.

The runtimes of the paper so far are ONNX Runtime and the NumPy engine written
for it. This adds Treelite (GTIL, its reference C++ interpreter), built from the
node table of each exported ONNX graph (edgeids.treelite_forest), so that it
serves the same graph rather than the training framework's model. Two builds:

  float32  thresholds and leaf weights in float32, Scaler applied in float32:
           the graph's own arithmetic;
  float64  thresholds and leaf weights in float64, Scaler in float64: the
           default-dtype mistake of the paper's float64 NumPy traversal.

Each is compared with ONNX Runtime on the whole test split (labels, and
probabilities at 1e-5) and on the release gate's data-free probes (exp19), for
the deployed forest and the twelve trained models. Also reported: Treelite's
throughput at batch size 2,000 on this machine next to ONNX Runtime's.
"""

import argparse
import json
import os
import time

import numpy as np

from edgeids.data import ARTIFACTS, RESULTS, load_test_split
from edgeids.datasets import ton_iot_test, unsw_test
from edgeids.gates import probe_gate, replay_gate
from edgeids.models import MODEL_KEYS
from edgeids.onnx_forest import load_onnx_forest
from edgeids.probes import datafree_probes
from edgeids.runtime import ort_probs, probabilities
from edgeids.treelite_forest import to_treelite, treelite_scores


def tl_probs(model, forest, x, dtype, batch=50000):
    return np.concatenate([probabilities(forest, treelite_scores(model, forest, x[i:i + batch], dtype))
                           for i in range(0, len(x), batch)])


def throughput(fn, x, batch=2000, seconds=3.0):
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        fn(x[(n // batch * batch) % (len(x) - batch):][:batch]); n += batch
    return n / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cicids", required=True)
    ap.add_argument("--unsw", default="")
    ap.add_argument("--ton-iot", default="")
    ap.add_argument("--replay-rows", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp22_treelite_runtime.json"))
    args = ap.parse_args()

    import onnxruntime as ort
    import treelite
    report = {"versions": {"treelite": treelite.__version__, "numpy": np.__version__, "onnxruntime": ort.__version__},
              "datasets": {}}
    x, _, _ = load_test_split(args.cicids)
    jobs = [("CICIDS2017", x, [("deployed_rf", os.path.join(ARTIFACTS, "ids_rf.onnx"))] +
             [(m, os.path.join(RESULTS, "exp5_models", f"{m}.onnx")) for m in MODEL_KEYS])]
    if args.unsw:
        jobs.append(("UNSW-NB15", unsw_test(args.unsw)[0], [(m, os.path.join(RESULTS, "exp10_models", "unsw_nb15", f"{m}.onnx")) for m in MODEL_KEYS]))
    if args.ton_iot:
        jobs.append(("ToN-IoT", ton_iot_test(args.ton_iot)[0], [(m, os.path.join(RESULTS, "exp10_models", "ton_iot", f"{m}.onnx")) for m in MODEL_KEYS]))

    for name, x, models in jobs:
        report["datasets"][name] = {}
        for key, path in models:
            forest = load_onnx_forest(path)
            _, probes = datafree_probes(forest)
            ref_replay, ref_probes = ort_probs(path, x), ort_probs(path, probes)
            rep = {}
            for tt, dtype in (("float32", np.float32), ("float64", np.float64)):
                t0 = time.perf_counter()
                model = to_treelite(forest, threshold_type=tt)
                rep[tt] = {"build_seconds": round(time.perf_counter() - t0, 1),
                           "replay": replay_gate(tl_probs(model, forest, x, dtype), ref_replay, args.replay_rows),
                           "datafree_probes": probe_gate(tl_probs(model, forest, probes, dtype), ref_probes)}
                if tt == "float32":
                    import treelite.gtil
                    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
                    inp = sess.get_inputs()[0].name
                    rep["rows_s_batch2000"] = {
                        "treelite_gtil": round(throughput(lambda b: treelite_scores(model, forest, b), x), 0),
                        "onnxruntime": round(throughput(lambda b: sess.run(None, {inp: b}), x), 0)}
            rep["probes"] = int(len(probes))
            report["datasets"][name][key] = rep
            line = {tt: (rep[tt]["replay"]["label_flips"], rep[tt]["replay"]["tolerances"]["1e-05"]["prob_replay_passes"],
                         rep[tt]["datafree_probes"]["label_flips"], rep[tt]["datafree_probes"]["rows_over_tolerance"],
                         rep[tt]["datafree_probes"]["max_delta"]) for tt in ("float32", "float64")}
            print(name, key, json.dumps(line), rep["rows_s_batch2000"], flush=True)
            with open(args.out, "w") as f:
                json.dump(report, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
