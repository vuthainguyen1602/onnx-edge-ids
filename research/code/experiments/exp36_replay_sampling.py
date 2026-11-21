"""Replay gate: repeated draws, stratified sampling and the confidence tolerance.

exp16 reports the chance that a 2,000-row replay passes a float64 traversal as
(1 - q)^2000, with q the share of the whole test split that trips the check.
That is exact for rows drawn uniformly with replacement. This measures it
directly and asks two further questions, on all thirteen models of exp16
(runtime parity: float64 traversal vs ONNX Runtime):

  * Repeated replay. Draw R replays of 2,000 rows without replacement and count
    how many pass, giving an empirical pass rate with a Wilson 95% interval,
    under three samplers: uniform; stratified by ONNX Runtime's predicted label
    with proportional allocation; and stratified with equal allocation
    (1,000 rows per class), which oversamples the minority class. Labels are
    the reference runtime's predictions, which a gate always has, not ground
    truth, which a recording may not.
  * Tolerance. For tolerances from 1e-7 to 1e-2: the share of rows on which the
    correct float32 traversal trips the confidence check (a false failure) and
    the chance a 2,000-row replay passes the float64 traversal.
"""

import argparse
import json
import math
import os
import time

import numpy as np

from edgeids.data import ARTIFACTS, RESULTS, load_test_split
from edgeids.datasets import ton_iot_test, unsw_test
from edgeids.models import MODEL_KEYS
from edgeids.onnx_forest import load_onnx_forest
from edgeids.runtime import label, numpy_probs, ort_probs

TOLERANCES = (1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 1e-4, 1e-3, 1e-2)


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def draws(rng, strata, n_rows, mode, rows):
    """Index arrays of `rows` rows, R times, by sampler `mode`."""
    if mode == "uniform":
        return lambda: rng.choice(n_rows, rows, replace=False)
    classes = [np.flatnonzero(strata == c) for c in np.unique(strata)]
    if mode == "proportional":
        alloc = [int(round(rows * len(c) / n_rows)) for c in classes]
        alloc[int(np.argmax(alloc))] += rows - sum(alloc)
    else:  # balanced
        alloc = [rows // len(classes)] * len(classes)
        alloc[0] += rows - sum(alloc)
    alloc = [min(a, len(c)) for a, c in zip(alloc, classes)]
    return lambda: np.concatenate([rng.choice(c, a, replace=False) for c, a in zip(classes, alloc)])


def run_model(path, x, rows, repeats, seed):
    forest = load_onnx_forest(path)
    ref = ort_probs(path, x)
    ref_label = label(ref)
    p32 = numpy_probs(forest, x, np.float32)
    p64 = numpy_probs(forest, x, np.float64)
    d32 = np.abs(p32 - ref).max(axis=1)
    d64 = np.abs(p64 - ref).max(axis=1)
    f32 = label(p32) != ref_label
    f64 = label(p64) != ref_label
    n = len(x)
    rep = {"rows": int(n), "classes": {int(c): int((ref_label == c).sum()) for c in np.unique(ref_label)},
           "float32_max_delta": float(d32.max()), "float32_label_flips": int(f32.sum()),
           "tolerance": {}, "sampling": {}}
    for tol in TOLERANCES:
        t32 = f32 | (d32 > tol)
        t64 = f64 | (d64 > tol)
        rep["tolerance"][f"{tol:g}"] = {
            "float32_rows_tripping": int(t32.sum()),
            "float32_prob_false_failure": float(1 - (1 - t32.mean()) ** rows),
            "float64_rows_tripping": int(t64.sum()),
            "float64_prob_replay_passes": float((1 - t64.mean()) ** rows)}
    checks = {"labels": f64, "labels+conf@1e-05": f64 | (d64 > 1e-5)}
    for mode in ("uniform", "proportional", "balanced"):
        rng = np.random.default_rng(seed)
        draw = draws(rng, ref_label, n, mode, rows)
        passes = {k: 0 for k in checks}
        for _ in range(repeats):
            idx = draw()
            for k, trips in checks.items():
                passes[k] += int(not trips[idx].any())
        rep["sampling"][mode] = {k: {"passes": v, "rate": v / repeats, "wilson95": wilson(v, repeats)}
                                 for k, v in passes.items()}
    rep["sampling"]["analytic_uniform"] = {k: float((1 - t.mean()) ** rows) for k, t in checks.items()}
    rep["tripping_rows_by_class"] = {k: {int(c): int(t[ref_label == c].sum()) for c in np.unique(ref_label)}
                                     for k, t in checks.items()}
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cicids", required=True, help="CICIDS2017 test_data.parquet directory")
    ap.add_argument("--unsw", default="", help="directory with UNSW_NB15_{training,testing}-set.csv")
    ap.add_argument("--ton-iot", default="", help="ToN-IoT train_test_network.csv")
    ap.add_argument("--replay-rows", type=int, default=2000)
    ap.add_argument("--repeats", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp36_replay_sampling.json"))
    args = ap.parse_args()

    import onnxruntime
    started = time.perf_counter()
    report = {"numpy": np.__version__, "onnxruntime": onnxruntime.__version__, "replay_rows": args.replay_rows,
              "repeats": args.repeats, "seed": args.seed, "datasets": {}}
    jobs = []
    x, _, _ = load_test_split(args.cicids)
    assert len(x) == 447275, len(x)
    jobs.append(("CICIDS2017", x, [("deployed_rf", os.path.join(ARTIFACTS, "ids_rf.onnx"))] +
                 [(m, os.path.join(RESULTS, "exp5_models", f"{m}.onnx")) for m in MODEL_KEYS]))
    if args.unsw:
        x, cols = unsw_test(args.unsw)
        assert len(cols) == 39 and len(x) == 82332, (len(cols), len(x))
        jobs.append(("UNSW-NB15", x, [(m, os.path.join(RESULTS, "exp10_models", "unsw_nb15", f"{m}.onnx")) for m in MODEL_KEYS]))
    if args.ton_iot:
        x, cols = ton_iot_test(args.ton_iot)
        assert len(cols) == 14 and len(x) == 63313, (len(cols), len(x))
        jobs.append(("ToN-IoT", x, [(m, os.path.join(RESULTS, "exp10_models", "ton_iot", f"{m}.onnx")) for m in MODEL_KEYS]))

    for name, x, models in jobs:
        report["datasets"][name] = {"rows": int(len(x)), "models": {}}
        for key, path in models:
            t0 = time.perf_counter()
            rep = run_model(path, x, args.replay_rows, args.repeats, args.seed)
            rep["seconds"] = round(time.perf_counter() - t0, 1)
            report["datasets"][name]["models"][key] = rep
            s = rep["sampling"]
            print(name, key, {m: {k: round(v["rate"], 3) for k, v in s[m].items()} for m in ("uniform", "proportional", "balanced")},
                  {k: round(v, 3) for k, v in s["analytic_uniform"].items()}, flush=True)
            with open(args.out, "w") as f:
                json.dump(report, f, indent=2)

    report["seconds"] = round(time.perf_counter() - started, 1)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
