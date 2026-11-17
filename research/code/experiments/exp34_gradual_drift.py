"""Gradual drift: the frozen detector on traffic that moves from CICIDS2017 to CSE-CIC-IDS2018 step by step.

exp4 froze the detector and ran it on the whole 2018 test split, one abrupt
shift. Here the same frozen gate, forest and threshold see windows of 10,000
flows in which a share p of the rows comes from 2018 and the rest from 2017,
for p from 0 to 1. Each window takes a run of consecutive rows from each split
(the splits keep their captures' label phases) at a random offset, and the two
runs are interleaved at random. Per window:

  * detection: classifier attack F1 and recall (ONNX Runtime labels), gate
    attack recall, two-stage F1;
  * serving parity: label differences of the float32 NumPy traversal (the
    fallback) against ONNX Runtime, and of exact early exit against the full
    forest;
  * the label-free signal of exp9: the share of flows the early-exit rule cannot
    settle at its first check, and whether exp9's alarm threshold for 10,000-row
    windows (the largest rate over its calibration windows) fires.

Five windows per share, seed 0.

    python exp34_gradual_drift.py --in-dist $DATA --shifted $DATA2018
"""

import argparse
import json
import os

import numpy as np

from edgeids.data import ARTIFACTS, RESULTS, Gate, load_test_split
from edgeids.forest import load_forest, margin_bounds, predict_early_exit
from edgeids.runtime import lockstep_labels, session_labels

WINDOW = 10000
REPEATS = 5
SHARES = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
BATCH = 2000


def f1(y, p):
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    return (2 * tp / (2 * tp + fp + fn)) if tp else 0.0, (tp / (tp + fn)) if (tp + fn) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dist", required=True)
    ap.add_argument("--shifted", required=True)
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp34_gradual_drift.json"))
    args = ap.parse_args()
    import onnxruntime as ort

    alarm = json.load(open(os.path.join(RESULTS, "exp9_drift_alarm.json")))["all_traffic"][str(WINDOW)][
        "thresholds"]["max_of_calibration_windows"]["threshold"]
    npz = os.path.join(ARTIFACTS, "ids_rf_numpy.npz")
    f32 = load_forest(npz, "float32")
    session = ort.InferenceSession(os.path.join(ARTIFACTS, "ids_rf.onnx"), providers=["CPUExecutionProvider"])
    gate = Gate()
    bounds = margin_bounds(f32)
    first = f32.n_trees // 2 + 4
    schedule = [first] + [8] * ((f32.n_trees - first + 7) // 8)
    x17, y17, _ = load_test_split(args.in_dist)
    x18, y18, _ = load_test_split(args.shifted)
    rng = np.random.default_rng(0)
    report = {"window": WINDOW, "repeats": REPEATS, "alarm_threshold_from_exp9": alarm, "shares": {}}
    for p in SHARES:
        rows = []
        for _ in range(REPEATS):
            n18 = int(round(p * WINDOW))
            n17 = WINDOW - n18
            s17 = int(rng.integers(0, len(x17) - n17 + 1))
            s18 = int(rng.integers(0, len(x18) - n18 + 1))
            x = np.concatenate([x17[s17:s17 + n17], x18[s18:s18 + n18]])
            y = np.concatenate([y17[s17:s17 + n17], y18[s18:s18 + n18]])
            order = rng.permutation(WINDOW)
            x, y = np.ascontiguousarray(x[order], dtype=np.float32), y[order]
            lab = session_labels(session, x)
            lab32 = lockstep_labels(f32, x)
            ee, used = zip(*[predict_early_exit(f32, x[i:i + BATCH], schedule, bounds=bounds)
                             for i in range(0, WINDOW, BATCH)])
            ee, used = np.concatenate(ee), np.concatenate(used)
            fwd = gate.score(x) >= gate.threshold
            cf1, crec = f1(y, lab)
            sf1, _ = f1(y, (fwd & (lab == 1)).astype(np.int64))
            und = float((used > first).mean())
            rows.append({"attack_rate": float(y.mean()), "classifier_f1": cf1, "classifier_recall": crec,
                         "gate_attack_recall": float(fwd[y == 1].mean()) if (y == 1).any() else None,
                         "two_stage_f1": sf1, "float32_vs_onnxruntime_label_differences": int((lab32 != lab).sum()),
                         "early_exit_vs_full_forest_label_differences": int((ee != lab32).sum()),
                         "undecided_at_first_check": und, "alarm": und > alarm})
        mean = lambda k: float(np.mean([r[k] for r in rows if r[k] is not None]))
        report["shares"][f"{p:g}"] = {
            "windows": rows,
            "mean": {k: mean(k) for k in ("classifier_f1", "classifier_recall", "two_stage_f1", "undecided_at_first_check")},
            "alarms": sum(r["alarm"] for r in rows),
            "float32_vs_onnxruntime_label_differences": sum(r["float32_vs_onnxruntime_label_differences"] for r in rows),
            "early_exit_vs_full_forest_label_differences": sum(r["early_exit_vs_full_forest_label_differences"] for r in rows),
        }
        s = report["shares"][f"{p:g}"]
        print(f"p={p:g}", {k: round(v, 3) for k, v in s["mean"].items()}, "alarms", s["alarms"], "/", REPEATS,
              "parity diffs", s["float32_vs_onnxruntime_label_differences"], s["early_exit_vs_full_forest_label_differences"],
              flush=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
