"""Exact early exit on the traffic the gate actually forwards to the classifier."""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgeids.data import ARTIFACTS, RESULTS, Gate, load_test_split
from edgeids.forest import load_forest, margin_bounds, predict_early_exit, predict_lockstep

BATCH = 2000


def run(fn, x):
    start = time.perf_counter()
    out = [fn(x[i:i + BATCH]) for i in range(0, len(x), BATCH)]
    return out, time.perf_counter() - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp3_exact_early_exit.json"))
    args = ap.parse_args()

    from sklearn.metrics import f1_score

    x, y, _ = load_test_split(args.data)
    forest = load_forest(os.path.join(ARTIFACTS, "ids_rf_numpy.npz"))
    bounds = margin_bounds(forest)
    # The label cannot be certain before more than half of the (equally weighted) trees have voted.
    first = forest.n_trees // 2 + 4
    schedule = [first] + [8] * ((forest.n_trees - first + 7) // 8)

    gate = Gate()
    forwarded = gate.score(x) >= gate.threshold
    report = {
        "rows": int(len(x)),
        "trees": forest.n_trees,
        "schedule_first_check_after": first,
        "gate_forward_rate": float(forwarded.mean()),
        "gate_attack_recall": float((forwarded & (y == 1)).sum() / (y == 1).sum()),
        "subsets": {},
    }

    for name, mask in (("gate_forwarded", forwarded), ("all_traffic", np.ones(len(x), dtype=bool))):
        xs, ys = x[mask], y[mask]
        full, t_full = run(lambda m: predict_lockstep(forest, m)[0], xs)
        ee, t_ee = run(lambda m: predict_early_exit(forest, m, schedule, bounds=bounds), xs)
        full = np.concatenate(full)
        labels = np.concatenate([e[0] for e in ee])
        used = np.concatenate([e[1] for e in ee])
        report["subsets"][name] = {
            "rows": int(len(xs)),
            "labels_identical_to_full_forest": int((labels == full).sum()),
            "attack_f1": round(float(f1_score(ys, labels)), 4),
            "mean_trees_used": round(float(used.mean()), 2),
            "rows_decided_at_first_check": float((used == first).mean()),
            "full_forest_seconds": round(t_full, 2),
            "early_exit_seconds": round(t_ee, 2),
            "speedup": round(t_full / t_ee, 2),
        }
        print(name, report["subsets"][name])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
