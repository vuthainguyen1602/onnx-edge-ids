"""Ablation of the block-lockstep traversal's block budget (pairs advanced per block).

The shipped traversal advances blocks of about PAIRS_PER_BLOCK = 8,192
(tree, row) pairs at a time (edgeids.forest); whole-forest lockstep, one block
of all 200 trees, was slower than the per-tree loop at batch size 2,000 on real
flows. This sweeps the budget from 1,024 pairs to whole-forest at the batch
sizes of exp1, on the same real flows and with the same best-of-five timer, and
checks that every budget returns the same labels.

    python research/code/experiments/exp23_lockstep_block_ablation.py --data test_data.parquet
    PYTHONPATH=research/code python research/code/experiments/exp23_lockstep_block_ablation.py --npy rows.npy   # on a board without pandas or the installed library

--export-npy writes the 20,000 rows used here, so a board can run the same rows.
"""

import argparse
import json
import os
import platform
import time

import numpy as np

import edgeids.forest as F
from edgeids.data import ARTIFACTS, RESULTS

PLAN = [(1, 500), (20, 3000), (100, 10000), (500, 10000), (2000, 20000)]  # (batch size, rows), as exp1
BUDGETS = [1024, 2048, 4096, 8192, 16384, 32768, 65536, None]          # None = whole forest in one block


def throughput(forest, x, batch, reps=5):
    best = np.inf
    for _ in range(reps):
        start = time.perf_counter()
        for i in range(0, len(x), batch):
            F.predict_lockstep(forest, x[i:i + batch])
        best = min(best, time.perf_counter() - start)
    return len(x) / best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="test_data.parquet directory")
    ap.add_argument("--npy", help="float32 rows saved with --export-npy")
    ap.add_argument("--export-npy", help="write the 20,000 rows used here to this .npy and exit")
    ap.add_argument("--npz", default=os.path.join(ARTIFACTS, "ids_rf_numpy.npz"))
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp23_lockstep_block_ablation.json"))
    args = ap.parse_args()

    if args.npy:
        x = np.load(args.npy)
    else:
        from edgeids.data import load_test_split
        x, _, _ = load_test_split(args.data, limit=20000)
    x = np.ascontiguousarray(x[:20000], dtype=np.float32)
    if args.export_npy:
        np.save(args.export_npy, x)
        print("wrote", args.export_npy, x.shape)
        return

    forest = F.load_forest(args.npz)
    shipped = F.PAIRS_PER_BLOCK
    F.PAIRS_PER_BLOCK = shipped
    ref, _ = F.predict_lockstep(forest, x[:5000])
    report = {"machine": f"{platform.machine()} {platform.system()} {platform.processor()}",
              "numpy": np.__version__, "trees": forest.n_trees, "shipped_budget": shipped, "results": []}
    for budget in BUDGETS:
        F.PAIRS_PER_BLOCK = budget if budget is not None else forest.n_trees * 20000
        labels, _ = F.predict_lockstep(forest, x[:5000])
        row = {"budget": budget if budget is not None else "whole forest",
               "labels_identical_5000": int((labels == ref).sum()), "rows_s": {}}
        for batch, n in PLAN:
            row["rows_s"][str(batch)] = round(throughput(forest, x[:n], batch), 1)
        report["results"].append(row)
        print(json.dumps(row), flush=True)
    F.PAIRS_PER_BLOCK = shipped

    # best budget per batch size, and how far the shipped budget is from it
    report["best_by_batch"] = {}
    for batch, _ in PLAN:
        best = max(report["results"], key=lambda r: r["rows_s"][str(batch)])
        ship = next(r for r in report["results"] if r["budget"] == shipped)
        report["best_by_batch"][str(batch)] = {"budget": best["budget"], "rows_s": best["rows_s"][str(batch)],
                                               "shipped_rows_s": ship["rows_s"][str(batch)],
                                               "shipped_share_of_best": round(ship["rows_s"][str(batch)] / best["rows_s"][str(batch)], 3)}
    print(json.dumps(report["best_by_batch"]))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
