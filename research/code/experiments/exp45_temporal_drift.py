"""Temporal evaluation on CSE-CIC-IDS2018: runtime parity and detection, day by day.

exp4 and exp34 shift the data between datasets; this keeps one dataset and moves
forward in time. CSE-CIC-IDS2018 records ten working days (14 Feb - 2 Mar 2018), each
with its own attack families, so a model trained on the first days meets new attack
types later (concept drift) as well as shifted feature distributions (feature drift).
The train and test splits of the thesis pipeline are pooled and re-split by day; the
features are the deployed model's 29 columns.

Two protocols, both with scikit-learn RF and XGBoost (StandardScaler + model, exported
as in exp5) and, as a third model, the deployed forest trained on CICIDS2017:

  * frozen: train once on 14-16 Feb, test on each later day;
  * expanding: for each later day d, retrain on every earlier day and test on d.

Per test day (at most --day-rows rows) and model, kept apart:

  1. runtime parity: the float32 traversal vs ONNX Runtime on the day's rows (label
     flips, largest probability deviation), and the float64 traversal's divergence;
     for every trained model, the gate's data-free crossing pair, which needs no rows,
     passes the float32 traversal and fails the float64 one;
  2. replay power: probability that a 2,000-row confidence-checking replay passes the
     float64 traversal when the recording comes from the training days vs from the day;
  3. detection: precision, recall, F1, false-positive rate, and recall on attack types
     seen in training vs unseen;
  4. drift: population stability index of each feature against the training days, and
     the attack rate and share of attacks of unseen types.

    python research/code/experiments/exp45_temporal_drift.py --data2018 $DATA2018
"""

import argparse
import glob
import json
import os
import sys
import tempfile
import time

import numpy as np

from edgeids.data import ARTIFACTS, RESULTS, feature_columns
from edgeids.models import build_models, export_pipeline
from edgeids.onnx_forest import load_onnx_forest
from edgeids.runtime import numpy_probs, ort_probs

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
from onnx_edge_ids.boundary_probe import CROSSING_VARIANTS, build_probes  # noqa: E402

TOL = 1e-5
REPLAY_ROWS = 2000
TRAIN_DAYS = ("2018-02-14", "2018-02-15", "2018-02-16")
MODELS = ("sklearn_rf", "xgboost_hist")


def load_days(root):
    import pandas as pd
    import pyarrow.parquet as pq
    cols = feature_columns()
    frames = []
    for split in ("train", "test"):
        parts = sorted(glob.glob(os.path.join(root, f"{split}_data.parquet", "*.parquet")))
        frames.append(pq.read_table(parts, columns=cols + ["timestamp", "label", "label_binary"]).to_pandas())
    df = pd.concat(frames, ignore_index=True)
    df["day"] = pd.to_datetime(df["timestamp"], dayfirst=True).dt.strftime("%Y-%m-%d")
    df = df[df["day"] >= "2018-01-01"]            # three rows carry a 1970 timestamp
    x = df[cols].to_numpy(dtype=np.float64, na_value=0.0)
    x[~np.isfinite(x)] = 0.0                       # the serving path's cleaning policy
    return x.astype(np.float32), df["label_binary"].to_numpy().astype(np.int64), df["label"].to_numpy(), df["day"].to_numpy()


def psi(ref, cur, bins=10):
    """Population stability index of each column of ``cur`` against ``ref`` (quantile bins of ``ref``)."""
    out = []
    for j in range(ref.shape[1]):
        edges = np.unique(np.quantile(ref[:, j], np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            out.append(0.0 if np.all(cur[:, j] == ref[0, j]) else float("nan")); continue
        r = np.histogram(np.clip(ref[:, j], edges[0], edges[-1]), edges)[0] / len(ref)
        c = np.histogram(np.clip(cur[:, j], edges[0], edges[-1]), edges)[0] / len(cur)
        r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
        out.append(float(np.sum((c - r) * np.log(c / r))))
    return np.array(out)


def detection(y, pred, names, seen):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else None
    att = y == 1
    unseen = att & ~np.isin(names, list(seen))
    return {"precision": round(p, 4), "recall": None if r is None else round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if r and p + r else 0.0,
            "fpr": round(fp / (fp + tn), 5) if fp + tn else None,
            "recall_seen_types": round(float(pred[att & ~unseen].mean()), 4) if (att & ~unseen).any() else None,
            "recall_unseen_types": round(float(pred[unseen].mean()), 4) if unseen.any() else None}


def runtime_parity(path, forest, x):
    ort = ort_probs(path, x)
    f32 = numpy_probs(forest, x, np.float32)
    f64 = numpy_probs(forest, x, np.float64)
    lab = lambda p: (p[:, 1] > 0.5).astype(np.int64)  # noqa: E731
    over64 = (lab(f64) != lab(ort)) | np.any(np.abs(f64 - ort) > TOL, axis=1)
    return lab(ort), {
        "float32_label_flips": int((lab(f32) != lab(ort)).sum()),
        "float32_max_deviation": float(np.abs(f32 - ort).max()),
        "float64_label_flips": int((lab(f64) != lab(ort)).sum()),
        "float64_rows_over_tolerance": int(over64.sum()),
        "float64_divergence_rate": float(over64.mean())}


def datafree_gate(path, forest):
    pr = build_probes(path, CROSSING_VARIANTS)
    ort = ort_probs(path, pr.rows)
    out = {"probes": len(pr)}
    for name, dt in (("float32", np.float32), ("float64", np.float64)):
        p = numpy_probs(forest, pr.rows, dt)
        fails = ((p[:, 1] > 0.5) != (ort[:, 1] > 0.5)) | np.any(np.abs(p - ort) > TOL, axis=1)
        out[name] = {"fails": bool(fails.any()), "failing_probes": int(fails.sum())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data2018", required=True, help="directory holding train_data.parquet and test_data.parquet")
    ap.add_argument("--day-rows", type=int, default=200000, help="rows sampled per test day")
    ap.add_argument("--train-rows", type=int, default=600000)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp45_temporal_drift.json"))
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    x, y, names, day = load_days(args.data2018)
    days = sorted(set(day))
    test_days = [d for d in days if d not in TRAIN_DAYS]
    tmp = tempfile.mkdtemp()

    def take(mask, limit):
        idx = np.flatnonzero(mask)
        return np.sort(rng.choice(idx, limit, replace=False)) if len(idx) > limit else idx

    test_idx = {d: take(day == d, args.day_rows) for d in test_days}
    frozen_train = take(np.isin(day, TRAIN_DAYS), args.train_rows)
    ref = x[frozen_train]
    report = {"settings": {"train_days": TRAIN_DAYS, "test_days": test_days, "day_rows": args.day_rows,
                           "train_rows": args.train_rows, "tolerance": TOL, "replay_rows": REPLAY_ROWS},
              "days": {}, "frozen": {}, "expanding": {}}

    # drift of each test day against the frozen training days
    seen_frozen = set(names[np.isin(day, TRAIN_DAYS) & (y == 1)])
    for d in test_days:
        i = test_idx[d]
        p = psi(ref, x[i])
        att = y[i] == 1
        report["days"][d] = {
            "rows": int((day == d).sum()), "rows_used": int(len(i)), "attack_rate": round(float(att.mean()), 4),
            "attack_types": sorted(set(names[i][att])),
            "unseen_attack_share_frozen": round(float((~np.isin(names[i][att], list(seen_frozen))).mean()), 4) if att.any() else None,
            "psi_mean": round(float(np.nanmean(p)), 3), "psi_max": round(float(np.nanmax(p)), 3),
            "features_psi_over_0.25": int(np.sum(p > 0.25))}
        print(d, report["days"][d], flush=True)

    def training_rate(path, forest, train_idx):
        """float64 divergence rate on a 50,000-row sample of the training rows (a recording made at training time)."""
        tr = np.sort(rng.choice(train_idx, min(50000, len(train_idx)), replace=False))
        return runtime_parity(path, forest, x[tr])[1]["float64_divergence_rate"]

    def evaluate(path, forest, train_rate, seen, d):
        i = test_idx[d]
        lab, par = runtime_parity(path, forest, x[i])
        # replay power against the float64 traversal: recording from the training days vs from the day
        return {**par, "float64_divergence_rate_training_days": train_rate,
                "replay_pass_float64_recording_training_days": float((1 - train_rate) ** REPLAY_ROWS),
                "replay_pass_float64_recording_this_day": float((1 - par["float64_divergence_rate"]) ** REPLAY_ROWS),
                "detection": detection(y[i], lab, names[i], seen)}

    # deployed forest: trained on CICIDS2017, frozen
    dep = os.path.join(ARTIFACTS, "ids_rf.onnx")
    dep_forest = load_onnx_forest(dep)
    report["deployed_rf"] = {"note": "trained on CICIDS2017; every 2018 attack type is unseen", "days": {}}
    for d in test_days:
        i = test_idx[d]
        lab, par = runtime_parity(dep, dep_forest, x[i])
        report["deployed_rf"]["days"][d] = {**par, "detection": detection(y[i], lab, names[i], set())}
        print("deployed_rf", d, par["float32_label_flips"], report["deployed_rf"]["days"][d]["detection"], flush=True)

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    for key in args.models.split(","):
        # frozen
        t0 = time.perf_counter()
        pipe = Pipeline([("scaler", StandardScaler()), ("model", build_models(0)[key][1])])
        pipe.fit(x[frozen_train], y[frozen_train])
        path = os.path.join(tmp, f"{key}_frozen.onnx")
        export_pipeline(pipe, x.shape[1], path)
        forest = load_onnx_forest(path)
        entry = {"train_seconds": round(time.perf_counter() - t0, 1), "gate": datafree_gate(path, forest), "days": {}}
        rate = training_rate(path, forest, frozen_train)
        for d in test_days:
            entry["days"][d] = evaluate(path, forest, rate, seen_frozen, d)
            print("frozen", key, d, entry["days"][d]["float32_label_flips"], entry["days"][d]["detection"], flush=True)
        report["frozen"][key] = entry
        # expanding window
        entry = {"days": {}}
        for d in test_days:
            past = take(day < d, args.train_rows)
            seen = set(names[(day < d) & (y == 1)])
            pipe = Pipeline([("scaler", StandardScaler()), ("model", build_models(0)[key][1])])
            pipe.fit(x[past], y[past])
            path = os.path.join(tmp, f"{key}_{d}.onnx")
            export_pipeline(pipe, x.shape[1], path)
            forest = load_onnx_forest(path)
            r = evaluate(path, forest, training_rate(path, forest, past), seen, d)
            r["gate"] = datafree_gate(path, forest)
            r["train_days"] = sorted(set(day[past]))
            entry["days"][d] = r
            print("expanding", key, d, r["float32_label_flips"], r["gate"]["float32"]["fails"], r["gate"]["float64"]["fails"],
                  r["detection"], flush=True)
            report["expanding"][key] = entry
            with open(args.out, "w") as f:
                json.dump(report, f, indent=1, default=str)
        report["expanding"][key] = entry

    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
