"""Does boundary-directed parity testing generalize beyond the deployed forest?

Trains tree ensembles of several families on the same 29 CICIDS2017 features,
with the same StandardScaler front end as the deployed pipeline, exports each
to ONNX, and runs the boundary probe of exp2 on every one of them:

  * threshold sharing: how many distinct (feature, threshold) pairs the splits
    use and how many trees share each pair (binned trainers vs exact splits);
  * runtime parity: a float64 and a float32 NumPy traversal of the ONNX graph
    against ONNX Runtime, on boundary probes and on the replayed test split;
  * converter parity: the training framework's own prediction against ONNX
    Runtime, on the same probes and replay (the double-to-float export issue).

The deployed forest is included as the reference point; its
numbers should reproduce exp2 (815 pairs, 0 / 15 replay flips).
"""

import argparse
import json
import os
import time

import numpy as np

from edgeids.data import ARTIFACTS, RESULTS, load_test_split
from edgeids.models import build_models, export_pipeline
from edgeids.probes import probe_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="train_data.parquet directory")
    ap.add_argument("--test", required=True, help="test_data.parquet directory")
    ap.add_argument("--train-rows", type=int, default=600000)
    ap.add_argument("--max-nodes", type=int, default=100000, help="internal nodes probed per model (all if fewer)")
    ap.add_argument("--models", default="deployed_rf,sklearn_rf,sklearn_hgb,lightgbm,xgboost_hist")
    ap.add_argument("--model-dir", default=os.path.join(RESULTS, "exp5_models"))
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp5_generalization.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import platform

    import lightgbm
    import onnxmltools
    import onnxruntime
    import skl2onnx
    import sklearn
    import xgboost
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    os.makedirs(args.model_dir, exist_ok=True)
    x_test, _, _ = load_test_split(args.test)
    x_train, y_train, _ = load_test_split(args.train)
    rng = np.random.default_rng(args.seed)
    if len(x_train) > args.train_rows:
        idx = np.sort(rng.choice(len(x_train), size=args.train_rows, replace=False))
        x_train, y_train = x_train[idx], y_train[idx]
    print(f"train {len(x_train):,} rows ({y_train.mean():.3f} attack) | test {len(x_test):,} rows", flush=True)

    report = {
        "machine": f"{platform.machine()} {platform.system()}",
        "versions": {"numpy": np.__version__, "onnxruntime": onnxruntime.__version__, "sklearn": sklearn.__version__,
                     "lightgbm": lightgbm.__version__, "xgboost": xgboost.__version__, "skl2onnx": skl2onnx.__version__,
                     "onnxmltools": onnxmltools.__version__},
        "train_rows": int(len(x_train)), "test_rows": int(len(x_test)), "max_nodes_probed": args.max_nodes,
        "models": {},
    }
    if os.path.exists(args.out):
        with open(args.out) as f:
            report["models"] = json.load(f).get("models", {})

    catalogue = build_models(args.seed)
    for key in args.models.split(","):
        t0 = time.perf_counter()
        if key == "deployed_rf":
            desc, onnx_path, native = "deployed RandomForest (32-bin histogram trainer)", os.path.join(ARTIFACTS, "ids_rf.onnx"), None
            print(f"== {key}: {desc}", flush=True)
            entry = {"description": desc, "train_seconds": None}
        else:
            desc, est = catalogue[key]
            print(f"== {key}: {desc}", flush=True)
            pipe = Pipeline([("scaler", StandardScaler()), ("model", est)])
            pipe.fit(x_train.astype(np.float64), y_train)
            train_s = time.perf_counter() - t0
            onnx_path = os.path.join(args.model_dir, f"{key}.onnx")
            ops = export_pipeline(pipe, x_train.shape[1], onnx_path)
            print(f"   trained in {train_s:.0f}s, exported: {ops}", flush=True)
            native = lambda m, p=pipe: np.asarray(p.predict(m.astype(np.float64))).astype(np.int64)
            entry = {"description": desc, "train_seconds": round(train_s, 1), "onnx_ops": ops,
                     "onnx_bytes": os.path.getsize(onnx_path)}
        entry.update(probe_model(onnx_path, native, x_test, args.max_nodes, args.seed))
        entry["seconds"] = round(time.perf_counter() - t0, 1)
        report["models"][key] = entry
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"   done in {entry['seconds']}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
