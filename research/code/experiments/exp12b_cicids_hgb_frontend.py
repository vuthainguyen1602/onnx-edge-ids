"""The skl2onnx HistGradientBoosting threshold-rounding defect on CICIDS2017.

exp12 shows the defect on a toy dataset; exp5's Table 3 reports a 1,362-flow
framework-vs-export gap for the CICIDS2017 HistGradientBoosting. This splits
that gap into its two causes, under the released converter and under main:

  native_float64_scaler_vs_ort : framework (fed the float64 scikit-learn scaler)
                                 vs ONNX Runtime -- the Table 3 number.
  native_float32_scaled_vs_ort : framework fed the graph's own float32 Scaler
                                 output -- isolates the trees from the front end.

Under skl2onnx 1.20.0 the second number is the threshold rounding; onnx/sklearn-onnx
PR 1227 (on main) takes it to 0. Run once per installed converter version:

    python exp12b_cicids_hgb_frontend.py --train train_data.parquet --test test_data.parquet
    pip install git+https://github.com/onnx/sklearn-onnx.git@main && python exp12b_cicids_hgb_frontend.py ...
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edgeids.data import RESULTS, load_test_split
from edgeids.onnx_forest import load_onnx_forest
from exp5_generalization import export_pipeline, ort_labels


def flips(a, b):
    return int((np.asarray(a).astype(np.int64) != np.asarray(b).astype(np.int64)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--train-rows", type=int, default=600000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import skl2onnx
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    x, _, _ = load_test_split(args.test)
    xtr, ytr, _ = load_test_split(args.train)
    rng = np.random.default_rng(0)
    if len(xtr) > args.train_rows:
        idx = np.sort(rng.choice(len(xtr), size=args.train_rows, replace=False))
        xtr, ytr = xtr[idx], ytr[idx]
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("model", HistGradientBoostingClassifier(max_iter=200, max_bins=255, random_state=0))])
    pipe.fit(xtr.astype(np.float64), ytr)

    model_dir = os.path.join(RESULTS, "exp12b_models")
    os.makedirs(model_dir, exist_ok=True)
    onnx_path = os.path.join(model_dir, "cicids_hgb.onnx")
    export_pipeline(pipe, x.shape[1], onnx_path)
    forest = load_onnx_forest(onnx_path)
    lo = ort_labels(onnx_path, x)
    z = (x - forest.offset.astype(np.float32)) * forest.scale.astype(np.float32)
    report = {"skl2onnx": skl2onnx.__version__, "test_rows": int(len(x)),
              "native_float64_scaler_vs_ort_flips": flips(pipe.predict(x.astype(np.float64)), lo),
              "native_float32_scaled_vs_ort_flips": flips(pipe.named_steps["model"].predict(z.astype(np.float64)), lo)}
    print(json.dumps(report, indent=2))
    out = args.out or os.path.join(RESULTS, f"exp12b_cicids_hgb_frontend_skl2onnx_{skl2onnx.__version__}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
