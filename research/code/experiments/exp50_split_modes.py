"""Probes for equality splits, categorical splits and missing values (the gate's builder handles LEQ/LT only).

exp47 met a graph the release gate cannot probe: onnxmltools's fix for LightGBM zero-as-missing
writes BRANCH_EQ nodes in front of the splits, and shares the split's default child between
the two (the tree becomes a DAG). This runs the extended builder of ``edgeids/probes_ext.py``
(crossing pair for monotone splits, the values on and around the threshold for equality
splits, a NaN probe per node) on three kinds of graph:

  zero_as_missing  LightGBM zero_as_missing=True on CICIDS2017, the released and the fixed
                   converter output of exp47;
  categorical      LightGBM on UNSW-NB15 with proto, service and state as categorical features;
  missing          XGBoost and LightGBM trained on CICIDS2017 with 5% of three features set to NaN.

Per graph: the split modes and missing-value flags it carries, whether the gate's own builder
accepts it, the extended probes (count, share reaching their node), and three comparisons on
them: the NumPy float32 traversal (onnx_forest, which reads equality modes and missing-value
flags) against ONNX Runtime, the training framework against ONNX Runtime, and the shipped
NumPy engine (src/, via the exporter) against ONNX Runtime, each as failing probes by kind.

    .lpvenv/bin/python research/code/experiments/exp50_split_modes.py --cicids-train $TRAIN --cicids-test $DATA --unsw $UNSW \\
        --fixed-python .e3venv/bin/python      # .e3venv with onnxmltools@b9f9169bfb, as left by exp47
"""

import argparse
import collections
import json
import os
import subprocess
import sys
import tempfile

import joblib
import numpy as np

from edgeids.data import RESULTS
from edgeids.datasets import choose_features, load_cicids, read_table
from edgeids.onnx_forest import load_onnx_forest, reaches
from edgeids.probes_ext import KINDS, build
from edgeids.runtime import numpy_probs, ort_probs

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
TOL = 1e-5


def convert_lgbm(m, nf, out):
    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType as FT
    onx = onnxmltools.convert_lightgbm(m, initial_types=[("input", FT([None, nf]))], target_opset=15, zipmap=False)
    open(out, "wb").write(onx.SerializeToString())
    return out


def convert_xgb(m, nf, out):
    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType as FT
    onx = onnxmltools.convert_xgboost(m, initial_types=[("input", FT([None, nf]))], target_opset=15)
    open(out, "wb").write(onx.SerializeToString())
    return out


def convert_with(python, kind, model, nf, out, tmp):
    """Convert a pickled model with another interpreter (e.g. the fixed onnxmltools of exp47's .e3venv)."""
    pkl = os.path.join(tmp, "m.pkl")
    joblib.dump(model, pkl)
    code = ("import joblib,onnxmltools,sys;from onnxmltools.convert.common.data_types import FloatTensorType as FT;"
            f"m=joblib.load({pkl!r});t=[('input',FT([None,{nf}]))];"
            + ("o=onnxmltools.convert_lightgbm(m,initial_types=t,target_opset=15,zipmap=False)" if kind == "lgbm"
               else "o=onnxmltools.convert_xgboost(m,initial_types=t,target_opset=15)")
            + f";open({out!r},'wb').write(o.SerializeToString());print(onnxmltools.__version__)")
    r = subprocess.run([python, "-c", code], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip().splitlines()[-1])
    return out


def failing(a, b):
    return ((a[:, 1] > 0.5) != (b[:, 1] > 0.5)) | np.any(np.abs(a - b) > TOL, axis=1) | (np.isnan(a).any(axis=1) != np.isnan(b).any(axis=1))


def by_kind(mask, kind):
    return {KINDS[k]: int(mask[kind == k].sum()) for k in np.unique(kind)}


def evaluate(name, path, model, tmp):
    f = load_onnx_forest(path)
    internal = ~f.is_leaf
    row = {"split_modes": {str(k): int(v) for k, v in collections.Counter(f.mode[internal].tolist()).items()},
           "nodes_with_missing_true": int(f.missing_true[internal].sum()), "internal_nodes": int(internal.sum())}
    from edgeids.onnx_forest import parents
    kids = np.concatenate([f.left[internal], f.right[internal]])
    row["shared_children"] = int((np.bincount(kids, minlength=len(f.left)) > 1).sum())
    from onnx_edge_ids.boundary_probe import CROSSING_VARIANTS, build_probes
    try:
        build_probes(path, CROSSING_VARIANTS)
        row["gate_builder"] = "ok"
    except Exception as e:
        row["gate_builder"] = f"{type(e).__name__}: {e}"[:200]
    pr = build(f)
    finite = ~np.isnan(pr.rows[np.arange(len(pr.node)), f.feature[pr.node]])
    reach = np.zeros(len(pr.node), dtype=bool)
    reach[finite] = reaches(f, pr.rows[finite], pr.node[finite])
    row["probes"] = {"total": int(len(pr.node)), "by_kind": by_kind(np.ones(len(pr.node), bool), pr.kind),
                     "reach_non_nan": round(float(reach[finite].mean()), 4)}
    ort = ort_probs(path, pr.rows)
    npy = numpy_probs(f, pr.rows, np.float32)
    fn = failing(npy, ort)
    row["numpy_float32_vs_ort"] = {"failing": int(fn.sum()), "by_kind": by_kind(fn, pr.kind),
                                   "max_deviation": float(np.nanmax(np.abs(npy - ort)))}
    fw = failing(model.predict_proba(pr.rows).astype(np.float64), ort)
    row["framework_vs_ort"] = {"failing": int(fw.sum()), "by_kind": by_kind(fw, pr.kind)}
    npz = os.path.join(tmp, f"{name}.npz")
    out = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "export_numpy.py"), "--model", path, "--out", npz],
                         capture_output=True, text=True)
    if out.returncode != 0:
        row["shipped_numpy_engine"] = {"export": (out.stderr.strip().splitlines() or ["failed"])[-1][:200]}
    else:
        from onnx_edge_ids import create_inference_engine
        eng = create_inference_engine("numpy", npz)
        _, conf = eng._infer(pr.rows)
        se = failing(np.asarray(conf, dtype=np.float64), ort)
        row["shipped_numpy_engine"] = {"export": "ok", "failing": int(se.sum()), "by_kind": by_kind(se, pr.kind)}
    print(name, json.dumps(row), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cicids-train", required=True)
    ap.add_argument("--cicids-test", required=True)
    ap.add_argument("--unsw", required=True)
    ap.add_argument("--fixed-python", default="", help="interpreter with the fixed onnxmltools (exp47's .e3venv)")
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp50_split_modes.json"))
    args = ap.parse_args()
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    import lightgbm, onnxmltools, onnxruntime, xgboost
    tmp = tempfile.mkdtemp()
    report = {"versions": {"onnxmltools": onnxmltools.__version__, "lightgbm": lightgbm.__version__,
                           "xgboost": xgboost.__version__, "onnxruntime": onnxruntime.__version__},
              "tolerance": TOL, "graphs": {}}

    # zero-as-missing: the graphs and model of exp47
    d47 = os.path.join(RESULTS, "exp47_models")
    m = joblib.load(os.path.join(d47, "lgbm_zero.pkl"))
    for tag in ("released", "fixed"):
        report["graphs"][f"zero_as_missing/{tag}"] = evaluate(f"zero_{tag}", os.path.join(d47, f"lgbm_zero__{tag}.onnx"), m, tmp)

    # categorical splits on UNSW-NB15
    import pandas as pd
    df = read_table(os.path.join(args.unsw, "UNSW_NB15_training-set.csv"))
    num, _ = choose_features(df, "label", [], [])
    cats = [c for c in ("proto", "service", "state") if c in df.columns]
    x = df[num].to_numpy(dtype=np.float64, na_value=0.0)
    x[~np.isfinite(x)] = 0.0
    xc = np.column_stack([pd.Categorical(df[c]).codes.astype(np.float64) for c in cats])
    X = np.hstack([x, xc]).astype(np.float32)
    y = df["label"].to_numpy().astype(np.int64)
    cat_idx = list(range(x.shape[1], X.shape[1]))
    m = LGBMClassifier(n_estimators=200, max_bin=255, random_state=0, n_jobs=8, verbose=-1).fit(X, y, categorical_feature=cat_idx)
    try:
        p = convert_lgbm(m, X.shape[1], os.path.join(tmp, "cat.onnx"))
        report["graphs"]["categorical/lightgbm"] = evaluate("categorical", p, m, tmp)
        if args.fixed_python:
            p = convert_with(args.fixed_python, "lgbm", m, X.shape[1], os.path.join(tmp, "cat_fixed.onnx"), tmp)
            report["graphs"]["categorical/lightgbm (fixed converter)"] = evaluate("categorical_fixed", p, m, tmp)
    except Exception as e:
        report["graphs"]["categorical/lightgbm"] = {"conversion": f"{type(e).__name__}: {e}"[:300]}
        print("categorical conversion failed:", e)
    report["graphs"]["categorical/lightgbm"]["categorical_features"] = cats

    # missing values
    xt, yt, _ = load_cicids(args.cicids_train, args.cicids_test)
    xt = xt.astype(np.float32).copy()
    rng = np.random.default_rng(0)
    for j in (11, 12, 13):                       # flow_bytes_s, flow_packets_s, flow_iat_mean
        xt[rng.random(len(xt)) < 0.05, j] = np.nan
    for name, model, conv in (("missing/xgboost", XGBClassifier(n_estimators=200, max_depth=8, tree_method="hist", max_bin=256,
                                                                 random_state=0, n_jobs=8), convert_xgb),
                              ("missing/lightgbm", LGBMClassifier(n_estimators=200, max_bin=255, random_state=0, n_jobs=8,
                                                                  verbose=-1), convert_lgbm)):
        model.fit(xt, yt)
        p = conv(model, xt.shape[1], os.path.join(tmp, name.replace("/", "_") + ".onnx"))
        report["graphs"][name] = evaluate(name.replace("/", "_"), p, model, tmp)
        if args.fixed_python:
            kind = "lgbm" if "lightgbm" in name else "xgb"
            p = convert_with(args.fixed_python, kind, model, xt.shape[1], os.path.join(tmp, name.replace("/", "_") + "_fixed.onnx"), tmp)
            report["graphs"][name + " (fixed converter)"] = evaluate(name.replace("/", "_") + "_fixed", p, model, tmp)

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1, default=str)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
