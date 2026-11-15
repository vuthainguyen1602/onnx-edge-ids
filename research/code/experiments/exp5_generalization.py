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
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgeids.data import ARTIFACTS, RESULTS, load_test_split
from edgeids.onnx_forest import load_onnx_forest, path_rows, predict, reaches, rows_on_a_threshold, threshold_sharing, trees_moved

N_TREES = 200


def build_models(seed):
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from xgboost import XGBClassifier

    return {
        # exact splits: thresholds are midpoints between observed values
        "sklearn_rf": ("scikit-learn RandomForest (exact midpoint splits)",
                       RandomForestClassifier(n_estimators=N_TREES, max_depth=15, n_jobs=-1, random_state=seed)),
        # binned trainers
        "sklearn_hgb": ("scikit-learn HistGradientBoosting (max_bins=255)",
                        HistGradientBoostingClassifier(max_iter=N_TREES, max_bins=255, random_state=seed)),
        "lightgbm": ("LightGBM (max_bin=255)",
                     LGBMClassifier(n_estimators=N_TREES, max_bin=255, random_state=seed, n_jobs=8, verbose=-1)),
        "xgboost_hist": ("XGBoost tree_method=hist (max_bin=256, BRANCH_LT)",
                         XGBClassifier(n_estimators=N_TREES, max_depth=8, tree_method="hist", max_bin=256,
                                       random_state=seed, n_jobs=8)),
    }


def export_pipeline(pipe, n_features, out):
    import onnx
    from skl2onnx import convert_sklearn, update_registered_converter
    from skl2onnx.common.data_types import FloatTensorType
    from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
    try:
        from lightgbm import LGBMClassifier
        from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm
        update_registered_converter(LGBMClassifier, "LightGbmLGBMClassifier", calculate_linear_classifier_output_shapes,
                                    convert_lightgbm, options={"nocl": [True, False], "zipmap": [True, False, "columns"]})
    except ImportError:
        pass
    try:
        from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
        from xgboost import XGBClassifier
        update_registered_converter(XGBClassifier, "XGBoostXGBClassifier", calculate_linear_classifier_output_shapes,
                                    convert_xgboost, options={"nocl": [True, False], "zipmap": [True, False, "columns"]})
    except ImportError:
        pass
    # skl2onnx 1.20 hands numpy booleans to an INTS attribute (nodes_missing_value_tracks_true) for
    # HistGradientBoosting under recent scikit-learn/onnx; protobuf refuses them. Cast at the boundary.
    from skl2onnx.common import _container
    if not getattr(_container.ModelComponentContainer.add_node, "_bool_safe", False):
        _orig = _container.ModelComponentContainer.add_node

        def add_node(self, *args, **attrs):
            for k, v in list(attrs.items()):
                if isinstance(v, (list, tuple)) and len(v) and all(isinstance(e, (bool, np.bool_, int, np.integer)) for e in v):
                    attrs[k] = [int(e) for e in v]
            return _orig(self, *args, **attrs)
        add_node._bool_safe = True
        _container.ModelComponentContainer.add_node = add_node
    model = convert_sklearn(pipe, initial_types=[("input", FloatTensorType([None, n_features]))],
                            options={id(pipe.steps[-1][1]): {"zipmap": False}},
                            target_opset={"": 17, "ai.onnx.ml": 3})
    onnx.save_model(model, out)
    return [n.op_type for n in model.graph.node]


def ort_outputs(path, x, batch=50000):
    """(label output, label derived from the probability output) of ONNX Runtime.

    The derived label is class 1 iff P(class 1) > 0.5, read from column 1 of the
    probability output. For a binary TreeEnsembleClassifier with a single weight
    column, ONNX Runtime's own label output changed meaning between releases
    (1.19: P > 0.5; 1.27: P > 0), so the derived label is what the parity tests
    compare against, and the disagreement between the two is reported separately.
    """
    import onnxruntime as ort
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    labs, probs = [], []
    for i in range(0, len(x), batch):
        out = s.run(None, {name: x[i:i + batch]})
        labs.append(np.asarray(out[0]).reshape(-1))
        probs.append(np.asarray(out[1]))
    lab = np.concatenate(labs).astype(np.int64)
    prob = np.concatenate(probs)
    derived = (prob[:, 1] > 0.5).astype(np.int64) if prob.shape[1] == 2 else np.argmax(prob, axis=1).astype(np.int64)
    return lab, derived


def ort_labels(path, x, batch=50000):
    return ort_outputs(path, x, batch)[1]


def flips(a, b):
    return int((np.asarray(a).astype(np.int64) != np.asarray(b).astype(np.int64)).sum())


def probe_model(onnx_path, native, x_test, max_nodes, seed):
    forest = load_onnx_forest(onnx_path)
    rep = {"trees": forest.n_trees, "nodes": int(len(forest.left)), "post_transform": forest.post_transform,
           "split_modes": sorted(set(int(m) for m in forest.mode[~forest.is_leaf]))}
    rep["sharing"] = threshold_sharing(forest)

    rng = np.random.default_rng(seed)
    internal = np.flatnonzero(~forest.is_leaf)
    if len(internal) > max_nodes:
        internal = np.sort(rng.choice(internal, size=max_nodes, replace=False))
    feat = forest.feature[internal]
    base = x_test[rng.choice(len(x_test), size=len(internal), replace=True)]
    rows = path_rows(forest, internal, base)
    scale, offset = forest.scale.astype(np.float64), forest.offset.astype(np.float64)
    raw = np.where(scale[feat] != 0, forest.threshold[internal].astype(np.float64) / np.where(scale[feat] != 0, scale[feat], 1) + offset[feat], offset[feat]).astype(np.float32)
    variants = {"on_threshold": raw,
                "one_ulp_below": np.nextafter(raw, np.float32(-np.inf)),
                "one_ulp_above": np.nextafter(raw, np.float32(np.inf))}

    rep["probes"] = {"nodes_probed": int(len(internal)), "variants": {}}
    tot = {"probes": 0, "f32": 0, "f64": 0, "native": 0, "reach": 0}
    for name, values in variants.items():
        probes = rows.copy()
        probes[np.arange(len(internal)), feat] = values
        on_path = reaches(forest, probes, internal)
        lo = ort_labels(onnx_path, probes)
        l32 = predict(forest, probes, np.float32)
        l64 = predict(forest, probes, np.float64)
        moved = trees_moved(forest, probes)
        flipped = l64 != lo
        v = {"probes": int(len(probes)), "probes_reaching_their_node": int(on_path.sum()),
             "numpy_float32_vs_ort_flips": flips(l32, lo), "numpy_float64_vs_ort_flips": flips(l64, lo),
             "probes_where_float64_moves_a_tree": int((moved > 0).sum()),
             "trees_moved_mean_when_any": float(moved[moved > 0].mean()) if (moved > 0).any() else 0.0,
             "trees_moved_mean_when_label_flips": float(moved[flipped].mean()) if flipped.any() else 0.0,
             "label_flip_rate_given_a_tree_moved": float(flipped[moved > 0].mean()) if (moved > 0).any() else 0.0}
        if native is not None:
            v["native_vs_ort_flips"] = flips(native(probes), lo)
            tot["native"] += v["native_vs_ort_flips"]
        tot["probes"] += len(probes); tot["f32"] += v["numpy_float32_vs_ort_flips"]; tot["f64"] += v["numpy_float64_vs_ort_flips"]
        tot["reach"] += int(on_path.sum())
        rep["probes"]["variants"][name] = v
        print("   ", name, v, flush=True)
    rep["probes"]["total"] = {"probes": tot["probes"], "reaching_their_node": tot["reach"],
                              "numpy_float32_vs_ort_flips": tot["f32"], "numpy_float64_vs_ort_flips": tot["f64"]}
    if native is not None:
        rep["probes"]["total"]["native_vs_ort_flips"] = tot["native"]

    lo_out, lo = ort_outputs(onnx_path, x_test)
    l32, l64 = predict(forest, x_test, np.float32), predict(forest, x_test, np.float64)
    moved = trees_moved(forest, x_test)
    r = {"rows": int(len(x_test)), "ort_label_output_vs_ort_probability_label_flips": flips(lo_out, lo),
         "rows_with_a_feature_exactly_on_a_threshold_float32": rows_on_a_threshold(forest, x_test),
         "rows_where_float64_moves_a_tree": int((moved > 0).sum()),
         "trees_moved_mean_when_any": float(moved[moved > 0].mean()) if (moved > 0).any() else 0.0,
         "numpy_float32_vs_ort_flips": flips(l32, lo), "numpy_float64_vs_ort_flips": flips(l64, lo)}
    if native is not None:
        r["native_vs_ort_flips"] = flips(native(x_test), lo)
    for k in ("numpy_float64_vs_ort_flips", "native_vs_ort_flips"):
        if k in r:
            rate = r[k] / len(x_test)
            r[k.replace("_flips", "_rate")] = rate
            r[k.replace("_flips", "_prob_2000_row_replay_misses")] = float((1 - rate) ** 2000)
    rep["replay"] = r
    print("    replay", r, flush=True)

    # density: flips per probe vs flips per replayed row
    for k in ("numpy_float64_vs_ort", "native_vs_ort"):
        pk = k + "_flips"
        if pk in rep["probes"]["total"] and rep["replay"].get(pk, 0) > 0:
            rep["probes"]["total"][k + "_density_vs_replay"] = (rep["probes"]["total"][pk] / tot["probes"]) / (rep["replay"][pk] / len(x_test))
    return rep


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

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import platform, sklearn, onnxruntime, lightgbm, xgboost, skl2onnx, onnxmltools

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
