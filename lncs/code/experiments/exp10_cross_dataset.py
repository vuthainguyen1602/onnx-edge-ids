"""Does boundary-directed parity testing carry over to other intrusion datasets?

exp5 trains several tree-ensemble families on CICIDS2017 only. This trains the
same families, with the same StandardScaler front end and the same exporters,
on another labelled flow dataset (UNSW-NB15, ToN-IoT, CSE-CIC-IDS2018, ...),
then runs exp5's probe on each export: threshold sharing, float32/float64
NumPy traversal vs ONNX Runtime, framework vs ONNX Runtime, on boundary probes
and on the dataset's own held-out rows. One run handles one dataset and adds
it to the results file under --name.

Each dataset has its own feature space. Features are the numeric columns of
the input minus the label columns and minus identifier columns (addresses,
ports, timestamps, row ids), which let a model learn the capture rather than
the traffic; the columns actually used are written to the report so the
choice can be audited, and --drop / --keep override it.

    python lncs/code/experiments/exp10_cross_dataset.py --name UNSW-NB15 \
        --train UNSW_NB15_training-set.csv --test UNSW_NB15_testing-set.csv --label-col label
    python lncs/code/experiments/exp10_cross_dataset.py --name ToN-IoT \
        --train train_test_network.csv --label-col label          # no --test: 70/30 split
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from edgeids.data import RESULTS
from edgeids.onnx_forest import load_onnx_forest
from exp5_generalization import build_models, export_pipeline, flips, ort_labels, probe_model

# Lower-cased, with spaces/dashes/dots as underscores. Identifiers of the
# capture, not properties of the flow.
IDENTIFIERS = {
    "id", "flow_id", "srcip", "dstip", "src_ip", "dst_ip", "source_ip", "destination_ip", "ip_src", "ip_dst",
    "sport", "dsport", "src_port", "dst_port", "source_port", "destination_port", "srcport", "dstport",
    "stime", "ltime", "ts", "timestamp", "time", "date", "datetime", "frame_time",
}
# Label-like columns other than --label-col (attack family, attack type).
LABEL_LIKE = {"label", "attack_cat", "attack", "type", "class", "category", "subcategory", "label_binary"}


def norm(name):
    return "_".join(name.replace("\ufeff", "").strip().lower().replace("-", " ").replace(".", " ").replace("/", " ").split())


def read_table(spec):
    """A CSV or Parquet file, a directory of them, or a glob, as one pandas DataFrame."""
    import pandas as pd
    paths = sorted(glob.glob(os.path.join(spec, "*"))) if os.path.isdir(spec) else sorted(glob.glob(spec))
    paths = [p for p in paths if p.endswith((".csv", ".parquet"))]
    if not paths:
        raise SystemExit(f"[ERR] no .csv/.parquet under {spec}")
    frames = [pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p, low_memory=False, encoding="utf-8-sig") for p in paths]
    return pd.concat(frames, ignore_index=True)


def binary_labels(column, benign):
    """1 = attack. A numeric 0/1 column is used as is; otherwise every value outside ``benign`` is an attack."""
    import pandas as pd
    if pd.api.types.is_numeric_dtype(column) and set(np.unique(column.dropna())) <= {0, 1}:
        return column.fillna(0).astype(np.int64).to_numpy()
    benign = {b.strip().lower() for b in benign}
    return (~column.astype(str).str.strip().str.lower().isin(benign)).astype(np.int64).to_numpy()


def choose_features(df, label_col, drop, keep):
    import pandas as pd
    if keep:
        missing = [c for c in keep if c not in df.columns]
        if missing:
            raise SystemExit(f"[ERR] --keep columns not in the data: {missing}")
        return list(keep), {}
    dropped = {}
    out = []
    drop = {norm(d) for d in drop}
    for c in df.columns:
        n = norm(c)
        if c == label_col or n in LABEL_LIKE:
            dropped[c] = "label"
        elif n in IDENTIFIERS or n in drop:
            dropped[c] = "identifier"
        elif not pd.api.types.is_numeric_dtype(df[c]):
            dropped[c] = f"non-numeric ({df[c].dtype})"
        elif df[c].nunique(dropna=True) <= 1:
            dropped[c] = "constant"
        else:
            out.append(c)
    return out, dropped


def matrix(df, cols):
    # same cleaning policy as the serving path: NaN/Inf -> 0, float32 on the wire
    x = df[cols].to_numpy(dtype=np.float64, na_value=0.0)
    x[~np.isfinite(x)] = 0.0
    return x.astype(np.float32)


def sample(rng, n, limit):
    return np.sort(rng.choice(n, size=limit, replace=False)) if limit and n > limit else np.arange(n)


def front_end_flips(onnx_path, pipe, x):
    """Framework vs ORT once the framework gets the graph's float32 Scaler output instead of the
    float64 scikit-learn scaler: what is left is a disagreement in the trees, not the front end."""
    forest = load_onnx_forest(onnx_path)
    z = (x - forest.offset.astype(np.float32)) * forest.scale.astype(np.float32)
    return flips(pipe.named_steps["model"].predict(z.astype(np.float64)), ort_labels(onnx_path, x))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True, help="dataset name in the report, e.g. UNSW-NB15")
    ap.add_argument("--train", required=True, help="CSV/Parquet file, directory or glob")
    ap.add_argument("--test", default="", help="held-out split; omit to split --train 70/30 (stratified)")
    ap.add_argument("--label-col", required=True)
    ap.add_argument("--benign", default="benign,normal,0",
                    help="label values counted as benign when the label column is not 0/1")
    ap.add_argument("--drop", default="", help="extra comma-separated columns to leave out")
    ap.add_argument("--keep", default="", help="comma-separated feature columns to use instead of the automatic choice")
    ap.add_argument("--models", default="sklearn_rf,sklearn_hgb,lightgbm,xgboost_hist")
    ap.add_argument("--train-rows", type=int, default=600000)
    ap.add_argument("--test-rows", type=int, default=500000, help="held-out rows replayed per model (0 = all)")
    ap.add_argument("--max-nodes", type=int, default=100000, help="internal nodes probed per model (all if fewer)")
    ap.add_argument("--model-dir", default=os.path.join(RESULTS, "exp10_models"))
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp10_cross_dataset.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import platform
    import lightgbm, onnxmltools, onnxruntime, skl2onnx, sklearn, xgboost
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(args.seed)
    train_df = read_table(args.train)
    if args.label_col not in train_df.columns:
        raise SystemExit(f"[ERR] --label-col {args.label_col!r} not in {list(train_df.columns)}")
    cols, dropped = choose_features(train_df, args.label_col,
                                    [d for d in args.drop.split(",") if d], [k for k in args.keep.split(",") if k])
    benign = args.benign.split(",")
    x_all, y_all = matrix(train_df, cols), binary_labels(train_df[args.label_col], benign)
    if args.test:
        test_df = read_table(args.test)
        x_train, y_train = x_all, y_all
        x_test, y_test = matrix(test_df, cols), binary_labels(test_df[args.label_col], benign)
        split = f"given test split {args.test}"
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x_all, y_all, test_size=0.3, stratify=y_all, random_state=args.seed)
        split = "stratified 70/30 split of --train"
    del train_df
    if len(np.unique(y_train)) < 2:
        raise SystemExit(f"[ERR] training labels have one class only; check --label-col/--benign")
    i = sample(rng, len(x_train), args.train_rows)
    x_train, y_train = x_train[i], y_train[i]
    i = sample(rng, len(x_test), args.test_rows)
    x_test, y_test = x_test[i], y_test[i]
    print(f"{args.name}: {len(cols)} features | train {len(x_train):,} ({y_train.mean():.3f} attack) | "
          f"test {len(x_test):,} ({y_test.mean():.3f} attack)", flush=True)
    print(f"features: {cols}", flush=True)
    if dropped:
        print(f"left out: {dropped}", flush=True)

    report = {"datasets": {}}
    if os.path.exists(args.out):
        with open(args.out) as f:
            report = json.load(f)
    report["machine"] = f"{platform.machine()} {platform.system()}"
    report["versions"] = {"numpy": np.__version__, "onnxruntime": onnxruntime.__version__, "sklearn": sklearn.__version__,
                          "lightgbm": lightgbm.__version__, "xgboost": xgboost.__version__,
                          "skl2onnx": skl2onnx.__version__, "onnxmltools": onnxmltools.__version__}
    entry = {
        "train": args.train, "test": args.test or None, "split": split, "label_col": args.label_col,
        "benign_values": benign, "features": cols, "left_out": dropped,
        "train_rows": int(len(x_train)), "test_rows": int(len(x_test)),
        "train_attack_share": float(y_train.mean()), "test_attack_share": float(y_test.mean()),
        "max_nodes_probed": args.max_nodes, "seed": args.seed, "models": {},
    }
    report["datasets"][args.name] = entry
    model_dir = os.path.join(args.model_dir, norm(args.name))
    os.makedirs(model_dir, exist_ok=True)

    from sklearn.metrics import f1_score

    catalogue = build_models(args.seed)
    for key in args.models.split(","):
        desc, est = catalogue[key]
        print(f"== {args.name} / {key}: {desc}", flush=True)
        t0 = time.perf_counter()
        pipe = Pipeline([("scaler", StandardScaler()), ("model", est)])
        pipe.fit(x_train.astype(np.float64), y_train)
        train_s = time.perf_counter() - t0
        onnx_path = os.path.join(model_dir, f"{key}.onnx")
        ops = export_pipeline(pipe, x_train.shape[1], onnx_path)
        native = lambda m, p=pipe: np.asarray(p.predict(m.astype(np.float64))).astype(np.int64)
        m = {"description": desc, "train_seconds": round(train_s, 1), "onnx_ops": ops,
             "onnx_bytes": os.path.getsize(onnx_path),
             "test_f1_native": float(f1_score(y_test, native(x_test)))}
        print(f"   trained in {train_s:.0f}s, test F1 {m['test_f1_native']:.3f}, exported: {ops}", flush=True)
        m.update(probe_model(onnx_path, native, x_test, args.max_nodes, args.seed))
        m["replay"]["native_on_float32_scaled_input_vs_ort_flips"] = front_end_flips(onnx_path, pipe, x_test)
        print(f"   framework fed the graph's float32 scaling: {m['replay']['native_on_float32_scaled_input_vs_ort_flips']} replay flips", flush=True)
        m["seconds"] = round(time.perf_counter() - t0, 1)
        entry["models"][key] = m
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"   done in {m['seconds']}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
