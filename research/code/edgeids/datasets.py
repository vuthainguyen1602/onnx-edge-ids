"""Flow datasets: tables, feature choice, and the CICIDS2017 / UNSW-NB15 / ToN-IoT splits the experiments use."""

import glob
import os

import numpy as np

from edgeids.data import load_test_split

LABEL_LIKE = {"label", "attack_cat", "attack", "type", "class", "category", "subcategory", "label_binary"}


IDENTIFIERS = {
    "id", "flow_id", "srcip", "dstip", "src_ip", "dst_ip", "source_ip", "destination_ip", "ip_src", "ip_dst",
    "sport", "dsport", "src_port", "dst_port", "source_port", "destination_port", "srcport", "dstport",
    "stime", "ltime", "ts", "timestamp", "time", "date", "datetime", "frame_time",
}


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


def load_cicids(train_dir, test_dir, train_rows=600000, seed=0):
    x_test, _, _ = load_test_split(test_dir)
    x_train, y_train, _ = load_test_split(train_dir)
    rng = np.random.default_rng(seed)
    if len(x_train) > train_rows:
        idx = np.sort(rng.choice(len(x_train), size=train_rows, replace=False))
        x_train, y_train = x_train[idx], y_train[idx]
    return x_train, y_train, x_test


def load_unsw(path_dir):
    train = read_table(os.path.join(path_dir, "UNSW_NB15_training-set.csv"))
    test = read_table(os.path.join(path_dir, "UNSW_NB15_testing-set.csv"))
    cols, _ = choose_features(train, "label", [], [])
    benign = ["benign", "normal", "0"]
    return matrix(train, cols), binary_labels(train["label"], benign), matrix(test, cols)


def load_ton_iot(csv):
    from sklearn.model_selection import train_test_split
    df = read_table(csv)
    cols, _ = choose_features(df, "label", [], [])
    x, y = matrix(df, cols), binary_labels(df["label"], ["benign", "normal", "0"])
    x_train, x_test, y_train, _ = train_test_split(x, y, test_size=0.3, stratify=y, random_state=0)
    return x_train, y_train, x_test


def unsw_test(path_dir):
    """UNSW-NB15 test matrix and the feature columns chosen on its training file."""
    train = read_table(os.path.join(path_dir, "UNSW_NB15_training-set.csv"))
    test = read_table(os.path.join(path_dir, "UNSW_NB15_testing-set.csv"))
    cols, _ = choose_features(train, "label", [], [])
    return matrix(test, cols), cols


def ton_iot_test(csv):
    """ToN-IoT held-out matrix of the stratified 70/30 split, and its feature columns."""
    from sklearn.model_selection import train_test_split
    df = read_table(csv)
    cols, _ = choose_features(df, "label", [], [])
    x, y = matrix(df, cols), binary_labels(df["label"], ["benign", "normal", "0"])
    _, x_test, _, _ = train_test_split(x, y, test_size=0.3, stratify=y, random_state=0)
    return x_test, cols
