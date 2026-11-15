"""Load the labelled CICIDS2017 test split and the anomaly gate."""

import glob
import json
import os

import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The served artifacts live once, at the repository root, next to the package that serves them.
ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(CODE_DIR)), "artifacts")
RESULTS = os.path.join(CODE_DIR, "results")


def feature_columns():
    with open(os.path.join(ARTIFACTS, "feature_columns.json")) as f:
        return json.load(f)


def load_test_split(parquet_dir, limit=None):
    """Returns (float32 matrix in deployed column order, binary labels, attack names)."""
    import pyarrow.parquet as pq

    cols = feature_columns()
    parts = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    if not parts:
        raise FileNotFoundError(f"No parquet parts under {parquet_dir}")
    table = pq.read_table(parts, columns=cols + ["label", "label_binary"])
    if limit:
        table = table.slice(0, limit)
    x = np.column_stack([table[c].to_numpy(zero_copy_only=False) for c in cols]).astype(np.float64)
    # Same cleaning policy as the serving path: NaN/Inf -> 0.
    x[~np.isfinite(x)] = 0.0
    y = table["label_binary"].to_numpy(zero_copy_only=False).astype(np.int64)
    names = np.asarray(table["label"].to_pylist())
    return x.astype(np.float32), y, names


class Gate:
    """The deployed first stage: autoencoder reconstruction error against a fixed threshold."""

    def __init__(self):
        import joblib

        self.model = joblib.load(os.path.join(ARTIFACTS, "anomaly_autoencoder.pkl"))
        self.scaler = joblib.load(os.path.join(ARTIFACTS, "anomaly_scaler.pkl"))
        with open(os.path.join(ARTIFACTS, "anomaly_threshold.json")) as f:
            self.threshold = float(json.load(f)["threshold"])

    def score(self, matrix, batch=50000):
        out = np.empty(matrix.shape[0], dtype=np.float64)
        for i in range(0, matrix.shape[0], batch):
            xs = self.scaler.transform(matrix[i:i + batch])
            xh = self.model.predict(xs)
            out[i:i + batch] = np.mean((xs - xh) ** 2, axis=1)
        return out
