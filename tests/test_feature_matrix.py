"""The vectorized matrix builder must match ``clean_value`` applied cell by cell."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from onnx_edge_ids import FeatureMatrixBuilder, clean_value  # noqa: E402


def _reference(columns, rows, dtype):
    out = np.empty((len(rows), len(columns)), dtype=dtype)
    for i, raw in enumerate(rows):
        for j, name in enumerate(columns):
            out[i, j] = clean_value(raw.get(name, 0.0))
    return out


def _builder(tmp_path, columns, dtype=np.float32):
    path = tmp_path / "features.json"
    path.write_text(json.dumps(columns))
    return FeatureMatrixBuilder(path, dtype=dtype)


def test_numeric_rows_match_reference(tmp_path):
    columns = [f"f{i}" for i in range(29)]
    rng = np.random.default_rng(0)
    rows = [{c: float(v) for c, v in zip(columns, rng.lognormal(3.0, 3.0, len(columns)))}
            for _ in range(200)]
    for dtype in (np.float32, np.float64):
        builder = _builder(tmp_path, columns, dtype)
        got = builder.build(rows)
        assert got.dtype == dtype
        assert np.array_equal(got, _reference(columns, rows, dtype))


def test_dirty_values_match_reference(tmp_path):
    columns = ["a", "b", "c", "d"]
    rows = [
        {"a": "1.5", "b": "nan", "c": "inf", "d": 7},              # CSV strings, int
        {"a": None, "b": float("-inf"), "c": True},                 # missing "d"
        {"a": "", "b": "abc", "c": [1, 2], "d": 10 ** 400},         # unparseable, overflow
        {"a": 1e39, "b": -0.0, "c": " 2.5 ", "d": float("nan")},    # beyond float32
    ]
    builder = _builder(tmp_path, columns)
    assert np.array_equal(builder.build(rows), _reference(columns, rows, np.float32))
    assert np.array_equal(builder.build(rows[:2]), _reference(columns, rows[:2], np.float32))


def test_empty_batch(tmp_path):
    builder = _builder(tmp_path, ["a", "b"])
    assert builder.build([]).shape == (0, 2)
