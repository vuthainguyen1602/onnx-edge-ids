#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import numpy as np


def load_feature_columns(features_path):
    path = Path(features_path)
    with path.open("r") as f:
        columns = json.load(f)
    if not isinstance(columns, list) or not all(isinstance(c, str) for c in columns):
        raise ValueError(f"Feature file must contain a JSON list of strings: {path}")
    return columns


def clean_value(value):
    if value is None:
        return 0.0
    try:
        v = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if v != v or v == float("inf") or v == float("-inf"):
        return 0.0
    return v


class FeatureMatrixBuilder:
    """Convert Kafka-style flow dictionaries to a dense numeric matrix."""

    def __init__(self, features_path, dtype=np.float32):
        self.feature_columns = load_feature_columns(features_path)
        self.dtype = dtype

    @property
    def n_features(self):
        return len(self.feature_columns)

    def build(self, raw_data_list):
        """Same values as applying ``clean_value`` to every cell, without the per-cell call.

        NumPy parses the whole batch in one go (numbers and numeric strings) in
        float64, as ``float()`` would; only a batch holding something it cannot
        parse falls back to ``clean_value`` per cell. Non-finite values become 0
        before the cast to ``dtype``, as in ``clean_value``.
        """
        shape = (len(raw_data_list), len(self.feature_columns))
        values = [[raw.get(name, 0.0) for name in self.feature_columns] for raw in raw_data_list]
        try:
            out = np.array(values, dtype=np.float64).reshape(shape)
        except (TypeError, ValueError, OverflowError):
            out = np.array([[clean_value(v) for v in row] for row in values],
                           dtype=np.float64).reshape(shape)
        out[~np.isfinite(out)] = 0.0
        return out.astype(self.dtype, copy=False)

    def build_single(self, raw_data):
        return self.build([raw_data])
