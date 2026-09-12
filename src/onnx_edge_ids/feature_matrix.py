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
    except (TypeError, ValueError):
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
        out = np.empty((len(raw_data_list), len(self.feature_columns)), dtype=self.dtype)
        for i, raw in enumerate(raw_data_list):
            for j, name in enumerate(self.feature_columns):
                out[i, j] = clean_value(raw.get(name, 0.0))
        return out

    def build_single(self, raw_data):
        return self.build([raw_data])
