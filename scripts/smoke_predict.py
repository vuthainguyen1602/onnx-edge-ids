#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from onnx_edge_ids import FeatureMatrixBuilder, create_inference_engine


def main():
    parser = argparse.ArgumentParser(description="Smoke-test ONNX-EdgeIDS inference")
    parser.add_argument("--engine", choices=["onnx", "numpy"], default="onnx")
    parser.add_argument("--model", default=str(ROOT / "artifacts" / "ids_rf.onnx"))
    parser.add_argument("--features", default=str(ROOT / "artifacts" / "feature_columns.json"))
    parser.add_argument("--rows", type=int, default=4)
    args = parser.parse_args()

    features = json.load(open(args.features))
    messages = [{c: 0.0 for c in features} for _ in range(args.rows)]

    builder = FeatureMatrixBuilder(args.features)
    engine = create_inference_engine(args.engine, args.model)
    result = engine.predict_batch(builder.build(messages))

    print("predictions:", result.predictions.tolist())
    print("confidences:", result.confidences.round(6).tolist())
    print("stats:", result.stats)


if __name__ == "__main__":
    main()
