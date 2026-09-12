#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export a deployed Spark ``PipelineModel`` to a pure-NumPy RandomForest artifact.

The exported ``.npz`` contains:
  * fitted StandardScaler offset/scale,
  * flattened RandomForest nodes,
  * per-leaf class scores weighted exactly like Spark's averaged forest output,
  * feature-column metadata for order checks.

Usage:
    python scripts/export_numpy.py \
        --model artifacts/ids_pipeline_model \
        --out   model/ids_rf_numpy.npz \
        --features model/feature_columns.json
"""

import argparse
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JETSON_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, JETSON_DIR)

from export_onnx import (  # noqa: E402
    _read_json,
    read_stages,
    load_assembler,
    load_scaler,
    load_forest,
)


def flatten_forest(by_tree, tree_weights, n_classes):
    tree_offsets = [0]
    left = []
    right = []
    feature = []
    threshold = []
    is_leaf = []
    leaf_scores = []

    total_weight = sum(float(w) for w in tree_weights.values()) or float(len(by_tree))

    for tid in sorted(by_tree):
        nodes = sorted(by_tree[tid], key=lambda n: n["id"])
        local_index = {int(n["id"]): i for i, n in enumerate(nodes)}
        base = tree_offsets[-1]
        weight = float(tree_weights.get(tid, 1.0)) / total_weight

        for node in nodes:
            node_left = int(node["leftChild"])
            node_right = int(node["rightChild"])
            leaf = node_left < 0 or node_right < 0

            is_leaf.append(leaf)
            if leaf:
                left.append(-1)
                right.append(-1)
                feature.append(-1)
                threshold.append(0.0)

                stats = np.asarray(node["impurityStats"], dtype=np.float64)
                total = stats.sum()
                dist = stats / total if total > 0 else np.full(n_classes, 1.0 / n_classes)
                leaf_scores.append(dist * weight)
            else:
                split = node["split"]
                if int(split["numCategories"]) >= 0:
                    raise NotImplementedError(
                        f"[ERR] Tree {tid} uses a categorical split; "
                        "the current NumPy export handles continuous splits only."
                    )

                left.append(base + local_index[node_left])
                right.append(base + local_index[node_right])
                feature.append(int(split["featureIndex"]))
                threshold.append(float(split["leftCategoriesOrThreshold"][0]))
                leaf_scores.append(np.zeros(n_classes, dtype=np.float64))

        tree_offsets.append(base + len(nodes))

    return {
        "tree_offsets": np.asarray(tree_offsets, dtype=np.int64),
        "left": np.asarray(left, dtype=np.int64),
        "right": np.asarray(right, dtype=np.int64),
        "feature": np.asarray(feature, dtype=np.int64),
        "threshold": np.asarray(threshold, dtype=np.float64),
        "is_leaf": np.asarray(is_leaf, dtype=np.bool_),
        "leaf_scores": np.vstack(leaf_scores).astype(np.float64),
    }


def numpy_predict(matrix, offset, scale, arrays):
    x = np.ascontiguousarray(matrix, dtype=np.float64)
    scaled = (x - offset) * scale
    n_classes = arrays["leaf_scores"].shape[1]
    scores = np.zeros((scaled.shape[0], n_classes), dtype=np.float64)

    tree_offsets = arrays["tree_offsets"]
    for t in range(len(tree_offsets) - 1):
        base = int(tree_offsets[t])
        node = np.full(scaled.shape[0], base, dtype=np.int64)
        active = np.ones(scaled.shape[0], dtype=bool)

        while np.any(active):
            rows = np.flatnonzero(active)
            cur = node[rows]
            leaf = arrays["is_leaf"][cur]
            if np.any(leaf):
                leaf_rows = rows[leaf]
                scores[leaf_rows] += arrays["leaf_scores"][node[leaf_rows]]
                active[leaf_rows] = False

            branch_rows = rows[~leaf]
            if branch_rows.size:
                cur_branch = node[branch_rows]
                feat = arrays["feature"][cur_branch]
                go_left = scaled[branch_rows, feat] <= arrays["threshold"][cur_branch]
                node[branch_rows] = np.where(
                    go_left, arrays["left"][cur_branch], arrays["right"][cur_branch])

    return np.argmax(scores, axis=1).astype(np.int64), scores


def validate(npz_path, spark_model_dir, csv_path, feature_columns, n_rows):
    from pyspark.sql import SparkSession
    from pyspark.ml import PipelineModel
    import pandas as pd

    df = pd.read_csv(csv_path, nrows=n_rows)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"[ERR] CSV is missing {len(missing)} feature columns: {missing[:5]}")

    x = df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    matrix = x.to_numpy(dtype=np.float64)

    artifact = np.load(npz_path, allow_pickle=False)
    numpy_labels, numpy_probs = numpy_predict(
        matrix,
        artifact["offset"].astype(np.float64),
        artifact["scale"].astype(np.float64),
        {k: artifact[k] for k in (
            "tree_offsets", "left", "right", "feature",
            "threshold", "is_leaf", "leaf_scores"
        )},
    )

    spark = (SparkSession.builder.appName("export_numpy_validate")
             .master("local[*]").config("spark.ui.enabled", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    sdf = spark.createDataFrame(x.astype(float))
    rows = (PipelineModel.load(spark_model_dir).transform(sdf)
            .select("prediction", "probability").collect())
    spark_labels = np.array([int(r["prediction"]) for r in rows], dtype=np.int64)
    spark_probs = np.array([list(r["probability"]) for r in rows], dtype=np.float64)
    spark.stop()

    agreement = float(np.mean(numpy_labels == spark_labels))
    max_delta = float(np.max(np.abs(numpy_probs - spark_probs)))
    print(f"\n[VALIDATE] rows={len(spark_labels)}")
    print(f"  label agreement : {agreement:.6%}")
    print(f"  max |dprob|     : {max_delta:.3e}")
    if agreement < 1.0:
        idx = np.flatnonzero(numpy_labels != spark_labels)[:5]
        print(f"  [WARN] disagreeing rows: {idx.tolist()}")
    return agreement, max_delta


def main():
    ap = argparse.ArgumentParser(description="Spark PipelineModel -> NumPy artifact")
    ap.add_argument("--model", required=True, help="Saved Spark PipelineModel directory")
    ap.add_argument("--out", required=True, help="Output .npz path")
    ap.add_argument("--features", default=None, help="feature_columns.json (order check)")
    ap.add_argument("--validate-csv", default=None, help="CSV to compare Spark vs NumPy on")
    ap.add_argument("--validate-rows", type=int, default=5000)
    args = ap.parse_args()

    stages = read_stages(args.model)
    classes = [s["meta"]["class"].split(".")[-1] for s in stages]
    if classes != ["VectorAssembler", "StandardScalerModel", "RandomForestClassificationModel"]:
        raise NotImplementedError(f"[ERR] Unsupported pipeline shape: {classes}")

    input_cols = load_assembler(stages[0])
    offset, scale = load_scaler(stages[1])
    by_tree, tree_weights, n_features, n_classes = load_forest(stages[2])

    if args.features:
        expected = _read_json(args.features)
        if list(expected) != list(input_cols):
            raise ValueError(
                "[ERR] Assembler column order differs from feature_columns.json; "
                "the served matrix would be permuted.\n"
                f"  assembler[:5]={list(input_cols)[:5]}\n"
                f"  features  [:5]={list(expected)[:5]}"
            )
        print(f"[OK] Feature order matches {args.features}")

    if len(input_cols) != n_features:
        raise ValueError(f"[ERR] assembler cols={len(input_cols)} != forest features={n_features}")

    arrays = flatten_forest(by_tree, tree_weights, n_classes)
    metadata = {
        "producer": "ONNX-EdgeIDS/scripts/export_numpy.py",
        "source_model": os.path.abspath(args.model),
        "feature_columns": list(input_cols),
        "n_features": int(n_features),
        "n_classes": int(n_classes),
        "n_trees": int(len(arrays["tree_offsets"]) - 1),
        "n_nodes": int(len(arrays["left"])),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        offset=offset.astype(np.float64),
        scale=scale.astype(np.float64),
        metadata_json=json.dumps(metadata),
        **arrays,
    )
    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"[OK] Wrote {args.out} ({size_mb:.2f} MB)")
    print(f"  Trees: {metadata['n_trees']} | nodes: {metadata['n_nodes']}")

    if args.validate_csv:
        validate(args.out, args.model, args.validate_csv, list(input_cols), args.validate_rows)


if __name__ == "__main__":
    main()
