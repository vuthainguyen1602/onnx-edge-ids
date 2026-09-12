#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export a deployed Spark ``PipelineModel`` (VectorAssembler + StandardScaler +
RandomForestClassifier) to a single ONNX graph for JVM-free edge serving.

Why not skl2onnx: there is no Spark -> sklearn -> ONNX path that preserves the
*validated* model. Rebuilding an equivalent sklearn forest either hacks
``sklearn.tree._tree.Tree`` internals or retrains, and retraining would break
the claim that the board serves exactly the model selected under the
leakage-aware offline protocol. Spark's saved format already stores the node
table as Parquet, and it maps one-to-one onto the ONNX ``TreeEnsembleClassifier``
operator, so this script reads the saved model directly. It needs no Spark and
no sklearn -- only pyarrow + onnx -- and can therefore run anywhere.

Equivalence argument (checked by --validate):
  * label: derived with ArgMax over the scores rather than taken from the
    classifier's own label output -- see the note in ``build_model``.
  * split semantics: Spark sends a continuous split LEFT when
    ``feature <= leftCategoriesOrThreshold[0]``   -> ONNX ``BRANCH_LEQ``
    with truenodeids = leftChild, falsenodeids = rightChild.
  * leaf output: Spark's ``predictRaw`` sums each tree's leaf class
    distribution normalized to sum 1, then ``raw2probability`` divides by the
    number of trees. TreeEnsembleClassifier has no aggregate_function (that is
    a TreeEnsembleRegressor attribute) and always sums the class weights, so
    the per-tree share is folded into the weights instead: class_weights =
    (leaf impurityStats normalized per leaf) * w_t / sum(w). Summing those
    reproduces Spark's averaged probability exactly, with post_transform NONE.
  * scaling: Spark StandardScaler computes (x - mean) * (1/std), with the
    scale forced to 0 where std == 0 -> ai.onnx.ml ``Scaler``.

Usage:
    python scripts/export_onnx.py \
        --model  artifacts/ids_pipeline_model \
        --out    model/ids_rf.onnx \
        --validate-csv /path/to/test_sample.csv     # optional, needs pyspark
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ----------------------------------------------------------------------
# Reading the saved Spark model (no Spark required)
# ----------------------------------------------------------------------

def _read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _read_parquet_dir(path):
    """Spark writes one part per partition; a cluster-saved model has many."""
    import pyarrow.parquet as pq

    parts = sorted(glob.glob(os.path.join(path, "*.parquet")))
    if not parts:
        raise FileNotFoundError(f"[ERR] No parquet parts under {path}")
    import pyarrow as pa
    return pa.concat_tables([pq.read_table(p) for p in parts])


def read_stages(model_dir):
    """Resolve stageUids -> stage directories (a dir may hold stale sibling stages)."""
    meta = _read_json(os.path.join(model_dir, "metadata", "part-00000"))
    uids = meta["paramMap"]["stageUids"]

    stages = []
    for i, uid in enumerate(uids):
        stage_dir = os.path.join(model_dir, "stages", f"{i}_{uid}")
        if not os.path.isdir(stage_dir):
            raise FileNotFoundError(f"[ERR] Stage dir missing: {stage_dir}")
        stage_meta = _read_json(os.path.join(stage_dir, "metadata", "part-00000"))
        stages.append({"uid": uid, "dir": stage_dir, "meta": stage_meta})
        print(f"[OK] Stage {i}: {stage_meta['class'].split('.')[-1]} ({uid})")
    return stages


def _param(meta, key, default=None):
    return meta["paramMap"].get(key, meta.get("defaultParamMap", {}).get(key, default))


def load_assembler(stage):
    return _param(stage["meta"], "inputCols", [])


def load_scaler(stage):
    """-> (offset, scale) implementing Spark's (x - mean) * (1/std)."""
    meta = stage["meta"]
    with_mean = bool(_param(meta, "withMean", False))
    with_std = bool(_param(meta, "withStd", True))

    table = _read_parquet_dir(os.path.join(stage["dir"], "data")).to_pydict()
    mean = np.asarray(table["mean"][0]["values"], dtype=np.float64)
    std = np.asarray(table["std"][0]["values"], dtype=np.float64)

    offset = mean if with_mean else np.zeros_like(mean)
    if with_std:
        scale = np.where(std != 0.0, 1.0 / np.where(std != 0.0, std, 1.0), 0.0)
    else:
        scale = np.ones_like(std)

    print(f"[OK] Scaler: withMean={with_mean} withStd={with_std} "
          f"dim={len(mean)} zero-std features={int(np.count_nonzero(std == 0.0))}")
    return offset.astype(np.float32), scale.astype(np.float32)


def load_forest(stage):
    """-> (trees, tree_weights, n_features, n_classes).

    trees[t] is a dict of arrays keyed by the dense node index of tree t.
    """
    meta = stage["meta"]
    n_features = int(meta["numFeatures"])
    n_classes = int(meta["numClasses"])

    weights_table = _read_parquet_dir(os.path.join(stage["dir"], "treesMetadata")).to_pydict()
    tree_weights = dict(zip(weights_table["treeID"], weights_table["weights"]))

    table = _read_parquet_dir(os.path.join(stage["dir"], "data")).to_pydict()
    tree_ids = table["treeID"]
    nodes = table["nodeData"]

    by_tree = {}
    for tid, node in zip(tree_ids, nodes):
        by_tree.setdefault(int(tid), []).append(node)

    print(f"[OK] Forest: trees={len(by_tree)} nodes={len(nodes)} "
          f"features={n_features} classes={n_classes}")
    return by_tree, tree_weights, n_features, n_classes


# ----------------------------------------------------------------------
# ONNX graph
# ----------------------------------------------------------------------

def build_tree_attributes(by_tree, tree_weights, n_classes):
    """Flatten Spark nodes into the TreeEnsembleClassifier attribute arrays."""
    nodes_treeids, nodes_nodeids = [], []
    nodes_featureids, nodes_values, nodes_modes = [], [], []
    nodes_truenodeids, nodes_falsenodeids = [], []
    class_treeids, class_nodeids, class_ids, class_weights = [], [], [], []

    # TreeEnsembleClassifier always sums class weights, so carry each tree's
    # share in the weights themselves. RF weights are uniform (1.0), giving the
    # plain 1/n_trees average Spark applies; the general form covers the rest.
    total_weight = sum(float(w) for w in tree_weights.values()) or float(len(by_tree))

    for tid in sorted(by_tree):
        nodes = by_tree[tid]
        # Spark ids are unique per tree but not guaranteed contiguous: remap.
        index = {int(n["id"]): i for i, n in enumerate(sorted(nodes, key=lambda n: n["id"]))}
        weight = float(tree_weights.get(tid, 1.0)) / total_weight

        for node in sorted(nodes, key=lambda n: n["id"]):
            nid = index[int(node["id"])]
            is_leaf = int(node["leftChild"]) < 0 or int(node["rightChild"]) < 0

            nodes_treeids.append(tid)
            nodes_nodeids.append(nid)

            if is_leaf:
                nodes_modes.append("LEAF")
                nodes_featureids.append(0)
                nodes_values.append(0.0)
                nodes_truenodeids.append(0)
                nodes_falsenodeids.append(0)

                stats = np.asarray(node["impurityStats"], dtype=np.float64)
                total = stats.sum()
                dist = stats / total if total > 0 else np.full(n_classes, 1.0 / n_classes)
                for c in range(n_classes):
                    class_treeids.append(tid)
                    class_nodeids.append(nid)
                    class_ids.append(c)
                    class_weights.append(float(dist[c]) * weight)
            else:
                split = node["split"]
                if int(split["numCategories"]) >= 0:
                    raise NotImplementedError(
                        f"[ERR] Tree {tid} node {nid} uses a categorical split; "
                        f"the deployed pipeline is all-continuous, so this needs "
                        f"BRANCH_EQ handling before export.")
                nodes_modes.append("BRANCH_LEQ")
                nodes_featureids.append(int(split["featureIndex"]))
                nodes_values.append(float(split["leftCategoriesOrThreshold"][0]))
                nodes_truenodeids.append(index[int(node["leftChild"])])
                nodes_falsenodeids.append(index[int(node["rightChild"])])

    return {
        "nodes_treeids": nodes_treeids,
        "nodes_nodeids": nodes_nodeids,
        "nodes_featureids": nodes_featureids,
        "nodes_values": nodes_values,
        "nodes_modes": nodes_modes,
        "nodes_truenodeids": nodes_truenodeids,
        "nodes_falsenodeids": nodes_falsenodeids,
        "nodes_missing_value_tracks_true": [0] * len(nodes_modes),
        "class_treeids": class_treeids,
        "class_nodeids": class_nodeids,
        "class_ids": class_ids,
        "class_weights": class_weights,
        "classlabels_int64s": list(range(n_classes)),
        "post_transform": "NONE",
    }


def build_model(offset, scale, tree_attrs, n_features, feature_columns):
    from onnx import helper, TensorProto

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, n_features])
    label = helper.make_tensor_value_info("label", TensorProto.INT64, [None])
    probs = helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None, None])

    scaler = helper.make_node(
        "Scaler", ["input"], ["scaled"],
        domain="ai.onnx.ml", name="standard_scaler",
        offset=offset.tolist(), scale=scale.tolist())

    forest = helper.make_node(
        "TreeEnsembleClassifier", ["scaled"], ["label_raw", "probabilities"],
        domain="ai.onnx.ml", name="random_forest", **tree_attrs)

    # ONNX Runtime mishandles the label output of a two-class
    # TreeEnsembleClassifier whose leaves carry a weight per class: the scores
    # are correct but the emitted label is always class 1 (reproduced on a
    # single-leaf ensemble, ort 1.27). Derive the label from the scores
    # instead, which is also what Spark does (probability2prediction = argmax,
    # first max on a tie). ``label_raw`` stays unused inside the graph.
    argmax = helper.make_node(
        "ArgMax", ["probabilities"], ["label"],
        axis=1, keepdims=0, name="probability2prediction")

    graph = helper.make_graph([scaler, forest, argmax], "ids_pipeline", [x], [label, probs])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17),
                       helper.make_opsetid("ai.onnx.ml", 3)],
        producer_name="ONNX-EdgeIDS/export_onnx.py")
    model.doc_string = json.dumps({"feature_columns": feature_columns}, indent=None)
    return model


# ----------------------------------------------------------------------
# Equivalence check against the Spark model
# ----------------------------------------------------------------------

def validate(onnx_path, spark_model_dir, csv_path, feature_columns, n_rows):
    """Agreement between the exported graph and the Spark model it came from.

    This is the number the OSP has to report: an export nobody checked is not
    the validated model.
    """
    import onnxruntime as ort
    from pyspark.sql import SparkSession
    from pyspark.ml import PipelineModel
    import pandas as pd

    df = pd.read_csv(csv_path, nrows=n_rows)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"[ERR] CSV is missing {len(missing)} feature columns: {missing[:5]}")

    x = df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    matrix = x.to_numpy(dtype=np.float32)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_labels, onnx_probs = sess.run(None, {sess.get_inputs()[0].name: matrix})
    onnx_labels = np.asarray(onnx_labels).reshape(-1)
    onnx_probs = np.asarray(onnx_probs).reshape(len(onnx_labels), -1)

    spark = (SparkSession.builder.appName("export_onnx_validate")
             .master("local[*]").config("spark.ui.enabled", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    sdf = spark.createDataFrame(x.astype(float))
    rows = (PipelineModel.load(spark_model_dir).transform(sdf)
            .select("prediction", "probability").collect())
    spark_labels = np.array([int(r["prediction"]) for r in rows], dtype=np.int64)
    spark_probs = np.array([list(r["probability"]) for r in rows], dtype=np.float64)
    spark.stop()

    agreement = float(np.mean(onnx_labels == spark_labels))
    max_delta = float(np.max(np.abs(onnx_probs - spark_probs)))
    print(f"\n[VALIDATE] rows={len(spark_labels)}")
    print(f"  label agreement : {agreement:.6%}")
    print(f"  max |dprob|     : {max_delta:.3e}")
    if agreement < 1.0:
        idx = np.flatnonzero(onnx_labels != spark_labels)[:5]
        print(f"  [WARN] disagreeing rows: {idx.tolist()}")
    return agreement, max_delta


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Spark PipelineModel -> ONNX (edge serving)")
    ap.add_argument("--model", required=True, help="Saved Spark PipelineModel directory")
    ap.add_argument("--out", required=True, help="Output .onnx path")
    ap.add_argument("--features", default=None, help="feature_columns.json (order check)")
    ap.add_argument("--validate-csv", default=None, help="CSV to compare Spark vs ONNX on")
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
                f"  features  [:5]={list(expected)[:5]}")
        print(f"[OK] Feature order matches {args.features}")

    if len(input_cols) != n_features:
        raise ValueError(f"[ERR] assembler cols={len(input_cols)} != forest features={n_features}")

    tree_attrs = build_tree_attributes(by_tree, tree_weights, n_classes)
    model = build_model(offset, scale, tree_attrs, n_features, list(input_cols))

    import onnx
    onnx.checker.check_model(model)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    onnx.save_model(model, args.out)
    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"[OK] Wrote {args.out} ({size_mb:.2f} MB)")

    if args.validate_csv:
        validate(args.out, args.model, args.validate_csv, list(input_cols), args.validate_rows)


if __name__ == "__main__":
    main()
