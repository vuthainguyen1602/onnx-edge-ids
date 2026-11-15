"""Build a Treelite model from the node table of an ONNX TreeEnsemble graph.

Treelite does not read ONNX. To make it a third runtime of the *same* exported
graph (not of the training framework's model), this rebuilds every tree from
the arrays edgeids.onnx_forest reads out of the graph: split feature,
threshold, comparison mode, children, and per-leaf target weights. Treelite then
traverses and sums; the graph's Scaler and post-transform stay outside it, as
they would in any runtime without an ONNX front end.
"""

import numpy as np

OPNAME = {0: "<=", 1: "<", 2: ">=", 3: ">"}  # BRANCH_LEQ, BRANCH_LT, BRANCH_GTE, BRANCH_GT


def to_treelite(forest, threshold_type="float32", leaf_output_type=None):
    from treelite.model_builder import Metadata, ModelBuilder, PostProcessorFunc, TreeAnnotation

    leaf_output_type = leaf_output_type or threshold_type
    k = int(forest.n_targets)
    if k == 1:
        meta = Metadata(num_feature=len(forest.scale), task_type="kRegressor", average_tree_output=False,
                        num_target=1, num_class=[1], leaf_vector_shape=(1, 1))
        ann = TreeAnnotation(num_tree=forest.n_trees, target_id=[0] * forest.n_trees, class_id=[0] * forest.n_trees)
    else:
        meta = Metadata(num_feature=len(forest.scale), task_type="kMultiClf", average_tree_output=False,
                        num_target=1, num_class=[k], leaf_vector_shape=(1, k))
        ann = TreeAnnotation(num_tree=forest.n_trees, target_id=[0] * forest.n_trees, class_id=[-1] * forest.n_trees)
    builder = ModelBuilder(threshold_type=threshold_type, leaf_output_type=leaf_output_type, metadata=meta,
                           tree_annotation=ann, postprocessor=PostProcessorFunc(name="identity"),
                           base_scores=[float(b) for b in np.asarray(forest.base_values, dtype=np.float64).reshape(-1)][:k] or [0.0] * k)
    cast = np.float32 if threshold_type == "float32" else np.float64
    for t in range(forest.n_trees):
        a, b = forest.tree_offsets[t], forest.tree_offsets[t + 1]
        builder.start_tree()
        for node in range(a, b):
            builder.start_node(int(node - a))
            if forest.is_leaf[node]:
                w = forest.leaf_scores[node]
                builder.leaf(float(cast(w[0])) if k == 1 else [float(cast(v)) for v in w])
            else:
                builder.numerical_test(int(forest.feature[node]), float(cast(forest.threshold[node])),
                                       default_left=True, opname=OPNAME[int(forest.mode[node])],
                                       left_child_key=int(forest.left[node] - a), right_child_key=int(forest.right[node] - a))
            builder.end_node()
        builder.end_tree()
    return builder.commit()


def treelite_scores(model, forest, x, dtype=np.float32):
    """Raw summed leaf weights (plus base values) for raw rows x: the graph's Scaler is applied first, in dtype."""
    import treelite.gtil
    z = (np.asarray(x, dtype=dtype) - forest.offset.astype(dtype)) * forest.scale.astype(dtype)
    out = np.asarray(treelite.gtil.predict(model, z, pred_margin=True), dtype=np.float64)
    return out.reshape(len(x), -1)
