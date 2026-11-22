"""The four trained ensemble families and their ONNX export (StandardScaler + model)."""

import numpy as np

MODEL_KEYS = ("sklearn_rf", "sklearn_hgb", "lightgbm", "xgboost_hist")

N_TREES = 200


def build_models(seed):
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from xgboost import XGBClassifier

    return {
        # exact splits: thresholds are midpoints between observed values
        "sklearn_rf": ("scikit-learn RandomForest (exact midpoint splits)",
                       RandomForestClassifier(n_estimators=N_TREES, max_depth=15, n_jobs=-1, random_state=seed)),
        # binned trainers
        "sklearn_hgb": ("scikit-learn HistGradientBoosting (max_bins=255)",
                        HistGradientBoostingClassifier(max_iter=N_TREES, max_bins=255, random_state=seed)),
        "lightgbm": ("LightGBM (max_bin=255)",
                     LGBMClassifier(n_estimators=N_TREES, max_bin=255, random_state=seed, n_jobs=8, verbose=-1)),
        "xgboost_hist": ("XGBoost tree_method=hist (max_bin=256, BRANCH_LT)",
                         XGBClassifier(n_estimators=N_TREES, max_depth=8, tree_method="hist", max_bin=256,
                                       random_state=seed, n_jobs=8)),
    }


def patch_skl2onnx_bools():
    """skl2onnx 1.20 hands numpy booleans to an INTS attribute (nodes_missing_value_tracks_true) for
    HistGradientBoosting under recent scikit-learn/onnx; protobuf refuses them. Cast at the boundary."""
    from skl2onnx.common import _container
    if getattr(_container.ModelComponentContainer.add_node, "_bool_safe", False):
        return
    _orig = _container.ModelComponentContainer.add_node

    def add_node(self, *args, **attrs):
        for k, v in list(attrs.items()):
            if isinstance(v, (list, tuple)) and len(v) and all(isinstance(e, (bool, np.bool_, int, np.integer)) for e in v):
                attrs[k] = [int(e) for e in v]
        return _orig(self, *args, **attrs)
    add_node._bool_safe = True
    _container.ModelComponentContainer.add_node = add_node


def export_pipeline(pipe, n_features, out):
    import onnx
    from skl2onnx import convert_sklearn, update_registered_converter
    from skl2onnx.common.data_types import FloatTensorType
    from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
    try:
        from lightgbm import LGBMClassifier
        from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm
        update_registered_converter(LGBMClassifier, "LightGbmLGBMClassifier", calculate_linear_classifier_output_shapes,
                                    convert_lightgbm, options={"nocl": [True, False], "zipmap": [True, False, "columns"]})
    except ImportError:
        pass
    try:
        from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
        from xgboost import XGBClassifier
        update_registered_converter(XGBClassifier, "XGBoostXGBClassifier", calculate_linear_classifier_output_shapes,
                                    convert_xgboost, options={"nocl": [True, False], "zipmap": [True, False, "columns"]})
    except ImportError:
        pass
    patch_skl2onnx_bools()
    model = convert_sklearn(pipe, initial_types=[("input", FloatTensorType([None, n_features]))],
                            options={id(pipe.steps[-1][1]): {"zipmap": False}},
                            target_opset={"": 17, "ai.onnx.ml": 3})
    onnx.save_model(model, out)
    return [n.op_type for n in model.graph.node]
