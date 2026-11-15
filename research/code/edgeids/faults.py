"""The six one-line arithmetic faults injected into the float32 traversal of the deployed forest."""

from dataclasses import replace

import numpy as np

from edgeids.forest import _sum_over_trees


def scores_with(forest, scaled, batch=2000):
    trees = np.arange(forest.n_trees)
    return np.concatenate([
        _sum_over_trees(forest, scaled[i:i + batch], trees, forest.leaf_scores)
        for i in range(0, len(scaled), batch)
    ]).astype(np.float64)


def fault_scorers(f32, f64):
    """name -> function(x float32) returning class scores. Mirrors exp11.faults line for line."""
    o32, s32 = f32.offset, f32.scale
    o64, s64 = f64.offset, f64.scale
    internal = ~f32.is_leaf

    strict = replace(f32, threshold=np.where(internal, np.nextafter(f32.threshold, np.float32(-np.inf)), f32.threshold))
    decimal = replace(f32, threshold=np.array([float(f"{t:.6g}") for t in f32.threshold], dtype=np.float32))
    raw_thr = f32.threshold.copy()
    raw_thr[internal] = f32.threshold[internal] / s32[f32.feature[internal]] + o32[f32.feature[internal]]
    raw = replace(f32, threshold=raw_thr.astype(np.float32))

    return {
        "float32 (deployed)": lambda x: scores_with(f32, (x - o32) * s32),
        "float64": lambda x: scores_with(f64, (x.astype(np.float64) - o64) * s64),
        "float64_scaler": lambda x: scores_with(f32, ((x.astype(np.float64) - o64) * s64).astype(np.float32)),
        "scaler_reordered": lambda x: scores_with(f32, x * s32 - o32 * s32),
        "raw_thresholds": lambda x: scores_with(raw, np.ascontiguousarray(x, dtype=np.float32)),
        "strict_less": lambda x: scores_with(strict, (x - o32) * s32),
        "decimal_thresholds": lambda x: scores_with(decimal, (x - o32) * s32),
    }


def fault_labels(f32, f64):
    """name -> function(x float32) returning labels, for the six faults (the deployed float32 traversal excluded)."""
    return {name: (lambda x, fn=fn: np.argmax(fn(x), axis=1).astype(np.int64))
            for name, fn in fault_scorers(f32, f64).items() if name != "float32 (deployed)"}
