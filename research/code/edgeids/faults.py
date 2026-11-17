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


# Exporter faults: one defect each, injected into the arrays of an exported NumPy artifact.

def up(a):
    return np.nextafter(a.astype(np.float32), np.float32(np.inf)).astype(np.float64)


def exporter_mutants(d):
    """name -> NumPy-artifact arrays with one exporter defect each (exp25, exp29); ``d`` is the exported .npz as a dict."""
    internal = np.flatnonzero(~d["is_leaf"])
    out = {}
    m = dict(d); m["threshold"] = np.where(d["is_leaf"], d["threshold"], up(d["threshold"])); out["thr_all_up"] = m
    m = dict(d); t = d["threshold"].copy(); t[internal[0]] = up(t[internal[0]:internal[0] + 1])[0]
    m["threshold"] = t; out["thr_one_up"] = m
    used = np.bincount(d["feature"][internal]).argsort()[::-1]
    fa, fb = int(used[0]), int(used[1])
    f = d["feature"].copy(); f[d["feature"] == fa], f[d["feature"] == fb] = fb, fa
    m = dict(d); m["feature"] = f; out["feature_swap"] = m
    m = dict(d); o = d["offset"].copy(); o[fa] = up(o[fa:fa + 1])[0]; m["offset"] = o; out["offset_one_up"] = m
    m = dict(d); m["scale"] = d["scale"].astype(np.float16).astype(np.float64); out["scale_fp16"] = m
    m = dict(d); ls = d["leaf_scores"].copy(); t0 = slice(int(d["tree_offsets"][0]), int(d["tree_offsets"][1]))
    ls[t0] = ls[t0, ::-1] if ls.shape[1] == 2 else -ls[t0]
    m["leaf_scores"] = ls; out["leaf_tree0"] = m
    if d["strict"].any():
        m = dict(d); m["strict"] = np.zeros_like(d["strict"]); out["mode_dropped"] = m
    if np.any(d["base_values"] != 0):
        m = dict(d); m["base_values"] = np.zeros_like(d["base_values"]); out["base_dropped"] = m
    return out


# Single-node faults: one node's threshold moved just past ONNX Runtime's float32 crossing (exp30, exp33).

def single_node_mutants(base, onnx_path, seed=0, count=20):
    """name -> artifact arrays with one node's threshold moved so that the raw value just past
    ONNX Runtime's float32 crossing now takes the true branch (visible at that node by construction)."""
    from onnx_edge_ids.boundary_probe import crossing_values, load_forest_arrays
    f = load_forest_arrays(onnx_path, np.float64)
    internal = np.flatnonzero(~f.is_leaf)
    internal = internal[f.scale[f.feature[internal]] > 0]
    nodes = np.sort(np.random.default_rng(seed).choice(internal, count, replace=False))
    g = f.feature[nodes]
    start = (f.threshold[nodes] / f.scale[g] + f.offset[g]).astype(np.float32)
    ok, last_true = crossing_values(f, nodes, start)
    first_false = np.nextafter(last_true, np.float32(np.inf))
    z = ((first_false - f.offset[g].astype(np.float32)) * f.scale[g].astype(np.float32)).astype(np.float32)
    out = {}
    for v, zv, good in zip(nodes, z, ok):
        if not good:
            continue
        m = dict(base)
        t = base["threshold"].copy()
        t[v] = float(np.nextafter(zv, np.float32(np.inf))) if base["strict"][v] else float(zv)
        m["threshold"] = t
        out[f"node_{int(v)}"] = m
    return out
