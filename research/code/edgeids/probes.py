"""Boundary-directed probes: rows routed to a split node, placed on its threshold and one float32 step either side.

Two forest representations are probed: the deployed NumPy artifact (edgeids.forest, BRANCH_LEQ only) and
any ONNX TreeEnsemble graph (edgeids.onnx_forest). probe_model is the per-model study of exp5/exp10.
"""

import numpy as np

from edgeids.forest import scale_input
from edgeids.onnx_forest import (
    load_onnx_forest,
    path_rows,
    predict,
    reaches,
    rows_on_a_threshold,
    threshold_sharing,
    trees_moved,
)
from edgeids.runtime import flips, ort_labels, ort_outputs


def artifact_parents(forest):
    parent = np.full(len(forest.left), -1, dtype=np.int64)
    went_left = np.zeros(len(forest.left), dtype=bool)
    internal = np.flatnonzero(~forest.is_leaf)
    parent[forest.left[internal]] = internal
    went_left[forest.left[internal]] = True
    parent[forest.right[internal]] = internal
    return parent, went_left


def artifact_path_rows(f64, targets, base):
    """One row per target node whose features satisfy every split on the way to it."""
    parent, went_left = artifact_parents(f64)
    rows = base.astype(np.float64).copy()
    for i, node in enumerate(targets):
        lo, hi = {}, {}
        cur = node
        while parent[cur] >= 0:
            p = parent[cur]
            g, t = int(f64.feature[p]), float(f64.threshold[p])
            if went_left[cur]:
                hi[g] = min(hi.get(g, np.inf), t)
            else:
                lo[g] = max(lo.get(g, -np.inf), t)
            cur = p
        for g in set(lo) | set(hi):
            a, b = lo.get(g, -np.inf), hi.get(g, np.inf)
            if np.isfinite(a) and np.isfinite(b):
                s = (a + b) / 2.0
            elif np.isfinite(b):
                s = b - 1.0
            else:
                s = a + 1.0
            rows[i, g] = s / f64.scale[g] + f64.offset[g]
    return rows.astype(np.float32)


def artifact_reaches(forest, rows, targets):
    """Whether row i passes through node targets[i] under this forest's arithmetic."""
    scaled = scale_input(forest, rows)
    tree = np.searchsorted(forest.tree_offsets, targets, side="right") - 1
    cur = forest.tree_offsets[tree].copy()
    idx = np.arange(len(targets))
    hit = cur == targets
    active = ~forest.is_leaf[cur] & ~hit
    while active.any():
        a = np.flatnonzero(active)
        c = cur[a]
        go_left = scaled[idx[a], forest.feature[c]] <= forest.threshold[c]
        cur[a] = np.where(go_left, forest.left[c], forest.right[c])
        hit[a] = cur[a] == targets[a]
        active[a] = ~forest.is_leaf[cur[a]] & ~hit[a]
    return hit


def artifact_probes(f64, internal, feat, base):
    rows = artifact_path_rows(f64, internal, base)
    raw = (f64.threshold[internal] / f64.scale[feat] + f64.offset[feat]).astype(np.float32)
    out = []
    for values in (raw, np.nextafter(raw, np.float32(-np.inf)), np.nextafter(raw, np.float32(np.inf))):
        p = rows.copy()
        p[np.arange(len(internal)), feat] = values
        out.append(p)
    return np.concatenate(out)


def datafree_probes(forest):
    internal = np.flatnonzero(~forest.is_leaf)
    feat = forest.feature[internal]
    base = np.repeat(forest.offset.astype(np.float32).reshape(1, -1), len(internal), axis=0)
    rows = path_rows(forest, internal, base)
    scale, offset = forest.scale.astype(np.float64), forest.offset.astype(np.float64)
    safe = np.where(scale[feat] != 0, scale[feat], 1.0)
    raw = np.where(scale[feat] != 0, forest.threshold[internal].astype(np.float64) / safe + offset[feat],
                   offset[feat]).astype(np.float32)
    out = []
    for values in (raw, np.nextafter(raw, np.float32(-np.inf)), np.nextafter(raw, np.float32(np.inf))):
        p = rows.copy()
        p[np.arange(len(internal)), feat] = values
        out.append(p)
    return internal, np.concatenate(out)


def probe_model(onnx_path, native, x_test, max_nodes, seed):
    forest = load_onnx_forest(onnx_path)
    rep = {"trees": forest.n_trees, "nodes": int(len(forest.left)), "post_transform": forest.post_transform,
           "split_modes": sorted(set(int(m) for m in forest.mode[~forest.is_leaf]))}
    rep["sharing"] = threshold_sharing(forest)

    rng = np.random.default_rng(seed)
    internal = np.flatnonzero(~forest.is_leaf)
    if len(internal) > max_nodes:
        internal = np.sort(rng.choice(internal, size=max_nodes, replace=False))
    feat = forest.feature[internal]
    base = x_test[rng.choice(len(x_test), size=len(internal), replace=True)]
    rows = path_rows(forest, internal, base)
    scale, offset = forest.scale.astype(np.float64), forest.offset.astype(np.float64)
    raw = np.where(scale[feat] != 0, forest.threshold[internal].astype(np.float64) / np.where(scale[feat] != 0, scale[feat], 1) + offset[feat], offset[feat]).astype(np.float32)
    variants = {"on_threshold": raw,
                "one_ulp_below": np.nextafter(raw, np.float32(-np.inf)),
                "one_ulp_above": np.nextafter(raw, np.float32(np.inf))}

    rep["probes"] = {"nodes_probed": int(len(internal)), "variants": {}}
    tot = {"probes": 0, "f32": 0, "f64": 0, "native": 0, "reach": 0}
    for name, values in variants.items():
        probes = rows.copy()
        probes[np.arange(len(internal)), feat] = values
        on_path = reaches(forest, probes, internal)
        lo = ort_labels(onnx_path, probes)
        l32 = predict(forest, probes, np.float32)
        l64 = predict(forest, probes, np.float64)
        moved = trees_moved(forest, probes)
        flipped = l64 != lo
        v = {"probes": int(len(probes)), "probes_reaching_their_node": int(on_path.sum()),
             "numpy_float32_vs_ort_flips": flips(l32, lo), "numpy_float64_vs_ort_flips": flips(l64, lo),
             "probes_where_float64_moves_a_tree": int((moved > 0).sum()),
             "trees_moved_mean_when_any": float(moved[moved > 0].mean()) if (moved > 0).any() else 0.0,
             "trees_moved_mean_when_label_flips": float(moved[flipped].mean()) if flipped.any() else 0.0,
             "label_flip_rate_given_a_tree_moved": float(flipped[moved > 0].mean()) if (moved > 0).any() else 0.0}
        if native is not None:
            v["native_vs_ort_flips"] = flips(native(probes), lo)
            tot["native"] += v["native_vs_ort_flips"]
        tot["probes"] += len(probes); tot["f32"] += v["numpy_float32_vs_ort_flips"]; tot["f64"] += v["numpy_float64_vs_ort_flips"]
        tot["reach"] += int(on_path.sum())
        rep["probes"]["variants"][name] = v
        print("   ", name, v, flush=True)
    rep["probes"]["total"] = {"probes": tot["probes"], "reaching_their_node": tot["reach"],
                              "numpy_float32_vs_ort_flips": tot["f32"], "numpy_float64_vs_ort_flips": tot["f64"]}
    if native is not None:
        rep["probes"]["total"]["native_vs_ort_flips"] = tot["native"]

    lo_out, lo = ort_outputs(onnx_path, x_test)
    l32, l64 = predict(forest, x_test, np.float32), predict(forest, x_test, np.float64)
    moved = trees_moved(forest, x_test)
    r = {"rows": int(len(x_test)), "ort_label_output_vs_ort_probability_label_flips": flips(lo_out, lo),
         "rows_with_a_feature_exactly_on_a_threshold_float32": rows_on_a_threshold(forest, x_test),
         "rows_where_float64_moves_a_tree": int((moved > 0).sum()),
         "trees_moved_mean_when_any": float(moved[moved > 0].mean()) if (moved > 0).any() else 0.0,
         "numpy_float32_vs_ort_flips": flips(l32, lo), "numpy_float64_vs_ort_flips": flips(l64, lo)}
    if native is not None:
        r["native_vs_ort_flips"] = flips(native(x_test), lo)
    for k in ("numpy_float64_vs_ort_flips", "native_vs_ort_flips"):
        if k in r:
            rate = r[k] / len(x_test)
            r[k.replace("_flips", "_rate")] = rate
            r[k.replace("_flips", "_prob_2000_row_replay_misses")] = float((1 - rate) ** 2000)
    rep["replay"] = r
    print("    replay", r, flush=True)

    # density: flips per probe vs flips per replayed row
    for k in ("numpy_float64_vs_ort", "native_vs_ort"):
        pk = k + "_flips"
        if pk in rep["probes"]["total"] and rep["replay"].get(pk, 0) > 0:
            rep["probes"]["total"][k + "_density_vs_replay"] = (rep["probes"]["total"][pk] / tot["probes"]) / (rep["replay"][pk] / len(x_test))
    return rep


def pair_table(f32):
    """Distinct (feature, threshold) pairs and a node -> pair-id map (-1 for leaves)."""
    internal = np.flatnonzero(~f32.is_leaf)
    bits = f32.threshold[internal].view(np.int32).astype(np.int64) & 0xFFFFFFFF
    codes, inv = np.unique(f32.feature[internal] * (1 << 32) + bits, return_inverse=True)
    node_pair = np.full(len(f32.left), -1, dtype=np.int64)
    node_pair[internal] = inv
    pair_feat = (codes >> 32).astype(np.int64)
    pair_thr = (codes & 0xFFFFFFFF).astype(np.uint32).view(np.float32)
    return pair_feat, pair_thr, node_pair, len(internal)


def walk_marking_pair(f32, rows, target_pair, node_pair, exercised):
    """Walk every tree on every row; mark visited nodes carrying target_pair.

    Returns per-row whether any such node was visited, and sets
    exercised[node] = True for each one.
    """
    scaled = scale_input(f32, rows)
    n = scaled.shape[0]
    node = np.repeat(f32.tree_offsets[:-1].reshape(-1, 1), n, axis=1)
    hit = np.zeros(n, dtype=bool)
    active = ~f32.is_leaf[node]
    while active.any():
        t_idx, r_idx = np.nonzero(active)
        cur = node[t_idx, r_idx]
        match = node_pair[cur] == target_pair
        if match.any():
            hit[r_idx[match]] = True
            exercised[np.unique(cur[match])] = True
        go_left = scaled[r_idx, f32.feature[cur]] <= f32.threshold[cur]
        nxt = np.where(go_left, f32.left[cur], f32.right[cur])
        node[t_idx, r_idx] = nxt
        active[t_idx, r_idx] = ~f32.is_leaf[nxt]
    return hit
