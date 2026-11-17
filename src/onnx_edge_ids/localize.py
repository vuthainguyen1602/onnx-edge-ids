#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Where does a failing gate point? Fault localization from the boundary probes.

Two levels, both needing only what the gate already has:

* ``summarize_failures`` (any two runtimes, black box): every probe is built
  for one split node, so the failing probes name their nodes. Counting them by
  (feature, threshold) pair, by feature and by tree gives a scope: one
  threshold, one or two features, or global (arithmetic or precision).
* ``first_divergences`` (NumPy artifact vs. ONNX graph, white box): on the
  failing probes, both forests are walked tree by tree in float32, and the
  first node where they part is recorded with its kind: a threshold, the
  scaler of a feature, a split feature, a split mode, a leaf weight or a base
  value. Unlike the structure check, it reports only differences that change
  an output on some probe.
"""

import json
from collections import Counter

import numpy as np

from .boundary_probe import load_forest_arrays
from .onnx_artifact import read_onnx_forest


def _pair(feature, threshold):
    return int(feature), float(np.float32(threshold))


def summarize_failures(model_path, probes, failing, share=0.9, top=10):
    """Rank what the failing probes point at, and name the smallest scope that explains ``share`` of them."""
    failing = np.asarray(failing, dtype=bool)
    n_fail = int(failing.sum())
    if n_fail == 0:
        return {"failing_probes": 0, "scope": "none"}
    f = load_forest_arrays(model_path, np.float64)
    nodes = probes.node[failing]
    feats = f.feature[nodes]
    trees = np.searchsorted(f.tree_offsets, nodes, side="right") - 1
    by_feature = Counter(feats.tolist())
    by_pair = Counter(_pair(g, t) for g, t in zip(feats, f.threshold[nodes]))
    by_tree = Counter(trees.tolist())
    probed = Counter(f.feature[probes.node].tolist())

    top_pair, top_pair_n = by_pair.most_common(1)[0]
    covered, k = 0, 0
    for _, c in by_feature.most_common():
        covered += c
        k += 1
        if covered >= share * n_fail:
            break
    if top_pair_n >= share * n_fail:
        scope, suspects = "threshold", [{"feature": top_pair[0], "threshold": top_pair[1]}]
    elif k <= 2:
        scope, suspects = "feature", [{"feature": g} for g, _ in by_feature.most_common(k)]
    else:
        scope, suspects = "global", []
    return {
        "failing_probes": n_fail,
        "scope": scope,
        "suspects": suspects,
        "features_implicated": len(by_feature),
        "features_probed": len(probed),
        "trees_implicated": len(by_tree),
        "top_features": [{"feature": g, "failing": c, "share_of_its_probes": c / probed[g]}
                         for g, c in by_feature.most_common(top)],
        "top_pairs": [{"feature": p[0], "threshold": p[1], "failing": c} for p, c in by_pair.most_common(top)],
        "top_trees": [{"tree": t, "failing": c} for t, c in by_tree.most_common(top)],
    }


def _artifact(npz_path):
    data = np.load(str(npz_path), allow_pickle=False)
    d = {k: data[k] for k in data.files}
    d.setdefault("strict", np.zeros(len(d["left"]), dtype=bool))
    d.setdefault("base_values", np.zeros(d["leaf_scores"].shape[1]))
    d["meta"] = json.loads(str(d["metadata_json"]))
    return d


def first_divergences(onnx_path, npz_path, rows, top=10):
    """Per row and tree, the first node where the artifact and the graph route or score differently."""
    g = read_onnx_forest(onnx_path)
    a = _artifact(npz_path)
    if len(a["left"]) != len(g["left"]) or not np.array_equal(a["tree_offsets"], g["tree_offsets"]):
        return {"comparable": False, "reason": "tree layout differs from the graph; see the structure check"}
    x = np.ascontiguousarray(rows, dtype=np.float32)
    zg = (x - g["offset"].astype(np.float32)) * g["scale"].astype(np.float32)
    za = (x - a["offset"].astype(np.float32)) * a["scale"].astype(np.float32)
    tg, ta = g["threshold"].astype(np.float32), a["threshold"].astype(np.float32)
    lg, la = g["leaf_scores"].astype(np.float32), a["leaf_scores"].astype(np.float32)
    leaf = g["is_leaf"]
    found = Counter()
    rows_hit = np.zeros(len(x), dtype=bool)

    def kind(n):
        if a["is_leaf"][n] != leaf[n] or a["left"][n] != g["left"][n] or a["right"][n] != g["right"][n]:
            return "tree layout"
        if a["feature"][n] != g["feature"][n]:
            return "split feature"
        if ta[n] != tg[n]:
            return "threshold"
        if a["strict"][n] != g["strict"][n]:
            return "split mode"
        return "scaler"

    for t in range(len(g["tree_offsets"]) - 1):
        cur = np.full(len(x), g["tree_offsets"][t], dtype=np.int64)
        idx = np.arange(len(x))
        while idx.size:
            c = cur[idx]
            at_leaf = leaf[c]
            if at_leaf.any():
                li, lc = idx[at_leaf], c[at_leaf]
                bad = np.any(la[lc] != lg[lc], axis=1)
                for n in np.unique(lc[bad]):
                    found[("leaf weight", t, int(n), -1)] += int((lc[bad] == n).sum())
                rows_hit[li[bad]] = True
                idx, c = idx[~at_leaf], c[~at_leaf]
                if not idx.size:
                    break
            fg, fa = g["feature"][c], a["feature"][c]
            vg, va = zg[idx, fg], za[idx, np.maximum(fa, 0)]
            go_g = np.where(g["strict"][c], vg < tg[c], vg <= tg[c])
            go_a = np.where(a["strict"][c], va < ta[c], va <= ta[c])
            part = go_g != go_a
            for n in np.unique(c[part]):
                found[(kind(n), t, int(n), int(fg[c == n][0]))] += int((c[part] == n).sum())
            rows_hit[idx[part]] = True
            keep = ~part
            idx, c, go_g = idx[keep], c[keep], go_g[keep]
            cur[idx] = np.where(go_g, g["left"][c], g["right"][c])
    base_differs = bool(np.any(a["base_values"].astype(np.float32) != g["base_values"].astype(np.float32)))
    by_kind, by_feature, by_tree = Counter(), Counter(), Counter()
    for (k, t, n, feat), cnt in found.items():
        by_kind[k] += cnt
        by_tree[t] += cnt
        if feat >= 0:
            by_feature[feat] += cnt
    return {
        "comparable": True,
        "rows": int(len(x)),
        "rows_with_a_divergence": int(rows_hit.sum()),
        "base_value_differs": base_differs,
        "divergent_nodes": len(found),
        "by_kind": dict(by_kind.most_common()),
        "top_nodes": [{"kind": k, "tree": t, "node": n, "feature": f, "rows": cnt}
                      for (k, t, n, f), cnt in found.most_common(top)],
        "top_features": [{"feature": f, "rows": cnt} for f, cnt in by_feature.most_common(top)],
        "top_trees": [{"tree": t, "rows": cnt} for t, cnt in by_tree.most_common(top)],
    }
