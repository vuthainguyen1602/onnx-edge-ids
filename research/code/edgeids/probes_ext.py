"""Data-free probes for every split mode of TreeEnsemble and for missing-value routing (exp50).

The release gate's probe builder handles BRANCH_LEQ and BRANCH_LT, the modes that the tree
converters in the paper emit for numeric splits. Converters also emit BRANCH_EQ (LightGBM's
zero-as-missing splits, categorical splits written as equality chains) and attach a
missing-value flag to each node. This builds, from the graph alone:

  * monotone splits (LEQ, LT, GTE, GT): the crossing pair, the last raw float32 value on one
    side of the node's float32 decision and the first on the other (Theorem 1 applies);
  * equality splits (EQ, NEQ): the decision changes twice, so the probes are the raw values
    whose float32-scaled value equals the threshold (the first and the last of them, if any),
    and the last raw value below and the first above them;
  * missing values: one probe with NaN in the node's feature, which a runtime routes by the
    node's missing-value flag.

Every probe is routed to its node by ``onnx_forest.path_rows`` with the unconstrained
features at the scaler offset, as the gate does.
"""

from dataclasses import dataclass

import numpy as np

from edgeids.onnx_forest import _go_true, parents, path_rows, reaches

KINDS = ("monotone_crossing", "equality_on", "equality_neighbour", "missing")


@dataclass
class Probes:
    rows: np.ndarray
    node: np.ndarray
    kind: np.ndarray       # index into KINDS


def _ordered(x):
    i = np.asarray(x, dtype=np.float32).view(np.int32).astype(np.int64)
    return np.where(i < 0, -(i & 0x7FFFFFFF), i)


def _unordered(k):
    k = np.asarray(k, dtype=np.int64)
    return np.where(k < 0, (-k) | 0x80000000, k).astype(np.uint32).view(np.int32).view(np.float32)


def _scaled(forest, g, r):
    return ((np.float32(r) - np.float32(forest.offset[g])) * np.float32(forest.scale[g])).astype(np.float32)


def _decide(forest, v, r):
    g = int(forest.feature[v])
    z = np.atleast_1d(_scaled(forest, g, r))
    return _go_true(np.array([forest.mode[v]] * len(z)), z, np.full(len(z), forest.threshold[v], dtype=np.float32))


def _switch(forest, v, start_k, target, direction):
    """From ordered key ``start_k`` step in ``direction`` (+1/-1), doubling, to the first key whose
    decision equals ``target``; return the adjacent pair (last != target, first == target) or None."""
    if _decide(forest, v, _unordered(start_k))[0] == target:
        return None
    step, last = 1, start_k
    for _ in range(64):
        k = start_k + direction * step
        if abs(k) > 0x7F7FFFFF:
            return None
        if _decide(forest, v, _unordered(k))[0] == target:
            lo, hi = last, k
            while abs(hi - lo) > 1:
                mid = (lo + hi) // 2
                if _decide(forest, v, _unordered(mid))[0] == target:
                    hi = mid
                else:
                    lo = mid
            return lo, hi
        last, step = k, step * 2
    return None


def _run_end(forest, v, k, on, direction):
    """Last key from ``k`` in ``direction`` whose decision is still ``on`` (``k`` itself is on)."""
    inside, step = k, 1
    while abs(inside + direction * step) <= 0x7F7FFFFF and _decide(forest, v, _unordered(inside + direction * step))[0] == on:
        inside, step = inside + direction * step, step * 2
    outside = inside + direction * step
    while abs(outside - inside) > 1:
        mid = (inside + outside) // 2
        if _decide(forest, v, _unordered(mid))[0] == on:
            inside = mid
        else:
            outside = mid
    return inside


def values_for_node(forest, v):
    """Raw float32 values of node v's feature to probe, with their kinds."""
    g, t, m = int(forest.feature[v]), float(forest.threshold[v]), int(forest.mode[v])
    s, o = float(forest.scale[g]), float(forest.offset[g])
    start = np.float32(t / s + o) if s != 0 else np.float32(o)
    k0 = int(_ordered(start))
    out = []
    if m in (0, 1, 2, 3):
        d0 = _decide(forest, v, start)[0]
        pair = _switch(forest, v, k0, not d0, +1) or _switch(forest, v, k0, not d0, -1)
        if pair:
            out += [(_unordered(pair[0]), 0), (_unordered(pair[1]), 0)]
    elif m in (4, 5):
        on = m == 4                            # EQ is true on the threshold, NEQ false
        # raw values that land on the threshold: search both ways from the start for the run
        hits = [k for k in range(k0 - 4, k0 + 5) if _decide(forest, v, _unordered(k))[0] == on]
        if hits:
            # the run of raw values landing on the threshold can be long where the scale is small:
            # find its ends by galloping out of it and bisecting back
            lo = _run_end(forest, v, min(hits), on, -1)
            hi = _run_end(forest, v, max(hits), on, +1)
            out += [(_unordered(lo), 1), (_unordered(hi), 1), (_unordered(lo - 1), 2), (_unordered(hi + 1), 2)]
        else:                                   # no float32 input lands on it: probe the two around it
            z0 = _scaled(forest, g, start)
            below = start if z0 < np.float32(t) else np.nextafter(start, np.float32(-np.inf))
            out += [(np.float32(below), 2), (np.nextafter(np.float32(below), np.float32(np.inf)), 2)]
    out.append((np.float32(np.nan), 3))
    return out


def repair(forest, rows, targets, max_steps=64):
    """Nudge routed rows that miss their node: where float32 rounding puts a value on the wrong
    side of an ancestor's threshold (a bound plus one, rounded back onto the bound), step that
    feature one float32 value at a time toward the side the path needs."""
    parent, went_true = parents(forest)
    rows = rows.copy()
    for i in np.flatnonzero(~reaches(forest, rows, targets)):
        path, cur = [], int(targets[i])
        while parent[cur] >= 0:
            path.append((int(parent[cur]), bool(went_true[cur]))); cur = int(parent[cur])
        for _ in range(max_steps):
            wrong = None
            for p, need in reversed(path):                     # root first
                g = int(forest.feature[p])
                z = np.atleast_1d(_scaled(forest, g, rows[i, g]))
                go = _go_true(np.array([forest.mode[p]]), z, np.array([forest.threshold[p]], dtype=np.float32))[0]
                if go != need:
                    wrong = (p, g, need); break
            if wrong is None:
                break
            p, g, need = wrong
            m = int(forest.mode[p])
            if m in (4, 5):
                break                                           # an equality on the wrong side: not a rounding issue
            up = (m in (0, 1)) != need                          # LEQ/LT true wants lower values, GTE/GT true wants higher
            if forest.scale[g] < 0:
                up = not up
            rows[i, g] = np.nextafter(np.float32(rows[i, g]), np.float32(np.inf if up else -np.inf))
    return rows


def build(forest, nodes=None):
    internal = np.flatnonzero(~forest.is_leaf) if nodes is None else np.asarray(nodes)
    base = np.repeat(forest.offset.astype(np.float32).reshape(1, -1), len(internal), axis=0)
    routed = repair(forest, path_rows(forest, internal, base), internal)
    rows, node, kind = [], [], []
    for i, v in enumerate(internal):
        g = int(forest.feature[v])
        for val, k in values_for_node(forest, int(v)):
            r = routed[i].copy()
            r[g] = val
            rows.append(r); node.append(int(v)); kind.append(k)
    return Probes(np.asarray(rows, dtype=np.float32), np.asarray(node, dtype=np.int64), np.asarray(kind, dtype=np.int64))


def reach_rate(forest, probes):
    ok = ~np.isnan(probes.rows[np.arange(len(probes.node)), forest.feature[probes.node]])
    return float(reaches(forest, probes.rows[ok], probes.node[ok]).mean()) if ok.any() else None
