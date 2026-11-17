"""When is a probe set complete? Exact per-node disagreement sets for arithmetic faults.

Let A be the reference arithmetic (ONNX Runtime: float32 scaler, then the node's
comparison) and B a faulty one. Call B *elementwise monotone* when, at every
node, B's decision depends only on the node's own raw feature value r and
switches once as r grows. A takes the true branch exactly for r <= c_A (in
float32 order), so where A and B disagree at node v is the set D(v) of float32
values between c_A and c_B. A probe set catches B at v, given that the probe is
routed to v, iff it places one of its values for v inside D(v).

Claim (crossing completeness): the pair {c_A, next(c_A)} hits D(v) whenever D(v)
is non-empty. If B agreed with A at both, B would be true at c_A and false at
next(c_A), so B's single switch would sit exactly where A's does and D(v) would be
empty. The three-probe set {float32(x*), one step either side} has no such
guarantee: it misses D(v) whenever the disagreement lies more than one raw
float32 step from float32(x*), which happens where the scale is small (exp24).

This computes c_A and c_B for every internal node by bisection over float32,
for the six arithmetic faults of exp11 written generically for any graph, on all
thirteen models, and reports per model and fault: the nodes where B can be told
apart from A at all (D non-empty), how many of them each probe set hits, and the
sizes of D, which bound what an exhaustive enumeration of the disagreement
would cost.

    python exp31_probe_completeness.py
"""

import json
import os
import sys

import numpy as np

from edgeids.data import ARTIFACTS, RESULTS
from edgeids.models import MODEL_KEYS

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
from onnx_edge_ids.boundary_probe import load_forest_arrays  # noqa: E402

F32 = np.float32
LO, HI = -3.0e38, 3.0e38


def ordered(x):
    b = np.asarray(x, dtype=F32).view(np.int32).astype(np.int64)
    return np.where(b >= 0, b, -(b & 0x7FFFFFFF))


def unordered(k):
    k = np.asarray(k, dtype=np.int64)
    return np.where(k >= 0, k, (-k) | 0x80000000).astype(np.uint32).view(F32)


def decisions(f, nodes):
    """name -> function(raw float32 values, one per node) -> bool (true branch), for A and each fault."""
    g = f.feature[nodes]
    o32, s32 = f.offset[g].astype(F32), f.scale[g].astype(F32)
    o64, s64 = f.offset[g].astype(np.float64), f.scale[g].astype(np.float64)
    t32 = f.threshold[nodes].astype(F32)
    strict = f.strict[nodes]
    cmp = lambda z, t, st: np.where(st, z < t, z <= t)
    t6 = np.array([float(f"{t:.6g}") for t in t32], dtype=F32)
    raw_t = (t32 / s32 + o32).astype(F32)
    with np.errstate(all="ignore"):
        return {
            "reference": lambda r: cmp(((r - o32) * s32).astype(F32), t32, strict),
            "float64": lambda r: cmp((r.astype(np.float64) - o64) * s64, t32.astype(np.float64), strict),
            "float64_scaler": lambda r: cmp(((r.astype(np.float64) - o64) * s64).astype(F32), t32, strict),
            "scaler_reordered": lambda r: cmp((r * s32 - o32 * s32).astype(F32), t32, strict),
            "raw_thresholds": lambda r: cmp(r, raw_t, strict),
            "comparison_flipped": lambda r: cmp(((r - o32) * s32).astype(F32), t32, ~strict),
            "decimal_thresholds": lambda r: cmp(((r - o32) * s32).astype(F32), t6, strict),
        }


def switch_point(fn):
    """Largest raw float32 (as an ordered int) for which fn is true; fn non-increasing in r."""
    def run(n):
        lo = np.full(n, int(ordered(F32(LO))), dtype=np.int64)
        hi = np.full(n, int(ordered(F32(HI))), dtype=np.int64)
        with np.errstate(all="ignore"):
            t_lo, t_hi = fn(unordered(lo)), fn(unordered(hi))
        for _ in range(70):
            mid = (lo + hi) // 2
            with np.errstate(all="ignore"):
                m = fn(unordered(mid))
            lo = np.where(m, mid, lo)
            hi = np.where(m, hi, mid)
        return np.where(~t_lo, lo - 1, np.where(t_hi, hi, lo)), t_lo, t_hi
    return run


def analyse(path):
    f = load_forest_arrays(path, np.float64)
    nodes = np.flatnonzero(~f.is_leaf)
    nodes = nodes[f.scale[f.feature[nodes]] > 0]
    dec = decisions(f, nodes)
    n = len(nodes)
    c_a, _, _ = switch_point(dec["reference"])(n)
    g = f.feature[nodes]
    x_star = (f.threshold[nodes] / f.scale[g] + f.offset[g]).astype(F32)
    k0 = ordered(x_star)
    three = np.stack([k0 - 1, k0, k0 + 1], axis=1)
    cross = np.stack([c_a, c_a + 1], axis=1)
    out = {"internal_nodes_with_positive_scale": int(n), "faults": {}}
    for name, fn in dec.items():
        if name == "reference":
            continue
        c_b, _, _ = switch_point(fn)(n)
        lo, hi = np.minimum(c_a, c_b), np.maximum(c_a, c_b)       # D(v) = (lo, hi] in float32 order
        size = hi - lo
        live = size > 0
        hit3 = np.any((three > lo[:, None]) & (three <= hi[:, None]), axis=1) & live
        hitx = np.any((cross > lo[:, None]) & (cross <= hi[:, None]), axis=1) & live
        # sanity: the crossing pair must disagree under B at every live node (direct evaluation, not the interval)
        va, vb = unordered(c_a), unordered(c_a + 1)
        direct = (fn(va) != dec["reference"](va)) | (fn(vb) != dec["reference"](vb))
        sizes = size[live]
        out["faults"][name] = {
            "nodes_where_B_differs_from_A": int(live.sum()),
            "three_probe_set_hits": int(hit3.sum()),
            "crossing_pair_hits": int(hitx.sum()),
            "crossing_pair_hits_by_direct_evaluation": int((direct & live).sum()),
            "disagreement_size_float32_values": ({"median": int(np.median(sizes)), "p90": int(np.percentile(sizes, 90)),
                                                  "max": int(sizes.max()), "total": int(sizes.sum())}
                                                 if live.any() else None),
        }
    return out


def main():
    jobs = [("CICIDS2017", "deployed_rf", os.path.join(ARTIFACTS, "ids_rf.onnx"))]
    for ds, d in (("CICIDS2017", "exp5_models"), ("UNSW-NB15", os.path.join("exp10_models", "unsw_nb15")),
                  ("ToN-IoT", os.path.join("exp10_models", "ton_iot"))):
        jobs += [(ds, k, os.path.join(RESULTS, d, f"{k}.onnx")) for k in MODEL_KEYS]
    report = {"models": {}}
    for ds, key, path in jobs:
        rep = analyse(path)
        report["models"][f"{ds}/{key}"] = rep
        for fault, r in rep["faults"].items():
            print(ds, key, fault, r["nodes_where_B_differs_from_A"], "3-probe", r["three_probe_set_hits"],
                  "crossing", r["crossing_pair_hits"], r["crossing_pair_hits_by_direct_evaluation"],
                  r["disagreement_size_float32_values"], flush=True)
    out = os.path.join(RESULTS, "exp31_probe_completeness.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
