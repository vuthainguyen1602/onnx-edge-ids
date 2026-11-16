"""Do the three boundary probes straddle the threshold after float32 scaling?

The probes of Section 4 place feature g(v) on float32(x*(v)), x*(v) = theta/s + o
(Eq. 1, evaluated in float64), and on its float32 predecessor and successor, in
RAW space. The served graph compares the float32 SCALED value
z = (x - o) * s against theta. This asks, on the data-free probe set of all
thirteen models (exp19):

  * straddle: for what share of internal nodes do the three probes land on both
    sides of theta in float32 scaled space (one goes to the true branch, one to
    the false branch)?
  * why not: when they do not, is it because the scale is below one (a raw
    float32 step moves z by less than a scaled float32 step) and how many raw
    float32 steps separate float32(x*) from the exact float32 crossing, i.e. the
    largest raw float32 a with go_true(z(a)) and its successor going false?
  * does it matter: a "crossing" probe set that places g(v) on that exact pair
    (a, next(a)) instead, with the same routing rows, run through the same gate
    (labels + confidences at 1e-5) for the float32 and float64 traversals
    against ONNX Runtime, next to the shipped three-probe set.

    python exp24_probe_straddle.py
"""

import json
import os

import numpy as np

from edgeids.data import ARTIFACTS, RESULTS
from edgeids.gates import probe_gate
from edgeids.models import MODEL_KEYS
from edgeids.onnx_forest import _go_true, load_onnx_forest, path_rows, reaches
from edgeids.probes import datafree_probes
from edgeids.runtime import numpy_probs, ort_probs


def ordered(x):
    """float32 -> int64 that is monotone in the float value (adjacent floats differ by 1)."""
    b = np.asarray(x, dtype=np.float32).view(np.int32).astype(np.int64)
    return np.where(b >= 0, b, -(b & 0x7FFFFFFF))


def unordered(k):
    k = np.asarray(k, dtype=np.int64)
    b = np.where(k >= 0, k, (-k) | 0x80000000).astype(np.uint32)
    return b.view(np.float32)


def go_true_raw(forest, feat, thr, mode, raw):
    with np.errstate(over="ignore", invalid="ignore"):  # the exponential search can step past float32 range
        z = (raw.astype(np.float32) - forest.offset[feat].astype(np.float32)) * forest.scale[feat].astype(np.float32)
    return _go_true(mode, z.astype(np.float32), thr)


def crossing(forest, internal, raw0):
    """Largest raw float32 a whose scaled value takes the true branch (LEQ/LT, s > 0); NaN where undefined."""
    feat, thr, mode = forest.feature[internal], forest.threshold[internal], forest.mode[internal]
    ok = (forest.scale[feat] > 0) & np.isin(mode, (0, 1)) & np.isfinite(raw0)
    k0 = ordered(raw0)
    lo, hi = k0.copy(), k0.copy()          # invariant once bracketed: go_true(lo) and not go_true(hi)
    t0 = go_true_raw(forest, feat, thr, mode, raw0)
    step = np.ones(len(internal), dtype=np.int64)
    found = ~ok
    for _ in range(40):                     # exponential search away from float32(x*)
        cand = np.where(t0, k0 + step, k0 - step)
        g = go_true_raw(forest, feat, thr, mode, unordered(cand))
        new = ~found & (g != t0)
        hi = np.where(new & t0, cand, hi)
        lo = np.where(new & ~t0, cand, lo)
        lo = np.where(~found & ~new & t0, cand, lo)
        hi = np.where(~found & ~new & ~t0, cand, hi)
        found |= new
        step *= 2
    ok &= found
    for _ in range(64):                     # bisection to adjacent floats
        active = ok & (hi - lo > 1)
        if not active.any():
            break
        mid = (lo + hi) // 2
        g = go_true_raw(forest, feat, thr, mode, unordered(mid))
        lo = np.where(active & g, mid, lo)
        hi = np.where(active & ~g, mid, hi)
    return ok, lo, k0


def run_model(path):
    forest = load_onnx_forest(path)
    internal, probes = datafree_probes(forest)
    n = len(internal)
    feat, thr, mode = forest.feature[internal], forest.threshold[internal], forest.mode[internal]
    vals = np.stack([probes[i * n:(i + 1) * n][np.arange(n), feat] for i in range(3)], axis=1)
    sides = np.stack([go_true_raw(forest, feat, thr, mode, vals[:, i]) for i in range(3)], axis=1)
    straddle = sides.any(axis=1) & (~sides).any(axis=1)
    small = forest.scale[feat] < 1

    ok, a, k0 = crossing(forest, internal, vals[:, 0])
    dist = np.where(ok, np.where(sides[:, 0], a + 1 - k0, k0 - a), 0)  # raw float32 steps from float32(x*) to the other side

    base = np.repeat(forest.offset.astype(np.float32).reshape(1, -1), n, axis=0)
    rows = path_rows(forest, internal, base)
    cross = []
    for k in (a, a + 1):
        p = rows.copy()
        p[np.arange(n), feat] = np.where(ok, unordered(k), vals[:, 0])
        cross.append(p)
    cross = np.concatenate(cross)
    reach_cross = reaches(forest, cross, np.tile(internal, 2))

    ref3, refc = ort_probs(path, probes), ort_probs(path, cross)
    pct = lambda q: int(np.percentile(dist[ok & ~straddle], q)) if (ok & ~straddle).any() else 0
    return {
        "internal_nodes": int(n),
        "three_probes_straddle": float(straddle.mean()),
        "scale_below_one": {"share_of_nodes": float(small.mean()),
                            "straddle_if_scale_below_one": float(straddle[small].mean()) if small.any() else None,
                            "straddle_if_scale_at_least_one": float(straddle[~small].mean()) if (~small).any() else None},
        "crossing_found": float(ok.mean()),
        "raw_steps_to_crossing_when_not_straddling": {"median": pct(50), "p90": pct(90), "max": pct(100)},
        "crossing_probes": {"probes": int(len(cross)), "reaching_their_node": float(reach_cross.mean()),
                            "straddle": float(ok.mean())},
        "gate_three_probes": {"float32": probe_gate(numpy_probs(forest, probes, np.float32), ref3),
                              "float64": probe_gate(numpy_probs(forest, probes, np.float64), ref3)},
        "gate_crossing_probes": {"float32": probe_gate(numpy_probs(forest, cross, np.float32), refc),
                                 "float64": probe_gate(numpy_probs(forest, cross, np.float64), refc)},
    }


def main():
    import onnxruntime
    report = {"numpy": np.__version__, "onnxruntime": onnxruntime.__version__, "datasets": {}}
    jobs = [("CICIDS2017", os.path.join(RESULTS, "exp5_models"), ["deployed_rf"] + list(MODEL_KEYS)),
            ("UNSW-NB15", os.path.join(RESULTS, "exp10_models", "unsw_nb15"), list(MODEL_KEYS)),
            ("ToN-IoT", os.path.join(RESULTS, "exp10_models", "ton_iot"), list(MODEL_KEYS))]
    for name, model_dir, keys in jobs:
        report["datasets"][name] = {}
        for key in keys:
            path = os.path.join(ARTIFACTS, "ids_rf.onnx") if key == "deployed_rf" else os.path.join(model_dir, f"{key}.onnx")
            rep = run_model(path)
            report["datasets"][name][key] = rep
            g3, gc = rep["gate_three_probes"], rep["gate_crossing_probes"]
            print(f"{name:10s} {key:13s} straddle {rep['three_probes_straddle']:.3f} "
                  f"steps(med/p90/max) {rep['raw_steps_to_crossing_when_not_straddling']} "
                  f"f64 flips 3p {g3['float64']['label_flips']} cross {gc['float64']['label_flips']} "
                  f"f32 fails 3p {g3['float32']['gate_fails']} cross {gc['float32']['gate_fails']}", flush=True)
    out = os.path.join(RESULTS, "exp24_probe_straddle.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
