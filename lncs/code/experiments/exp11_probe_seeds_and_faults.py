"""Two robustness checks on the boundary probes of exp2, on the deployed forest.

(a) Base rows. exp2 draws one base row per node with seed 0. Repeat the draw
    with several seeds, and once with every unconstrained feature held at the
    scaler offset (the data-free probe set the release gate uses), and report
    how much the float64 flip count moves. On the data-free set the float64
    traversal flips no label, so its confidence deviation from ONNX Runtime,
    which is what the release gate's tolerance catches, is reported too.

(b) Injected faults. exp2 studies one faulty traversal (float64). Here six
    single-line deviations from the float32 traversal are each compared with
    ONNX Runtime on the probes and on the replayed test split:

      float64            scaler and comparison in float64 (exp2's variant)
      float64_scaler     scaler in float64, result cast to float32, compare in float32
      scaler_reordered   x * scale - offset * scale instead of (x - offset) * scale
      raw_thresholds     thresholds folded back to raw space, raw x compared with them
      strict_less        x < t instead of x <= t
      decimal_thresholds thresholds round-tripped through 6 significant digits

    Every fault is expressed as (scaled matrix, threshold table) and run by the
    same lockstep routine, so only the stated line differs.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edgeids.data import ARTIFACTS, RESULTS, load_test_split
from edgeids.forest import _sum_over_trees, load_forest, predict_lockstep
from exp2_boundary_parity import numpy_labels, onnx_labels, path_rows


def build_probes(f64, internal, feat, base):
    rows = path_rows(f64, internal, base)
    raw = (f64.threshold[internal] / f64.scale[feat] + f64.offset[feat]).astype(np.float32)
    out = []
    for values in (raw, np.nextafter(raw, np.float32(-np.inf)), np.nextafter(raw, np.float32(np.inf))):
        p = rows.copy()
        p[np.arange(len(internal)), feat] = values
        out.append(p)
    return np.concatenate(out)


def labels_with(forest, scaled, batch=2000):
    trees = np.arange(forest.n_trees)
    return np.concatenate([
        np.argmax(_sum_over_trees(forest, scaled[i:i + batch], trees, forest.leaf_scores), axis=1)
        for i in range(0, len(scaled), batch)
    ]).astype(np.int64)


def confidence_deltas(session, forests, probes, tolerance=1e-5, batch=20000):
    """Largest |P(class) - ONNX Runtime's| per traversal, and rows beyond the release gate's tolerance."""
    name = session.get_inputs()[0].name
    ref = np.concatenate([np.asarray(session.run(None, {name: probes[i:i + batch]})[1]) for i in range(0, len(probes), batch)])
    out = {}
    for key, forest in forests.items():
        scores = np.concatenate([predict_lockstep(forest, probes[i:i + 2000])[1] for i in range(0, len(probes), 2000)]).astype(np.float64)
        delta = np.abs(scores / scores.sum(axis=1, keepdims=True) - ref).max(axis=1)
        out[key] = {"max_delta": float(delta.max()), "rows_over_tolerance": int((delta > tolerance).sum()), "tolerance": tolerance}
    return out


def faults(f32, f64):
    """name -> function(x float32) returning labels."""
    o32, s32 = f32.offset, f32.scale
    o64, s64 = f64.offset, f64.scale
    internal = ~f32.is_leaf

    strict = replace(f32, threshold=np.where(internal, np.nextafter(f32.threshold, np.float32(-np.inf)), f32.threshold))
    decimal = replace(f32, threshold=np.array([float(f"{t:.6g}") for t in f32.threshold], dtype=np.float32))
    raw_thr = f32.threshold.copy()
    raw_thr[internal] = f32.threshold[internal] / s32[f32.feature[internal]] + o32[f32.feature[internal]]
    raw = replace(f32, threshold=raw_thr.astype(np.float32))

    return {
        "float64": lambda x: labels_with(f64, (x.astype(np.float64) - o64) * s64),
        "float64_scaler": lambda x: labels_with(f32, ((x.astype(np.float64) - o64) * s64).astype(np.float32)),
        "scaler_reordered": lambda x: labels_with(f32, x * s32 - o32 * s32),
        "raw_thresholds": lambda x: labels_with(raw, np.ascontiguousarray(x, dtype=np.float32)),
        "strict_less": lambda x: labels_with(strict, (x - o32) * s32),
        "decimal_thresholds": lambda x: labels_with(decimal, (x - o32) * s32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="test_data.parquet directory")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp11_probe_seeds_and_faults.json"))
    args = ap.parse_args()

    import onnxruntime as ort

    started = time.perf_counter()
    x, _, _ = load_test_split(args.data)
    npz = os.path.join(ARTIFACTS, "ids_rf_numpy.npz")
    f32, f64 = load_forest(npz, "float32"), load_forest(npz, "float64")
    session = ort.InferenceSession(os.path.join(ARTIFACTS, "ids_rf.onnx"), providers=["CPUExecutionProvider"])
    internal = np.flatnonzero(~f32.is_leaf)
    feat = f32.feature[internal]

    report = {"numpy": np.__version__, "onnxruntime": ort.__version__, "internal_nodes": int(len(internal)),
              "replay_rows": int(len(x))}

    # (a) base rows
    runs, probes0, ort0 = [], None, None
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        base = x[rng.choice(len(x), size=len(internal), replace=True)]
        probes = build_probes(f64, internal, feat, base)
        lab = onnx_labels(session, probes)
        r = {"seed": seed, "probes": int(len(probes)),
             "float32_flips": int((numpy_labels(f32, probes) != lab).sum()),
             "float64_flips": int((numpy_labels(f64, probes) != lab).sum())}
        runs.append(r)
        print("base rows", r, flush=True)
        if seed == 0:
            probes0, ort0 = probes, lab
    c = np.array([r["float64_flips"] for r in runs])
    base = np.repeat(f32.offset.reshape(1, -1), len(internal), axis=0)
    probes = build_probes(f64, internal, feat, base)
    lab = onnx_labels(session, probes)
    conf = confidence_deltas(session, {"float32": f32, "float64": f64}, probes)
    report["base_rows"] = {
        "runs": runs,
        "float64_flips_mean": float(c.mean()), "float64_flips_sd": float(c.std(ddof=1)),
        "float64_flips_min": int(c.min()), "float64_flips_max": int(c.max()),
        "float32_flips_max": int(max(r["float32_flips"] for r in runs)),
        "scaler_offset_base": {"probes": int(len(probes)),
                               "float32_flips": int((numpy_labels(f32, probes) != lab).sum()),
                               "float64_flips": int((numpy_labels(f64, probes) != lab).sum()),
                               "confidence_vs_onnx": conf},
    }
    print("offset base", report["base_rows"]["scaler_offset_base"], flush=True)

    # (b) injected faults, on the seed-0 probes and on replay
    ort_replay = onnx_labels(session, x)
    report["faults"] = {}
    for name, fn in faults(f32, f64).items():
        p = int((fn(probes0) != ort0).sum())
        r = int((fn(x) != ort_replay).sum())
        rate = r / len(x)
        entry = {"probe_flips": p, "probes": int(len(probes0)), "replay_flips": r,
                 "prob_2000_row_replay_misses": float((1.0 - rate) ** 2000),
                 "probe_density_vs_replay": (p / len(probes0)) / rate if r else None}
        report["faults"][name] = entry
        print(name, entry, flush=True)

    report["seconds"] = round(time.perf_counter() - started, 1)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
