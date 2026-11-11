"""A real attack flow given two verdicts by a mixed fleet.

The paper's fleet claim (Section on boundary parity): when some boards serve the
primary runtime (float32, the deployed one) and others the fallback, a flow
whose features sit on shared thresholds can receive different verdicts depending
on which board sees it. Here we construct such a flow from a real attack record
and only features a flow's own sender shapes, and measure how far it has to move.

For each attack flow in the test split that ONNX Runtime labels Attack, and each
attacker-controllable feature, we snap that one feature to the raw float32 values
that bracket each distinct split threshold on it, leaving every other feature at
its recorded value. A hit is a single-feature variant that ONNX Runtime (and the
deployed float32 traversal) still labels Attack while a float64 traversal of the
same graph labels Benign: an evasion against the fallback-running boards only.
We report how many attack flows admit one, the size of the feature change, and
how often the change is within the feature's own recorded range.

    python exp14_fleet_divergence.py --data test_data.parquet [--attack-rows 4000]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgeids.data import RESULTS, feature_columns, load_test_split
from edgeids.forest import load_forest, predict_lockstep

# Features a flow's own sender shapes by choosing its packet sizes, counts and
# timing: forward-direction and whole-flow statistics. Backward-direction
# features depend on the victim's responses, so we treat them as not controllable.
CONTROLLABLE = [
    "flow_duration", "total_fwd_packets", "total_length_of_fwd_packets",
    "fwd_packet_length_max", "fwd_packet_length_min", "fwd_packet_length_mean",
    "flow_bytes_s", "flow_packets_s", "flow_iat_mean", "flow_iat_std",
    "flow_iat_max", "flow_iat_min", "fwd_iat_total", "fwd_iat_mean",
    "fwd_psh_flags", "packet_length_mean", "average_packet_size",
]

# Of the controllable features, these are integer counts/flags: a remote sender
# cannot place them strictly between two integers, so a crafted value that is not
# an integer is not realizable. The continuous features (durations, byte/packet
# rates, inter-arrival times, mean/averaged lengths) can take intermediate values.
INTEGER_VALUED = {"total_fwd_packets", "total_length_of_fwd_packets", "fwd_psh_flags"}


def ort_session():
    import onnxruntime as ort
    return ort.InferenceSession(os.path.join(os.path.dirname(RESULTS), "..", "..", "artifacts", "ids_rf.onnx"),
                                providers=["CPUExecutionProvider"])


def ort_labels(session, x, batch=20000):
    name = session.get_inputs()[0].name
    return np.concatenate([np.asarray(session.run(None, {name: x[i:i + batch]})[0]).reshape(-1)
                           for i in range(0, len(x), batch)]).astype(np.int64)


def np_labels(forest, x, batch=20000):
    return np.concatenate([predict_lockstep(forest, x[i:i + batch])[0] for i in range(0, len(x), batch)])


def thresholds_per_feature(f32):
    internal = ~f32.is_leaf
    out = {}
    for feat in np.unique(f32.feature[internal]):
        ts = np.unique(f32.threshold[internal & (f32.feature == feat)])
        out[int(feat)] = ts
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--attack-rows", type=int, default=4000, help="attack flows scanned (0 = all)")
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp14_fleet_divergence.json"))
    args = ap.parse_args()

    cols = feature_columns()
    ctrl = [cols.index(c) for c in CONTROLLABLE]
    x, y, names = load_test_split(args.data)
    npz = os.path.join(os.path.dirname(RESULTS), "..", "..", "artifacts", "ids_rf_numpy.npz")
    f32, f64 = load_forest(npz, "float32"), load_forest(npz, "float64")
    session = ort_session()

    # real attack flows that ONNX Runtime already labels Attack
    is_attack = (y == 1) & (ort_labels(session, x) == 1)
    idx = np.flatnonzero(is_attack)
    rng = np.random.default_rng(0)
    if args.attack_rows and len(idx) > args.attack_rows:
        idx = np.sort(rng.choice(idx, size=args.attack_rows, replace=False))

    thr = thresholds_per_feature(f32)
    scale, offset = f32.scale.astype(np.float64), f32.offset.astype(np.float64)

    hits = []  # one per evaded attack flow (first controllable single-feature variant found)
    changes_examined = 0
    for row_i in idx:
        base = x[row_i]
        found = None
        for feat in ctrl:
            raw_thr = (thr[feat].astype(np.float64) / scale[feat] + offset[feat]).astype(np.float32)
            cand = np.unique(np.concatenate([raw_thr, np.nextafter(raw_thr, np.float32(-np.inf)),
                                             np.nextafter(raw_thr, np.float32(np.inf))]))
            trial = np.repeat(base.reshape(1, -1), len(cand), axis=0).astype(np.float32)
            trial[:, feat] = cand
            changes_examined += len(cand)
            lo = ort_labels(session, trial)
            l64 = np_labels(f64, trial)
            l32 = np_labels(f32, trial)
            ok = np.flatnonzero((lo == 1) & (l32 == 1) & (l64 == 0))
            if len(ok):
                j = ok[0]
                found = {"row": int(row_i), "attack_name": str(names[row_i]), "feature": cols[feat],
                         "recorded_value": float(base[feat]), "crafted_value": float(cand[j]),
                         "abs_change": float(abs(cand[j] - base[feat])),
                         "within_feature_recorded_range": bool(x[:, feat].min() <= cand[j] <= x[:, feat].max()),
                         "feature_is_integer_valued": cols[feat] in INTEGER_VALUED,
                         "crafted_value_is_integer": float(cand[j]).is_integer()}
                break
        if found:
            hits.append(found)

    report = {
        "numpy": np.__version__, "attack_flows_scanned": int(len(idx)),
        "single_feature_variants_examined": int(changes_examined),
        "controllable_features": CONTROLLABLE,
        "attack_flows_with_a_fleet_divergence": len(hits),
        "fraction": len(hits) / len(idx) if len(idx) else 0.0,
        "within_recorded_range": int(sum(h["within_feature_recorded_range"] for h in hits)),
        "via_continuous_feature": int(sum(not h["feature_is_integer_valued"] for h in hits)),
        "via_continuous_feature_within_range": int(sum((not h["feature_is_integer_valued"]) and h["within_feature_recorded_range"] for h in hits)),
        "abs_change_continuous_median": float(np.median([h["abs_change"] for h in hits if not h["feature_is_integer_valued"]])) if any(not h["feature_is_integer_valued"] for h in hits) else None,
        "features_used": {c: int(sum(h["feature"] == c for h in hits)) for c in CONTROLLABLE if any(h["feature"] == c for h in hits)},
        "examples": sorted([h for h in hits if not h["feature_is_integer_valued"]], key=lambda h: h["abs_change"])[:6],
    }
    print(json.dumps({k: v for k, v in report.items() if k not in ("controllable_features",)}, indent=2))
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
