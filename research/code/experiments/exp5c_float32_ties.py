"""Where does a float32 traversal still disagree with ONNX Runtime?

exp5 over five training seeds finds a handful of float32 probe flips on the
scikit-learn RandomForest export (none on replay, none on the other forests).
This script rebuilds those probes and reports, for every flipped one, ONNX
Runtime's P(attack) and the float32 traversal's summed score. If both sit
within summation error of 0.5 the flip is a tie broken by the order in which
200 leaf weights are added, not a routing difference.

    python exp5c_float32_ties.py --test $DATA --model sklearn_rf \
        --run 2:/tmp/exp5_models_seed2 --run 3:/tmp/exp5_models_seed3 --run 4:/tmp/exp5_models_seed4
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgeids.data import RESULTS, load_test_split
from edgeids.onnx_forest import labels_from_scores, load_onnx_forest, path_rows, raw_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--model", default="sklearn_rf")
    ap.add_argument("--run", action="append", required=True, help="seed:model_dir, as passed to exp5")
    ap.add_argument("--max-nodes", type=int, default=100000)
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp5c_float32_ties.json"))
    args = ap.parse_args()

    import onnxruntime as ort

    x, _, _ = load_test_split(args.test)
    report = {"model": args.model, "onnxruntime": ort.__version__, "numpy": np.__version__, "runs": []}
    for spec in args.run:
        seed, model_dir = spec.split(":", 1)
        seed, path = int(seed), os.path.join(model_dir, f"{args.model}.onnx")
        forest = load_onnx_forest(path)
        # same draws as exp5_generalization.probe_model
        rng = np.random.default_rng(seed)
        internal = np.flatnonzero(~forest.is_leaf)
        if len(internal) > args.max_nodes:
            internal = np.sort(rng.choice(internal, size=args.max_nodes, replace=False))
        feat = forest.feature[internal]
        rows = path_rows(forest, internal, x[rng.choice(len(x), size=len(internal), replace=True)])
        scale, offset = forest.scale.astype(np.float64), forest.offset.astype(np.float64)
        raw = np.where(scale[feat] != 0, forest.threshold[internal].astype(np.float64) / np.where(scale[feat] != 0, scale[feat], 1) + offset[feat], offset[feat]).astype(np.float32)
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        name = session.get_inputs()[0].name
        flips = []
        for variant, values in (("on_threshold", raw), ("one_ulp_below", np.nextafter(raw, np.float32(-np.inf))),
                                ("one_ulp_above", np.nextafter(raw, np.float32(np.inf)))):
            probes = rows.copy()
            probes[np.arange(len(internal)), feat] = values
            prob = np.concatenate([np.asarray(session.run(None, {name: probes[i:i + 50000]})[1]) for i in range(0, len(probes), 50000)])
            scores = raw_scores(forest, probes, np.float32)
            mine = scores[:, 0] if forest.n_targets == 1 else scores[:, 1]
            bad = np.flatnonzero(labels_from_scores(forest, scores) != (prob[:, 1] > 0.5).astype(np.int64))
            flips += [{"variant": variant, "ort_p_attack": float(prob[b, 1]), "float32_score": float(mine[b]),
                       "abs_difference": float(abs(prob[b, 1] - mine[b]))} for b in bad]
        run = {"seed": seed, "probes": int(3 * len(internal)), "float32_flips": len(flips),
               "max_distance_from_one_half": max([max(abs(f["ort_p_attack"] - 0.5), abs(f["float32_score"] - 0.5)) for f in flips], default=0.0),
               "flips": flips}
        report["runs"].append(run)
        print({k: v for k, v in run.items() if k != "flips"}, flush=True)
    report["float32_flips_total"] = sum(r["float32_flips"] for r in report["runs"])
    report["max_distance_from_one_half"] = max(r["max_distance_from_one_half"] for r in report["runs"])
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
