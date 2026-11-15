"""Summarize exp5 across training seeds: seed 0 is results/exp5_generalization.json,
seeds 1.. are results/exp5_seeds/seed<N>.json (exp5_generalization.py --seed N --out ...).

For each trained forest: probe and replay flip counts per seed, the probe
density relative to replay, and the probability that a 2,000-row replay misses.
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgeids.data import RESULTS

MODELS = ["sklearn_rf", "sklearn_hgb", "lightgbm", "xgboost_hist"]
KINDS = {"float64": "numpy_float64_vs_ort", "framework": "native_vs_ort"}


def main():
    files = [os.path.join(RESULTS, "exp5_generalization.json")] + sorted(glob.glob(os.path.join(RESULTS, "exp5_seeds", "seed*.json")))
    runs = [json.load(open(f))["models"] for f in files]
    out = {"seeds": len(runs), "files": [os.path.relpath(f, RESULTS) for f in files], "models": {}}
    for m in MODELS:
        entry = {}
        for kind, key in KINDS.items():
            probes, replay, density, miss, f32 = [], [], [], [], 0
            for r in runs:
                p, rp = r[m]["probes"]["total"], r[m]["replay"]
                probes.append(p[key + "_flips"]); replay.append(rp[key + "_flips"])
                miss.append(rp[key + "_prob_2000_row_replay_misses"])
                density.append(p.get(key + "_density_vs_replay"))
                f32 += p["numpy_float32_vs_ort_flips"] + rp["numpy_float32_vs_ort_flips"]
            entry[kind] = {"probe_flips": probes, "replay_flips": replay, "prob_2000_row_replay_misses": miss,
                           "probe_density_vs_replay": density}
            entry["float32_flips_all_seeds"] = f32
        entry["pairs"] = [r[m]["sharing"]["distinct_feature_threshold_pairs"] for r in runs]
        out["models"][m] = entry
        print(m, json.dumps(entry))
    path = os.path.join(RESULTS, "exp5_seeds_summary.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
