"""Summarize exp10 across training seeds for UNSW-NB15 and ToN-IoT, as exp5_seeds_summary does for CICIDS2017.

Seed 0 is results/exp10_cross_dataset.json; seeds 1.. are
results/exp10_seeds/<dataset>_seed<N>.json, written by
exp10_cross_dataset.py --seed N --out ... (for ToN-IoT the seed also redraws the 70/30 split).

For each dataset and trained forest: probe and replay flip counts per seed for the
float64 traversal and the training framework, the probability that a 2,000-row
label-comparing replay misses, and float32 flips summed over seeds.
"""

import glob
import json
import os

from edgeids.data import RESULTS

MODELS = ["sklearn_rf", "sklearn_hgb", "lightgbm", "xgboost_hist"]
KINDS = {"float64": "numpy_float64_vs_ort", "framework": "native_vs_ort"}
DATASETS = {"UNSW-NB15": "unsw_nb15", "ToN-IoT": "ton_iot"}


def main():
    seed0 = json.load(open(os.path.join(RESULTS, "exp10_cross_dataset.json")))["datasets"]
    out = {"datasets": {}}
    for name, slug in DATASETS.items():
        files = sorted(glob.glob(os.path.join(RESULTS, "exp10_seeds", f"{slug}_seed*.json")))
        runs = [seed0[name]["models"]] + [json.load(open(f))["datasets"][name]["models"] for f in files]
        ds = {"seeds": len(runs), "files": ["exp10_cross_dataset.json"] + [os.path.relpath(f, RESULTS) for f in files],
              "models": {}}
        for m in MODELS:
            entry, f32 = {}, 0
            for kind, key in KINDS.items():
                probes, replay, miss = [], [], []
                for r in runs:
                    p, rp = r[m]["probes"]["total"], r[m]["replay"]
                    probes.append(p[key + "_flips"])
                    replay.append(rp[key + "_flips"])
                    miss.append(rp[key + "_prob_2000_row_replay_misses"])
                entry[kind] = {"probe_flips": probes, "replay_flips": replay, "prob_2000_row_replay_misses": miss}
            for r in runs:
                f32 += r[m]["probes"]["total"]["numpy_float32_vs_ort_flips"] + r[m]["replay"]["numpy_float32_vs_ort_flips"]
            entry["float32_flips_all_seeds"] = f32
            entry["pairs"] = [r[m]["sharing"]["distinct_feature_threshold_pairs"] for r in runs]
            ds["models"][m] = entry
            print(name, m, json.dumps(entry))
        out["datasets"][name] = ds
    path = os.path.join(RESULTS, "exp10_seeds_summary.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
