"""Compare exp27 runs pairwise: label flips, probability differences, ties.

A run is a tag: m3, x86, jetson (ONNX Runtime) or m3_numpy, x86_numpy,
jetson_numpy (the shipped NumPy engine). For every probe set of a pair
REF:OTHER: rows whose label differs, the
largest probability difference, and how many probes are ties in the reference
(P(class 1) within 1e-6 of one half), where a different summation order can
flip a label without any routing difference.

    python exp27_compare.py --dir DIR --pair m3:x86 --pair m3:jetson --pair m3_numpy:x86_numpy \
        --pair m3_numpy:jetson_numpy --pair m3:m3_numpy --pair jetson:jetson_numpy
"""

import argparse
import glob
import json
import os

import numpy as np

from edgeids.data import RESULTS

TIE = 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--pair", action="append", required=True, help="REF:OTHER")
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp27_cross_arch_ties.json"))
    args = ap.parse_args()
    root = os.path.join(args.dir, "outputs")
    pairs = [p.split(":") for p in args.pair]
    tags = sorted({t for p in pairs for t in p})
    env = {t: json.load(open(os.path.join(root, t, "env.json"))) for t in tags}
    report = {"environments": env, "tie_band": TIE, "comparisons": {}}
    for ref, other in pairs:
        sets, totals = {}, {"probes": 0, "label_flips": 0, "ties_in_reference": 0, "flips_on_ties": 0, "max_prob_diff": 0.0}
        for ref_file in sorted(glob.glob(os.path.join(root, ref, "*.npz"))):
            stem = os.path.basename(ref_file)[:-4]
            a, b = np.load(ref_file), np.load(os.path.join(root, other, stem + ".npz"))
            flips = a["labels"] != b["labels"]
            tie = np.abs(a["probs"][:, -1].astype(np.float64) - 0.5) <= TIE
            row = {"probes": int(len(flips)), "label_flips": int(flips.sum()),
                   "ties_in_reference": int(tie.sum()), "flips_on_ties": int((flips & tie).sum()),
                   "max_prob_diff": float(np.abs(a["probs"].astype(np.float64) - b["probs"]).max())}
            sets[stem] = row
            for k in ("probes", "label_flips", "ties_in_reference", "flips_on_ties"):
                totals[k] += row[k]
            totals["max_prob_diff"] = max(totals["max_prob_diff"], row["max_prob_diff"])
        report["comparisons"][f"{ref}:{other}"] = {"totals": totals, "sets": sets}
        print(f"{ref}:{other}", totals)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
