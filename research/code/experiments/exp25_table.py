"""Print the exporter-fault table (LaTeX rows) from exp25_exporter_mutations.json.

    python research/code/experiments/exp25_table.py
"""

import json
import os

from edgeids.data import RESULTS

MODELS = {"deployed_rf": "deployed RF", "xgboost_hist": "XGBoost"}
FAULTS = {"thr_all_up": "every threshold $+1$ step", "thr_one_up": "one threshold $+1$ step",
          "feature_swap": "two split features swapped", "offset_one_up": "one scaler offset $+1$ step",
          "scale_fp16": "scaler scale via float16", "leaf_tree0": "tree 0 leaf weights wrong",
          "mode_dropped": "\\texttt{BRANCH\\_LT} read as \\texttt{LEQ}", "base_dropped": "base value dropped"}


def pct(p):
    v = 100.0 * p
    return "100" if v >= 99.95 else ("0" if v < 0.05 else (f"{v:.1f}" if v < 10 else f"{v:.0f}"))


def probes(r):
    n = lambda v: f"{v:,}".replace(",", "{,}")
    return "pass" if r["tripping_conf"] == 0 else f"{n(r['label_flips'])}\\,/\\,{n(r['tripping_conf'])}"


def main():
    rep = json.load(open(os.path.join(RESULTS, "exp25_exporter_mutations.json")))["models"]
    for i, (key, label) in enumerate(MODELS.items()):
        first = True
        for fault, name in FAULTS.items():
            if fault not in rep[key]:
                continue
            r = rep[key][fault]
            if fault == "thr_one_up" and not r.get("shift_visible_to_some_float32_input", True):
                name += "$^{*}$"
            cells = [label if first else "", name, "fail" if r["structure_mismatches"] else "pass",
                     pct(r["replay"]["prob_passes_labels_only"]), pct(r["replay"]["prob_passes_with_conf"]),
                     probes(r["probes_from_npz"]), probes(r["probes_from_onnx"]), probes(r["crossing_probes_from_onnx"])]
            print("    " + " & ".join(cells) + r" \\")
            first = False
        if i == 0:
            print(r"    \midrule")


if __name__ == "__main__":
    main()
