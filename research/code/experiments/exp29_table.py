"""Print the fault-localization table (LaTeX rows) from exp29_fault_localization.json.

Columns: what the fault changed; L1 (scope from the failing probes' target
nodes): the scope it reports and whether it is right, and for feature faults
whether the faulty features rank first among its suspects; L2 (first
divergence, artifact vs. graph): the kind it reports, top-1 (its most frequent
first-divergence node, or the base value, is a faulty element) and precision
(share of reported first divergences, weighted by probes, on faulty elements).

    python research/code/experiments/exp29_table.py
"""

import json
import os

from edgeids.data import RESULTS

MODELS = {"deployed_rf": "deployed RF", "xgboost_hist": "XGBoost"}
CHANGED = {"thr_all_up": "every threshold", "thr_one_up": "one threshold", "feature_swap": "two split features",
           "offset_one_up": "one scaler offset", "scale_fp16": "every scaler scale", "leaf_tree0": "tree 0's leaves",
           "mode_dropped": "every split mode", "base_dropped": "the base value"}
YES, NO = r"\checkmark", r"$\times$"


def main():
    rep = json.load(open(os.path.join(RESULTS, "exp29_fault_localization.json")))
    for i, (model, label) in enumerate(MODELS.items()):
        first = True
        for fault, changed in CHANGED.items():
            r = rep["exporter_faults"].get(f"{model}/{fault}")
            if r is None:
                continue
            cells = [label if first else "", changed]
            first = False
            if not r["failing_probes"]:
                cells += [r"\multicolumn{5}{c}{no output changes; only the structure check sees it}"]
            else:
                rank = ("--" if r["l1_truth_features_in_top_k"] is None
                        else (YES if r["l1_truth_features_in_top_k"] else NO))
                kind = ", ".join(r["l2_by_kind"]) or ("base value" if r["l2_base_value_differs"] else "none")
                top1 = r["l2_base_correct"] if r["l2_top_node_correct"] is None else r["l2_top_node_correct"]
                prec = "--" if r["l2_precision"] is None else f"{100 * r['l2_precision']:.0f}\\%"
                cells += [f"{r['l1_scope']} {YES if r['l1_correct'] else NO}", rank, kind, YES if top1 else NO, prec]
            print("    " + " & ".join(cells) + r" \\")
        print(r"    \midrule")
    ar = rep["arithmetic_faults"]
    n = len(ar)
    l1 = sum(r["l1_correct"] for r in ar.values())
    l2 = sum(r["l2_says_artifact_matches_graph"] for r in ar.values())
    print(f"    deployed RF & {n} arithmetic faults & global {l1}/{n} & -- & "
          f"\\multicolumn{{3}}{{l}}{{no divergence (engine arithmetic) {l2}/{n}}} \\\\")


if __name__ == "__main__":
    main()
