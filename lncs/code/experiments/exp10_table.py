"""Print the cross-dataset parity table (LaTeX rows) from exp10_cross_dataset.json.

Every number comes from the results file; a dataset or model that has not been
run does not appear.

    python lncs/code/experiments/exp10_table.py
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgeids.data import RESULTS

NAMES = {"sklearn_rf": r"\texttt{scikit-learn} RF", "sklearn_hgb": r"\texttt{scikit-learn} HGB",
         "lightgbm": r"\texttt{LightGBM}", "xgboost_hist": r"\texttt{XGBoost-Hist}"}


def pct(n, d):
    return f"${n:,}$ (${100.0 * n / d:.2f}\\%$)".replace(",", "{,}") if d else "--"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(RESULTS, "exp10_cross_dataset.json"))
    args = ap.parse_args()
    with open(args.results) as f:
        report = json.load(f)

    print(r"% Dataset & Model & Features & Internal nodes (probed) & Probes & Float64/ORT & Float32/ORT & Native/ORT & Replay float64/ORT")
    for name, ds in report["datasets"].items():
        for key, m in ds["models"].items():
            tot, rep = m["probes"]["total"], m["replay"]
            nodes = m["sharing"]["internal_nodes"]
            probed = m["probes"]["nodes_probed"]
            nodes_s = f"${nodes:,}$".replace(",", "{,}") + ("" if probed == nodes else f" (${probed:,}$)".replace(",", "{,}"))
            print(" & ".join([
                name, NAMES.get(key, key), str(len(ds["features"])), nodes_s, f"${tot['probes']:,}$".replace(",", "{,}"),
                pct(tot["numpy_float64_vs_ort_flips"], tot["probes"]),
                pct(tot["numpy_float32_vs_ort_flips"], tot["probes"]),
                pct(tot.get("native_vs_ort_flips", 0), tot["probes"]),
                pct(rep["numpy_float64_vs_ort_flips"], rep["rows"]),
            ]) + r" \\")


if __name__ == "__main__":
    main()
