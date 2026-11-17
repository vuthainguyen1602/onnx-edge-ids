"""Print the probe-completeness table (LaTeX rows) from exp31_probe_completeness.json.

One row per arithmetic fault, over the thirteen models: the nodes at which the
fault can be told apart from ONNX Runtime at all, the share of them the
three-probe set and the crossing pair hit (lowest and highest model), and the
median and largest number of float32 values in a node's disagreement set.

    python research/code/experiments/exp31_table.py
"""

import json
import os

from edgeids.data import RESULTS

FAULTS = {"float64": "float64 traversal", "float64_scaler": "float64 scaler, float32 comparison",
          "scaler_reordered": "scaler as $xs - os$", "raw_thresholds": "thresholds folded to raw space",
          "comparison_flipped": "$<$ and $\\le$ exchanged", "decimal_thresholds": "thresholds via 6 significant digits"}


def pct(v):
    p = round(100 * v)
    return "99" if p == 100 and v < 1 else str(p)


def main():
    models = json.load(open(os.path.join(RESULTS, "exp31_probe_completeness.json")))["models"]
    for key, label in FAULTS.items():
        rows = [m["faults"][key] for m in models.values()]
        live = sum(r["nodes_where_B_differs_from_A"] for r in rows)
        three = [r["three_probe_set_hits"] / r["nodes_where_B_differs_from_A"] for r in rows]
        cross = [r["crossing_pair_hits"] / r["nodes_where_B_differs_from_A"] for r in rows]
        meds = [r["disagreement_size_float32_values"]["median"] for r in rows]
        mx = max(r["disagreement_size_float32_values"]["max"] for r in rows)
        f3 = f"{pct(min(three))}--{pct(max(three))}" if pct(min(three)) != pct(max(three)) else pct(min(three))
        fx = f"{pct(min(cross))}--{pct(max(cross))}" if pct(min(cross)) != pct(max(cross)) else pct(min(cross))
        exp = f"{mx:.1e}".replace("e+0", "e").split("e")
        print(f"    {label} & " + f"{live:,}".replace(",", "{,}") +
              f" & {f3} & {fx} & {min(meds)}--{max(meds)} & ${exp[0]}\\times10^{{{int(exp[1])}}}$ \\\\")


if __name__ == "__main__":
    main()
