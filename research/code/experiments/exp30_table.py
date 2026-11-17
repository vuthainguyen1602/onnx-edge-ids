"""Print the testing-baselines table (LaTeX rows) from exp30_testing_baselines.json.

One row per model and fault group: how many faults each input generator
detects at a 2,000-input budget and at the model's probe budget ("a / b").
Engine mutants count only those that some generator detects (the rest change
no output beyond the tolerance and are treated as equivalent). Single-node faults
are shown twice, comparing probabilities and comparing margins.

    python research/code/experiments/exp30_table.py
"""

import json
import os

from edgeids.data import RESULTS

METHODS = ["replay", "uniform", "boundary", "coverage", "probes", "crossing"]
MODELS = {"deployed_rf": "deployed RF", "xgboost_hist": "XGBoost"}
GROUPS = [("arithmetic", "", "arithmetic"), ("exporter", "", "exporter"), ("engine", "", "engine mutants"),
          ("single_node", "", "single-node, probabilities"), ("single_node", "margin:", "single-node, margins")]


def main():
    rep = json.load(open(os.path.join(RESULTS, "exp30_testing_baselines.json")))["models"]
    for i, (model, label) in enumerate(MODELS.items()):
        e = rep[model]
        small, full = e["budgets"]
        first = True
        for kind, pre, name in GROUPS:
            fs = [v for v in e["faults"].values() if v["kind"] == kind]
            if pre and (model == "deployed_rf" or not any(f"{pre}replay@{small}" in v for v in fs)):
                continue  # the deployed forest's output is not transformed: its margins are its probabilities
            if kind == "engine":
                fs = [v for v in fs if any(v.get(f"{m}@{b}") for m in METHODS for b in (small, full))]
            if kind == "exporter":
                fs = [v for v in fs if any(v.get(f"{p}{m}@{b}") for p in ("", "margin:") for m in METHODS
                                           for b in (small, full))]
            if not fs:
                continue
            cells = [f"{sum(bool(v.get(f'{pre}{m}@{small}')) for v in fs)} / {sum(bool(v.get(f'{pre}{m}@{full}')) for v in fs)}"
                     for m in METHODS]
            print("    " + " & ".join([label if first else "", f"{name} ({len(fs)})"] + cells) + r" \\")
            first = False
        if i == 0:
            print(r"    \midrule")


if __name__ == "__main__":
    main()
