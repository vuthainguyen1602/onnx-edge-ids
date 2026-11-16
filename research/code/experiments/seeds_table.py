"""Print the five-seed appendix table (LaTeX rows) from exp5_seeds_summary.json and exp10_seeds_summary.json.

One row per trained model: the min-max over seeds 0-4 of probe flips, replay flips
and the probability that a 2,000-row label-comparing replay passes, for the float64
traversal and for the training framework, and float32 flips summed over the seeds.
Every number comes from the two summary files.

    python research/code/experiments/seeds_table.py
"""

import json
import os

from edgeids.data import RESULTS

MODELS = {"sklearn_rf": "sklearn RF", "sklearn_hgb": "sklearn HGB", "lightgbm": "LightGBM", "xgboost_hist": "XGBoost"}


def rng(values, fmt):
    lo, hi = min(values), max(values)
    a, b = fmt(lo), fmt(hi)
    return a if a == b else f"{a}--{b}"


def count(n):
    return f"{n:,}".replace(",", "{,}")


def passes(p):
    v = 100.0 * p
    if v >= 99.95:
        return "100"
    if v < 0.05:
        return "0"
    return f"{v:.1f}" if v < 10 else f"{v:.0f}"


def main():
    cic = json.load(open(os.path.join(RESULTS, "exp5_seeds_summary.json")))["models"]
    other = json.load(open(os.path.join(RESULTS, "exp10_seeds_summary.json")))["datasets"]
    blocks = [("CICIDS2017", cic), ("UNSW-NB15", other["UNSW-NB15"]["models"]), ("ToN-IoT", other["ToN-IoT"]["models"])]
    for name, models in blocks:
        for i, (key, label) in enumerate(MODELS.items()):
            e = models[key]
            cells = [name if i == 0 else "", label]
            for kind in ("float64", "framework"):
                k = e[kind]
                cells += [rng(k["probe_flips"], count), rng(k["replay_flips"], count),
                          rng(k["prob_2000_row_replay_misses"], passes)]
            cells.append(count(e["float32_flips_all_seeds"]))
            print("    " + " & ".join(cells) + r" \\")
        if name != "ToN-IoT":
            print(r"    \midrule")


if __name__ == "__main__":
    main()
