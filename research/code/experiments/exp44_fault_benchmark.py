"""A fault benchmark: every fault the experiments inject or meet, with its class, provenance and outcomes.

No new runs: this collects the faults of exp30 (arithmetic, exporter, single-node and
engine-code mutants), exp40 (coupled one-element faults), exp43 (converter faults written
into the graph, including multi-node faults, with Leafparity) and the real defects of
exp39 (Hummingbird), exp41 (ONNX Runtime 1.26/1.27) and exp12 (skl2onnx 1.20), and labels
each with

  * where it is injected: runtime arithmetic, NumPy engine code, exported NumPy artifact,
    ONNX graph (converter), third-party runtime, runtime release;
  * its fault model and what the paper can say about it: ``proved`` (single-switch at the
    node where it acts, so Theorem 1 guarantees the crossing pair sees it at every node it
    reaches), ``structure`` (the gate's structure check compares the artifact with the
    graph), or ``empirical`` (outside both: only measured rates);
  * provenance: synthetic, or real with its upstream reference;
  * the outcome of each check that was run on it.

It writes the manifest and a summary of detection by fault model and check.

    python research/code/experiments/exp44_fault_benchmark.py
"""

import json
import os
from collections import defaultdict

from edgeids.data import RESULTS

# single-switch at the node where they act -> covered by Theorem 1
PROVED = {
    "arithmetic/float64", "arithmetic/float64_scaler", "arithmetic/scaler_reordered", "arithmetic/raw_thresholds",
    "arithmetic/strict_less", "arithmetic/decimal_thresholds",
    "exporter/thr_all_up", "exporter/thr_one_up", "exporter/offset_one_up", "exporter/scale_fp16",
    "exporter/mode_dropped",
    "engine/leq_to_lt", "engine/lt_to_leq", "engine/strict_ignored", "engine/children_swapped",
    "engine/offset_sign", "engine/scaler_reordered", "engine/input_float64",
}
STRUCTURE = {"exporter/feature_swap", "exporter/leaf_tree0", "exporter/base_dropped"}

FAULT_MODEL = {
    "arithmetic": "single-switch, every node",
    "exporter/thr_all_up": "single-switch, every node", "exporter/thr_one_up": "single-switch, one node",
    "exporter/offset_one_up": "single-switch, nodes on one feature", "exporter/scale_fp16": "single-switch, every node",
    "exporter/mode_dropped": "single-switch, every node", "exporter/feature_swap": "feature mapping, many nodes",
    "exporter/leaf_tree0": "leaf weights, one tree", "exporter/base_dropped": "base value",
    "single_node": "single-switch, one node",
    "engine/leq_to_lt": "single-switch, every node", "engine/lt_to_leq": "single-switch, every node",
    "engine/strict_ignored": "single-switch, every node", "engine/children_swapped": "single-switch, every node",
    "engine/offset_sign": "single-switch, every node", "engine/scaler_reordered": "single-switch, every node",
    "engine/input_float64": "single-switch, every node",
    "engine/scores_float64": "score accumulation", "engine/first_tree_skipped": "tree dropped",
    "engine/last_block_short": "tree dropped", "engine/base_dropped": "base value",
    "engine/tie_to_class1": "label rule", "engine/logistic_sign": "post-transform",
    "engine/argmax_to_argmin": "label rule", "engine/block_size_halved": "batching",
}

# real defects met while building the paper
REAL = {
    "ort_1.26_1.27": "ONNX Runtime 1.26.0/1.27.0 label single-column binary graphs as attack when the score is positive "
                     "(onnxruntime issue 28557; PRs 27552, 28685; fixed in 1.28.0)",
    "skl2onnx_1.20_hgb": "skl2onnx 1.20 rounds HistGradientBoosting thresholds to the nearest float32 "
                         "(fixed by sklearn-onnx PR 1227)",
    "hummingbird_base": "Hummingbird 0.4.12 drops the graph's base_values (observed in exp39)",
    "own_float64": "the project's own early exporter traversed in float64 (mirrored by arithmetic/float64)",
}


def load(name):
    p = os.path.join(RESULTS, name)
    return json.load(open(p)) if os.path.exists(p) else None


def fault_model(key):
    if key in FAULT_MODEL:
        return FAULT_MODEL[key]
    return FAULT_MODEL.get(key.split("/")[0], key.split("/")[0])


def coverage(key):
    if key in PROVED or key.startswith("single_node/"):
        return "proved"
    if key in STRUCTURE:
        return "structure"
    return "empirical"


def main():
    entries = []

    exp30 = load("exp30_testing_baselines.json")
    for model, v in exp30["models"].items():
        full = max(v["budgets"])
        for key, f in v["faults"].items():
            if key.startswith("control"):
                continue
            checks = {"replay_2000": f.get("replay@2000"), "crossing": f.get(f"crossing@{full}"),
                      # margins are the check for a logistic output; on the forest they are its raw votes,
                      # compared at a looser tolerance, so they are not counted there
                      "crossing_margin": f.get(f"margin:crossing@{full}") if model == "xgboost_hist" else None}
            changes = any(val for k, val in f.items() if "@" in k)
            entries.append({
                "id": f"exp30/{model}/{key}", "model": model,
                "injected_into": {"arithmetic": "runtime arithmetic", "exporter": "NumPy artifact",
                                  "single_node": "NumPy artifact", "engine": "NumPy engine code"}[key.split("/")[0]],
                "fault_model": fault_model(key), "theorem": coverage(key),
                "provenance": "synthetic (mirrors a real defect)" if key == "arithmetic/float64" else "synthetic",
                "changes_output": changes, "checks": checks})

    exp40 = load("exp40_fault_class_ablation.json")
    for model, v in exp40["models"].items():
        full = max(v["budgets"])
        for key, f in v["faults"].items():
            entries.append({
                "id": f"exp40/{model}/{key}", "model": model, "injected_into": "NumPy artifact",
                "fault_model": "leaf weight, one leaf" if "leaf" in key else "feature mapping, one node",
                "theorem": "empirical", "provenance": "synthetic", "changes_output": True,
                "checks": {"structure": f.get("structure"), "replay_2000": f.get("replay@2000"),
                           "crossing": f.get(f"crossing@{full}"),
                           "crossing_margin": f.get(f"margin:crossing@{full}") if model == "xgboost_hist" else None}})

    exp43 = load("exp43_converter_faults_leafparity.json")
    if exp43:
        for model, v in exp43["models"].items():
            for key, f in v["faults"].items():
                if key.startswith("control"):
                    continue
                cls = f["class"]
                proved = cls.startswith("single-switch") or cls == "multi-node: threshold"
                lp = f["leafparity"]
                entries.append({
                    "id": f"exp43/{model}/{key}", "model": f"{model} (no scaler)", "injected_into": "ONNX graph (converter)",
                    "fault_model": cls, "theorem": "proved" if proved else "empirical", "provenance": "synthetic",
                    "changes_output": True,
                    "checks": {"replay_2000": f["replay"]["prob_passes"] < 0.5, "crossing": f["probes"]["detected"],
                               "crossing_margin": f["probes"].get("margin_detected"),
                               # a run that ends in an error checked nothing: counted as not detected
                               "leafparity": bool(lp.get("detected")), "leafparity_error": lp.get("error")}})

    for ver in ("1.26.0", "1.27.0"):
        d = load(f"exp41_ort_release_gate_{ver}.json")
        for ds, models in d["datasets"].items():
            for model, f in models.items():
                if not isinstance(f, dict) or "crossing" not in f:
                    continue
                if not (f["crossing"]["gate_fails"] or f["replay"]["pass_conf"] < 1):
                    continue  # the regression touches single-column binary graphs only
                entries.append({
                    "id": f"exp41/{ver}/{ds}/{model}", "model": f"{ds} {model}", "injected_into": "runtime release",
                    "fault_model": "label rule / probability row", "theorem": "empirical",
                    "provenance": "real: " + REAL["ort_1.26_1.27"], "changes_output": True,
                    "checks": {"replay_2000": f["replay"]["pass_conf"] < 0.5, "crossing": f["crossing"]["gate_fails"]}})

    d = load("exp39_hummingbird.json")
    for ds, models in d["datasets"].items():
        for model, f in models.items():
            bad = {var: r for var, r in f.get("variants", {}).items()
                   if isinstance(r, dict) and "crossing" in r and r["crossing"]["gate_fails"]
                   and "float64" not in var}  # the float64 variants are a deliberate control, not a defect
            if not bad:
                continue  # one entry per model: the defect is the same in every Hummingbird implementation
            entries.append({
                "id": f"exp39/{ds}/{model}", "model": f"{ds} {model}", "injected_into": "third-party runtime",
                "fault_model": "base value", "theorem": "empirical", "provenance": "real: " + REAL["hummingbird_base"],
                "changes_output": True, "variants_failing": sorted(bad),
                "checks": {"replay_2000": all(r["replay"]["pass_conf"] < 0.5 for r in bad.values()),
                           "crossing": all(r["crossing"]["gate_fails"] for r in bad.values())}})

    entries.append({"id": "exp12/skl2onnx_1.20/CICIDS2017/sklearn_hgb", "model": "CICIDS2017 sklearn_hgb",
                    "injected_into": "converter release", "fault_model": "single-switch, every node (rounded thresholds)",
                    "theorem": "proved", "provenance": "real: " + REAL["skl2onnx_1.20_hgb"], "changes_output": True,
                    "checks": {}, "note": "measured as converter divergence in exp12/exp12b/exp19"})

    summary = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for e in entries:
        if not e["changes_output"]:
            continue
        group = (e["theorem"], e["fault_model"].split(",")[0].split(":")[0])
        for check, val in e["checks"].items():
            if val is None or check.endswith("_error"):
                continue
            s = summary[" | ".join(group)][check]
            s[0] += bool(val); s[1] += 1
    def category(e):
        fm = e["fault_model"]
        if fm.startswith("multi-node"):
            return "multi-node, single-switch only" if e["theorem"] == "proved" else "multi-node, mixed kinds"
        if fm.startswith("single-switch"):
            return "single-switch, one node" if "one node" in fm else "single-switch, many nodes"
        if fm.startswith("feature mapping"):
            return "feature mapping"
        if fm.startswith("leaf weight"):
            return "leaf weights"
        if fm == "base value":
            return "base value"
        if fm.startswith("label rule / probability"):
            return "runtime release: decision rule"
        return "aggregation, label rule, post-transform"
    table = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for e in entries:
        if not e["changes_output"]:
            continue
        row = table[category(e)]
        row["faults"][0] += 1
        for check, val in e["checks"].items():
            if val is None or check.endswith("_error"):
                continue
            row[check][0] += bool(val); row[check][1] += 1
    by_category = {c: {k: (v[0] if k == "faults" else f"{v[0]}/{v[1]}") for k, v in r.items()} for c, r in table.items()}

    out = {"entries": entries, "by_fault_model": by_category,
           "counts": {"faults": len(entries), "changing_output": sum(e["changes_output"] for e in entries),
                      "real": sum(e["provenance"].startswith("real") for e in entries),
                      "by_theorem": {t: sum(e["theorem"] == t for e in entries) for t in ("proved", "structure", "empirical")}},
           "summary_detected_over_total": {g: {c: f"{a}/{b}" for c, (a, b) in v.items()} for g, v in sorted(summary.items())},
           "real_defects": REAL}
    path = os.path.join(RESULTS, "exp44_fault_benchmark.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["counts"], indent=1))
    for g, v in out["summary_detected_over_total"].items():
        print(f"{g:55s} " + "  ".join(f"{c}={x}" for c, x in v.items()))
    print()
    for c, r in by_category.items():
        print(f"{c:42s} " + "  ".join(f"{k}={v}" for k, v in r.items()))
    print("wrote", path)


if __name__ == "__main__":
    main()
