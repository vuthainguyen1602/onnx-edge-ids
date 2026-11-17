"""Boundary probes against standard testing baselines, on the same faults and input budget.

Faults (each a faulty implementation compared with its reference):
  * the six arithmetic faults of exp11 in the float32 traversal of the deployed
    forest (reference: the deployed float32 traversal, equal to ONNX Runtime);
  * the exporter faults of exp25 that change some output, served by the shipped
    NumPy engine (reference: ONNX Runtime), on the deployed forest and on the
    CICIDS2017 XGBoost model;
  * the float64 traversal of the XGBoost graph (reference: ONNX Runtime);
  * mutants of the shipped NumPy engine's own code (mutation testing): one
    textual mutation of numpy_engine.py each, serving the correct artifact;
  * single-node faults: the threshold of one random node moved just far enough
    that the first raw float32 value past ONNX Runtime's crossing changes branch
    (visible by construction, but confined to one node), twenty per model.

Input generators, each given the same budget N:
  replay      rows drawn from the test split;
  uniform     each feature uniform between its test-split minimum and maximum;
  boundary    a random test row with one feature set onto a random split
              threshold or one float32 step either side, not routed to the node;
  coverage    coverage-guided fuzzing: mutate rows of a corpus (a feature onto a
              threshold, or a multiplicative perturbation) and keep a mutant when
              it reaches a leaf no earlier input reached;
  probes      the gate's data-free three-probe set (a random subset when N is smaller);
  crossing    the gate's data-free crossing pair (two probes per node, so at the
              larger budget it uses two thirds of it).

A fault is detected when some input changes the label (as each implementation
returns it) or moves a probability by more than 1e-5. For faults in the NumPy
artifact (exporter and single-node faults, and a correct artifact as a control)
detection is also measured on margins: a copy of the graph and of the artifact
with the post-transform set to NONE, so a logistic output that saturates near 0
or 1 cannot hide a changed tree (tolerance 1e-4: margins of correct runtimes differ
by up to ~1e-5 from float32 summation). Budgets: 2,000 (a replay gate's size) and the model's probe count.

    python exp30_testing_baselines.py --data $DATA
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

from edgeids.data import ARTIFACTS, RESULTS, load_test_split
from edgeids.faults import exporter_mutants, fault_scorers
from edgeids.forest import load_forest
from edgeids.onnx_forest import load_onnx_forest
from edgeids.runtime import numpy_probs

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
from onnx_edge_ids import create_inference_engine  # noqa: E402
from onnx_edge_ids.boundary_probe import CROSSING_VARIANTS, build_probes, crossing_values, load_forest_arrays  # noqa: E402

TOL = 1e-5
MARGIN_TOL = 1e-4   # margins sum ~200 leaves of magnitude ~10: float32 noise reaches ~1e-5 on correct runtimes
SEED = 0
MODELS = {"deployed_rf": os.path.join(ARTIFACTS, "ids_rf.onnx"),
          "xgboost_hist": os.path.join(RESULTS, "exp5_models", "xgboost_hist.onnx")}

# One textual mutation of the shipped NumPy engine each: (name, old, new).
ENGINE_MUTATIONS = [
    ("leq_to_lt", "go_left = value <= thr\n", "go_left = value < thr\n"),
    ("lt_to_leq", "np.where(self.strict[cur], value < thr, value <= thr)", "np.where(self.strict[cur], value <= thr, value <= thr)"),
    ("strict_ignored", "if self.any_strict:", "if False:"),
    ("children_swapped", "nxt = np.where(go_left, self.left[cur], self.right[cur])", "nxt = np.where(go_left, self.right[cur], self.left[cur])"),
    ("offset_sign", "scaled = (x - self.offset) * self.scale", "scaled = (x + self.offset) * self.scale"),
    ("scaler_reordered", "scaled = (x - self.offset) * self.scale", "scaled = x * self.scale - self.offset * self.scale"),
    ("input_float64", "x = np.ascontiguousarray(matrix, dtype=self.inference_dtype)", "x = np.ascontiguousarray(matrix, dtype=np.float64)"),
    ("scores_float64", "scores = np.zeros((n_rows, self.n_targets), dtype=self.inference_dtype)", "scores = np.zeros((n_rows, self.n_targets), dtype=np.float64)"),
    ("first_tree_skipped", "for start in range(0, n_trees, block):", "for start in range(1, n_trees, block):"),
    ("last_block_short", "roots = self.tree_offsets[start:min(start + block, n_trees)]", "roots = self.tree_offsets[start:min(start + block, n_trees - 1)]"),
    ("base_dropped", "return self._outputs(scores + self.base_values)", "return self._outputs(scores)"),
    ("tie_to_class1", "return (p1 > half).astype(np.int64), probs", "return (p1 >= half).astype(np.int64), probs"),
    ("logistic_sign", "p1 = one / (one + np.exp(-s)) if self.post_transform == \"LOGISTIC\" else s", "p1 = one / (one + np.exp(s)) if self.post_transform == \"LOGISTIC\" else s"),
    ("argmax_to_argmin", "return np.argmax(probs, axis=1).astype(np.int64), probs", "return np.argmin(probs, axis=1).astype(np.int64), probs"),
    ("block_size_halved", "block = max(1, self.PAIRS_PER_BLOCK // max(n_rows, 1))", "block = max(1, self.PAIRS_PER_BLOCK // max(n_rows, 1) // 2)"),
]


def engine_mutant_class(name, old, new):
    src = open(os.path.join(REPO, "src", "onnx_edge_ids", "numpy_engine.py")).read()
    if src.count(old) != 1:
        return None
    src = src.replace(old, new).replace("from .inference_engine import", "from onnx_edge_ids.inference_engine import")
    path = os.path.join(tempfile.mkdtemp(), f"engine_{name}.py")
    with open(path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location(f"engine_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.NumpyInferenceEngine


def probs_of(engine, x, batch=2000):
    """(labels, probability rows) from a shipped-style engine, in batches of 2,000 (the gate's probe batch)."""
    labels, out = [], []
    for i in range(0, len(x), batch):
        lab, p = engine._infer(np.ascontiguousarray(x[i:i + batch], dtype=np.float32))
        labels.append(np.asarray(lab)); out.append(np.asarray(p, dtype=np.float64))
    return np.concatenate(labels), np.concatenate(out)


def as_pair(v):
    """(labels, probabilities); a bare probability array is labelled by its argmax."""
    return v if isinstance(v, tuple) else (np.argmax(v, axis=1), v)


def detected(ref, got):
    (lr, pr), (lg, pg) = as_pair(ref), as_pair(got)
    if pg.shape != pr.shape or lg.shape != lr.shape:
        return True
    return bool(np.any((lr != lg) | (np.abs(pr - pg).max(axis=1) > TOL)))


# ----------------------------------------------------------------------------- generators

def gen_replay(rng, n, ctx):
    return ctx["x"][rng.integers(0, len(ctx["x"]), n)]


def gen_uniform(rng, n, ctx):
    lo, hi = ctx["lo"], ctx["hi"]
    return (lo + rng.random((n, len(lo))) * (hi - lo)).astype(np.float32)


def boundary_values(ctx, rng, n):
    f = ctx["forest"]
    nodes = ctx["internal"][rng.integers(0, len(ctx["internal"]), n)]
    g = f.feature[nodes]
    raw = (f.threshold[nodes] / f.scale[g] + f.offset[g]).astype(np.float32)
    step = rng.integers(-1, 2, n)
    raw = np.where(step < 0, np.nextafter(raw, np.float32(-np.inf)), np.where(step > 0, np.nextafter(raw, np.float32(np.inf)), raw))
    return g, raw


def gen_boundary(rng, n, ctx):
    rows = gen_replay(rng, n, ctx).copy()
    g, raw = boundary_values(ctx, rng, n)
    rows[np.arange(n), g] = raw
    return rows


def leaves(ctx, x):
    f = ctx["forest"]
    z = ((x - f.offset.astype(np.float32)) * f.scale.astype(np.float32)).astype(np.float32)
    thr = f.threshold.astype(np.float32)
    out = np.empty((len(x), len(f.tree_offsets) - 1), dtype=np.int64)
    idx = np.arange(len(x))
    for t in range(len(f.tree_offsets) - 1):
        cur = np.full(len(x), f.tree_offsets[t], dtype=np.int64)
        active = ~f.is_leaf[cur]
        while active.any():
            a = np.flatnonzero(active)
            c = cur[a]
            v = z[idx[a], f.feature[c]]
            go = np.where(f.strict[c], v < thr[c], v <= thr[c])
            cur[a] = np.where(go, f.left[c], f.right[c])
            active[a] = ~f.is_leaf[cur[a]]
        out[:, t] = cur
    return out


def gen_coverage(rng, n, ctx, batch=1000):
    corpus = list(gen_replay(rng, 100, ctx))
    seen = set(np.unique(leaves(ctx, np.asarray(corpus))).tolist())
    produced = []
    while sum(len(b) for b in produced) < n:
        m = min(batch, n - sum(len(b) for b in produced))
        parents = np.asarray(corpus)[rng.integers(0, len(corpus), m)].copy()
        on_threshold = rng.random(m) < 0.5
        g, raw = boundary_values(ctx, rng, m)
        feat = np.where(on_threshold, g, rng.integers(0, parents.shape[1], m))
        noise = (parents[np.arange(m), feat] * rng.normal(1.0, 0.1, m)).astype(np.float32)
        parents[np.arange(m), feat] = np.where(on_threshold, raw, noise)
        lv = leaves(ctx, parents)
        for i in range(m):
            new = set(lv[i].tolist()) - seen
            if new:
                seen |= new
                corpus.append(parents[i])
        produced.append(parents)
    ctx["coverage_leaves"] = len(seen)
    return np.concatenate(produced)[:n]


def gen_probes(rng, n, ctx):
    p = ctx["probes"]
    return p if n >= len(p) else p[np.sort(rng.choice(len(p), n, replace=False))]


def gen_crossing(rng, n, ctx):
    p = ctx["crossing"]
    return p if n >= len(p) else p[np.sort(rng.choice(len(p), n, replace=False))]


GENERATORS = {"replay": gen_replay, "uniform": gen_uniform, "boundary": gen_boundary,
              "coverage": gen_coverage, "probes": gen_probes, "crossing": gen_crossing}


def single_node_mutants(base, onnx_path, count=20):
    """name -> artifact arrays with one node's threshold moved so that the raw value just past
    ONNX Runtime's float32 crossing now takes the true branch (visible at that node by construction)."""
    f = load_forest_arrays(onnx_path, np.float64)
    internal = np.flatnonzero(~f.is_leaf)
    internal = internal[f.scale[f.feature[internal]] > 0]
    nodes = np.sort(np.random.default_rng(SEED).choice(internal, count, replace=False))
    g = f.feature[nodes]
    start = (f.threshold[nodes] / f.scale[g] + f.offset[g]).astype(np.float32)
    ok, last_true = crossing_values(f, nodes, start)
    first_false = np.nextafter(last_true, np.float32(np.inf))
    z = ((first_false - f.offset[g].astype(np.float32)) * f.scale[g].astype(np.float32)).astype(np.float32)
    out = {}
    for v, zv, good in zip(nodes, z, ok):
        if not good:
            continue
        m = dict(base)
        t = base["threshold"].copy()
        t[v] = float(np.nextafter(zv, np.float32(np.inf))) if base["strict"][v] else float(zv)
        m["threshold"] = t
        out[f"node_{int(v)}"] = m
    return out


# ----------------------------------------------------------------------------- faults

def none_graph(onnx_path, out_path):
    """A copy of the graph whose tree ensemble returns raw scores (post_transform NONE)."""
    import onnx
    m = onnx.load(onnx_path)
    for node in m.graph.node:
        if node.op_type == "TreeEnsembleClassifier":
            for a in node.attribute:
                if a.name == "post_transform":
                    a.s = b"NONE"
    onnx.save(m, out_path)
    return out_path


def none_artifact(npz_path, out_path):
    d = dict(np.load(npz_path, allow_pickle=False))
    meta = json.loads(str(d["metadata_json"]))
    meta["post_transform"] = "NONE"
    d["metadata_json"] = json.dumps(meta)
    np.savez(out_path, **d)
    return out_path


def faults_for(model, onnx_path, tmp):
    """name -> (kind, reference_fn, faulty_fn, margin_reference_fn, margin_faulty_fn); each fn maps float32
    rows to probability rows or (labels, probability rows); the margin pair is None where not applicable."""
    npz = os.path.join(tmp, f"{model}.npz")
    subprocess.run([sys.executable, os.path.join(REPO, "scripts", "export_numpy.py"), "--model", onnx_path,
                    "--out", npz], check=True, capture_output=True)
    ort_eng = create_inference_engine("onnx", onnx_path)

    def ort_ref(x, batch=50000):
        labels, probs = [], []
        for i in range(0, len(x), batch):
            r = ort_eng._infer(np.ascontiguousarray(x[i:i + batch], dtype=np.float32))
            labels.append(r[0]); probs.append(np.asarray(r[1], dtype=np.float64))
        return np.concatenate(labels), np.concatenate(probs)

    ort_none = create_inference_engine("onnx", none_graph(onnx_path, os.path.join(tmp, f"{model}_none.onnx")))

    def margin_ref(x, batch=50000):
        return np.concatenate([np.asarray(ort_none._infer(np.ascontiguousarray(x[i:i + batch], dtype=np.float32))[1],
                                          dtype=np.float64) for i in range(0, len(x), batch)])

    def margin_of(path):
        eng = create_inference_engine("numpy", none_artifact(path, path[:-4] + "_none.npz"))
        return lambda x: probs_of(eng, x)[1]
    out = {}
    if model == "deployed_rf":
        f32 = load_forest(os.path.join(ARTIFACTS, "ids_rf_numpy.npz"), "float32")
        f64 = load_forest(os.path.join(ARTIFACTS, "ids_rf_numpy.npz"), "float64")
        sc = fault_scorers(f32, f64)
        ref = sc["float32 (deployed)"]
        for name, fn in sc.items():
            if name != "float32 (deployed)":
                out[f"arithmetic/{name}"] = ("arithmetic", ref, fn, None, None)
    else:
        forest = load_onnx_forest(onnx_path)
        out["arithmetic/float64"] = ("arithmetic", lambda x: ort_ref(x)[1], lambda x: numpy_probs(forest, x, np.float64),
                                     None, None)
    base = dict(np.load(npz, allow_pickle=False))
    good = create_inference_engine("numpy", npz)
    out["control/correct_artifact"] = ("control", ort_ref, lambda x, e=good: probs_of(e, x), margin_ref, margin_of(npz))
    for name, arrays in exporter_mutants(base).items():
        path = os.path.join(tmp, f"{model}_{name}.npz")
        np.savez(path, **arrays)
        eng = create_inference_engine("numpy", path)
        out[f"exporter/{name}"] = ("exporter", ort_ref, lambda x, e=eng: probs_of(e, x), margin_ref, margin_of(path))
    for name, arrays in single_node_mutants(base, onnx_path).items():
        path = os.path.join(tmp, f"{model}_{name}.npz")
        np.savez(path, **arrays)
        eng = create_inference_engine("numpy", path)
        out[f"single_node/{name}"] = ("single_node", ort_ref, lambda x, e=eng: probs_of(e, x), margin_ref,
                                      margin_of(path))
    for name, old, new in ENGINE_MUTATIONS:
        cls = engine_mutant_class(name, old, new)
        if cls is None:
            continue
        eng = cls(npz)
        out[f"engine/{name}"] = ("engine", ort_ref, lambda x, e=eng: probs_of(e, x), None, None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CICIDS2017 test_data.parquet directory")
    args = ap.parse_args()
    x, _, _ = load_test_split(args.data)
    x = np.ascontiguousarray(x, dtype=np.float32)
    tmp = tempfile.mkdtemp()
    report = {"tolerance": TOL, "margin_tolerance": MARGIN_TOL, "seed": SEED, "models": {}}
    for model, onnx_path in MODELS.items():
        forest = load_forest_arrays(onnx_path, np.float64)
        probes = build_probes(onnx_path).rows
        ctx = {"x": x, "lo": x.min(axis=0), "hi": x.max(axis=0), "forest": forest,
               "internal": np.flatnonzero(~forest.is_leaf), "probes": probes,
               "crossing": build_probes(onnx_path, CROSSING_VARIANTS).rows}
        faults = faults_for(model, onnx_path, tmp)
        budgets = [2000, len(probes)]
        entry = {"probes": len(probes), "budgets": budgets, "faults": {}, "generation_seconds": {}, "inputs": {}}
        inputs = {}
        for b in budgets:
            for method, gen in GENERATORS.items():
                rng = np.random.default_rng(SEED)
                t0 = time.perf_counter()
                inputs[(method, b)] = gen(rng, b, ctx)
                entry["generation_seconds"][f"{method}@{b}"] = round(time.perf_counter() - t0, 2)
                if method == "coverage":
                    entry["inputs"][f"coverage_leaves@{b}"] = ctx["coverage_leaves"]
        refs = {}
        for fname, (kind, ref_fn, bad_fn, mref_fn, mbad_fn) in faults.items():
            row = {"kind": kind}
            for (method, b), xin in inputs.items():
                key = (id(ref_fn), method, b)
                if key not in refs:
                    refs[key] = ref_fn(xin)
                try:
                    row[f"{method}@{b}"] = detected(refs[key], bad_fn(xin))
                except Exception as e:  # a crashing mutant is detected by any input
                    row[f"{method}@{b}"] = True
                    row["crashes"] = type(e).__name__
                if mref_fn is not None:
                    mkey = (id(mref_fn), method, b)
                    if mkey not in refs:
                        refs[mkey] = mref_fn(xin)
                    # the last column is the margin s in both (ONNX Runtime returns [-s, s] for a single
                    # raw score, the NumPy engine [1 - s, s]); a two-column graph's last column is its class-1 score
                    row[f"margin:{method}@{b}"] = bool(np.any(np.abs(refs[mkey][:, -1] - mbad_fn(xin)[:, -1]) > MARGIN_TOL))
            entry["faults"][fname] = row
            print(model, fname, json.dumps(row), flush=True)
        report["models"][model] = entry
    out = os.path.join(RESULTS, "exp30_testing_baselines.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
