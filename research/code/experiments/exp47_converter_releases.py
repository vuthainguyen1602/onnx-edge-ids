"""Real converter defects in the latest converter releases, and which checks catch them.

exp43 wrote synthetic faults into graphs. These four defects are real: each was reported
upstream, changes the converted model's output, and is fixed on the converter's main
branch but in no release as of November 2026 (onnxmltools 1.16.0 and skl2onnx 1.20.0 are
the latest):

  lgbm_threshold   onnxmltools PR 810 (issue 806): LightGBM float64 thresholds are rounded
                   to the nearest float32, so the input equal to float32(t) > t takes the
                   other branch; a single-switch fault.
  lgbm_zero        onnxmltools PR 814: with zero_as_missing=True LightGBM sends 0 to the
                   split's default child; the converter compares it with the threshold.
                   Not single-switch: the decision changes at 0 and at the threshold.
  xgb_dart         onnxmltools PR 813: the converter ignores DART tree weights; every
                   tree's leaves are off by a factor (a leaf-weight fault).
  hgb_threshold    skl2onnx PR 1227: HistGradientBoosting thresholds rounded to the nearest
                   float32 (single-switch); Leafparity does not support HistGradientBoosting.

Models are trained once on CICIDS2017 without a scaler (as exp43) and pickled; each is
converted by the released converter and by the fixed one (for lgbm_zero also by the build
right after PR 810, to separate the two LightGBM defects). Every graph is checked against
the training model: Leafparity, the data-free crossing pair (probabilities and margins),
and a 2,000-row replay of the test split (probabilities and margins).

Three steps, the convert step once per converter build:

    .lpvenv/bin/python exp47_converter_releases.py train --cicids-train $TRAIN --cicids-test $DATA
    .e3venv/bin/python exp47_converter_releases.py convert --tag released          # onnxmltools 1.16.0, skl2onnx 1.20.0
    .e3venv/bin/pip install --no-deps --force-reinstall git+https://github.com/onnx/onnxmltools@edef44c56a     # right after PR 810
    .e3venv/bin/python exp47_converter_releases.py convert --tag after810 --models lgbm_zero
    .e3venv/bin/pip install --no-deps --force-reinstall git+https://github.com/onnx/onnxmltools@b9f9169bfb git+https://github.com/onnx/sklearn-onnx@360404cc56
    .e3venv/bin/python exp47_converter_releases.py convert --tag fixed
    .lpvenv/bin/python exp47_converter_releases.py evaluate --cicids-test $DATA
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

from edgeids.data import RESULTS

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODEL_DIR = os.path.join(RESULTS, "exp47_models")
TOL = 1e-5
MARGIN_TOL = 1e-4
REPLAY_ROWS = 2000
DEFECTS = {
    "lgbm_threshold": "onnxmltools PR 810 (issue 806): float64 LightGBM thresholds rounded to nearest float32",
    "lgbm_zero": "onnxmltools PR 814: zero_as_missing splits ignore the default child for 0",
    "xgb_dart": "onnxmltools PR 813: DART tree weights ignored",
    "hgb_threshold": "skl2onnx PR 1227: HistGradientBoosting thresholds rounded to nearest float32",
}


def build(name):
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from xgboost import XGBClassifier
    n = 200
    if name == "lgbm_threshold":
        return LGBMClassifier(n_estimators=n, max_bin=255, random_state=0, n_jobs=8, verbose=-1)
    if name == "lgbm_zero":
        return LGBMClassifier(n_estimators=n, max_bin=255, random_state=0, n_jobs=8, verbose=-1, zero_as_missing=True)
    if name == "xgb_dart":
        return XGBClassifier(n_estimators=n, max_depth=8, tree_method="hist", max_bin=256, booster="dart",
                             rate_drop=0.1, skip_drop=0.5, random_state=0, n_jobs=8)
    if name == "hgb_threshold":
        return HistGradientBoostingClassifier(max_iter=n, max_bins=255, random_state=0)
    raise ValueError(name)


def train(args):
    import joblib
    from edgeids.datasets import load_cicids
    x_train, y_train, _ = load_cicids(args.cicids_train, args.cicids_test)
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name in args.models.split(","):
        t0 = time.perf_counter()
        m = build(name).fit(x_train.astype(np.float32), y_train)
        joblib.dump(m, os.path.join(MODEL_DIR, f"{name}.pkl"))
        print("trained", name, round(time.perf_counter() - t0, 1), "s", flush=True)


def convert(args):
    import joblib
    import onnxmltools
    import skl2onnx
    from onnxmltools.convert.common.data_types import FloatTensorType as FT
    from skl2onnx.common.data_types import FloatTensorType
    versions = {"onnxmltools": onnxmltools.__version__, "skl2onnx": skl2onnx.__version__}
    for pkg in ("onnxmltools", "skl2onnx"):
        try:  # a git install records the commit it came from
            from importlib.metadata import distribution
            d = distribution(pkg if pkg == "onnxmltools" else "skl2onnx")
            direct = d.read_text("direct_url.json")
            if direct:
                versions[pkg] += " @" + json.loads(direct).get("vcs_info", {}).get("commit_id", "")[:10]
        except Exception:
            pass
    for name in args.models.split(","):
        m = joblib.load(os.path.join(MODEL_DIR, f"{name}.pkl"))
        nf = m.n_features_in_
        try:
            if name.startswith("lgbm"):
                onx = onnxmltools.convert_lightgbm(m, initial_types=[("input", FT([None, nf]))], target_opset=15, zipmap=False)
            elif name.startswith("xgb"):
                onx = onnxmltools.convert_xgboost(m, initial_types=[("input", FT([None, nf]))], target_opset=15)
            else:
                from edgeids.models import patch_skl2onnx_bools
                patch_skl2onnx_bools()  # a packaging defect of 1.20 unrelated to the outputs (see models.py)
                onx = skl2onnx.convert_sklearn(m, initial_types=[("input", FloatTensorType([None, nf]))],
                                               options={id(m): {"zipmap": False}}, target_opset={"": 17, "ai.onnx.ml": 3})
            with open(os.path.join(MODEL_DIR, f"{name}__{args.tag}.onnx"), "wb") as f:
                f.write(onx.SerializeToString())
            status = "ok"
        except Exception as e:
            status = f"{type(e).__name__}: {e}"
        meta_path = os.path.join(MODEL_DIR, "converters.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        meta.setdefault(args.tag, {"versions": versions, "models": {}})["models"][name] = status
        meta[args.tag]["versions"] = versions
        json.dump(meta, open(meta_path, "w"), indent=1)
        print(args.tag, name, versions, status, flush=True)


def margin_graph(path, out):
    import onnx
    m = onnx.load(path)
    for n in m.graph.node:
        if n.op_type == "TreeEnsembleClassifier":
            for a in n.attribute:
                if a.name == "post_transform":
                    a.s = b"NONE"
    onnx.save_model(m, out)
    return out


def framework_margin(m, x):
    if hasattr(m, "booster_"):                      # LightGBM
        return m.predict(x, raw_score=True).astype(np.float64)
    if hasattr(m, "get_booster"):                   # XGBoost
        return m.predict(x, output_margin=True).astype(np.float64)
    return m.decision_function(x).astype(np.float64)  # HistGradientBoosting


def compare(ref_p, got_p, ref_m, got_m):
    flips = (ref_p[:, 1] > 0.5) != (got_p[:, 1] > 0.5)
    fail = flips | np.any(np.abs(ref_p - got_p) > TOL, axis=1)
    mfail = np.abs(ref_m - got_m) > MARGIN_TOL
    return flips, fail, mfail


def evaluate(args):
    import joblib
    sys.path.insert(0, os.path.join(REPO, "src"))
    from edgeids.data import load_test_split
    from edgeids.runtime import ort_probs
    from onnx_edge_ids.boundary_probe import CROSSING_VARIANTS, build_probes
    x_test, _, _ = load_test_split(args.cicids_test)
    x_test = np.ascontiguousarray(x_test, dtype=np.float32)
    meta = json.load(open(os.path.join(MODEL_DIR, "converters.json")))
    tmp = tempfile.mkdtemp()
    report = {"converters": {t: v["versions"] for t, v in meta.items()}, "defects": DEFECTS,
              "settings": {"tolerance": TOL, "margin_tolerance": MARGIN_TOL, "replay_rows": REPLAY_ROWS,
                           "probe_set": "crossing", "leafparity_input_dtype": "float32"},
              "models": {}}
    try:
        from leafparity import analyze
    except ImportError:
        analyze = None
    for name in args.models.split(","):
        m = joblib.load(os.path.join(MODEL_DIR, f"{name}.pkl"))
        ref_r = m.predict_proba(x_test).astype(np.float64)
        ref_rm = framework_margin(m, x_test)
        entry = {"defect": DEFECTS[name], "graphs": {}}
        for tag in meta:
            path = os.path.join(MODEL_DIR, f"{name}__{tag}.onnx")
            if not os.path.exists(path):
                continue
            row = {}
            got = ort_probs(path, x_test)
            mg = margin_graph(path, os.path.join(tmp, "m.onnx"))
            got_m = ort_probs(mg, x_test)[:, -1]
            fl, fa, mf = compare(ref_r, got, ref_rm, got_m)
            row["replay"] = {"label_flips": int(fl.sum()), "rows_failing": int(fa.sum()), "rows_failing_margin": int(mf.sum()),
                             "max_deviation": float(np.abs(ref_r - got).max()),
                             "prob_passes": float((1 - fa.mean()) ** REPLAY_ROWS),
                             "margin_prob_passes": float((1 - mf.mean()) ** REPLAY_ROWS)}
            t0 = time.perf_counter()
            try:
                pr = build_probes(path, CROSSING_VARIANTS)
            except Exception as e:  # a split mode the gate's probe builder does not handle is recorded, not fatal
                pr = None
                row["probes"] = {"error": f"{type(e).__name__}: {e}", "detected": None, "margin_detected": None,
                                 "failing": None, "failing_margin": None, "probes": 0}
            if pr is not None:
                pp = m.predict_proba(pr.rows).astype(np.float64)
                fl, fa, mf = compare(pp, ort_probs(path, pr.rows), framework_margin(m, pr.rows), ort_probs(mg, pr.rows)[:, -1])
                row["probes"] = {"probes": len(pr), "label_flips": int(fl.sum()), "failing": int(fa.sum()),
                                 "failing_margin": int(mf.sum()), "detected": bool(fa.any()), "margin_detected": bool(mf.any()),
                                 "seconds": round(time.perf_counter() - t0, 2)}
            if analyze is not None and not name.startswith("hgb"):
                # in a child process with a time limit: an analysis that does not finish counts as not detected
                t0 = time.perf_counter()
                try:
                    out = subprocess.run([sys.executable, __file__, "leafparity", "--models", name, "--graph", path],
                                         capture_output=True, text=True, timeout=args.leafparity_timeout)
                    last = [ln for ln in out.stdout.splitlines() if ln.startswith("{")]
                    row["leafparity"] = json.loads(last[-1]) if last else {"error": out.stderr[-500:], "detected": False}
                except subprocess.TimeoutExpired:
                    row["leafparity"] = {"error": f"timeout after {args.leafparity_timeout} s", "detected": False}
                row["leafparity"]["seconds"] = round(time.perf_counter() - t0, 1)
            else:
                row["leafparity"] = {"not_applicable": "HistGradientBoosting is not supported by Leafparity 0.1.2"}
            entry["graphs"][tag] = row
            lp = row["leafparity"]
            print(f"{name:15s} {tag:9s} replay flips={row['replay']['label_flips']:6d} fail={row['replay']['rows_failing']:6d} "
                  f"P(pass)={row['replay']['prob_passes']:.3f}/m{row['replay']['margin_prob_passes']:.3f} | probes "
                  f"fail={row['probes']['failing']}/m{row['probes']['failing_margin']} of {row['probes']['probes']} | "
                  f"LP={lp.get('verdict', lp.get('error', 'n/a'))}", flush=True)
        report["models"][name] = entry
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1, default=str)
    print("wrote", args.out)


def leafparity_one(args):
    """Leafparity on one graph against its training model; prints the verdict as one JSON line."""
    import joblib
    from leafparity import analyze
    m = joblib.load(os.path.join(MODEL_DIR, f"{args.models}.pkl"))
    try:
        r = analyze(m, args.graph, input_dtype="float32", allow_missing=False, worst_case_seconds=30)
        v = r.verdict
        res = {"verdict": v["status"], "detected": v["status"] != "EQUIVALENT",
               "class_change_found": bool(v.get("label_flip_found")), "problem_kinds": v.get("problem_kinds"),
               "max_raw_difference_found": v.get("max_raw_difference_found"),
               "rounding_allowance": v.get("rounding_allowance")}
    except Exception as e:
        res = {"error": f"{type(e).__name__}: {e}", "detected": False}
    print(json.dumps(res, default=str), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["train", "convert", "evaluate", "leafparity"])
    ap.add_argument("--graph")
    ap.add_argument("--leafparity-timeout", type=int, default=600)
    ap.add_argument("--cicids-train")
    ap.add_argument("--cicids-test")
    ap.add_argument("--tag")
    ap.add_argument("--models", default=",".join(DEFECTS))
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp47_converter_releases.json"))
    args = ap.parse_args()
    {"train": train, "convert": convert, "evaluate": evaluate, "leafparity": leafparity_one}[args.step](args)


if __name__ == "__main__":
    main()
