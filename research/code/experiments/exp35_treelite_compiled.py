"""Treelite compiled to C (TL2cgen) as a further runtime of the same ONNX graphs.

exp22 ran Treelite's interpreter (GTIL). A compiled runtime is a different code
path: TL2cgen emits C for the forest, which the compiler may optimise and which
sums the trees in its own order. This builds the float32 Treelite model of each
of the thirteen graphs (edgeids.treelite_forest, as exp22), compiles it with
the system C compiler (TL2cgen 1.0, clang -O2 by default), applies the graph's
Scaler in float32, and compares the compiled runtime with ONNX Runtime on the
gate's data-free probes, both the crossing pair (the gate's default) and the
three-probe set: label differences, probabilities over 1e-5, the largest
probability difference, and label differences at a tie (both within 1e-6 of
one half). It also compares compiled and interpreted Treelite.

    .tlvenv/bin/python exp35_treelite_compiled.py
"""

import json
import os
import platform
import sys
import tempfile

import numpy as np

from edgeids.data import ARTIFACTS, RESULTS
from edgeids.models import MODEL_KEYS
from edgeids.onnx_forest import load_onnx_forest
from edgeids.runtime import ort_probs, probabilities
from edgeids.treelite_forest import to_treelite, treelite_scores

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
from onnx_edge_ids.boundary_probe import CROSSING_VARIANTS, VARIANTS, build_probes  # noqa: E402

TOL = 1e-5
TIE = 1e-6


def compiled_scores(predictor, forest, x, batch=50000):
    import tl2cgen
    out = []
    for i in range(0, len(x), batch):
        z = ((x[i:i + batch].astype(np.float32) - forest.offset.astype(np.float32))
             * forest.scale.astype(np.float32)).astype(np.float32)
        s = np.asarray(predictor.predict(tl2cgen.DMatrix(z, dtype="float32"), pred_margin=True), dtype=np.float64)
        out.append(s.reshape(len(z), -1))
    return np.concatenate(out)


def compare(ref, got):
    flips = np.argmax(ref, 1) != np.argmax(got, 1)
    tie = (np.abs(ref[:, -1] - 0.5) <= TIE) & (np.abs(got[:, -1] - 0.5) <= TIE)
    d = np.abs(ref - got).max(axis=1)
    return {"label_differences": int(flips.sum()), "at_a_tie": int((flips & tie).sum()),
            "over_tolerance": int((d > TOL).sum()), "max_probability_difference": float(d.max())}


def main():
    import tl2cgen
    import treelite
    jobs = [("CICIDS2017", "deployed_rf", os.path.join(ARTIFACTS, "ids_rf.onnx"))]
    for ds, d in (("CICIDS2017", "exp5_models"), ("UNSW-NB15", os.path.join("exp10_models", "unsw_nb15")),
                  ("ToN-IoT", os.path.join("exp10_models", "ton_iot"))):
        jobs += [(ds, k, os.path.join(RESULTS, d, f"{k}.onnx")) for k in MODEL_KEYS]
    tmp = tempfile.mkdtemp()
    report = {"versions": {"treelite": treelite.__version__, "tl2cgen": tl2cgen.__version__, "numpy": np.__version__},
              "machine": f"{platform.machine()} {platform.system()}", "tolerance": TOL, "tie_band": TIE, "models": {}}
    for ds, key, path in jobs:
        forest = load_onnx_forest(path)
        model = to_treelite(forest, "float32")
        lib = os.path.join(tmp, f"{ds}_{key}.dylib" if platform.system() == "Darwin" else f"{ds}_{key}.so")
        tl2cgen.export_lib(model, toolchain="clang" if platform.system() == "Darwin" else "gcc", libpath=lib,
                           params={"parallel_comp": 32}, verbose=False)
        predictor = tl2cgen.Predictor(lib)
        entry = {}
        for name, variants in (("crossing", CROSSING_VARIANTS), ("three", VARIANTS)):
            x = build_probes(path, variants).rows
            ref = ort_probs(path, x)
            comp = probabilities(forest, compiled_scores(predictor, forest, x))   # margins include base values
            gtil = probabilities(forest, treelite_scores(model, forest, x, np.float32))
            entry[name] = {"probes": int(len(x)), "compiled_vs_onnxruntime": compare(ref, comp),
                           "compiled_vs_interpreter": compare(gtil, comp)}
        report["models"][f"{ds}/{key}"] = entry
        print(ds, key, {k: (v["compiled_vs_onnxruntime"]["label_differences"], v["compiled_vs_onnxruntime"]["over_tolerance"],
                            f"{v['compiled_vs_onnxruntime']['max_probability_difference']:.1e}",
                            v["compiled_vs_interpreter"]["label_differences"]) for k, v in entry.items()}, flush=True)
    out = os.path.join(RESULTS, "exp35_treelite_compiled.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
