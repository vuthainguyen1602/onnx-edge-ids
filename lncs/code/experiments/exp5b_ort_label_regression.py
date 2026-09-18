"""Which ONNX Runtime releases label a stock skl2onnx RandomForest export the way scikit-learn does?

skl2onnx exports a binary RandomForestClassifier as a TreeEnsembleClassifier with
a single weight column (P(class 1) per leaf, summed over trees) and
post_transform NONE. ONNX Runtime's label output for that graph is
``P > 0.5`` in most releases but ``P > 0`` in some, which labels every row with
any attack vote as an attack. This script runs one exported model through a
list of ONNX Runtime releases, each installed into its own throwaway venv, and
records how many rows of the replayed test split change label. It is the
replay-visible counterpart of the boundary probes: a version regression this
gross needs no probes to find, and a boundary probe alone would not size it.

Usage:
    python exp5b_ort_label_regression.py --model results/exp5_models/sklearn_rf.onnx \
        --data /path/to/test_data.parquet --versions 1.25.1,1.26.0,1.27.0,1.28.0,1.30.0
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgeids.data import RESULTS, load_test_split

CHILD = r'''
import sys, json, numpy as np, onnxruntime as ort
x = np.load(sys.argv[2])["x"]
s = ort.InferenceSession(sys.argv[1], providers=["CPUExecutionProvider"])
name = s.get_inputs()[0].name
labs, probs = [], []
for i in range(0, len(x), 50000):
    out = s.run(None, {name: x[i:i + 50000]})
    labs.append(np.asarray(out[0]).reshape(-1)); probs.append(np.asarray(out[1]))
lab = np.concatenate(labs).astype(np.int64); prob = np.concatenate(probs)
p1 = prob[:, 1]
print(json.dumps({
    "onnxruntime": ort.__version__,
    "rows": int(len(x)),
    "label_output_attack_rows": int((lab == 1).sum()),
    "p_gt_half_rows": int((p1 > 0.5).sum()),
    "p_gt_zero_rows": int((p1 > 0).sum()),
    "label_output_vs_p_gt_half_flips": int((lab != (p1 > 0.5)).sum()),
    "probability_column0_is_negative": bool((prob[:, 0] < 0).any()),
}))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="test_data.parquet directory")
    ap.add_argument("--versions", default="1.19.2,1.22.1,1.25.1,1.26.0,1.27.0,1.28.0,1.30.0")
    ap.add_argument("--venv-dir", default=os.path.join(tempfile.gettempdir(), "exp5b_ort_venvs"))
    ap.add_argument("--out", default=os.path.join(RESULTS, "exp5b_ort_label_regression.json"))
    args = ap.parse_args()

    x, _, _ = load_test_split(args.data)
    os.makedirs(args.venv_dir, exist_ok=True)
    rows_file = os.path.join(args.venv_dir, "rows.npz")
    np.savez(rows_file, x=x)
    child = os.path.join(args.venv_dir, "child.py")
    with open(child, "w") as f:
        f.write(CHILD)

    report = {"model": os.path.basename(args.model), "rows": int(len(x)), "runs": []}
    for v in args.versions.split(","):
        venv = os.path.join(args.venv_dir, f"ort-{v}")
        py = os.path.join(venv, "bin", "python")
        if not os.path.exists(py):
            subprocess.run([sys.executable, "-m", "venv", venv], check=True)
            subprocess.run([os.path.join(venv, "bin", "pip"), "install", "-q", f"onnxruntime=={v}", "numpy"], check=True)
        res = subprocess.run([py, child, args.model, rows_file], capture_output=True, text=True)
        if res.returncode != 0:
            print(f"onnxruntime {v}: failed\n{res.stderr[-500:]}")
            report["runs"].append({"onnxruntime": v, "error": res.stderr.strip().splitlines()[-1] if res.stderr else "failed"})
            continue
        run = json.loads(res.stdout.strip().splitlines()[-1])
        report["runs"].append(run)
        print(f"onnxruntime {run['onnxruntime']}: label output flips vs P>0.5 = {run['label_output_vs_p_gt_half_flips']:,} / {run['rows']:,}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
