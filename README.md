# ONNX-EdgeIDS

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22731927.svg)](https://doi.org/10.5281/zenodo.22731927)

This repository is the software and the workspace for a **Software Impacts**
article about it. A separate, longer systems paper that studies the same
software lives under [`lncs/`](lncs/) with its own experiment code; the two
manuscripts are independent and the Software Impacts article does not depend on
anything in `lncs/`.

Software Impacts publishes short software publications. The submission should
make the software artifact citable and reusable, with an impact overview and
metadata about the current code version. The journal page and indexed examples
describe the format as a short descriptive paper of about three pages with an
Impact Overview statement and code/repository metadata.

## Proposed Software Name

**ONNX-EdgeIDS**

## Proposed Title

ONNX-EdgeIDS: A Lightweight Two-Stage Edge Serving Toolkit for Network Intrusion Detection

## Software Pitch

ONNX-EdgeIDS serves portable ONNX or transparent NumPy IDS artifacts on Jetson-class edge devices. The package focuses on the serving side of the problem: flow replay over ZeroMQ or Kafka, a first-stage gate, a second-stage classifier node, and scripts for measuring throughput and latency without carrying a heavyweight training runtime into the inference path.

## Why This Fits Software Impacts

The core contribution is reusable software:

- artifact preparation: ONNX artifact to lightweight NumPy fallback;
- runtime: ONNX/NumPy classifier interface for network flows streamed over ZeroMQ or Kafka;
- validation: ONNX-versus-NumPy prediction checks;
- benchmark scripts: edge inference cost, latency, memory, and energy.

The research context that exposed the software need is that an IDS model can be accurate while the serving stack is still too heavy for edge hardware. This article focuses on the resulting software artifact and its impact, not on a full distributed IDS evaluation.

## Files

- `src/onnx_edge_ids/`: reusable ONNX/NumPy inference package.
- `artifacts/`: the served ONNX and NumPy artifacts plus the feature-column metadata.
- `scripts/export_numpy.py`: convert an ONNX tree-ensemble artifact to the NumPy format.
- `scripts/export_artifacts.sh`: create the NumPy artifact from an ONNX artifact.
- `scripts/validate_parity.py`: check that the NumPy artifact reproduces the ONNX artifact on replayed rows and on a boundary probe of every split threshold; exits non-zero outside tolerance.
- `scripts/init_topics.py`: create Kafka input/suspicious topics.
- `scripts/csv_producer.py`: replay CICIDS2017-derived CSV flow rows over ZeroMQ (float32 batches) or Kafka at a target rate.
- `scripts/gate_node.py`: Jetson #1 anomaly gate, forwarding suspicious flows over ZeroMQ or Kafka.
- `scripts/classifier_node.py`: minimal Jetson #2 classifier loop over ZeroMQ or Kafka.
- `scripts/smoke_predict.py`: local smoke test for exported artifacts.
- `docs/data.md`: CICIDS2017 dataset note and citation requirement.
- `docs/deploy_2jetson.md`: two-Jetson test plan using thesis gate + ONNX classifier.
- `deploy/sync_to_jetson.sh`: rsync this repo to a Jetson node.
- `experiments/plan.md`: validation and benchmark plan.
- `notes/paper_pitch.md`: idea notes and contribution boundaries.
- `lncs/`: LNCS systems paper (manuscript, experiment code, result files): boundary-directed runtime parity testing, its generalization to scikit-learn/LightGBM/XGBoost exports and to converter parity, an ONNX Runtime release sweep, block-lockstep traversal, exact early exit, behaviour under distribution shift. See `lncs/README.md`.

## Dataset

The deployment replay uses a local CICIDS2017-derived flow CSV. The raw IDS 2017/CICIDS2017 dataset is not committed; see `docs/data.md` for the official source and citation.

## Quick Local Test

Optional NumPy artifact export from the ONNX model:

```bash
ONNX_MODEL=artifacts/ids_rf.onnx \
FEATURES_JSON=artifacts/feature_columns.json \
./scripts/export_artifacts.sh

python scripts/smoke_predict.py --engine onnx --model artifacts/ids_rf.onnx
python scripts/smoke_predict.py --engine numpy --model artifacts/ids_rf_numpy.npz
```

Validate that the two backends agree before deploying. The script fails the
command rather than only printing a warning, and it runs two checks:

- replay: both backends over the same rows (a replay CSV, or synthetic rows
  when `--csv` is omitted), reporting label agreement and the largest
  confidence difference;
- boundary probes (on by default, `--no-boundary-probes` to skip): one row
  per internal node of the forest, routed to that node and placed exactly on
  its split threshold and one float32 step either side. The probes are built
  from the NumPy artifact alone, so this check needs no dataset. For the
  shipped forest that is 287,223 probes over 815 distinct feature/threshold
  pairs, 99.9% of which reach their node; it takes a few seconds for ONNX
  Runtime and about half a minute for the NumPy engine on a laptop. Replayed
  traffic almost never sits on a threshold, which is the only place two tree
  runtimes can disagree, so whether the replay check catches a fallback that
  scales or compares in the wrong precision depends on which rows happened to
  be sampled; the probes exercise every threshold, so they catch it every time.

```bash
python scripts/validate_parity.py \
  --csv /path/to/replay_cicids2017.csv --rows 2000 \
  --report-json results/parity_report.json
```

The package's own tests (no dataset needed) cover the probe construction and
the parity of the shipped artifacts:

```bash
pip install -e ".[onnx]" pytest
pytest tests/
```

CI (`.github/workflows/ci.yml`) runs these tests and the full parity gate on
synthetic rows on every push and pull request.

## Jetson #2 Minimal Classifier Test

Copy this repo to Jetson #2, copy `artifacts/ids_rf.onnx` and
`artifacts/feature_columns.json`, install the optional runtime, then run:

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[kafka,zeromq]"
pip install -r requirements_jetson_onnx.txt

python scripts/classifier_node.py \
  --bootstrap <mac-ip>:9092 \
  --topic ids-suspicious-flow \
  --engine onnx \
  --model artifacts/ids_rf.onnx \
  --features artifacts/feature_columns.json
```

If ONNX Runtime is not available on the Jetson image, use:

```bash
python scripts/classifier_node.py \
  --bootstrap <mac-ip>:9092 \
  --topic ids-suspicious-flow \
  --engine numpy \
  --model artifacts/ids_rf_numpy.npz \
  --features artifacts/feature_columns.json
```

## Before Submission

- [x] Software release archived (Zenodo DOIs in `CITATION.cff` and the article).
- [x] Export validation on replay flows on four hosts/runtimes: `validation/`.
- [x] Workflow figure (train / export / release gate / serve) in the article.
- [ ] Fill the Software Impacts submission form (repository URL, tag, license,
      archive DOI, contact e-mail) from `submission_checklist.md`.

## Citation And License

The current release is `v0.4.0`, under the MIT license, archived on Zenodo
(concept DOI [10.5281/zenodo.22731927](https://doi.org/10.5281/zenodo.22731927);
per-version DOIs in `CITATION.cff`).
