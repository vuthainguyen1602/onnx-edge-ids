# Systems paper: experiment code and results

Full systems-paper treatment of the software in this repository: a framework-agnostic ONNX/NumPy serving toolkit for a two-stage Kafka edge intrusion detection pipeline on Jetson Orin Nano Super.

The manuscript is a journal submission and is kept outside this repository; the directory keeps its original name (`lncs/`) so that archived releases and paths stay valid. This is a different manuscript from the short Software Impacts article about the same software, and the two are meant to be read independently.

## Research question

A detector served by a primary runtime (ONNX Runtime) and a fallback runtime (NumPy) promises the same verdict from both. How do you check that promise, what does the fallback cost at streaming batch sizes, and how much of the forest does real traffic need if verdicts may not change?

The experiments in `code/` cover boundary-directed parity probes and replay gates (comparing labels, and labels and confidences), their generalization to other trainers, converters and intrusion datasets, an ONNX Runtime release sweep, a block-lockstep NumPy traversal, an exact early-exit rule, and behaviour under a cross-dataset shift. Every result file is under `code/results*/`, and [`REPRODUCE.md`](REPRODUCE.md) maps each one to the script that writes it.

## Code

Experiment code is in [`code/`](code/): the traversals, the early-exit rule, the boundary prober, the result files quoted in the paper, and tests that run without the dataset. See [`code/README.md`](code/README.md). It reads the served forests from `../artifacts/` and a test pins its traversal to the packaged `NumpyInferenceEngine`, so the paper and the package cannot drift apart silently. Since v0.4.0 the package's release gate (`../scripts/validate_parity.py`) runs the same probe construction, dataset-free, from `../src/onnx_edge_ids/boundary_probe.py`.

The serving software itself is the rest of this repository (`../src/onnx_edge_ids/`, `../scripts/`). The block-lockstep traversal measured in the paper is the one shipped in `../src/onnx_edge_ids/numpy_engine.py` since commit `894238a`; the on-board figures in the paper predate it and describe the per-tree traversal.

## Reproduce

A step-by-step guide from a bare clone to regenerated result files, written for a machine that is not a Jetson, is in [`REPRODUCE.md`](REPRODUCE.md). The summary below maps the steps onto the paper's tables.

No Jetson is needed for Steps 1 and 3: the experiments under `code/` and the parity check run on any x86 or ARM64 machine, and their disagreement and label-identity counts come out the same there (throughput differs and is labelled by machine wherever it appears). Step 2's Kafka pipeline runs on any two Linux/macOS hosts; only the power column of Table `tab:engines` requires the boards (`tegrastats`).

**Step 1 — export the NumPy fallback from an ONNX artifact, then validate parity (fills Table `tab:parity`):**

```bash
# from the repository root
ONNX_MODEL=artifacts/ids_rf.onnx FEATURES_JSON=artifacts/feature_columns.json ./scripts/export_artifacts.sh
python scripts/validate_parity.py --csv <path to a CICIDS2017-derived replay CSV> --rows 2000 \
  --report-json results/parity_report.json
```

**Step 2 — two-Jetson deployment, one classifier-node run per backend and per batch size (fills Table `tab:engines`, batch-size sensitivity in Sect. "Results"):**

Follow `../docs/deploy_2jetson.md`: gate on Jetson #1, `classifier_node.py --engine onnx|numpy --batch-size 1|20|500 --metrics-csv ...` on Jetson #2.

**Step 3 — regenerate the boundary-parity, traversal, early-exit and generalization tables** with `code/experiments/` (see `code/README.md`; `exp5` needs the training split and `lightgbm xgboost skl2onnx onnxmltools`, `exp5b` creates one venv per ONNX Runtime release). `exp1` and `exp3` were also re-run on Jetson #2 (`code/results_jetson/`), which the paper's traversal and early-exit tables quote.

**Step 4 — end-to-end run over Wi-Fi (Table `tab:e2e`):**

From the coordinator, `code/experiments/exp7_e2e_run.sh` starts the gate on Jetson #1 and the ONNX classifier on Jetson #2 with `--latency-csv`, replays the CSV at a given rate, and takes each board's clock offset before and after (`exp7_clock_offset.py`); `exp7_e2e_summary.py` writes `code/results_jetson/exp7_e2e.json`. All three hosts are on Wi-Fi; a wired re-run would only shrink the transport share, a few milliseconds next to the batching delays.

## Metrics to report

- Classifier-node throughput (rows/s) and p95 (ms/row), per backend and per batch size
- Cold-start time to first verdict, per backend
- RSS (MB) and total-board power (W, `tegrastats`, not idle-subtracted)
- Label agreement / max confidence deviation between backends (parity gate)
- Artifact size (ONNX vs. NumPy)
- End-to-end (producer-to-verdict) latency per stage over Wi-Fi, light and maximum load (`exp7`) — done 2026-10-11
- Boundary-probe and replay disagreement counts per runtime (`exp2`), and the routed-vs-unrouted probe ablation (`exp2b`)
- The same counts, plus framework-vs-ONNX-Runtime parity and trees moved per probe, on scikit-learn/LightGBM/XGBoost exports (`exp5`); label agreement across ONNX Runtime releases (`exp5b`)
- Detector, gate, parity and early-exit behaviour on shifted traffic (`exp4`)
- Block-lockstep and exact-early-exit throughput, re-measured on Jetson #2 — done 2026-10-11 (`code/results_jetson/`)

## Manuscript

The manuscript is not in this repository. `REPRODUCE.md` maps each of its tables and figures to the result files under `code/`.

## Cross-reference

- Software Impacts article on the same software: `../notes/paper_pitch.md`, `../submission_checklist.md`
- Earlier distributed-deployment work and the thesis (out of scope here): [`vuthainguyen1602/Thesis_IDS`](https://github.com/vuthainguyen1602/Thesis_IDS)
