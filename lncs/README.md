# LNCS systems paper

Full systems-paper (Springer LNCS) treatment of the software in this repository: a framework-agnostic ONNX/NumPy serving toolkit for a two-stage Kafka edge intrusion detection pipeline on Jetson Orin Nano Super.

**Venue:** TBD. This is a different manuscript from the short **Software Impacts** article about the same software (`../notes/paper_pitch.md`, `../submission_checklist.md`; that manuscript is kept outside the repository). The two are meant to be read independently, not merged.

## Research question

A detector served by a primary runtime (ONNX Runtime) and a fallback runtime (NumPy) promises the same verdict from both. How do you check that promise, what does the fallback cost at streaming batch sizes, and how much of the forest does real traffic need if verdicts may not change?

Findings, all regenerable from `code/`:

- **Boundary-directed parity testing.** One probe per split node, on the threshold and one float32 step either side (287,223 probes, 99.9% reach their node). A float64 NumPy traversal flips 328 labels against ONNX Runtime on the probes but only 15 on the whole 447,275-row labelled test split, so a 2,000-row replay gate misses it 93.5% of the time. The deployed float32 traversal flips none.
- **Why labels flip.** The forest's 95,741 splits share 815 distinct (feature, threshold) pairs, so one boundary value moves many trees at once; the same fact makes an exhaustive gate cheap.
- **Block-lockstep traversal.** 71x at batch 1, 11x at 20, 1.4x at 500, and 0.9x at 2,000 (where the old per-tree loop still wins), on real flows.
- **Exact early exit.** Gate-forwarded traffic needs 104.1 of 200 trees on average with labels provably identical to the full forest: 2.0x, the rule's ceiling.
- **Other forests, other converters.** On scikit-learn RandomForest and HistGradientBoosting, LightGBM and XGBoost exported with skl2onnx/onnxmltools, the same probes find float64-traversal and framework-vs-ONNX-Runtime divergence 3 to 169x more densely than replay; the float32 traversal never disagrees. Threshold sharing is a matter of bin count (72 trees per pair at Spark's 32 bins, 4 at 255, 2 for exact splits). ONNX Runtime 1.26.0 and 1.27.0 relabel 85% of flows for a stock scikit-learn export (P > 0 instead of P > 0.5); replay catches that, probes are not meant to.
- **Distribution shift.** This work does not make the detector robust to drift. Frozen and run on CSE-CIC-IDS2018, classifier F1 falls 0.98 to 0.24 and gate attack recall 0.72 to 0.01. Runtime parity and early-exit exactness hold on all 477,771 shifted flows, and the share of flows early exit cannot settle rises 4.8% to 28.2% without labels.

## Title

*Two Runtimes, One Verdict: Boundary-Probed Parity for Tree-Ensemble Intrusion Detection at the Edge*

## Code

Experiment code is in [`code/`](code/): the traversals, the early-exit rule, the boundary prober, the result files quoted in the paper, and tests that run without the dataset. See [`code/README.md`](code/README.md). It reads the served forests from `../artifacts/` and a test pins its traversal to the packaged `NumpyInferenceEngine`, so the paper and the package cannot drift apart silently.

The serving software itself is the rest of this repository (`../src/onnx_edge_ids/`, `../scripts/`). The block-lockstep traversal measured in the paper is the one shipped in `../src/onnx_edge_ids/numpy_engine.py` since commit `894238a`; the on-board figures in the paper predate it and describe the per-tree traversal.

## Reproduce

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

**Step 3 — regenerate the boundary-parity, traversal, early-exit and generalization tables** with `code/experiments/` (see `code/README.md`; `exp5` needs the training split and `lightgbm xgboost skl2onnx onnxmltools`, `exp5b` creates one venv per ONNX Runtime release), then re-run `exp1` and `exp3` on the Jetson boards: their throughput numbers were taken on an ARM64 development machine.

**Step 4 — wired-link end-to-end run (the pending measurement):**

Repeat Step 2 over a wired inter-board link instead of Wi-Fi, and report producer-timestamp-to-verdict latency/throughput, not just in-process classifier cost.

## Metrics to report

- Classifier-node throughput (rows/s) and p95 (ms/row), per backend and per batch size
- Cold-start time to first verdict, per backend
- RSS (MB) and total-board power (W, `tegrastats`, not idle-subtracted)
- Label agreement / max confidence deviation between backends (parity gate)
- Artifact size (ONNX vs. NumPy)
- End-to-end (producer-to-verdict) throughput/latency over a **wired** link — pending
- Boundary-probe and replay disagreement counts per runtime (`exp2`)
- The same counts, plus framework-vs-ONNX-Runtime parity and trees moved per probe, on scikit-learn/LightGBM/XGBoost exports (`exp5`); label agreement across ONNX Runtime releases (`exp5b`)
- Detector, gate, parity and early-exit behaviour on shifted traffic (`exp4`)
- Block-lockstep and exact-early-exit throughput, re-measured on Jetson hardware — pending

## Manuscript

The English draft is in `manuscript/` (Springer LNCS template, XeLaTeX; `llncs.cls`/`splncs04.bst`/font setup vendored under `manuscript/vendor/`). Compile with:

```bash
cd lncs/manuscript
./compile.sh
```

Compiles cleanly (17 pages, 0 undefined citations/references, no overfull boxes). On-board numbers come from the software's own validation run; the boundary-parity, traversal, early-exit and generalization tables come from `code/results/*.json`. The remaining `\ph{...}` markers are the Jetson re-measurement of those two throughput tables and the wired-link latency.

## Cross-reference

- Software Impacts article on the same software: `../notes/paper_pitch.md`, `../submission_checklist.md`
- Earlier distributed-deployment work and the thesis (out of scope here): [`vuthainguyen1602/Thesis_IDS`](https://github.com/vuthainguyen1602/Thesis_IDS)
