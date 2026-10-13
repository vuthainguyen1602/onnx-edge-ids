# LNCS systems paper

Full systems-paper (Springer LNCS) treatment of the software in this repository: a framework-agnostic ONNX/NumPy serving toolkit for a two-stage Kafka edge intrusion detection pipeline on Jetson Orin Nano Super.

**Venue:** DIMVA 2027, Enschede, 30 June–2 July 2027 (LNCS; CFP out, checked 2026-10-13: Cycle 1 paper deadline 10 Dec 2026, Cycle 2 18 Feb 2027, 18 pages main text + 5 bibliography + 2 appendix, a Cycle 1 reject may not go to Cycle 2); backup ACNS 2027 Cycle 2 (deadline 21 Jan 2027 AoE, 20 pages excluding references). Both fit the current 20-page draft (main text ends three lines into p. 18, at the DIMVA limit) without cuts; no double submission. This is a different manuscript from the short **Software Impacts** article about the same software (`../notes/paper_pitch.md`, `../submission_checklist.md`; that manuscript is kept outside the repository). The two are meant to be read independently, not merged.

## Research question

A detector served by a primary runtime (ONNX Runtime) and a fallback runtime (NumPy) promises the same verdict from both. How do you check that promise, what does the fallback cost at streaming batch sizes, and how much of the forest does real traffic need if verdicts may not change?

Findings, all regenerable from `code/`:

- **Boundary-directed parity testing.** One probe per split node, on the threshold and one float32 step either side (287,223 probes, 99.9% reach their node). A float64 NumPy traversal flips 328 labels against ONNX Runtime on the probes but only 15 on the whole 447,275-row labelled test split, so a 2,000-row replay gate misses it 93.5% of the time. The deployed float32 traversal flips none.
- **Why labels flip.** The forest's 95,741 splits share 815 distinct (feature, threshold) pairs, so one boundary value moves many trees at once; the same fact makes an exhaustive gate cheap. The path-satisfying construction matters: the same boundary values written into unrouted random rows exercise 27% of the nodes and find 2.5x fewer flips.
- **Block-lockstep traversal.** On Jetson #2: 80x at batch 1, 20x at 20, 2.2x at 500, 1.2x at 2,000, on real flows. On an Apple M3 the old per-tree loop wins at 2,000 (0.9x), so the crossover depends on the host.
- **Exact early exit.** Gate-forwarded traffic needs 104.1 of 200 trees on average with labels provably identical to the full forest: 1.9x on Jetson #2 (2.05x on the M3), the rule's ceiling.
- **Other forests, other converters.** On scikit-learn RandomForest and HistGradientBoosting, LightGBM and XGBoost exported with skl2onnx/onnxmltools, the same probes find float64-traversal and framework-vs-ONNX-Runtime divergence 3 to 169x more densely than replay; the float32 traversal never disagrees. Threshold sharing is a matter of bin count (72 trees per pair at the deployed trainer's 32 bins, 4 at 255, 2 for exact splits). ONNX Runtime 1.26.0 and 1.27.0 relabel 85% of flows for a stock scikit-learn export (P > 0 instead of P > 0.5); replay catches that, probes are not meant to.
- **Distribution shift.** This work does not make the detector robust to drift. Frozen and run on CSE-CIC-IDS2018, classifier F1 falls 0.98 to 0.24 and gate attack recall 0.72 to 0.01. Runtime parity and early-exit exactness hold on all 477,771 shifted flows, and the share of flows early exit cannot settle rises 4.8% to 28.2% without labels. Calibrated as an alarm on windows of 10,000 flows (threshold = max over calibration windows), it raises 0 false alarms on 22 held-out in-distribution windows and fires on 47 of 47 shifted windows; on 500 gate-forwarded flows, 0/51 and 23/23.

## Title

*Two Runtimes, One Verdict: Boundary-Directed Parity Testing for Tree-Ensemble Intrusion Detection at the Edge*

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

The English draft is in `manuscript/` (Springer LNCS template, XeLaTeX; `llncs.cls`/`splncs04.bst`/font setup vendored under `manuscript/vendor/`). Compile with:

```bash
cd lncs/manuscript
./compile.sh
```

Compiles cleanly (17 pages, 0 undefined citations/references, no overfull boxes). On-board numbers come from the software's own validation run; the boundary-parity, traversal, early-exit and generalization tables come from `code/results/*.json`. The traversal and early-exit tables report Jetson #2 (`code/results_jetson/`); the gate row of the engine table comes from `code/results_jetson/exp6_gate_node.json`; the end-to-end table comes from `code/results_jetson/exp7_e2e.json`; the classifier and cold-start tables are five-run means from `code/results_jetson/exp8_summary.json`. No `\ph{...}` markers remain.

## Cross-reference

- Software Impacts article on the same software: `../notes/paper_pitch.md`, `../submission_checklist.md`
- Earlier distributed-deployment work and the thesis (out of scope here): [`vuthainguyen1602/Thesis_IDS`](https://github.com/vuthainguyen1602/Thesis_IDS)
