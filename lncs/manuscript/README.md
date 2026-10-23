# Manuscript — LNCS systems paper

Self-contained English draft: `llncs.cls`, `splncs04.bst`, and the XeLaTeX font setup are vendored under `vendor/`, so this compiles without any Springer template checkout.

## Compile

```bash
cd lncs/manuscript
./compile.sh
```

Compiles cleanly as of this draft (20 pages: main text ends on p. 18, references to p. 20; 0 undefined citations/references, no overfull boxes), exactly at the DIMVA 18-page main-text limit since the UNSW-NB15 table (`tab:unsw`) went in: anything added now needs a cut elsewhere. The replay-parity table was folded into the prose of Sect. 3.3 to make room earlier.

## TODO before submission

Venue: DIMVA 2027 (Enschede, 30 June–2 July 2027; CFP checked 2026-10-13 at dimva.org/dimva2027). Limits: 18 pages main text + 5 bibliography + 2 appendix, Springer LNCS template. Cycle 1: abstract 3 Dec 2026, paper 10 Dec 2026 (firm), notification 28 Jan 2027, https://dimva27r1.hotcrp.com/. Cycle 2: abstract 11 Feb 2027, paper 18 Feb 2027 (firm), notification 8 Apr 2027, https://dimva27r2.hotcrp.com/. A Cycle 1 reject may not be resubmitted to Cycle 2. Camera-ready 27 May 2027. Backup: ACNS 2027 Cycle 2, 21 Jan 2027 AoE, 20 pages excluding references, no parallel submission. See `../README.md`.

Done:

- [x] Venue picked and CFP confirmed (above); the draft fits the 18+5+2 limit without cuts
- [x] Single author (Thai Nguyen Vu, UTC), matching `../../CITATION.cff`; the co-authors of the earlier companion paper were removed
- [x] No trainer-framework-specific content anywhere in `main.tex`/`references.bib` or in `../code/`; the deployed forest's trainer is described only as a 32-bin histogram trainer (Sect. 3.1, 4), which is all the threshold-sharing explanation needs
- [x] No `\input`/`\bibliography` paths outside this repository; no `\ph{...}` placeholders left
- [x] Boundary-directed parity experiment (`../code/experiments/exp2_boundary_parity.py`): 287,223 probes, float32 0 flips, float64 328 flips vs 15 on full replay
- [x] Block-lockstep traversal measured on real flows, including the batch size where it loses (`exp1`); shipped in `src/onnx_edge_ids/numpy_engine.py` since `894238a`
- [x] Exact early exit measured on gate-forwarded traffic (`exp3`)
- [x] `exp1` and `exp3` re-run on Jetson #2 (`../code/results_jetson/`); the traversal and early-exit tables quote the board
- [x] Distribution-shift stress test on CSE-CIC-IDS2018 (`exp4`): detector and gate collapse, serving guarantees hold, undecided rate rises
- [x] Probes on four more forests and their converters (`exp5`, Table `tab:generalize`): float32 0 flips everywhere, 24 to 169x denser than replay where divergence is rare, trees-moved diagnostic
- [x] Probes on a second dataset (`exp10`, UNSW-NB15, Table `tab:unsw`): float32 0 flips on all four models; probes alone find LightGBM's divergence, replay is as dense for XGBoost; framework-vs-ORT disagreement traced to the float64 scaler (all of XGBoost's, about half of HGB's)
- [x] ONNX Runtime release sweep (`exp5b`): 1.26.0/1.27.0 relabel 85% of flows for a stock scikit-learn export (upstream PRs 27536, 28685, issue 28557), cited in Sect. 4.1
- [x] Gate-node row of Table `tab:engines` from a Kafka replay on Jetson #2 (`exp6`)
- [x] End-to-end latency measured over Wi-Fi (`exp7`, Table `tab:e2e`); a wired re-run was dropped because the inter-board link carries 3.4% of its bandwidth and the ceiling is the pure-Python Kafka client
- [x] Classifier steady-state and cold-start figures repeated five times per backend on Jetson #2, 25 W mode (`exp8`); Tables `tab:engines` and `tab:coldstart` report mean ± SD

Open before the chosen cycle:

- [ ] Choose Cycle 1 (10 Dec 2026) or Cycle 2 (18 Feb 2027); register the abstract on HotCRP a week before the paper deadline
- [ ] Wait for the Software Impacts decision before citing it: cite only once accepted, otherwise do not mention it (the two manuscripts stay independent and must not duplicate claims)
- [x] Undecided-rate drift alarm calibrated (`exp9`, 2026-10-13, Table `tab:alarm`): max-over-calibration-windows threshold, 0 false alarms on held-out 2017 windows, 47/47 shifted windows at W=10,000, 23/23 on 500 forwarded flows; still one shift, windows are consecutive split rows
- [x] Boundary prober is part of `scripts/validate_parity.py` since v0.4.0 (2026-10-13): `src/onnx_edge_ids/boundary_probe.py`, dataset-free, on by default; Sect. 3.3 and the Discussion say so, numbers from `validation/parity_boundary_m3_20261013.json`
- [ ] Only if a CUDA/TensorRT provider is ever deployed: re-run `exp2` against it (a different provider is a different runtime); not needed for the CPU-only deployment described
