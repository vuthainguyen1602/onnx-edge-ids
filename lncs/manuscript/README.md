# Manuscript — LNCS systems paper

Self-contained English draft: `llncs.cls`, `splncs04.bst`, and the XeLaTeX font setup are vendored under `vendor/`, so this compiles without any Springer template checkout.

## Compile

```bash
cd lncs/manuscript
./compile.sh
```

Compiles cleanly as of this draft (17 pages, 0 undefined citations/references, no overfull boxes).

## TODO before submission

- [x] Venue picked: target DIMVA 2027 (cycle 2 ~Feb 2027 by the 2026 pattern; CFP not yet out), backup ACNS 2027 cycle 2 (Jan 21, 2027). See `../README.md`
- [ ] When the DIMVA 2027 CFP appears, confirm dates/page limits and register the HotCRP submission
- [x] Single author (Thai Nguyen Vu, UTC), matching `../../CITATION.cff`; the co-authors of the earlier companion paper were removed
- [x] No trainer-framework-specific content anywhere in `main.tex`/`references.bib` or in `../code/`; the deployed forest's trainer is described only as a 32-bin histogram trainer (Sect. 3.1, 4), which is all the threshold-sharing explanation needs
- [x] No `\input`/`\bibliography` paths outside this repository
- [x] Boundary-directed parity experiment (`../code/experiments/exp2_boundary_parity.py`): 287,223 probes, float32 0 flips, float64 328 flips vs 15 on full replay
- [x] Block-lockstep traversal measured on real flows, including the batch size where it loses (`exp1`); shipped in `src/onnx_edge_ids/numpy_engine.py` since `894238a`
- [x] Exact early exit measured on gate-forwarded traffic (`exp3`)
- [x] Distribution-shift stress test on CSE-CIC-IDS2018 (`exp4`): detector and gate collapse, serving guarantees hold, undecided rate rises
- [x] Probes on four more forests and their converters (`exp5`, Table `tab:generalize`): float32 0 flips everywhere, 24 to 169x denser than replay where divergence is rare, trees-moved diagnostic
- [x] ONNX Runtime release sweep (`exp5b`): 1.26.0/1.27.0 relabel 85% of flows for a stock scikit-learn export (upstream PRs 27536, 28685, issue 28557), cited in Sect. 4.1
- [ ] If the undecided-rate drift signal is to be claimed as more than an observation: calibrate an alarm threshold and measure false alarms on in-distribution windows
- [ ] Re-run `exp1` and `exp3` on the Jetson Cortex-A78AE boards and merge into Table `tab:engines`
- [ ] Re-run `exp2` against any other ONNX Runtime provider that will be deployed (CUDA/TensorRT): a different provider is a different runtime
- [ ] Decide whether the boundary prober should become part of `scripts/validate_parity.py`, the package's release gate
- [ ] Run the wired-link end-to-end benchmark and fill the remaining `\ph{...}`
- [ ] Repeat the steady-state/cold-start/parity measurements across multiple runs (the software's own acceptance criteria call for this; current numbers are a single run)
- [ ] Verify target venue's page limit/template requirements once chosen; at 17 pages, Table `tab:parity` and Sect. 6.3 are the first cuts if 15 is the limit
- [ ] Decide whether/how to cross-reference the parallel Software Impacts submission for the same software once that one is finalized, so the two don't duplicate claims
