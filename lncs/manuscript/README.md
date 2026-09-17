# Manuscript — LNCS systems paper

Self-contained English draft: `llncs.cls`, `splncs04.bst`, and the XeLaTeX font setup are vendored under `vendor/`, so this compiles without any Springer template checkout.

## Compile

```bash
cd lncs/manuscript
./compile.sh
```

Compiles cleanly as of this draft (15 pages, 0 undefined citations/references — only cosmetic under/overfull-hbox warnings in the bibliography).

## TODO before submission

- [ ] Pick a venue and deadline, update `../README.md` and the title page
- [x] Author names and affiliations (copied from an earlier companion paper — re-confirm order/roles before submission; `../../CITATION.cff` lists a single author for the software, so confirm this paper's author list is what's intended)
- [x] No PySpark/Spark content anywhere in `main.tex`/`references.bib`
- [x] No `\input`/`\bibliography` paths outside this repository
- [x] Boundary-directed parity experiment (`../code/experiments/exp2_boundary_parity.py`): 287,223 probes, float32 0 flips, float64 328 flips vs 15 on full replay
- [x] Block-lockstep traversal measured on real flows, including the batch size where it loses (`exp1`); shipped in `src/onnx_edge_ids/numpy_engine.py` since `894238a`
- [x] Exact early exit measured on gate-forwarded traffic (`exp3`)
- [x] Distribution-shift stress test on CSE-CIC-IDS2018 (`exp4`): detector and gate collapse, serving guarantees hold, undecided rate rises
- [ ] If the undecided-rate drift signal is to be claimed as more than an observation: calibrate an alarm threshold and measure false alarms on in-distribution windows
- [ ] Re-run `exp1` and `exp3` on the Jetson Cortex-A78AE boards and merge into Table `tab:engines`
- [ ] Re-run `exp2` against any other ONNX Runtime provider that will be deployed (CUDA/TensorRT): a different provider is a different runtime
- [ ] Decide whether the boundary prober should become part of `scripts/validate_parity.py`, the package's release gate
- [ ] Run the wired-link end-to-end benchmark and fill the remaining `\ph{...}`
- [ ] Repeat the steady-state/cold-start/parity measurements across multiple runs (the software's own acceptance criteria call for this; current numbers are a single run)
- [ ] Verify target venue's page limit/template requirements once chosen
- [ ] Decide whether/how to cross-reference the parallel Software Impacts submission for the same software once that one is finalized, so the two don't duplicate claims
