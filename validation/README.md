# Artifact validation records

`parity_boundary_m3_20261013.json` is the first run of the release gate with the
boundary probes (v0.4.0, Apple M3, NumPy 2.5.3, ONNX Runtime 1.30.0): 2,000
synthetic replay rows plus 287,223 boundary probes (95,741 internal nodes x 3
values, 815 distinct feature/threshold pairs, 99.92% reaching their node under
float32), 0 label disagreements, maximum confidence difference 4.2e-7, 16 s.

`parity_hosts_20261013.json` collects the `scripts/validate_parity.py` reports
quoted by the Software Impacts article (Table "Artifact validation"). The same
ONNX artifact (`artifacts/ids_rf.onnx`, md5 `60de6235…`) and its NumPy export
were run on four Python environments over the first 2,000 and all 10,000 rows
of the CICIDS2017-derived replay CSV (`replay_cicids2017.csv`, md5
`05bf0e1c…`, the upstream thesis repository's `jetson/sender/` copy of
2026-09-30), batch size 20:

| Environment | Python | NumPy | ONNX Runtime |
|---|---|---|---|
| Workstation, Apple M3 (project venv) | 3.12.4 | 2.5.3 | 1.30.0 |
| Workstation, Apple M3 (system Python) | 3.12.4 | 1.26.4 | 1.27.0 |
| Jetson #1, Orin Nano Super | 3.10.12 | 2.2.6 | 1.23.2 |
| Jetson #2, Orin Nano Super | 3.10.12 | 2.2.6 | 1.23.2 |

Every run returned label agreement 2000/2000 and 10000/10000, a maximum
confidence difference of 0.0, and the same attack counts (317 of 2,000 and
1,562 of 10,000). The NumPy engine under test is the released per-tree
traversal of `v0.2.0` (the boards hold a copy of that tree; the workstation
runs were made from a `v0.2.0` git worktree). The per-row timings inside the
reports are single runs and are not quoted anywhere; the article's Table 2
uses `lncs/code/results_jetson/exp8_summary.json`.

Re-run on any host:

```bash
python scripts/validate_parity.py --csv /path/to/replay_cicids2017.csv --rows 10000 \
  --report-json results/parity_report.json
```
