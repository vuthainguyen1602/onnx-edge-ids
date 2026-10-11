# Software Impacts Submission Checklist

## Manuscript

- [x] Short descriptive paper, roughly three pages.
- [x] Abstract focuses on software and reuse.
- [x] Code metadata table filled.
- [x] Impact overview section included.
- [x] No overclaiming benchmark results before final measurement.

Note: Software Impacts requires only the Code metadata table (C1-C9). The
Software metadata table (S1-S8) belongs to the SoftwareX template and was
removed.

## Software Artifact

- [x] Public Git repository.
- [x] Tagged release matching manuscript version: v0.2.0, matching C1.
- [x] Open-source license.
- [x] README with install, export, validation, and inference examples.
- [x] Requirements file or `pyproject.toml`.
- [x] Example data or instructions for obtaining replay data.
- [x] Tests or smoke checks.
- [x] Archive DOI via Zenodo, Software Heritage, or equivalent:
      10.5281/zenodo.22731928 (v0.2.0); concept DOI 10.5281/zenodo.22731927.
      v0.3.0 (2026-10-11, block-lockstep NumPy, --latency-csv, lncs/ code):
      10.5281/zenodo.23299232, with ORCID, affiliation and keywords from
      .zenodo.json. The Software Impacts article still describes v0.2.0.

## Validation Data Measured

- [x] Number of replay rows used for export validation: 2,000.
- [x] ONNX vs NumPy label agreement: 2000/2000 (100.0000%).
- [x] ONNX vs NumPy probability delta: 0.0 max.
- [x] Artifact size: 7.38 MB ONNX, 1.23 MB NumPy.
- [x] Jetson throughput, p95 latency, memory, and energy:
      classifier 20,185 rows/s, p95 0.0676 ms/row, 155.6 MB RSS, 4.64 W (ONNX);
      152 rows/s, p95 8.2267 ms/row, 46.2 MB RSS, 4.93 W (NumPy);
      gate 13,915 rows/s, p95 0.0783 ms/row, 28.64% forward rate (first
      validation run). Re-measured 2026-10-11 on Jetson #2 serving a 20,000-record
      Kafka replay (lncs/code/results_jetson/exp6_gate_node.json): 21,191 rows/s,
      p95 0.0569 ms/row, 123.6 MB RSS, 5.25 W input rail (4.75 W idle), 11.8%
      forwarded. Both papers now quote the re-measurement.
- [x] Repeated 2026-10-11, five runs per backend on Jetson #2, 25 W mode
      (lncs/code/results_jetson/exp8_summary.json; the article's Table 2 now
      quotes these): ONNX 15,174 +/- 179 rows/s, p95 0.0847 ms/row, 156.8 MB,
      4.66 W; per-tree NumPy (v0.2.0) 127 +/- 1 rows/s, p95 9.50 ms/row,
      47.6 MB, 4.99 W. Cold start, five runs per board: ONNX 373 ms load,
      7 ms first prediction, 670 ms to first verdict; NumPy 80, 114, 482 ms.
      The single-run ONNX capacity below (20,185) was not reproduced.
- [x] Engine cold-start load time (mean of both boards, page cache dropped):
      ONNX Runtime 294.5 ms to build the session, 3.1 ms first prediction,
      542.3 ms process start to first verdict; NumPy 79.7 ms to load,
      115.1 ms first prediction, 368.4 ms to first verdict.
- [x] Transport link stated: both boards and the workstation are on one Wi-Fi
      LAN, 5-8 ms median round trip (setup paragraph of the use case). End-to-end
      latency over that link: lncs/code/results_jetson/exp7_e2e.json (64 rows/s:
      p50 1.58 s, p95 2.11 s; max rate: p50 3.55 s, p95 6.44 s, gate drains ~965
      rows/s). Not re-taken on a wired link: the inter-board link carries 3.4%
      of its bandwidth and the ceiling is the pure-Python Kafka client, which
      runs 2.6-4.5x faster on a laptop CPU with the same code.

## Known Gaps Before Submission

- All three hosts run over Wi-Fi (no Ethernet link). The articles say so, and
  the end-to-end latency was measured over that link (exp7).
- Fixed 2026-10-11: the v0.2.0 Zenodo record now carries the ORCID,
  affiliation and keywords (edited and republished, DOI unchanged). It had
  lacked them because `.zenodo.json` was added after the v0.2.0 tag; v0.3.0
  picked the file up automatically.
- v0.1.0 stays where it is. It is already published and must keep describing
  the tree archived under that name.
- `artifacts/anomaly_*` (the gate autoencoder, scaler and threshold) come from
  the upstream thesis repository and are not redistributed here.

## Scope Boundary

This article is about ONNX-EdgeIDS as reusable software. The distributed Kafka
deployment it grew out of is background and motivation only. Do not recast that
systems work here.
