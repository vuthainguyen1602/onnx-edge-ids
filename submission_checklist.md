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

## Validation Data Measured

- [x] Number of replay rows used for export validation: 2,000.
- [x] ONNX vs NumPy label agreement: 2000/2000 (100.0000%).
- [x] ONNX vs NumPy probability delta: 0.0 max.
- [x] Artifact size: 7.38 MB ONNX, 1.23 MB NumPy.
- [x] Jetson throughput, p95 latency, memory, and energy:
      classifier 20,185 rows/s, p95 0.0676 ms/row, 155.6 MB RSS, 4.64 W (ONNX);
      152 rows/s, p95 8.2267 ms/row, 46.2 MB RSS, 4.93 W (NumPy);
      gate 13,915 rows/s, p95 0.0783 ms/row, 28.64% forward rate.
- [x] Engine cold-start load time (mean of both boards, page cache dropped):
      ONNX Runtime 294.5 ms to build the session, 3.1 ms first prediction,
      542.3 ms process start to first verdict; NumPy 79.7 ms to load,
      115.1 ms first prediction, 368.4 ms to first verdict.
- [x] Transport link stated: Jetson #2 runs over Wi-Fi (setup paragraph of the
      use case). Not re-taken on a wired link: the inter-board link carries 3.4%
      of its bandwidth and the ceiling is the pure-Python Kafka client, which
      runs 2.6-4.5x faster on a laptop CPU with the same code.

## Known Gaps Before Submission

- Jetson #2 has no Ethernet link and runs over Wi-Fi. The article says so;
  wired transport figures remain optional, since Wi-Fi is not the bound.
- The Zenodo record for v0.2.0 carries the right title and author name, taken
  from CITATION.cff, but no ORCID, affiliation or keywords: `.zenodo.json` was
  added after the v0.2.0 tag, so it is absent from the archived tree. Fix by
  editing the record at https://zenodo.org/records/22731928 and republishing,
  which keeps the DOI. Later releases pick the file up automatically.
- v0.1.0 stays where it is. It is already published and must keep describing
  the tree archived under that name.
- `artifacts/anomaly_*` (the gate autoencoder, scaler and threshold) come from
  the upstream thesis repository and are not redistributed here.

## Scope Boundary

This article is about ONNX-EdgeIDS as reusable software. The distributed Kafka
deployment it grew out of is background and motivation only. Do not recast that
systems work here.
