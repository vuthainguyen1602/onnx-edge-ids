# Software Impacts Submission Checklist

## Manuscript

- [x] Short descriptive paper, roughly three pages.
- [x] Abstract focuses on software and reuse.
- [x] Code metadata table filled.
- [x] Impact overview section included.
- [ ] References include prior SOICT system paper or thesis artifact when citeable.
- [x] No overclaiming benchmark results before final measurement.

Note: Software Impacts requires only the Code metadata table (C1-C9). The
Software metadata table (S1-S8) belongs to the SoftwareX template and was
removed.

## Software Artifact

- [x] Public Git repository.
- [ ] Tagged release matching manuscript version.
- [x] Open-source license.
- [x] README with install, export, validation, and inference examples.
- [x] Requirements file or `pyproject.toml`.
- [x] Example data or instructions for obtaining replay data.
- [x] Tests or smoke checks.
- [ ] Archive DOI via Zenodo, Software Heritage, or equivalent.

## Validation Data Measured

- [x] Number of replay rows used for export validation: 2,000.
- [x] ONNX vs NumPy label agreement: 2000/2000 (100.0000%).
- [x] ONNX vs NumPy probability delta: 0.0 max.
- [x] Artifact size: 7.38 MB ONNX, 1.23 MB NumPy.
- [x] Jetson throughput, p95 latency, memory, and energy:
      classifier 20,185 rows/s, p95 0.0676 ms/row, 155.6 MB RSS, 4.64 W (ONNX);
      152 rows/s, p95 8.2267 ms/row, 46.2 MB RSS, 4.93 W (NumPy);
      gate 13,915 rows/s, p95 0.0783 ms/row, 28.64% forward rate.
- [ ] Engine cold-start load time.
- [ ] End-to-end throughput over a wired link on both boards.

## Known Gaps Before Submission

- Jetson #2 has no Ethernet link and currently runs over Wi-Fi. Inference
  latency, memory and power are measured in-process and unaffected, but
  end-to-end transport figures should be re-taken on a wired link.
- The v0.1.0 tag predates the artifacts, the batching fix and
  `validate_parity.py`; retag before submitting.
- `artifacts/anomaly_*` (the gate autoencoder, scaler and threshold) come from
  the upstream thesis repository and are not redistributed here.

## Scope Boundary

This article is about ONNX-EdgeIDS as reusable software. The distributed Kafka
deployment from SOICT is background and motivation. Do not recast the whole
SOICT paper here.
