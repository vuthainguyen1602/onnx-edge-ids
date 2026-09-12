# Software Impacts Submission Checklist

## Manuscript

- [ ] Short descriptive paper, roughly three pages.
- [ ] Abstract focuses on software and reuse.
- [ ] Code metadata table filled.
- [ ] Software metadata table filled.
- [ ] Impact overview section included.
- [ ] References include prior SOICT system paper or thesis artifact when citeable.
- [ ] No overclaiming benchmark results before final measurement.

## Software Artifact

- [x] Public Git repository.
- [ ] Tagged release matching manuscript version.
- [x] Open-source license.
- [ ] README with install, export, validation, and inference examples.
- [ ] Requirements file or `pyproject.toml`.
- [ ] Example data or instructions for obtaining replay data.
- [ ] Tests or smoke checks.
- [ ] Archive DOI via Zenodo, Software Heritage, or equivalent.

## Validation Data To Fill

- [ ] Number of replay rows used for export validation.
- [ ] Spark vs ONNX label agreement.
- [ ] Spark vs ONNX probability delta.
- [ ] Artifact size.
- [ ] Engine load time.
- [ ] Jetson throughput, p95 latency, memory, and energy.

## Scope Boundary

This article is about ONNX-EdgeIDS as reusable software. The distributed Kafka
and PySpark deployment from SOICT is background and motivation. Do not recast
the whole SOICT paper here.
