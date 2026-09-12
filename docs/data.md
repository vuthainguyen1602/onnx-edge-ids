# Dataset note: CICIDS2017 / IDS 2017

ONNX-EdgeIDS is designed to replay labeled network-flow records derived from the Canadian Institute for Cybersecurity IDS 2017 dataset (CICIDS2017). The raw dataset is not redistributed in this repository.

Use the official CIC dataset page to obtain the data and follow its citation requirements:

- Official page: https://www.unb.ca/cic/datasets/ids-2017.html
- Paper to cite: Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization,” ICISSP 2018.

Expected local replay input: a CSV of derived flow records, kept outside Git,
whose path you pass explicitly. Nothing in the repository assumes a location.

```text
<your-path>/replay_cicids2017.csv
```

The replay CSV should contain the selected feature columns listed in `artifacts/feature_columns.json`. For deployment tests, keep generated CICIDS2017 slices outside Git, then pass the CSV path to `scripts/csv_producer.py` with `--csv`.
