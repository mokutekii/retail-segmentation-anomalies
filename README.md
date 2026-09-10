# Customer Segmentation & Anomaly Detection in Retail Transactions

**Authors:** Matthew Nguyen, Kevin Su

The authoritative workflow is a reproducible Python pipeline for customer-level RFM-style segmentation and Isolation Forest anomaly ranking. It validates the raw schema, uses a stable preprocessing contract, saves the fitted model with its feature schema, and records a dataset hash plus run metadata.

The original notebook is retained for exploratory visual work and the optional autoencoder/LSTM experiments. Its committed figures and CSVs are legacy course outputs; new pipeline runs write only to ignored `artifacts/` directories.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,notebooks]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m retail_segmentation.train --fast
```

The `--fast` run samples up to 50,000 valid transaction rows while keeping every selected invoice intact, so basket features are not built from partial checkouts. It is suitable for a smoke test. Omit it for a full run:

```powershell
.\.venv\Scripts\python.exe -m retail_segmentation.train --output-dir artifacts/full-run
```

Each run produces:

- `segmentation_model.joblib` — scaler, K-Means, Isolation Forest, and feature contract
- `customer_features.csv`, `cluster_labels.csv`, and `cluster_profiles.csv`
- `clustering_metrics.csv`, `anomaly_scores.csv`, and `anomaly_top10.csv`
- `run_metadata.json` plus three diagnostic figures

## Design choices

- The clustering matrix uses `log1p` followed by `RobustScaler` for heavy-tailed monetary and count features.
- The derived `rfm_score` is reported as a business summary only; it is not clustered alongside the source features, avoiding double weighting.
- The highest silhouette score selects `k`, while Davies–Bouldin and Calinski–Harabasz are recorded separately rather than plotted on an incomparable shared axis.
- Isolation Forest scores are a ranked screening signal, not proof of fraud or customer intent; investigate the accompanying customer-level evidence before acting.

## Notebook experiments

To run the legacy TensorFlow notebook, use Python 3.10 or 3.11 and install its optional dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[notebooks,deep-learning]"
```

## Docker

With Docker Desktop running:

```powershell
docker compose run --rm segment
```

Outputs are mounted into the local ignored `artifacts/` folder. Docker was not available on this machine during verification.
