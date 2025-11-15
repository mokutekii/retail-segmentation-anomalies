# Customer Segmentation & Anomaly Detection in Retail Transactions

**Authors:** Matthew Nguyen, Kevin Su  

This project builds **actionable customer segments** and a **ranked anomaly list** from the *Online Retail II* dataset, using only DSII-friendly, reproducible techniques:
- Interpretable features (RFM-style): `tx_count`, `spend_sum`, `item_qty_sum`, `basket_size_mean`, `recency_days`, `RFM_Score`
- Baseline **K-Means** segmentation with metrics sweep (silhouette, CH, DB)
- **Autoencoder** (tabular) → embeddings → comparative clustering
- **LSTM** (sequence) → embeddings → comparative clustering (habit vs variety)
- **Isolation Forest** + **AE reconstruction MSE** for anomalies
- Exports all figures/CSVs your slides/poster need

---

## Quick Start

1) Environment

> Windows + VS Code; Python 3.10–3.11 works best. In powershell:
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

2) Open the notebook

Open retail_segmentation.ipynb in VS Code (or Jupyter).
Select the .venv kernel when prompted and Run All.

The notebook prints the data date range, customer counts, and writes outputs to:

data/processed/ (engineered tables, metrics CSVs)

reports/figures/ (all charts used for slides/poster)

3) What you’ll see generated

Data coverage & distributions: daily_tx.png, hists_core.png, price_qty_hex.png, pareto_spend.png

Segmentation quality: silhouette_k_features.png, silhouette_k_ae.png, silhouette_k_seq.png

Cluster profiles (heatmaps): cluster_profile_heatmap_features.png, cluster_profile_heatmap_ae.png, cluster_profile_heatmap_seq.png

Training curves: ae_loss.png, lstm_loss.png

Business impact visuals: segment_lift_spend.png, segment_reengage_share.png

Anomaly evidence: if_score_hist.png, ae_mse_hist.png, anomaly_agreement_scatter.png, anomaly_cume_spend.png