# Customer Segmentation & Anomaly Detection in Retail Transactions

**Authors:** Matthew Nguyen, Kevin Su

This project builds **actionable customer segments** and a **ranked anomaly list** from the *Online Retail II* dataset with a fully reproducible DSII pipeline:
- Interpretable **RFM-style features**: `tx_count`, `spend_sum`, `item_qty_sum`, `basket_size_mean`, `recency_days`, `RFM_Score`
- Baseline **K-Means** with a metrics sweep (**silhouette**, **Calinski–Harabasz**, **Davies–Bouldin**)
- **Autoencoder (tabular)** → embeddings → comparative clustering
- **LSTM (sequence)** → embeddings → comparative clustering (habit vs. variety)
- **Isolation Forest** + **AE reconstruction MSE** for anomaly ranking
- Exports all figures/CSVs needed for slides/poster

> The notebook uses **relative paths**. Clone anywhere; outputs land under `data/processed/` and `reports/figures/`.

---

## Quick Start

### 1) Environment
- Windows / macOS / Linux, Python **3.10–3.11** recommended.

**Create venv + install deps**
```bash
# from the repo root
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
