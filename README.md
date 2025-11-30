# Retail Customer Purchase Prediction — Notebook-First Repo

Central deliverable is a **master Jupyter Notebook** that generates all plots/tables for: **slides, poster, and NeurIPS-style report**.

## Quick Start
```bash
git clone <your_repo_url>
cd retail-customer-purchase-prediction
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab  # open notebooks/retail_purchase_prediction.ipynb
```

## Data
- `data/online_shoppers_intention.csv` (UCI — primary)
- `data/online_retail_II.csv` (optional; RFM aggregation). Large files are git-ignored.

## Notebook Workflow
EDA → preprocessing → models (LR, DT/RF/(GB*), MLP) → metrics (ROC/F1/PR) → ablations → **save figures/tables** to `outputs/` → conclusions.

## Presentations
`presentations/slides/` and `/poster/` for exports; `/templates/` holds UH poster template and a reference deck.

## Report
NeurIPS LaTeX stub at `report/report.tex`. Add `neurips_2023.sty` to compile.

_Last updated: 2025-11-18_
