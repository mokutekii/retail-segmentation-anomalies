# TODO — Retail Customer Purchase Prediction
## 0) Repo Ready (must do first)
- [ ] Pull the latest repo; create venv; `pip install -r requirements.txt`.
- [ ] Open `notebooks/retail_purchase_prediction.ipynb` and run the first cells.
  - Owner: ____
  - DoD: `outputs/figures/` and `outputs/tables/` are created automatically and the notebook runs without import errors.

## 1) Data & EDA (fast)
- [ ] Load `data/online_shoppers_intention.csv` and run EDA cells.
- [ ] Produce at least two quick plots (e.g., purchase rate by `Month`, by `VisitorType`).
- [ ] Save figures to `outputs/figures/`.
- [ ] Write 3–5 bullets under the **Data** section (class imbalance, notable features, quirks).
  - Owner: ____
  - DoD: `class_counts.png` plus two additional EDA figures saved; Data bullets added in the notebook.

## 2) Baselines
- [ ] Run Logistic Regression and Decision Tree cells (already scaffolded).
- [ ] Export `outputs/tables/validation_metrics.csv`.
- [ ] Add a validation confusion matrix for Logistic Regression and save it.
  - Owner: ____
  - DoD: `validation_metrics.csv` contains LR and DT rows; `confusion_matrix_val_logreg.png` saved.

## 3) Ensembles (likely primary model)
- [ ] Train Random Forest; quick tune (`n_estimators=300`; try `max_depth=None/10/20`).
- [ ] If permitted by course rules, add Gradient Boosting.
- [ ] If a tree model leads, save top-20 feature importance plot.
  - Owner: ____
  - DoD: RF (and GB if used) appended to `validation_metrics.csv`; `feature_importance_top20.png` saved if applicable.

## 4) MLP (only if time allows)
- [ ] Implement shallow MLP (1–2 hidden layers, ReLU, dropout or L2, early stopping).
- [ ] Record metrics; add a simple learning curve (epochs vs val loss or val F1).
  - Owner: ____
  - DoD: MLP row added to `validation_metrics.csv`; `mlp_learning_curve.png` saved.

## 5) Lock Model and Test Evaluation
- [ ] Select best model by validation ROC-AUC (tie-break on F1).
- [ ] Run test evaluation; save metrics and plots.
  - Owner: ____
  - DoD: `outputs/tables/test_metrics.csv`, `outputs/figures/roc_curve_test.png`, and `outputs/figures/confusion_matrix_test.png` saved.

## 6) Mini Ablation (keep small)
- [ ] Define two feature groups and rerun the best model:
  - Behavioral: page counts/durations, bounce/exit metrics
  - Temporal/Tech: Month, Weekend, Browser/OS/Region
- [ ] Summarize the performance delta.
  - Owner: ____
  - DoD: `outputs/tables/ablations.csv` saved; `outputs/figures/ablation_delta_auc.png` with AUC changes.

## 7) Business Takeaways
- [ ] Write 5–7 bullets: which signals most influence purchase probability, and one recommended operating point (precision/recall trade-off at threshold 0.5 or adjusted).
  - Owner: ____
  - DoD: Bullets added under the “Business Takeaways” notebook section.

## 8) Slides
- [ ] Create `presentations/slides/final_slides.pptx` using `presentations/slides/slides_outline.md`.
- [ ] Include: Problem, Data, Methods, Results (ROC, confusion matrix, feature importance), Ablations, Takeaways, Future Work, Acknowledgments.
  - Owner: ____
  - DoD: Slide deck completed with minimal text and clear figures.

## 9) Poster
- [ ] Start from `presentations/templates/poster-template-uh.pptx`.
- [ ] Sections: Abstract (≤150 words), Background, Data, Methods, Results, Conclusion, Future Direction, Acknowledgments.
- [ ] Use saved figures from `outputs/figures/`.
  - Owner: ____
  - DoD: `presentations/poster/final_poster.pptx` completed.

## 10) Report Stubs (just enough for submission)
- [ ] In `report/report.tex`, add 2–3 sentences per section and figure references. Use Overleaf or local TeX (add `neurips_2023.sty` later if required).
  - Owner: ____
  - DoD: `report/report.tex` compiles with placeholders; figure references noted.

## 11) Reproducibility and Sanity Checks
- [ ] Add a “Reproducibility” cell at the end of the notebook (random seeds, versions); run `pip freeze > requirements-lock.txt`.
- [ ] Confirm preprocessing is inside an sklearn Pipeline (fit on train only) to avoid leakage.
  - Owner: ____
  - DoD: `requirements-lock.txt` created; leakage guardrail noted in the notebook.

## 12) Export Assets (optional convenience)
Paste this cell at the bottom of the notebook to collect final figures for slides/poster:

```python
from pathlib import Path
src = Path('../outputs/figures')
dst = Path('../presentations/assets')
dst.mkdir(parents=True, exist_ok=True)

order = [
    'class_counts.png',
    'roc_curve_test.png',
    'confusion_matrix_test.png',
    'feature_importance_top20.png',
    'ablation_delta_auc.png',
    'mlp_learning_curve.png'
]
i = 1
for name in order:
    p = src / name
    if p.exists():
        (dst / f'{i:02d}_{name}').write_bytes(p.read_bytes())
        i += 1
print('Exported to', dst.resolve())

- Owner: ____
- DoD: presentations/assets/ contains numbered PNGs ready to paste.
