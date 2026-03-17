# Part 3 README

This folder contains the Part 3 predictive-modeling handoff for Lab 2. The pipeline compares three classifier families across four feature blocks using the realistic by-image holdout splits from Part 1, then refits the selected model on all supervised labeled rows for unlabeled-image sanity checks.

Search mode used for this run: `fast`.

## Scripts
- `code/part3/dataset.py`
  - loads and validates the supervised modeling table
  - aligns Part 1 split files to Part 2 features
  - builds the unlabeled feature table used for final inference
- `code/part3/train_models.py`
  - tunes and evaluates logistic regression, random forest, and HistGradientBoosting
  - writes model-selection tables, diagnostics, and the final fitted model
- `code/part3/predict_unlabeled.py`
  - loads the saved final model
  - scores the three fixed unlabeled images and writes probability maps and masks

## Main outputs
- `results/part3/model_selection_summary.csv`
- `results/part3/outer_split_metrics.csv`
- `results/part3/selected_model_config.json`
- `results/part3/models/final_model.joblib`
- `results/part3/seed_stability.csv`
- `results/part3/spatial_robustness_metrics.csv`
- `results/part3/threshold_sensitivity.csv`
- `results/part3/diagnostics/`
- `results/part3/predictions/`

## Selected model
- family: `HistGradientBoosting`
- feature block: `B3_context_ae`
- mean outer-test ROC AUC: `0.9543`
- mean outer-test PR AUC: `0.9052`
- mean outer-test balanced accuracy: `0.9003`

## Integrity checks
- supervised rows: `207681`
- duplicate supervised keys: `0`
- rows with missing selected features: `0`
- split integrity table: `results/part3/split_integrity.csv`

## Recommended reading order
1. `documents/part3/meeting_summary.md`
2. `results/part3/model_selection_summary.csv`
3. `results/part3/outer_split_metrics.csv`
4. `results/part3/selected_model_config.json`
5. `results/part3/diagnostics/best_model/`

