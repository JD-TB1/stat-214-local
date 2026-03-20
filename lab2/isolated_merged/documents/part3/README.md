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
- mean outer-test ROC AUC: `0.9547`
- mean outer-test PR AUC: `0.9073`
- mean outer-test balanced accuracy: `0.9001`

## How to interpret the current results
- Treat these as strong preliminary results from the current `fast` search, not as the final exhaustive tuning run.
- The leading combination is `HistGradientBoosting + B3_context_ae`, which suggests the best current performance comes from combining expert features, engineered features, local spatial context, and the selected autoencoder features.
- The step from `B2_context` to `B3_context_ae` is meaningful.
  - For HistGradientBoosting, mean outer-test ROC AUC improved from `0.9401` to `0.9543`, and mean PR AUC improved from `0.8717` to `0.9052`.
  - That is evidence that the selected AE dimensions are adding useful signal beyond the hand-engineered/context block.
- The results are not equally easy across all held-out images.
  - The best model is very strong on `O013257` and `O013490`, but weaker on `O012791`.
  - That means the average performance is good, but we should still inspect the per-image error maps and diagnostics rather than relying on a single mean score.
- Read the metrics in roles:
  - ROC AUC is the main ranking metric across model/block combinations.
  - PR AUC helps judge positive-class retrieval quality.
  - Balanced accuracy reflects the thresholded classification behavior after threshold selection on validation data.
- Practical conclusion: the current pipeline is working and the AE-enhanced context block is promising, but the final reported model should still be confirmed with a `full` search rerun.

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

