# STAT 214 Lab 2 Part 3 - Meeting Summary

## Goal
Part 3 turns the Part 2 feature set into actual cloud-vs-non-cloud classifiers and evaluates them under realistic image-level generalization.

## Evaluation design
- Primary benchmark: by-image holdout splits from Part 1
- Tuning: validation split inside each holdout
- Secondary robustness: usable spatial splits only
- Excluded from aggregate spatial evaluation: `O012791_x_gt_q80`, because its test set has no cloud pixels

## Models compared
- Logistic Regression
- Random Forest
- HistGradientBoosting

## Best result
- Selected family: `HistGradientBoosting`
- Selected feature block: `B3_context_ae`
- Mean outer-test ROC AUC: `0.9543`
- Mean outer-test PR AUC: `0.9052`
- Mean outer-test balanced accuracy: `0.9003`

## Best models in the ranking table
- `HistGradientBoosting` + `B3_context_ae`: mean ROC AUC=0.9543, mean PR AUC=0.9052, mean balanced accuracy=0.9003
- `HistGradientBoosting` + `B2_context`: mean ROC AUC=0.9401, mean PR AUC=0.8717, mean balanced accuracy=0.8835
- `Random Forest` + `B3_context_ae`: mean ROC AUC=0.9273, mean PR AUC=0.8897, mean balanced accuracy=0.8465
- `HistGradientBoosting` + `B1_engineered`: mean ROC AUC=0.9259, mean PR AUC=0.8503, mean balanced accuracy=0.8610
- `HistGradientBoosting` + `B0_base`: mean ROC AUC=0.9217, mean PR AUC=0.8344, mean balanced accuracy=0.8687
- `Random Forest` + `B2_context`: mean ROC AUC=0.9210, mean PR AUC=0.8600, mean balanced accuracy=0.8380

## Best block per family
- `HistGradientBoosting` best block `B3_context_ae`: mean ROC AUC=0.9543, mean PR AUC=0.9052
- `Random Forest` best block `B3_context_ae`: mean ROC AUC=0.9273, mean PR AUC=0.8897
- `Logistic Regression` best block `B2_context`: mean ROC AUC=0.8551, mean PR AUC=0.7905

## Why this model won
- It had the best mean outer-test ROC AUC across the realistic by-image splits.
- Tie-breaking favored stronger PR AUC and balanced accuracy.
- The final model was then refit on all supervised labeled data and saved for unlabeled-image sanity checks.

## What to show in the meeting
- `results/part3/model_selection_summary.csv`
- `results/part3/outer_split_metrics.csv`
- `results/part3/diagnostics/best_model/`
- `results/part3/selected_model_config.json`

