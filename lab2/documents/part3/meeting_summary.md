# STAT 214 Lab 2 Part 3 - Meeting Summary

## Why Part 3 matters
Part 1 established which train/test splits are credible, and Part 2 produced a shortlist of promising predictors. Part 3 is where we test whether those predictors actually support cloud-vs-non-cloud prediction on new images rather than only looking good inside the same scene. The main risk here is overestimating performance through spatial leakage, so the whole pipeline is built around realistic generalization.

## Data used for modeling
- supervised modeling table = Part 2 engineered features joined with the Part 2 autoencoder embeddings
- supervised rows used: `207681`
- cloud rows: `80965`
- non-cloud rows: `126716`
- duplicate keys on `image_id/x/y/label`: `0`
- rows with missing selected features: `0`

## Feature blocks and why we compared them
- `B0_base`: `SD`, `NDAI`, `AF`, `AN`, `BF`
  - This is the compact baseline using the strongest raw/expert features from Part 2.
- `B1_engineered`: `B0_base + ndai_x_sd + af_df_gap + front_back_ratio + rad_cv + rad_range`
  - This tests whether simple domain-informed interactions and radiance summaries add usable signal beyond the base block.
- `B2_context`: `B1_engineered + local_SD_mean3 + local_NDAI_std3 + local_SD_std3 + local_rad_std_std3 + local_rad_mean_std3`
  - This tests whether local spatial context improves discrimination, which should matter because clouds form spatially structured regions.
- `B3_context_ae`: `B2_context + ae4 + ae5 + ae0`
  - This tests whether learned patch embeddings add information beyond the hand-engineered features.

The point of the block comparison is not only to find the best model, but also to measure the marginal value of engineering, local context, and learned features.

## Evaluation design and reasoning
- Primary benchmark: by-image holdout splits from Part 1.
  - This is the most realistic estimate of how well the model transfers to a new labeled image.
- Hyperparameter tuning: validation subset inside each outer holdout split.
  - This prevents tuning on the outer test image.
- Threshold selection: choose the probability threshold that maximizes validation balanced accuracy.
  - This is more appropriate than fixing `0.5` because the score calibration differs by model/split.
- Secondary robustness check: spatial-within-image splits for `O013257` and `O013490`.
  - These are useful supporting evidence, but weaker than by-image generalization because train and test still come from the same image.
- Excluded from aggregate spatial evaluation: `O012791_x_gt_q80`.
  - Its test split has zero cloud pixels, so threshold-based classification metrics are not meaningful there.

## Model families and why they were included
- Logistic Regression
  - Interpretable linear baseline. This tells us how far we can get with a relatively simple and explainable decision boundary.
- Random Forest
  - Nonlinear tree ensemble that can capture interactions and threshold effects without feature scaling.
- HistGradientBoosting
  - Strong tabular learner that usually handles mixed nonlinear structure more efficiently than random forests. This was the main candidate for the strongest predictive performance.

## Metrics and why we used them
- ROC AUC is the primary model-selection metric.
  - It measures ranking quality across thresholds and is the cleanest main comparison for the outer test splits.
- PR AUC is the secondary metric.
  - It is especially useful because cloud detection is still a positive-class detection problem, and precision-recall behavior matters.
- Balanced accuracy is used for threshold selection and tie-breaking.
  - This avoids letting the majority class dominate the threshold choice.
- F1, sensitivity, specificity, and Brier score are reported to show operational tradeoffs and calibration quality.

## What the current implementation actually ran
Search mode used for the saved results: `fast`.

This is a complete Part 3 pipeline, but the saved run is a reduced hyperparameter search rather than the exhaustive search:
- Logistic Regression: same grid in both modes (`3` values of `C`)
- Random Forest: fast grid = `4` settings, full grid = `18`
- HistGradientBoosting: fast grid = `4` settings, full grid = `16`

Across `3` by-image holdouts, `4` feature blocks, and `3` model families:
- fast search evaluates `132` primary candidates
- full search evaluates `444` primary candidates

So the current results should be described as preliminary model-selection results from the fast search, not the final exhaustive search. The pipeline itself is not partial; only the tuning breadth is reduced.

## Preliminary results from the current run
- Selected family: `HistGradientBoosting`
- Selected feature block: `B3_context_ae`
- Mean outer-test ROC AUC: `0.9543`
- Mean outer-test PR AUC: `0.9052`
- Mean outer-test balanced accuracy: `0.9003`

## Best models in the ranking table
- `HistGradientBoosting` + `B3_context_ae`: mean ROC AUC=`0.9543`, mean PR AUC=`0.9052`, mean balanced accuracy=`0.9003`
- `HistGradientBoosting` + `B2_context`: mean ROC AUC=`0.9401`, mean PR AUC=`0.8717`, mean balanced accuracy=`0.8835`
- `Random Forest` + `B3_context_ae`: mean ROC AUC=`0.9273`, mean PR AUC=`0.8897`, mean balanced accuracy=`0.8465`
- `HistGradientBoosting` + `B1_engineered`: mean ROC AUC=`0.9259`, mean PR AUC=`0.8503`, mean balanced accuracy=`0.8610`
- `HistGradientBoosting` + `B0_base`: mean ROC AUC=`0.9217`, mean PR AUC=`0.8344`, mean balanced accuracy=`0.8687`
- `Random Forest` + `B2_context`: mean ROC AUC=`0.9210`, mean PR AUC=`0.8600`, mean balanced accuracy=`0.8380`

## Best block per family
- `HistGradientBoosting`: best block `B3_context_ae`
- `Random Forest`: best block `B3_context_ae`
- `Logistic Regression`: best block `B2_context`

## How to interpret the current ranking
- `HistGradientBoosting + B3_context_ae` is currently the leader, which suggests that the strongest performance comes from combining expert features, engineered interactions, local context, and AE features.
- Within HistGradientBoosting, moving from `B2_context` to `B3_context_ae` improved mean ROC AUC from `0.9401` to `0.9543` and mean PR AUC from `0.8717` to `0.9052`.
  - That is evidence that the selected AE dimensions add useful signal beyond the hand-engineered/context features.
- Logistic Regression performed best with `B2_context`, not `B3_context_ae`.
  - This suggests the AE features may be more useful when the model can exploit nonlinear interactions.

## Per-holdout results for the current best model
- `holdout_O012791`: ROC AUC `0.8957`, PR AUC `0.8036`, balanced accuracy `0.8297`
- `holdout_O013257`: ROC AUC `0.9780`, PR AUC `0.9272`, balanced accuracy `0.9302`
- `holdout_O013490`: ROC AUC `0.9890`, PR AUC `0.9847`, balanced accuracy `0.9412`

The variation across holdouts matters. The model is very strong on two images and clearly harder-tested on `O012791`, so the next pass should pay attention to what is special about that image in the error maps.

## Diagnostics and post-hoc work already produced
- logistic diagnostics: coefficients, odds ratios, VIF, calibration, and feature-correlation plots
- random-forest and boosting diagnostics: permutation importance, plus RF impurity importance
- best-model post-hoc EDA: error maps, feature-distribution comparisons, error-by-quantile plots, and FP/FN summaries
- robustness checks: seed stability, spatial robustness, and threshold sensitivity
- final sanity check on three unlabeled images:
  - `O002539`: predicted cloud fraction `0.9616`
  - `O045178`: predicted cloud fraction `0.6718`
  - `O119738`: predicted cloud fraction `0.6687`

## Recommendation for the next run
- Keep the current pipeline structure.
- Rerun Part 3 in `full` search mode before locking the final model for submission.
- Use the current fast-mode outputs as the meeting/preliminary results, not as the final tuned benchmark.

## What to show in the meeting
- `results/part3/model_selection_summary.csv`
- `results/part3/outer_split_metrics.csv`
- `results/part3/diagnostics/best_model/`
- `results/part3/selected_model_config.json`
