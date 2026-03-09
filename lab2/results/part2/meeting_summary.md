# STAT 214 Lab 2 Part 2 - Meeting Summary

## 1. Goal of this pass
This Part 2 pass was focused on building a concrete, shareable predictor shortlist for feature engineering before classifier work. The work covers four predictor families:
- expert features from the lab/paper: `NDAI`, `SD`, `CORR`
- raw multi-angle radiances: `DF`, `CF`, `BF`, `AF`, `AN`
- hand-engineered scalar combinations and contrasts
- patch-based autoencoder latent features derived from `code/autoencoder.py`

## 2. Scripts added for this pass
- `code/part2/feature_engineering.py`
  - loads the 3 labeled images
  - creates engineered scalar and 3x3 neighborhood predictors
  - screens predictors with separation AUC, mutual information, and Cohen's d
  - writes `labeled_engineered_features.csv`, `feature_screening.csv`, and `predictor_catalog.md`
- `code/part2/autoencoder_features.py`
  - uses `code/autoencoder.py` plus `code/checkpoints/gsi-model.ckpt`
  - extracts latent embeddings for supervised pixels from 9x9 patches
  - writes `autoencoder_embeddings_supervised.csv`, `autoencoder_feature_screening.csv`, and `autoencoder_feature_notes.md`

## 3. Main findings from feature screening
### 3.1 Best expert / baseline predictors
- `SD`: AUC 0.935, MI 0.428
- `NDAI`: AUC 0.823, MI 0.259
- `CORR` is weak in the current screening: AUC 0.524, MI 0.010

Interpretation:
- `SD` is the strongest single interpretable baseline.
- `NDAI` is still useful and should remain in the baseline set.
- `CORR` should probably not be one of the three headline Part 2 predictors unless it helps in combinations or patch summaries.

### 3.2 Best radiance predictors
- `AF`: separation AUC 0.799
- `AN`: separation AUC 0.794
- `BF`: separation AUC 0.770

Interpretation:
- the aft / nadir-side radiances are much more informative than `DF` and `CF`
- a compact radiance baseline should likely start with `AF`, `AN`, and maybe `BF`

### 3.3 Best engineered scalar predictors
- `ndai_x_sd`: AUC 0.944, MI 0.428
- `rad_cv`: AUC 0.825, MI 0.260
- `front_back_ratio`: AUC 0.801, MI 0.238
- `af_df_gap`: separation AUC 0.794, |d| 1.500
- `rad_range`: AUC 0.775, MI 0.232

Interpretation:
- the strongest cheap nonlinear predictor is `ndai_x_sd`
- angular contrasts like `af_df_gap` and `front_back_ratio` are useful because they reflect multi-angle cloud/surface differences
- simple brightness variability summaries are also competitive

### 3.4 Best patch-local predictors
- `local_SD_mean3`: AUC 0.936, MI 0.450
- `local_NDAI_std3`: AUC 0.927, MI 0.360
- `local_SD_std3`: AUC 0.916, MI 0.386
- `local_rad_std_std3`: AUC 0.916, MI 0.323
- `local_rad_mean_std3`: AUC 0.889, MI 0.289

Interpretation:
- local context is clearly valuable
- patch-based summaries are competitive with or better than the strongest raw single-pixel predictors
- neighborhood texture should be part of the final Part 2 story, not just an optional extra

### 3.5 Best autoencoder latent predictors
From the checkpoint-backed embedding screen:
- `ae4`: AUC 0.897, MI 0.306
- `ae5`: AUC 0.867, MI 0.230
- `ae0`: AUC 0.845, MI 0.216
- `ae3`: AUC 0.799, MI 0.159
- `ae1`: AUC 0.779, MI 0.174

Interpretation:
- the autoencoder already gives several strong latent predictors
- `ae4`, `ae5`, and `ae0` are strong enough to include in the Part 2 meeting shortlist
- the embeddings should be treated as complementary features, not replacements for interpretable variables like `SD` and `NDAI`

## 4. Recommended meeting shortlist
If the meeting needs a short list instead of the full catalog, discuss these first:
- `SD`
- `NDAI`
- `AF`
- `AN`
- `ndai_x_sd`
- `af_df_gap`
- `front_back_ratio`
- `local_SD_mean3`
- `local_NDAI_std3`
- `local_SD_std3`
- `ae4`
- `ae5`
- `ae0`

## 5. Recommended Part 2 feature-engineering methods
- Keep a baseline block of interpretable predictors: `SD`, `NDAI`, `AF`, `AN`, and optionally `BF`
- Add low-cost scalar combinations: `ndai_x_sd`, `af_df_gap`, `front_back_ratio`, `rad_cv`, `rad_range`
- Add local 3x3 patch summaries to capture context: local means, local standard deviations, and centered-within-patch values
- Add the top few autoencoder embedding dimensions (`ae4`, `ae5`, `ae0`) as a second feature family
- In modeling, compare:
  - interpretable-only set
  - interpretable + local patch set
  - interpretable + local patch + autoencoder set

## 6. Caveats to mention during the meeting
- These are univariate screening metrics, not final model rankings.
- Some high-ranking features may be redundant with each other.
- Final selection still needs split-aware validation using the realistic train/val/test strategies from Part 1.
- `CORR` underperformed here, but could still matter in multivariate models or local summaries.

## 7. Files to open during the meeting
- `results/part2/predictor_catalog.md`
- `results/part2/feature_screening.csv`
- `results/part2/autoencoder_feature_notes.md`
- `results/part2/autoencoder_feature_screening.csv`
