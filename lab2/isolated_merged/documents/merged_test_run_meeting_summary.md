# Merged Part 1/2 Pipeline Test Run

## Goal

The point of this test was not to replace the existing Lab 2 pipeline immediately. The point was to answer a narrower question:

- if we take the group-merged Part 1 and Part 2 code
- adapt it so it runs cleanly in our current directory structure
- and then feed its Part 2 output into our existing Part 3 training pipeline

does the downstream modeling result improve, change, or stay essentially the same?

## What The Raw Group-Merged Scripts Were Doing

### Group `eda.py`

Source file:
- [`../../code/eda.py`](../../code/eda.py)

What it does:
- loads the three labeled `.npz` images through `utils_npz`
- visualizes the five radiance-angle image channels
- makes a small set of labeled scatterplots:
  - `corr` vs `ndai`
  - `corr` vs `sd`
  - `ra_df` vs `ra_an`
  - `ra_bf` vs `ra_an`
  - `ra_cf` vs `ra_an`

Interpretation:
- this is useful as a visual sanity-check script
- it confirms that multi-angle radiance behavior is visually different across classes
- but it is not a full EDA pipeline

What it does not do:
- no label maps
- no pooled feature rankings
- no split diagnostics
- no data quality report
- no summary artifacts that directly support Part 3 benchmarking

So the main reasoning here is:
- keep it as a lightweight visualization branch
- do not treat it as a full replacement for the original Part 1 EDA pipeline

### Group `feature_engineering_merged.py`

Source file:
- [`../../code/feature_engineering_merged.py`](../../code/feature_engineering_merged.py)

What it does:
- loads labeled pixels using `utils_npz`
- builds standard engineered features:
  - radiance aggregates
  - front vs aft contrasts
  - simple ratios and gaps
  - interactions such as `ndai_x_sd`
- adds a new angle-pair feature family:
  - `ndai_cf_an`
  - `ndai_bf_an`
  - `ndai_af_an`
  - `ndai_df_af`
  - `ndai_angular_range`
  - `ndai_angular_std`
  - `cov_df_an`
  - `cov_cf_af`
  - `cov_df_bf`
  - `sd_proxy_*`
  - `sd_angular_range`
  - `sd_angular_cv`
- builds local `3x3` features with reflect padding and finite-value fallback
- screens features using:
  - symmetric AUC
  - mutual information
  - Cohen's d
  - per-image AUC

Interpretation:
- this script is much more substantial than the group EDA script
- the new contribution is mainly in the angle-pair feature family and the more edge-stable local-feature construction
- it is still a univariate screening script, not a modeling script

So the main reasoning here is:
- the merged Part 2 script gives us new candidate predictors
- the only fair way to judge whether they matter is to plug them into the same Part 3 model-selection pipeline

## Why We Built An Isolated Directory

Isolated directory:
- [`../`](../)

Reasoning:
- we did not want to break the main `lab2` workflow
- the raw merged scripts had path assumptions and schema assumptions that were incompatible with the current pipeline
- Part 3 expects a specific feature-table contract, especially the uppercase feature names and the selected feature-block definitions

So instead of overwriting the main pipeline, we:
- copied the current Part 3 code into the isolate
- adapted the merged Part 1 and Part 2 scripts there
- fixed all paths so they point to the original lab data and isolated output folders
- preserved the original pipeline untouched

This makes the comparison cleaner:
- same data
- same Part 3 training logic
- only the Part 2 feature source changes

## What Had To Be Adapted

### Utilities

Added local helper:
- [`../code/utils_npz.py`](../code/utils_npz.py)

Reason:
- the raw merged scripts depended on `utils_npz.py`
- that loader was not already wired into the reorganized `code/part1` and `code/part2` structure
- we needed a stable loader that can find the original `data/image_data` directory from the isolate

### Part 1 adaptation

Adapted file:
- [`../code/part1/eda.py`](../code/part1/eda.py)

Reasoning:
- keep the merged EDA behavior intact
- write outputs into `results/part1/eda/merged_visuals`
- remove reliance on GUI display and make it headless-safe

### Part 2 adaptation

Adapted file:
- [`../code/part2/feature_engineering.py`](../code/part2/feature_engineering.py)

Reasoning:
- preserve the merged feature logic
- but emit a Part 3-compatible engineered table
- keep the original selected feature names used by the downstream blocks:
  - `SD`, `NDAI`, `AF`, `AN`, `BF`
  - `ndai_x_sd`, `af_df_gap`, `front_back_ratio`, `rad_cv`, `rad_range`
  - local features used in `B2_context`
- add the merged angle-pair family on top of that

This adaptation is the key step. Without it, the merged script and the Part 3 pipeline would not connect cleanly.

### Part 3 reuse

Reused files:
- [`../code/part3/dataset.py`](../code/part3/dataset.py)
- [`../code/part3/train_models.py`](../code/part3/train_models.py)
- [`../code/part3/predict_unlabeled.py`](../code/part3/predict_unlabeled.py)

Reasoning:
- Part 3 should stay fixed during the comparison
- otherwise we would not know whether any change in results came from the new features or from a changed training pipeline

Only path fixes and light logging changes were made.

## Differences Against The Original Pipeline

### Part 1

Original Part 1:
- [`../../code/part1/eda.py`](../../code/part1/eda.py)

Merged isolate Part 1:
- [`../code/part1/eda.py`](../code/part1/eda.py)

Main difference:
- original Part 1 is a full assignment-grade EDA workflow
- merged Part 1 is a targeted visual sanity-check workflow

Conclusion:
- the merged Part 1 script is useful, but it is narrower and should be treated as supplementary

### Part 2

Original Part 2:
- [`../../code/part2/feature_engineering.py`](../../code/part2/feature_engineering.py)

Merged isolate Part 2:
- [`../code/part2/feature_engineering.py`](../code/part2/feature_engineering.py)

Main differences:
- original uses the reorganized uppercase schema directly
- merged raw version started from lowercase `utils_npz` columns and had to be standardized
- merged adds the angle-pair family
- merged uses reflect padding instead of constant-NaN padding for local features
- merged screening reports richer per-feature diagnostics

Reasoning impact:
- these are real feature changes, not just cosmetic refactors
- so it was worth testing them through Part 3 rather than judging them by screening alone

## What We Ran

1. merged Part 1 EDA
2. merged Part 2 engineered features
3. existing Part 2 AE embeddings
4. existing Part 3 model training
5. unlabeled sanity prediction

Important note:
- this run used the current Part 3 default `fast` search mode
- so the pipeline is complete, but the tuning breadth is still preliminary
- for a final benchmark run, rerun Part 3 with `--search_mode full`

## Preliminary Results

Main ranking file:
- [`../results/part3/model_selection_summary.csv`](../results/part3/model_selection_summary.csv)

Selected config:
- [`../results/part3/selected_model_config.json`](../results/part3/selected_model_config.json)

Best merged result:
- `HistGradientBoosting + B3_context_ae`
- mean ROC AUC `0.954720`
- mean PR AUC `0.907329`
- mean balanced accuracy `0.900115`

Original best result for comparison:
- mean ROC AUC `0.954256`
- mean PR AUC `0.905172`
- mean balanced accuracy `0.900345`

Interpretation:
- the winner did not change
- the merged features give a tiny gain in ROC AUC and PR AUC
- balanced accuracy is essentially unchanged
- so the merged Part 2 changes look directionally helpful, but not transformative

This is the most important conclusion of the test run:
- the merged Part 2 ideas appear compatible with the existing modeling pipeline
- but they do not change the overall modeling story

## Which New Features Actually Surfaced

Merged screening output:
- [`../results/part2/feature_screening.csv`](../results/part2/feature_screening.csv)

Top merged features still include the strong existing signals:
- `ndai_x_sd`
- `local_SD_mean3`
- `SD`
- `local_NDAI_std3`
- `local_SD_std3`

Main new merged features that entered near the top:
- `cov_df_bf`
- `ndai_angular_range`
- `ndai_df_af`
- `ndai_angular_std`

Interpretation:
- the merged angle-pair family is not noise
- it does surface as competitive screening signal
- but the downstream model still benefits most from the established context-plus-AE block

## Unlabeled Sanity Check

Summary file:
- [`../results/part3/unlabeled_predictions/cloud_fraction_summary.csv`](../results/part3/unlabeled_predictions/cloud_fraction_summary.csv)

Predicted cloud fractions:
- `O002539`: `0.9636`
- `O045178`: `0.6727`
- `O119738`: `0.6683`

Interpretation:
- the merged-feature final model produces stable unlabeled outputs and does not fail operationally
- this matters because some feature changes can look fine on labeled splits but break when pushed to full-image inference

## Recommendation

Short version:
- keep the current main Part 3 pipeline
- keep the merged Part 2 ideas as a tested alternative branch
- if the group wants a final unified Part 2 script, the angle-pair family is the main contribution worth carrying forward

What I would tell the team:
- the merged Part 1 EDA script is useful as supplementary visualization, not as the full EDA replacement
- the merged Part 2 script contains legitimate new feature ideas
- the downstream gain is real but small in the current `fast` search run
- before adopting it as the default branch, run the same comparison again with `--search_mode full`
