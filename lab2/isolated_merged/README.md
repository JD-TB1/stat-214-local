# Lab 2 Isolated Merged Pipeline

This directory is a sandboxed copy of the Lab 2 workflow used to test the group-merged Part 1 and Part 2 code without changing the main pipeline in [`../`](../).

The purpose of this isolate is to:
- keep the original `code/part1`, `code/part2`, and `code/part3` workflow intact
- adapt the group-merged Part 1 and Part 2 scripts into a reproducible structure
- run the existing Part 3 modeling pipeline on the merged Part 2 feature table
- compare the merged-feature results against the original baseline

## Contents

Core scripts:
- [`code/part1/eda.py`](code/part1/eda.py): adapted version of the group-merged EDA script
- [`code/part2/feature_engineering.py`](code/part2/feature_engineering.py): adapted version of the group-merged Part 2 feature script
- [`code/part2/autoencoder_features.py`](code/part2/autoencoder_features.py): reused AE embedding script with path fixes only
- [`code/part3/train_models.py`](code/part3/train_models.py): reused Part 3 training pipeline
- [`code/part3/predict_unlabeled.py`](code/part3/predict_unlabeled.py): unlabeled sanity-check scoring
- [`code/utils_npz.py`](code/utils_npz.py): local helper so the merged scripts can load `.npz` files reliably

Reference inputs from the main repo:
- [`../code/eda.py`](../code/eda.py): raw group-merged Part 1 script
- [`../code/feature_engineering_merged.py`](../code/feature_engineering_merged.py): raw group-merged Part 2 script
- [`../code/utils_npz.py`](../code/utils_npz.py): original utility file the merged scripts depended on

Outputs and notes:
- [`results/part1`](results/part1)
- [`results/part2`](results/part2)
- [`results/part3`](results/part3)
- [`documents/part2/predictor_catalog.md`](documents/part2/predictor_catalog.md)
- [`documents/part3/README.md`](documents/part3/README.md)
- [`documents/merged_test_run_meeting_summary.md`](documents/merged_test_run_meeting_summary.md)

## What Changed Relative To The Main Pipeline

Part 1:
- the raw merged [`../code/eda.py`](../code/eda.py) is a narrow visual sanity-check script
- it plots radiance-angle images and a few scatter comparisons
- it does not recreate the broader original EDA outputs such as feature rankings, split diagnostics, or data-quality summaries
- in this isolate, it is treated as a supplementary visualization branch rather than a replacement for the main Part 1 analysis

Part 2:
- the raw merged [`../code/feature_engineering_merged.py`](../code/feature_engineering_merged.py) uses the `utils_npz` lowercase schema such as `ndai`, `sd`, and `ra_df`
- the main pipeline [`../code/part2/feature_engineering.py`](../code/part2/feature_engineering.py) uses the standardized uppercase schema expected by Part 3
- the isolated merged Part 2 script preserves the merged logic but writes a Part 3-compatible table
- the main merged additions are angle-pair features, reflect-padded local features with edge-safe fallbacks, and richer feature screening with AUC, MI, Cohen's d, and per-image AUC

Part 3:
- the Part 3 training and evaluation logic was intentionally reused so the comparison stays fair
- this isolate answers one question: if we swap in the merged Part 2 features, does the downstream model-selection outcome materially change

## How To Reproduce

Run from this directory with `env_214`.

1. Part 1 merged EDA
```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp/.cache \
/opt/homebrew/Caskroom/miniforge/base/envs/env_214/bin/python code/part1/eda.py
```

2. Part 2 merged engineered features
```bash
PYTHONDONTWRITEBYTECODE=1 \
/opt/homebrew/Caskroom/miniforge/base/envs/env_214/bin/python code/part2/feature_engineering.py
```

3. Part 2 autoencoder embeddings
```bash
PYTHONDONTWRITEBYTECODE=1 \
/opt/homebrew/Caskroom/miniforge/base/envs/env_214/bin/python code/part2/autoencoder_features.py
```

4. Part 3 training
```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp/.cache LOKY_MAX_CPU_COUNT=1 \
/opt/homebrew/Caskroom/miniforge/base/envs/env_214/bin/python code/part3/train_models.py
```

5. Part 3 unlabeled sanity check
```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp/.cache \
/opt/homebrew/Caskroom/miniforge/base/envs/env_214/bin/python code/part3/predict_unlabeled.py
```

## How To Interpret The Current Test Run

This isolate reused the current Part 3 script in its default `fast` search mode, not the exhaustive `full` search. The pipeline is complete and the comparison is meaningful, but the tuning breadth is still preliminary.

Main result:
- the winning configuration did not change
- the best model is still `HistGradientBoosting + B3_context_ae`
- the merged Part 2 features produced a very small improvement in mean ROC AUC and PR AUC, but not a meaningful change in the overall modeling conclusion

Comparison against the main pipeline:
- original best mean ROC AUC: `0.954256`
- merged-feature best mean ROC AUC: `0.954720`
- original best mean PR AUC: `0.905172`
- merged-feature best mean PR AUC: `0.907329`
- original best mean balanced accuracy: `0.900345`
- merged-feature best mean balanced accuracy: `0.900115`

Interpretation:
- the merged Part 2 additions seem useful enough to survive downstream evaluation
- the improvement is incremental, not structural
- the autoencoder-augmented block `B3_context_ae` still clearly outperforms the simpler blocks, so the overall Part 3 story remains the same
- because the winner is unchanged, the safest conclusion is that the merged Part 2 feature ideas are compatible with the existing pipeline, but they do not overturn the previous modeling choice

Part 2 screening takeaways:
- the top merged screening features still include strong existing terms such as `ndai_x_sd`, `local_SD_mean3`, and `SD`
- the main new merged entrants near the top are `cov_df_bf`, `ndai_angular_range`, and `ndai_df_af`
- this suggests the angle-pair family adds modest incremental signal beyond the original engineered set

Unlabeled sanity outputs:
- [`results/part3/unlabeled_predictions/cloud_fraction_summary.csv`](results/part3/unlabeled_predictions/cloud_fraction_summary.csv)
- predicted cloud fractions for the current best merged model are approximately `0.964` for `O002539`, `0.673` for `O045178`, and `0.668` for `O119738`

## Recommended Use

Use this isolate when you want to:
- inspect the group-merged Part 1 and Part 2 logic without disturbing the main lab workflow
- compare merged features against the original Part 3 benchmark
- prepare a cleaner final merged script later if the group decides to adopt the new angle-pair features

Do not treat this isolate as the canonical lab directory. The main working pipeline is still [`../`](../).
