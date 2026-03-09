# Part 2 README

This folder contains the Part 2 feature-engineering handoff for Lab 2. The goal of this pass was to produce a meeting-ready shortlist of predictors and feature-engineering methods, including both interpretable engineered features and autoencoder-derived latent features.

## Files created in this pass
### Scripts
- `code/part2/feature_engineering.py`
  - builds engineered predictors from the three labeled images
  - creates scalar contrasts and local 3x3 patch summaries
  - writes the main screening outputs below
- `code/part2/autoencoder_features.py`
  - uses `code/original/autoencoder.py` as the starting point for patch-based feature extraction
  - loads `code/original/checkpoints/gsi-model.ckpt`
  - extracts and screens latent embedding coordinates for supervised rows

### Main notes
- `results/part2/meeting_summary.md`
  - short summary for discussion
  - includes recommended predictor shortlist and suggested Part 2 methods
- `results/part2/predictor_catalog.md`
  - screening-based list of the strongest expert, radiance, engineered, and patch-local predictors
- `results/part2/autoencoder_feature_notes.md`
  - screening-based list of the strongest latent embedding dimensions

### Data outputs
- `results/part2/labeled_engineered_features.csv`
  - labeled dataset with raw + engineered + local-context features
- `results/part2/feature_screening.csv`
  - ranking table for classical / interpretable engineered predictors
- `results/part2/autoencoder_embeddings_supervised.csv`
  - latent coordinates for supervised rows only
- `results/part2/autoencoder_feature_screening.csv`
  - ranking table for autoencoder latent dimensions

## Recommended reading order
1. `results/part2/meeting_summary.md`
2. `results/part2/predictor_catalog.md`
3. `results/part2/autoencoder_feature_notes.md`
4. `results/part2/feature_screening.csv`
5. `results/part2/autoencoder_feature_screening.csv`

## How the scripts were run
Use the `env_214` Python directly.

Classical engineered predictors:
```bash
/opt/homebrew/Caskroom/miniforge/base/envs/env_214/bin/python code/part2/feature_engineering.py
```

Autoencoder-derived predictors:
```bash
/opt/homebrew/Caskroom/miniforge/base/envs/env_214/bin/python code/part2/autoencoder_features.py
```

## What to reuse next
For Part 3 modeling, the most promising starting blocks are:
- interpretable baseline: `SD`, `NDAI`, `AF`, `AN`, `BF`
- engineered scalar block: `ndai_x_sd`, `af_df_gap`, `front_back_ratio`, `rad_cv`, `rad_range`
- local-context block: `local_SD_mean3`, `local_NDAI_std3`, `local_SD_std3`
- autoencoder block: `ae4`, `ae5`, `ae0`

## Important caveat
These rankings are screening results, not the final modeling answer. Final feature selection still needs to be checked under the realistic split protocol from Part 1.
