# Part 2 Predictor Catalog

## Meeting-ready shortlist

Recommended predictors to discuss first:
- `ndai_x_sd` (engineered_scalar): separation AUC=0.944, MI=0.428, |d|=1.134
- `local_SD_mean3` (patch_local): separation AUC=0.936, MI=0.450, |d|=1.289
- `SD` (expert): separation AUC=0.935, MI=0.428, |d|=1.170
- `local_NDAI_std3` (patch_local): separation AUC=0.927, MI=0.360, |d|=1.109
- `local_SD_std3` (patch_local): separation AUC=0.916, MI=0.386, |d|=0.848
- `local_rad_std_std3` (patch_local): separation AUC=0.916, MI=0.323, |d|=1.208
- `local_rad_mean_std3` (patch_local): separation AUC=0.889, MI=0.289, |d|=0.820
- `local_NDAI_mean3` (patch_local): separation AUC=0.830, MI=0.267, |d|=1.463

## Predictor families and why they matter

- `expert`: physically motivated features from the paper/instructions, especially NDAI and SD.
- `radiance`: raw multi-angle signal that may pick up cloud altitude and angular response.
- `engineered_scalar`: low-cost combinations of radiance channels and expert features.
- `patch_local`: neighborhood texture and local-context summaries that can capture cloud smoothness / edges.

## Top candidates by family

### Expert features
- `SD`: AUC=0.935, MI=0.428, |d|=1.170
- `NDAI`: AUC=0.823, MI=0.259, |d|=1.415
- `CORR`: AUC=0.524, MI=0.010, |d|=0.115
### Radiance features
- `AF`: AUC=0.799, MI=0.201, |d|=1.156
- `AN`: AUC=0.794, MI=0.196, |d|=1.152
- `BF`: AUC=0.770, MI=0.169, |d|=0.979
- `DF`: AUC=0.554, MI=0.123, |d|=0.127
- `CF`: AUC=0.643, MI=0.087, |d|=0.539
### Engineered scalar features
- `ndai_x_sd`: AUC=0.944, MI=0.428, |d|=1.134
- `rad_cv`: AUC=0.825, MI=0.260, |d|=1.389
- `front_back_ratio`: AUC=0.801, MI=0.238, |d|=1.125
- `af_df_gap`: AUC=0.794, MI=0.258, |d|=1.500
- `rad_range`: AUC=0.775, MI=0.232, |d|=1.366
### Patch-local features
- `local_SD_mean3`: AUC=0.936, MI=0.450, |d|=1.289
- `local_NDAI_std3`: AUC=0.927, MI=0.360, |d|=1.109
- `local_SD_std3`: AUC=0.916, MI=0.386, |d|=0.848
- `local_rad_std_std3`: AUC=0.916, MI=0.323, |d|=1.208
- `local_rad_mean_std3`: AUC=0.889, MI=0.289, |d|=0.820

## Suggested Part 2 feature engineering methods

- Keep `NDAI`, `SD`, and the strongest angle radiances as baseline predictors.
- Add multi-angle contrasts such as `an_df_gap`, `af_df_gap`, and `forward_backward_gap` to expose altitude-sensitive angular differences.
- Add aggregate radiance summaries like `rad_mean`, `rad_std`, `rad_range`, and `rad_cv` to capture brightness and texture at the pixel level.
- Add local 3x3 neighborhood summaries like `local_NDAI_mean3`, `local_SD_std3`, and `local_rad_std_mean3`-style terms to encode context around each pixel.
- Use autoencoder embeddings as a second feature family rather than replacing the interpretable predictors.
