# Part 2 Predictor Catalog

## Meeting-ready shortlist

Recommended predictors to discuss first:
- `ndai_x_sd` (engineered_scalar): symmetric AUC=0.944, MI=0.428, |d|=1.134
- `local_SD_mean3` (patch_local): symmetric AUC=0.936, MI=0.450, |d|=1.289
- `SD` (expert): symmetric AUC=0.935, MI=0.428, |d|=1.170
- `local_NDAI_std3` (patch_local): symmetric AUC=0.927, MI=0.360, |d|=1.109
- `local_SD_std3` (patch_local): symmetric AUC=0.916, MI=0.386, |d|=0.848
- `local_rad_std_std3` (patch_local): symmetric AUC=0.916, MI=0.323, |d|=1.208
- `local_rad_mean_std3` (patch_local): symmetric AUC=0.889, MI=0.289, |d|=0.820
- `cov_df_bf` (angle_pair): symmetric AUC=0.854, MI=0.296, |d|=1.331

## Predictor families and why they matter

- `expert`: paper-style features such as NDAI, SD, and CORR.
- `radiance`: raw multi-angle radiance channels.
- `engineered_scalar`: simple gaps, ratios, and interactions built from expert and radiance features.
- `angle_pair`: extended angular relationships, covariance-style proxies, and angular spread measures from the merged group script.
- `patch_local`: 3x3 neighborhood summaries that capture local texture and context.

## Top candidates by family

### Expert features
- `SD`: symmetric AUC=0.935, MI=0.428, |d|=1.170
- `NDAI`: symmetric AUC=0.823, MI=0.259, |d|=1.415
- `CORR`: symmetric AUC=0.524, MI=0.010, |d|=0.115
### Radiance features
- `AF`: symmetric AUC=0.799, MI=0.201, |d|=1.156
- `AN`: symmetric AUC=0.794, MI=0.196, |d|=1.152
- `BF`: symmetric AUC=0.770, MI=0.169, |d|=0.979
- `DF`: symmetric AUC=0.554, MI=0.123, |d|=0.127
- `CF`: symmetric AUC=0.643, MI=0.087, |d|=0.539
### Engineered scalar features
- `ndai_x_sd`: symmetric AUC=0.944, MI=0.428, |d|=1.134
- `rad_cv`: symmetric AUC=0.825, MI=0.260, |d|=1.389
- `df_an_ratio`: symmetric AUC=0.823, MI=0.259, |d|=1.153
- `af_df_gap`: symmetric AUC=0.794, MI=0.258, |d|=1.500
- `front_back_ratio`: symmetric AUC=0.801, MI=0.238, |d|=1.125
### Angle-pair features
- `cov_df_bf`: symmetric AUC=0.854, MI=0.296, |d|=1.331
- `ndai_angular_range`: symmetric AUC=0.837, MI=0.282, |d|=1.526
- `ndai_df_af`: symmetric AUC=0.833, MI=0.284, |d|=1.493
- `ndai_angular_std`: symmetric AUC=0.835, MI=0.279, |d|=1.509
- `sd_proxy_df`: symmetric AUC=0.808, MI=0.268, |d|=1.573
### Patch-local features
- `local_SD_mean3`: symmetric AUC=0.936, MI=0.450, |d|=1.289
- `local_NDAI_std3`: symmetric AUC=0.927, MI=0.360, |d|=1.109
- `local_SD_std3`: symmetric AUC=0.916, MI=0.386, |d|=0.848
- `local_rad_std_std3`: symmetric AUC=0.916, MI=0.323, |d|=1.208
- `local_rad_mean_std3`: symmetric AUC=0.889, MI=0.289, |d|=0.820
