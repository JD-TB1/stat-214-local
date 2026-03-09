# STAT 214 Lab 2 Part 1 (EDA) - Meeting Summary

## 1) Sufficiency decision
**Decision: YES, current `code/part1/eda.py` is sufficient for Part 1 EDA**, with two minor caveats to mention in the meeting/report:
- You still need to **state your final chosen split protocol** for downstream modeling (the script provides diagnostics for two strong options).
- The `CORR` range check may be too strict for this dataset representation; verify definition/scaling before final cleaning rules.

## 1.1 Cleaning status update
- In the original EDA, cleaning was mostly **diagnostic** (quality checks + excluding non-finite rows in plotting/stat summaries).
- We now added a dedicated minimal-cleaning script:
  - `code/part1/clean_lab2.py`
- It performs conservative cleaning aligned with lab expectations:
  - keep only valid labels in `{-1, 0, +1}`
  - remove non-finite rows across required columns
  - enforce conservative physical bounds:
    - `NDAI in [-1, 1]`
    - `CORR in [-1, 1]` (updated from earlier `[0,1]` assumption)
    - `SD >= 0`, radiances `>= 0`
  - drop duplicate `(image_id, x, y)` rows if present
- Outputs from cleaning run (when executed):
  - `results/part1/cleaning/labeled_cleaned_<image_id>.csv`
  - `results/part1/cleaning/labeled_cleaned_all.csv`
  - `results/part1/cleaning/labeled_supervised_all.csv`
  - `results/part1/cleaning/cleaning_report.csv`
  - `results/part1/cleaning/cleaning_summary.json`

## 2) Requirement checklist vs. outputs
Part 1 instruction items from `lab2-instructions.tex`:

1. Plot expert labels on X/Y map for 3 labeled images.
- Done.
- Files: `label_map_O013257.png`, `label_map_O013490.png`, `label_map_O012791.png`, plus labeled-only maps.

2. Explore radiance relationships and class differences (cloud vs no-cloud), including CORR/NDAI/SD.
- Done.
- Files: `radiance_corr_*.png`, `radiance_pairs_*.png`, `feature_dist_<feature>_*.png`, `feature_ranking*.csv`.
- Includes quantitative metrics: Cohen's d, AUC, mutual information.

3. Split into train/val/test (or train/test + CV) and justify future-use realism.
- Done as diagnostics/proposal.
- Files: `results/part1/eda/split_diagnostics.csv`, `documents/part1/split_notes.md`.
- Includes by-image and spatial holdout analyses, and distribution-shift metrics.

4. Identify/handle real-world data imperfections.
- Done at EDA level (diagnostic + recommendations).
- File: `results/part1/eda/data_quality_report.csv`.
- Cleaning recommendations documented in `documents/part1/split_notes.md`.

## 3) Key EDA observations to present

### Label composition (class balance varies by image)
- `O013257`: cloud 20,468 (17.8%), non-cloud 50,358 (43.8%), unlabeled 44,174 (38.4%).
- `O013490`: cloud 39,253 (34.1%), non-cloud 42,830 (37.2%), unlabeled 32,949 (28.6%).
- `O012791`: cloud 21,244 (18.5%), non-cloud 33,528 (29.2%), unlabeled 60,201 (52.4%).

Interpretation: strong image-to-image class prevalence shifts, so evaluation must stress generalization across scenes.

### Pooled feature separation (cloud vs non-cloud)
Top features by absolute effect size from `feature_ranking.csv`:
1. **NDAI**: Cohen's d = 1.415, AUC = 0.823.
2. **SD**: Cohen's d = 1.170, AUC = 0.935.
3. **AF**: Cohen's d = -1.156, AUC = 0.799.
4. **AN**: Cohen's d = -1.152, AUC = 0.794.
5. **BF**: Cohen's d = -0.979, AUC = 0.770.

Interpretation: NDAI/SD are highly discriminative; several angular radiances (especially AF/AN/BF) also provide strong class signal.

### Per-image heterogeneity in "best" features
- `O013257` top-3: CF, BF, DF.
- `O013490` top-3: NDAI, SD, AF.
- `O012791` top-3: NDAI, SD, DF.

Interpretation: feature utility is not perfectly stable across images; this reinforces robust split strategy and model stability checks.

### Split diagnostics (distribution shift)
Largest shift cases by mean abs standardized mean difference (abs-SMD):
- `spatial_within_image / O013490_x_gt_q80`: **0.692**
- `by_image / holdout_O012791`: **0.610**
- `spatial_within_image / O013257_x_gt_q80`: **0.608**

Interpretation: realistic splits induce substantial shift; random pixel split would likely be optimistic due to spatial autocorrelation.

### Data quality diagnostics
- Non-finite rows: 0 for each labeled image and pooled.
- Duplicate `(x,y)` within image: 0 (good).
- NDAI out-of-range `[-1,1]`: 0.
- Negative radiances: 0.
- CORR can be negative in this dataset; use `[-1,1]` as the physically valid range for cleaning checks.

## 4) How this aligns with Yu et al. (2008)
From `yu2008.pdf` (extracted text):
- The paper emphasizes three physically motivated features: **CORR, SD, NDAI** for separating cloud-free surface from cloud.
- It highlights multi-angle behavior as the key signal and supports threshold/discriminant-style pipelines.

Your EDA aligns well:
- You explicitly quantified separability for NDAI/SD/CORR and angular radiances.
- You observed strong predictive value for NDAI/SD, consistent with the paper's feature rationale.
- You documented split realism and spatial dependence concerns, which matches operational generalization concerns in MISR cloud detection.

## 5) Required comments to include in your report/meeting
- Random pixel split is risky because neighboring pixels are autocorrelated and can leak spatial patterns.
- Use by-image and/or contiguous spatial holdout to better mimic future deployment.
- NDAI and SD appear robustly informative across images; radiance-angle contributions vary by scene.
- Treat unlabeled pixels (`label=0`) as unknown for supervision; use only +/-1 for class-separation metrics.
- Use minimal, conservative cleaning:
  - remove non-finite rows
  - enforce physical bounds with `CORR in [-1,1]` (not `[0,1]`)
  - check and handle duplicate coordinates only if present
  - avoid aggressive transformations before baseline modeling.

## 6) Suggested final statement for group alignment
"Part 1 EDA is complete and reproducible. We have label maps, class-conditioned feature analyses, quantitative ranking metrics, split-shift diagnostics, and data-quality checks. The EDA supports using NDAI/SD plus selected radiance features, and recommends by-image/spatial splits for reliable generalization estimates in Part 2/3."
