# STAT 214 Lab 2 Part 1 (EDA) - Annotated Meeting Summary

## 1. Why Part 1 exists
Part 1 is not just a plotting step. It exists to answer four practical questions before we start feature engineering or classification:
- what the labeled data actually looks like in space
- whether cloud and non-cloud are separable in the available variables
- how we should split the data so evaluation is realistic
- whether there are data quality problems that would quietly break later modeling

The main logic is:
- if we do not understand class balance, spatial structure, and feature behavior first, then any Part 2 or Part 3 result could be misleading
- EDA is where we decide what is signal, what is possible leakage, and what is likely to generalize

## 2. Sufficiency decision
**Decision: yes, the current `code/part1/eda.py` is sufficient for Part 1 EDA**, with two caveats that should still be stated clearly in the meeting or report:
- we still need to commit to a final split protocol for downstream modeling
- the valid range for `CORR` should be treated carefully, because this dataset representation allows negative values

Why this matters:
- Part 1 is considered complete when it does more than produce figures; it must support downstream decisions
- the current EDA already does that by giving label maps, feature separability checks, split diagnostics, and quality diagnostics
- the remaining caveats are about interpretation and final policy, not about missing core analysis

## 3. Cleaning status and why it was handled conservatively
In the original EDA, cleaning was mostly diagnostic. That was the right first step, because aggressive cleaning too early can hide real data issues or silently change the analysis population.

We later added a dedicated minimal-cleaning script:
- `code/part1/clean_lab2.py`

What it does:
- keeps only valid labels in `{-1, 0, +1}`
- removes non-finite rows across required variables
- enforces conservative physical bounds
- drops duplicate `(image_id, x, y)` rows only if they exist

Why the cleaning is intentionally conservative:
- the goal is to remove clearly invalid rows, not to "improve" the data by subjective filtering
- at this stage, we want reproducibility and defensibility more than aggressive preprocessing
- this keeps Part 3 modeling grounded in a dataset that still resembles the original lab input

Cleaning outputs, when generated:
- `results/part1/cleaning/labeled_cleaned_<image_id>.csv`
- `results/part1/cleaning/labeled_cleaned_all.csv`
- `results/part1/cleaning/labeled_supervised_all.csv`
- `results/part1/cleaning/cleaning_report.csv`
- `results/part1/cleaning/cleaning_summary.json`

## 4. How Part 1 maps to the assignment requirements
Part 1 of the lab effectively asks us to do four things, and each one serves a separate purpose.

### 4.1 Plot expert labels on the x/y map for the three labeled images
Done.

Files include:
- `label_map_O013257.png`
- `label_map_O013490.png`
- `label_map_O012791.png`

Reasoning:
- before talking about statistics, we need to see where cloud, non-cloud, and unlabeled pixels are located
- this reveals whether labels are spatially clustered, whether unlabeled regions are large, and whether the images differ a lot from each other
- that visual structure is crucial for deciding whether random pixel splits are trustworthy

### 4.2 Explore radiance relationships and class differences
Done.

Files include:
- `radiance_corr_*.png`
- `radiance_pairs_*.png`
- `feature_dist_<feature>_*.png`
- `feature_ranking*.csv`

Reasoning:
- this tells us whether any variables actually separate cloud from non-cloud
- it also tells us whether the physically motivated paper features are useful in our dataset
- it gives the first evidence for which variables are worth carrying into Part 2

### 4.3 Design train/validation/test splits and justify realism
Done as a diagnostic and proposal stage.

Files include:
- `results/part1/eda/split_diagnostics.csv`
- `documents/part1/split_notes.md`

Reasoning:
- the way we split the data determines whether our later accuracy estimates are believable
- a split is not just a technical detail; it encodes what kind of future deployment scenario we think the model should face
- because this is spatial image data, random pixel splits can create leakage through neighboring pixels

### 4.4 Identify real-world data imperfections
Done at the diagnostic level.

Files include:
- `results/part1/eda/data_quality_report.csv`
- `documents/part1/split_notes.md`

Reasoning:
- if there are missing, impossible, duplicated, or out-of-range values, later modeling results may fail silently or become hard to interpret
- the goal here is not just to clean, but to document what was checked and what assumptions are safe

## 5. Key EDA observations and how to explain the logic

### 5.1 Label composition varies a lot by image
- `O013257`: cloud 20,468 (17.8%), non-cloud 50,358 (43.8%), unlabeled 44,174 (38.4%)
- `O013490`: cloud 39,253 (34.1%), non-cloud 42,830 (37.2%), unlabeled 32,949 (28.6%)
- `O012791`: cloud 21,244 (18.5%), non-cloud 33,528 (29.2%), unlabeled 60,201 (52.4%)

Why this matters:
- the class balance is not stable from one image to another
- that means a model trained in one scene may face a different class mix in another scene
- this is one reason we should care about generalization across scenes, not just pooled performance

How to explain it in a meeting:
- the three images are not interchangeable samples from the same simple distribution
- because class prevalence shifts across images, evaluation should test robustness to scene-level change

### 5.2 Pooled feature separation shows strong signal in a few variables
Top pooled features by absolute effect size:
1. `NDAI`: Cohen's d = 1.415, AUC = 0.823
2. `SD`: Cohen's d = 1.170, AUC = 0.935
3. `AF`: Cohen's d = -1.156, AUC = 0.799
4. `AN`: Cohen's d = -1.152, AUC = 0.794
5. `BF`: Cohen's d = -0.979, AUC = 0.770

Why this matters:
- it shows the problem is not hopeless; the classes are meaningfully separated in several variables
- `NDAI` and `SD` support the paper's physically motivated feature logic
- the angular radiances also carry signal, which suggests the multi-angle geometry is informative

How to explain the metrics:
- AUC tells us how well a single variable separates cloud from non-cloud over all thresholds
- Cohen's d tells us how large the class difference is in standardized units
- together they show both ranking strength and effect magnitude

### 5.3 The "best" features are not identical in every image
- `O013257` top-3: `CF`, `BF`, `DF`
- `O013490` top-3: `NDAI`, `SD`, `AF`
- `O012791` top-3: `NDAI`, `SD`, `DF`

Why this matters:
- a feature that looks strong in the pooled data may not be equally strong in every scene
- this is an early warning against overfitting our feature decisions to one image
- it also motivates Part 2 to keep multiple feature families instead of relying on one "winner"

How to explain it in the meeting:
- the basic cloud signal is real, but the exact ranking of variables changes by image
- so later modeling should include stability checks and realistic validation, not just pooled fitting

### 5.4 Split diagnostics show that realistic splits create real shift
Largest shift cases by mean absolute standardized mean difference:
- `spatial_within_image / O013490_x_gt_q80`: 0.692
- `by_image / holdout_O012791`: 0.610
- `spatial_within_image / O013257_x_gt_q80`: 0.608

Why this matters:
- a realistic split should be hard enough to reflect future use
- these large shifts show that by-image and contiguous spatial holdouts are genuinely different from the training data
- that is exactly why they are better tests of generalization than random pixels

How to explain the logic:
- if we split neighboring pixels randomly, train and test both see nearly the same structures
- that makes the test set too easy
- by-image and spatial holdout splits remove that shortcut and create a more honest estimate of model performance

### 5.5 Data quality diagnostics were reassuring, with one important nuance
- non-finite rows: 0 for each labeled image and pooled
- duplicate `(x, y)` within image: 0
- NDAI out-of-range `[-1, 1]`: 0
- negative radiances: 0
- `CORR` can be negative in this dataset

Why this matters:
- it means there is no evidence of major corruption in the labeled data
- we do not need aggressive rescue steps before proceeding
- the main nuance is that `CORR` should not be forced into `[0, 1]` without checking the data definition

How to explain it:
- the data appears clean enough for baseline modeling
- our cleaning policy should therefore be minimal and defensible, not heavy-handed

## 6. Why unlabeled pixels are treated differently
The dataset uses `label = 0` for unlabeled pixels.

Reasoning:
- these are not confirmed non-cloud pixels
- treating them as a class label would contaminate the supervised analysis
- for EDA accounting and visual maps they are useful, but for separability metrics they should be excluded

This is important to say explicitly, because otherwise cloud vs non-cloud comparisons would mix in unknown cases.

## 7. How this aligns with Yu et al. (2008)
The paper emphasizes `CORR`, `SD`, and `NDAI` as physically motivated features, and it treats multi-angle behavior as the key cloud-detection signal.

Why our EDA is aligned:
- we explicitly checked the same expert features instead of ignoring the paper's logic
- we found that `SD` and `NDAI` are indeed strong in this dataset
- we also found that some raw radiance angles are useful, which is consistent with the broader multi-angle reasoning
- we went beyond the paper by also evaluating split realism and spatial dependence, which is important for a modern reproducible workflow

## 8. Main conclusions to carry into Part 2 and Part 3
- `NDAI` and `SD` should stay central in the baseline feature set
- selected radiance channels should also be retained because they carry strong class signal
- random pixel splits are not credible as the main evaluation strategy
- by-image and/or contiguous spatial holdouts are better aligned with realistic deployment
- cleaning should remain conservative because the data quality is mostly good

This is the key logic chain:
- label maps showed spatial structure
- feature separability showed that useful predictors exist
- per-image variation showed that the signal is not perfectly stable
- split diagnostics showed random splitting would be optimistic
- quality checks showed that the dataset is clean enough for cautious downstream modeling

## 9. Suggested meeting framing
The cleanest way to explain Part 1 is:
- first, we visualized labels to understand spatial structure and unlabeled regions
- second, we measured which variables separate cloud from non-cloud
- third, we checked whether those relationships were stable across images
- fourth, we designed split strategies that avoid spatial leakage
- fifth, we verified that only minimal cleaning is needed

In one sentence:
Part 1 established that the cloud-detection task has real signal, real scene-to-scene variation, and real spatial dependence, so Part 2 and Part 3 must use strong predictors and realistic splits.

## 10. Required points to mention in the meeting or report
- Random pixel split is risky because neighboring pixels are autocorrelated.
- By-image and spatial holdout splits better mimic future deployment.
- `NDAI` and `SD` appear robustly informative, while radiance-angle importance varies by scene.
- Unlabeled pixels should remain outside supervised class-separation metrics.
- Cleaning should remove only clearly invalid rows and keep assumptions conservative.

## 11. Suggested final statement for group alignment
"Part 1 EDA is complete and reproducible. We now understand the label geometry, the strongest baseline predictors, the scene-to-scene shifts, the risks of spatial leakage, and the minimal cleaning rules. That gives us a solid foundation for Part 2 feature engineering and Part 3 modeling."
