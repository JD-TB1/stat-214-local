# STAT 214 Lab 2 Part 2 - Annotated Meeting Summary

## 1. Why Part 2 exists
Part 1 told us what the data looks like and where the main difficulties are. Part 2 exists because raw columns alone are usually not the best final model inputs. We need a step that turns the raw MISR measurements into a predictor set that is both:
- scientifically defensible enough to explain in the lab report
- flexible enough to give Part 3 classifiers stronger signal than the original columns alone

The underlying logic is:
- cloud detection here is not just a brightness problem
- snow, ice, and clouds can all look bright, so single-pixel raw intensity can be ambiguous
- MISR is valuable because it measures the same location from multiple angles
- that means the most useful predictors may come from angular contrasts, local neighborhood structure, or learned patch patterns rather than from one raw value by itself

So the goal of this pass was to create a meeting-ready shortlist of feature candidates before classifier work begins.

## 2. Overall strategy and reasoning
This pass deliberately tested four predictor families, because each family answers a different question.

### 2.1 Expert features from the paper/instructions
These include `NDAI`, `SD`, and `CORR`.

Reasoning:
- these were already proposed by the lab or paper, so they are the most defensible starting point
- if they perform well, we can say our results are consistent with domain knowledge
- if they perform poorly, that is also useful because it tells us where simple expert features may be insufficient for this dataset

### 2.2 Raw multi-angle radiances
These include `DF`, `CF`, `BF`, `AF`, and `AN`.

Reasoning:
- even if engineered features are useful, we still need to know which raw viewing angles carry the most direct signal
- this helps us understand which parts of the sensor geometry matter most
- it also gives us a simple baseline feature block for Part 3

### 2.3 Hand-engineered scalar features
These are low-cost combinations like differences, ratios, interactions, and brightness-variability summaries.

Reasoning:
- if clouds and surface differ mainly through multi-angle behavior, then contrasts and ratios may isolate that behavior better than raw channels
- if one expert feature captures one aspect of cloud signal and another captures a different aspect, then an interaction term may combine them more effectively
- these features are still interpretable, which is important for explaining the report

### 2.4 Patch-based features
These include local 3x3 summaries and autoencoder latent features from `code/original/autoencoder.py`.

Reasoning:
- a pixel is not independent of its neighbors; clouds usually appear as spatial structures
- local summaries test whether neighborhood smoothness, variation, or texture helps
- autoencoder embeddings test whether a learned representation of a larger patch can capture patterns that hand-designed summaries miss

## 3. What the two scripts do and why both are needed

### 3.1 `code/part2/feature_engineering.py`
This script handles the interpretable side of Part 2.

What it does:
- loads the 3 labeled images
- keeps the main expert and raw radiance variables
- creates engineered scalar features such as products, angle gaps, ratios, and brightness summaries
- creates 3x3 local neighborhood summaries such as local means and local standard deviations
- screens each candidate feature with AUC, mutual information, and Cohen's d
- writes:
  - `results/part2/labeled_engineered_features.csv`
  - `results/part2/feature_screening.csv`
  - `documents/part2/predictor_catalog.md`

Why it matters:
- this script gives us the explainable feature set
- it lets us say not only which variables were strong, but also why we constructed them
- it is the main bridge from Part 1 EDA to Part 3 supervised modeling

### 3.2 `code/part2/autoencoder_features.py`
This script handles the learned patch-representation side of Part 2.

What it does:
- uses `code/original/autoencoder.py` and the existing checkpoint
- builds 9x9 patches around supervised pixels
- passes those patches through the encoder
- extracts latent coordinates for each supervised pixel
- screens those latent coordinates with the same metrics as above
- writes:
  - `results/part2/autoencoder_embeddings_supervised.csv`
  - `results/part2/autoencoder_feature_screening.csv`
  - `documents/part2/autoencoder_feature_notes.md`

Why it matters:
- this script tests whether a learned representation of local structure is useful
- it uses the GSI-provided autoencoder instead of starting from scratch, so it fits the assignment requirement and keeps the workflow reproducible
- it gives us a second feature family that may improve Part 3 even when the learned variables are not directly interpretable

## 4. How the screening metrics should be explained
The feature screening in this pass is univariate. That means each predictor is tested on its own, not inside a full classifier yet.

### 4.1 AUC
AUC here is the ROC AUC of a single feature used as a score for cloud vs non-cloud.

Reasoning:
- AUC asks how well one feature separates the two classes across all possible thresholds
- it is useful because it does not depend on picking one cutoff in advance
- it is especially convenient for ranking candidate predictors before model fitting

How it was computed:
- keep supervised rows only
- convert the labels to binary cloud vs non-cloud
- use one feature column as the numeric score
- compute `roc_auc_score`

Important note:
- some features are strong but in the opposite direction
- for screening, we care about separation strength more than sign
- that is why we report separation AUC, effectively `max(AUC, 1 - AUC)`

### 4.2 Mutual information
Reasoning:
- AUC mainly captures rank-based class separation
- mutual information can pick up nonlinear dependence that may not show up as a simple monotone separation
- using it alongside AUC reduces the risk of over-trusting one metric

### 4.3 Cohen's d
Reasoning:
- Cohen's d measures the standardized mean difference between cloud and non-cloud
- it gives a more direct sense of effect size
- it is helpful when explaining whether the class difference is practically large, not just statistically detectable

## 5. Main findings and how to explain the logic

### 5.1 Expert / baseline predictors
- `SD`: AUC 0.935, MI 0.428
- `NDAI`: AUC 0.823, MI 0.259
- `CORR`: AUC 0.524, MI 0.010

Reasoning and interpretation:
- `SD` is the strongest single interpretable baseline, so it should remain central in the Part 2 story
- `NDAI` is weaker than `SD` but still clearly useful, so it belongs in the baseline feature block
- `CORR` is weak in this screen, which means we should not feature it as a headline predictor unless it later helps in combinations or multivariate modeling

This is important for the meeting because it shows that we did not blindly keep all paper features equally. We checked them and retained the ones that actually separate the classes in our data.

### 5.2 Raw radiance predictors
- `AF`: separation AUC 0.799
- `AN`: separation AUC 0.794
- `BF`: separation AUC 0.770

Reasoning and interpretation:
- the aft/nadir-side views carry noticeably more direct signal than `DF` and `CF`
- this suggests that the geometry of the viewing angle is genuinely informative for distinguishing cloud from surface
- therefore, the raw-radiance baseline for Part 3 should not include all angles equally by default; it should begin with the strongest ones

### 5.3 Engineered scalar predictors
- `ndai_x_sd`: AUC 0.944, MI 0.428
- `rad_cv`: AUC 0.825, MI 0.260
- `front_back_ratio`: AUC 0.801, MI 0.238
- `af_df_gap`: separation AUC 0.794, |d| 1.500
- `rad_range`: AUC 0.775, MI 0.232

Reasoning and interpretation:
- `ndai_x_sd` is the product of `NDAI` and `SD`
  - logic: if each feature captures a different part of the cloud signal, their interaction may amplify pixels where both signals are present
  - result: it slightly outperforms either feature alone, so the interaction is worth keeping
- `rad_cv` is the coefficient of variation across radiance channels
  - logic: clouds and surfaces may differ not only in brightness but also in how variable the angular measurements are
  - result: a cheap summary of cross-angle variability is competitive
- `front_back_ratio` compares mean forward-like and aft-like radiances
  - logic: clouds may respond differently across viewing directions than surface backgrounds do
  - result: a single interpretable ratio captures that asymmetry
- `af_df_gap` is the difference between two angles
  - logic: angle-to-angle gaps directly probe the multi-angle behavior that MISR is designed to measure
  - result: a large effect size makes it useful even when its raw AUC is not as high as the best expert features
- `rad_range` summarizes brightness spread across angles
  - logic: larger or smaller across-angle spread may correspond to different physical structures
  - result: another low-cost feature worth retaining

The overall lesson is that engineered contrasts and interactions are justified because the problem is multi-angle by nature.

### 5.4 Patch-local predictors
- `local_SD_mean3`: AUC 0.936, MI 0.450
- `local_NDAI_std3`: AUC 0.927, MI 0.360
- `local_SD_std3`: AUC 0.916, MI 0.386
- `local_rad_std_std3`: AUC 0.916, MI 0.323
- `local_rad_mean_std3`: AUC 0.889, MI 0.289

Reasoning and interpretation:
- `local_SD_mean3` is the mean `SD` value in a 3x3 neighborhood
  - logic: a pixel inside a cloud patch should often be surrounded by similar values
  - result: local context helps at least as much as the center pixel alone
- `local_NDAI_std3` is the neighborhood standard deviation of `NDAI`
  - logic: cloud edges and textured regions may differ from smoother background regions
  - result: neighborhood texture is informative
- `local_SD_std3` measures local variation in `SD`
  - logic: this captures whether the patch is homogeneous or locally changing
  - result: it supports the claim that spatial structure matters
- the `local_rad_*` terms extend the same idea to brightness summaries
  - logic: neighborhood variation in brightness-based quantities can reveal cloud fields and boundaries
  - result: local summaries consistently compete with strong single-pixel features

This is one of the most important Part 2 conclusions: neighborhood information is not optional decoration. It is a major source of signal.

### 5.5 Autoencoder latent predictors
From the checkpoint-backed embedding screen:
- `ae4`: AUC 0.897, MI 0.306
- `ae5`: AUC 0.867, MI 0.230
- `ae0`: AUC 0.845, MI 0.216
- `ae3`: AUC 0.799, MI 0.159
- `ae1`: AUC 0.779, MI 0.174

Reasoning and interpretation:
- these features come from a learned compression of local 9x9 patches
- each `ae*` coordinate is not directly interpretable like `SD` or `AF`
- however, their performance shows that the encoder is capturing useful local structure

How to explain them in the meeting:
- we should not pretend to know the exact physical meaning of `ae4` or `ae5`
- the correct claim is that they are learned summaries of patch shape, texture, and multi-channel structure
- because they perform well, they are good candidates to add as complementary predictors
- because they are less interpretable, they should supplement the expert and engineered features rather than replace them

## 6. Shortlist with plain-language explanations
If the meeting needs a concise shortlist, discuss these first.

- `SD`
  - strongest interpretable expert feature; good baseline anchor
- `NDAI`
  - useful expert feature that should stay in the baseline set
- `AF`
  - one of the strongest raw angle radiances
- `AN`
  - another strong raw angle radiance
- `ndai_x_sd`
  - interaction feature showing that combining expert signals improves separation
- `af_df_gap`
  - angle contrast that directly uses MISR's multi-angle design
- `front_back_ratio`
  - compact summary of directional asymmetry
- `local_SD_mean3`
  - local-context feature showing neighborhood average matters
- `local_NDAI_std3`
  - local texture feature showing patch variability matters
- `local_SD_std3`
  - another local texture feature supporting spatial reasoning
- `ae4`
  - strongest learned latent patch feature
- `ae5`
  - second strong learned latent patch feature
- `ae0`
  - third useful latent patch feature

## 7. Recommended Part 2 message for the meeting
The clearest way to present the logic is:
- start from expert features because they are the most defensible baseline
- check raw radiances to see which viewing angles matter most
- engineer contrasts, ratios, and interactions because cloud detection is fundamentally multi-angle
- add local summaries because clouds are spatial structures, not isolated pixels
- add autoencoder latents because learned patch representations may capture structure that simple hand-designed summaries miss

In one sentence:
Part 2 turns raw MISR measurements into a layered predictor set, moving from interpretable single-pixel variables to local-context and learned patch features so Part 3 has stronger inputs.

## 8. Recommended feature-engineering blocks for Part 3
- Interpretable baseline block: `SD`, `NDAI`, `AF`, `AN`, and optionally `BF`
- Engineered scalar block: `ndai_x_sd`, `af_df_gap`, `front_back_ratio`, `rad_cv`, `rad_range`
- Local-context block: `local_SD_mean3`, `local_NDAI_std3`, `local_SD_std3`
- Learned patch block: `ae4`, `ae5`, `ae0`

Recommended modeling comparison:
- interpretable-only
- interpretable + engineered scalar + local context
- interpretable + engineered scalar + local context + autoencoder features

The reasoning behind this staged comparison is that it lets us measure whether extra complexity actually buys better performance.

## 9. Caveats to mention during the meeting
- These are screening results, not final model rankings.
- High-ranking features may be redundant with each other.
- Final feature selection still has to respect the realistic train/validation/test split logic from Part 1.
- `CORR` underperformed here, but that does not prove it is useless in a multivariate model.
- Autoencoder features can help prediction even if they are harder to interpret.

## 10. Files to open during the meeting
- `documents/part2/predictor_catalog.md`
- `results/part2/feature_screening.csv`
- `documents/part2/autoencoder_feature_notes.md`
- `results/part2/autoencoder_feature_screening.csv`
