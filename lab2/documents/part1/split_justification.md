# Split Justification

## Why these split strategies
- Pixel-wise random splitting can leak local spatial structure into both train and test because neighboring pixels are strongly autocorrelated.
- By-image holdout better reflects transfer to a new image/orbit condition.
- Spatial holdout (rightmost contiguous x block) stress-tests within-image spatial generalization.

## Concrete rules in this generator
1. By-image strategy:
- Test: one full labeled image (holdout).
- Train/Val inside the other two images: contiguous x-based split.
- Validation is the rightmost 20% of the pre-test pool in each training image.

2. Spatial-within-image strategy:
- Test: rightmost 20% by x in a single image.
- Validation: rightmost 20% by x of the remaining pre-test region.
- Train: the remainder.

## Label usage
- Supervised splits use only expert-labeled rows (`label` in {-1, +1}).
- Unlabeled rows (`label=0`) are excluded from train/val/test supervision files.
