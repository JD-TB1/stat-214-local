# Split Diagnostics Notes

## Why pixel-wise random split is risky
A random pixel split leaks spatially adjacent pixels from the same cloud structures into both train and test. Because nearby pixels are highly autocorrelated, this inflates test performance and understates generalization error.

## Recommended split strategies
1. By-image split: train on two labeled images and test on the held-out image.
2. Within-image spatial split: hold out a contiguous spatial block (for example, rightmost 20% of x).

## Cleaning recommendations (analysis-only stage)
- Keep all rows for core accounting, but exclude non-finite rows in plots/statistics.
- Track and investigate out-of-range NDAI/CORR values before modeling.
- Preserve duplicate (x, y) findings for QA and dedup policy in the modeling phase.
