# Autoencoder Feature Notes

Checkpoint used: `/Users/jayding/Desktop/DUKE/YY/歪歪长官的任务/Stat 214/stat-214-local/lab2/code/checkpoints/gsi-model.ckpt`

## What this script does
- Reuses the patch-based autoencoder architecture from `code/autoencoder.py`.
- Builds 9x9 patches around supervised pixels from the eight raw MISR channels/features.
- Screens each latent coordinate as a candidate Part 2 predictor.

## Meeting-ready autoencoder candidates
- `ae4`: separation AUC=0.897, MI=0.306, |d|=1.712
- `ae5`: separation AUC=0.867, MI=0.230, |d|=1.503
- `ae0`: separation AUC=0.845, MI=0.216, |d|=1.244
- `ae3`: separation AUC=0.799, MI=0.159, |d|=1.009
- `ae1`: separation AUC=0.779, MI=0.174, |d|=0.928

## How to use these in Part 2/3
- Treat the embedding dimensions as complementary predictors alongside interpretable expert and radiance features.
- Keep only the strongest latent dimensions instead of all eight if you want lower-variance downstream models.
- Prefer combining embeddings with NDAI, SD, and a small radiance contrast set rather than relying on embeddings alone.
