# STAT 214 Lab 2

This directory contains the local working copy of Lab 2 for MISR-based cloud detection. The layout is organized so collaborators can clearly separate starter code, Part 1 work, Part 2 work, reproducible outputs, and hand-written documentation.

## Layout
- `code/original/`: GSI-provided starter scripts and config files.
- `code/part1/`: Part 1 EDA, cleaning, and split-generation scripts.
- `code/part2/`: Part 2 feature-engineering and autoencoder-feature scripts.
- `code/part3/`: Part 3 predictive-modeling, diagnostics, and unlabeled-inference scripts.
- `data/image_data/`: local MISR `.npz` image files.
- `results/part1/`: reproducible Part 1 outputs.
- `results/part2/`: reproducible Part 2 outputs.
- `results/part3/`: reproducible Part 3 outputs.
- `documents/part1/`: Part 1 summaries and split notes.
- `documents/part2/`: Part 2 readmes, summaries, and feature notes.
- `documents/part3/`: Part 3 readme and meeting summary.

## Recommended reading order
1. `documents/part1/README.md`
2. `documents/part1/meeting_summary.md`
3. `documents/part2/README.md`
4. `documents/part2/meeting_summary.md`
5. `documents/part3/README.md`
6. `documents/part3/meeting_summary.md`

## Reproduction
Run commands from the `lab2` root.

Part 1 EDA:
```bash
python code/part1/eda.py
```

Part 1 cleaning:
```bash
python code/part1/clean_lab2.py
```

Part 1 splits:
```bash
python code/part1/make_splits.py
```

Part 2 engineered predictors:
```bash
/opt/homebrew/Caskroom/miniforge/base/envs/env_214/bin/python code/part2/feature_engineering.py
```

Part 2 autoencoder-derived predictors:
```bash
/opt/homebrew/Caskroom/miniforge/base/envs/env_214/bin/python code/part2/autoencoder_features.py
```

Part 3 predictive modeling:
```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp/.cache LOKY_MAX_CPU_COUNT=1 \
/opt/homebrew/Caskroom/miniforge/base/envs/env_214/bin/python code/part3/train_models.py
```

Part 3 unlabeled-image sanity check:
```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp/.cache \
/opt/homebrew/Caskroom/miniforge/base/envs/env_214/bin/python code/part3/predict_unlabeled.py
```

Original GSI autoencoder training:
```bash
/opt/homebrew/Caskroom/miniforge/base/envs/env_214/bin/python code/original/run_autoencoder.py code/original/configs/default.yaml
```

Original GSI embedding export:
```bash
/opt/homebrew/Caskroom/miniforge/base/envs/env_214/bin/python code/original/get_embedding.py code/original/configs/default.yaml results/part2/autoencoder/checkpoints/default-epoch=009.ckpt
```

## Output conventions
- `code/part1/` defaults to `results/part1/...` and `documents/part1/...`.
- `code/part2/` defaults to `results/part2/...` and `documents/part2/...`.
- `code/part3/` defaults to `results/part3/...` and `documents/part3/...`.
- `code/original/` writes generated autoencoder artifacts under `results/part2/autoencoder/...` when using the provided config.
