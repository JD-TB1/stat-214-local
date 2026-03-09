# Part 1 README

This folder contains the Part 1 documentation handoff for Lab 2. Part 1 covers exploratory data analysis, split design, and conservative cleaning guidance for the three labeled MISR images.

## Scripts
- `code/part1/eda.py`
  - produces label maps, radiance plots, class-conditioned feature plots, rankings, split diagnostics, and a summary JSON
- `code/part1/make_splits.py`
  - produces reproducible by-image and spatial train/val/test splits
- `code/part1/clean_lab2.py`
  - produces conservative cleaned labeled datasets for downstream modeling

## Documents
- `documents/part1/meeting_summary.md`
- `documents/part1/split_notes.md`
- `documents/part1/split_justification.md`

## Reproducible outputs
- `results/part1/eda/`
- `results/part1/splits/`
- `results/part1/cleaning/` when generated

## Run order
1. `python code/part1/eda.py`
2. `python code/part1/make_splits.py`
3. `python code/part1/clean_lab2.py`
