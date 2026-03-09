# STAT 214 Lab 2 - Current Stage README

This README summarizes the current Lab 2 pipeline status, scripts added in this stage, how to run them, and where outputs/notes are recorded.

## Project root
`/Users/jayding/Desktop/DUKE/YY/歪歪长官的任务/Stat 214/stat-214-local/lab2`

## 1) Scripts added in this stage

### `code/part1/eda.py`
Purpose:
- Part 1 EDA for MISR labeled images (`O013257`, `O013490`, `O012791`).
- Produces label maps, feature/radiance relationship plots, split diagnostics, quality diagnostics, and summary JSON.

Run:
```bash
python code/part1/eda.py
# or
python -m code.eda --max_points 50000 --seed 214 --out_dir results/eda
```

Main outputs:
- `results/eda/summary.json`
- `results/eda/label_map_*.png`
- `results/eda/label_map_labeled_only_*.png`
- `results/eda/radiance_corr_*.png`
- `results/eda/radiance_pairs_*.png`
- `results/eda/feature_dist_<feature>_<scope>.png`
- `results/eda/feature_ranking.csv` (+ per-image rankings)
- `results/eda/split_diagnostics.csv`
- `results/eda/data_quality_report.csv`
- `results/eda/split_notes.md`

Current result highlights (from `summary.json`):
- Top pooled features by |Cohen's d|: `NDAI`, `SD`, `AF`.
- Largest shift split diagnostic: `spatial_within_image / O013490_x_gt_q80` (abs SMD ~0.692).

---

### `code/part1/make_splits.py`
Purpose:
- Generate reproducible train/val/test files for two strategies:
1. by-image holdout
2. spatial within-image holdout

Run:
```bash
python code/part1/make_splits.py
# or
python -m code.make_splits --test_frac 0.2 --val_frac 0.2 --seed 214 --out_dir results/splits
```

Main outputs:
- `results/splits/by_image/holdout_<image_id>/{train,val,test}.csv`
- `results/splits/spatial_within_image/<image_id>_x_gt_q80/{train,val,test}.csv`
- `results/splits/split_manifest.csv`
- `results/splits/split_manifest.json`
- `results/splits/split_justification.md`

Current status:
- Split files are already generated in `results/splits/`.

---

### `code/part1/clean_lab2.py`
Purpose:
- Minimal, conservative data cleaning (separate from EDA diagnostics).
- Intended for pre-modeling dataset preparation.

Cleaning rules:
- keep labels in `{-1,0,+1}`
- remove non-finite rows in required fields
- enforce physical bounds:
  - `NDAI in [-1,1]`
  - `CORR in [-1,1]`
  - `SD >= 0`
  - radiances `>= 0`
- remove duplicate `(image_id, x, y)` rows if present

Run:
```bash
python code/part1/clean_lab2.py
# or
python -m code.clean_lab2 --out_dir results/cleaning
```

Expected outputs:
- `results/cleaning/labeled_cleaned_<image_id>.csv`
- `results/cleaning/labeled_cleaned_all.csv`
- `results/cleaning/labeled_supervised_all.csv`
- `results/cleaning/cleaning_report.csv`
- `results/cleaning/cleaning_summary.json`

Current status:
- Script is implemented and syntax-checked.
- `results/cleaning/` has **not** been generated yet in this workspace.

---

### `code/__init__.py`
Purpose:
- Enables `python -m code.eda`, `python -m code.make_splits`, `python -m code.clean_lab2`.

## 2) Meeting/report documentation files

### Meeting summary (main handoff note)
- `results/eda/part1_meeting_summary.md`

Contains:
- Part 1 sufficiency decision
- requirement checklist vs generated artifacts
- key EDA quantitative findings
- split justification points
- cleaning status update and recommended conservative cleaning policy

### Additional notes
- `results/eda/split_notes.md`: split-risk rationale (why random pixel split is risky)
- `results/splits/split_justification.md`: concrete split rules used by generator

## 3) Recommended execution order from this point
1. Run/refresh cleaning:
```bash
python code/part1/clean_lab2.py
```
2. (Optional) regenerate splits if parameters change:
```bash
python code/part1/make_splits.py --test_frac 0.2 --val_frac 0.2 --seed 214
```
3. Use `results/cleaning/labeled_supervised_all.csv` + `results/splits/...` for Part 2/3 modeling.

## 4) Quick artifact map
- EDA artifacts: `results/eda/`
- Split files: `results/splits/`
- Cleaning outputs: `results/cleaning/` (to be generated)
- Code scripts: `code/`

