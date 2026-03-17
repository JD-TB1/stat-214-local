# Lab 1

This directory contains the Lab 1 PECARN TBI workflow.

## Layout
- `code/`: core cleaning and modeling scripts
- `scripts/`: one-off analysis, plotting, and inspection helpers
- `data/`: local lab data and documentation files
- `documents/`: notes and preserved inspection transcripts
- `output/`: generated tables, cleaned datasets, and figures
- `report/`: report-facing figure assets

## Reorganization notes
- The stray repository-level `output/inspect_raw/` folder was a Lab 1 raw-inspection artifact from `scripts/inspect_raw.py`.
- That inspection output now belongs under `lab1/output/inspect_raw/`.
- The shell transcript from the earlier raw-inspection run lives under `documents/raw_inspection/`.
- The stability figure now writes into `lab1/output/figures/` instead of the old standalone `lab1/figs/` folder.

## Reproduction
- Cleaning: `python lab1/code/clean.py`
- Modeling: `python lab1/code/models.py`
- Raw inspection: `python lab1/scripts/inspect_raw.py`
- Stability figure: `python lab1/scripts/stability_check.py`
