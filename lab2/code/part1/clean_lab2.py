#!/usr/bin/env python3
"""Minimal data cleaning for STAT 214 Lab 2 MISR labeled images.

Purpose:
- Apply conservative, domain-aligned cleaning before modeling.
- Keep cleaning transparent and reversible.

Default outputs (under `results/part1/cleaning`):
- labeled_cleaned_all.csv
- labeled_cleaned_<image_id>.csv
- labeled_supervised_all.csv  (label in {-1, +1})
- cleaning_report.csv
- cleaning_summary.json

Run:
- python code/part1/clean_lab2.py
- python -m code.part1.clean_lab2 --out_dir results/part1/cleaning
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

LABELED_IDS = ["O013257", "O013490", "O012791"]
BASE_COLUMNS = ["y", "x", "NDAI", "SD", "CORR", "DF", "CF", "BF", "AF", "AN"]
ALL_COLUMNS = BASE_COLUMNS + ["label"]
RADIANCE_COLS = ["DF", "CF", "BF", "AF", "AN"]


def log(msg: str) -> None:
    print(f"[CLEAN] {msg}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal cleaning for MISR Lab 2 labeled data")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/part1/cleaning",
        help="Output directory relative to lab2 root unless absolute",
    )
    return parser.parse_args()


def resolve_paths(out_dir_arg: str) -> Tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parents[2]
    data_dir = root / "data" / "image_data"
    out_dir = Path(out_dir_arg)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return root, data_dir, out_dir


def pick_main_array(npz_obj: np.lib.npyio.NpzFile) -> Tuple[str, np.ndarray]:
    candidates: List[Tuple[str, np.ndarray]] = []
    for key in npz_obj.files:
        arr = npz_obj[key]
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] in (10, 11):
            candidates.append((key, arr))
    if not candidates:
        first_key = npz_obj.files[0]
        arr = npz_obj[first_key]
        raise ValueError(
            f"No usable 2D array with 10/11 columns found. keys={npz_obj.files}, first_shape={getattr(arr, 'shape', None)}"
        )
    return max(candidates, key=lambda kv: kv[1].shape[0])


def load_labeled(npz_path: Path) -> pd.DataFrame:
    with np.load(npz_path, allow_pickle=True) as npz_obj:
        log(f"Inspecting {npz_path.name}: keys={list(npz_obj.files)}")
        _, arr = pick_main_array(npz_obj)
    arr = np.asarray(arr)
    if arr.shape[1] != 11:
        raise ValueError(f"Expected 11-column labeled array for {npz_path.name}, got {arr.shape}")

    df = pd.DataFrame(arr, columns=ALL_COLUMNS)
    for col in ALL_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["label"] = df["label"].astype("Int64")
    df["image_id"] = npz_path.stem
    df["row_id"] = np.arange(len(df), dtype=np.int64)
    return df


def clean_minimal(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    report: Dict[str, int] = {}
    start_n = len(df)
    report["n_start"] = int(start_n)

    # 1) Keep recognized labels only.
    label_ok = df["label"].isin([-1, 0, 1])
    report["drop_bad_label"] = int((~label_ok).sum())
    out = df[label_ok].copy()

    # 2) Remove non-finite rows across required numeric columns.
    required = ["y", "x", "NDAI", "SD", "CORR", *RADIANCE_COLS, "label"]
    finite_mask = np.isfinite(out[required].to_numpy(dtype=float)).all(axis=1)
    report["drop_non_finite"] = int((~finite_mask).sum())
    out = out[finite_mask].copy()

    # 3) Conservative physical-range checks.
    ndai_ok = out["NDAI"].between(-1.0, 1.0, inclusive="both")
    corr_ok = out["CORR"].between(-1.0, 1.0, inclusive="both")
    sd_ok = out["SD"] >= 0.0
    rad_ok = (out[RADIANCE_COLS] >= 0.0).all(axis=1)
    phys_ok = ndai_ok & corr_ok & sd_ok & rad_ok

    report["drop_bad_ndai"] = int((~ndai_ok).sum())
    report["drop_bad_corr"] = int((~corr_ok).sum())
    report["drop_bad_sd"] = int((~sd_ok).sum())
    report["drop_bad_radiance"] = int((~rad_ok).sum())
    report["drop_bad_physical_total"] = int((~phys_ok).sum())
    out = out[phys_ok].copy()

    # 4) Resolve duplicate pixel coordinates within image (keep first occurrence).
    dup_xy = out.duplicated(subset=["image_id", "x", "y"], keep="first")
    report["drop_duplicate_xy"] = int(dup_xy.sum())
    out = out[~dup_xy].copy()

    report["n_end"] = int(len(out))
    report["n_removed_total"] = int(start_n - len(out))
    return out, report


def run() -> None:
    args = parse_args()
    root, data_dir, out_dir = resolve_paths(args.out_dir)
    log(f"root={root}")
    log(f"data_dir={data_dir}")
    log(f"out_dir={out_dir}")

    cleaned_parts: List[pd.DataFrame] = []
    report_rows: List[Dict[str, int]] = []
    summary: Dict[str, object] = {"per_image": {}, "pooled": {}}

    for image_id in LABELED_IDS:
        path = data_dir / f"{image_id}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing labeled image: {path}")
        raw = load_labeled(path)
        cleaned, rep = clean_minimal(raw)
        rep_row = {"scope": image_id, **rep}
        report_rows.append(rep_row)
        summary["per_image"][image_id] = rep
        cleaned_parts.append(cleaned)
        cleaned.to_csv(out_dir / f"labeled_cleaned_{image_id}.csv", index=False)
        log(f"Saved cleaned image file: labeled_cleaned_{image_id}.csv")

    pooled = pd.concat(cleaned_parts, ignore_index=True)
    pooled.to_csv(out_dir / "labeled_cleaned_all.csv", index=False)

    supervised = pooled[pooled["label"].isin([-1, 1])].copy()
    supervised.to_csv(out_dir / "labeled_supervised_all.csv", index=False)

    pooled_rep = {
        "n_start": int(sum(r["n_start"] for r in report_rows)),
        "n_end": int(len(pooled)),
        "n_removed_total": int(sum(r["n_removed_total"] for r in report_rows)),
        "n_supervised_end": int(len(supervised)),
        "n_unlabeled_end": int((pooled["label"] == 0).sum()),
    }
    summary["pooled"] = pooled_rep
    report_rows.append({"scope": "pooled", **pooled_rep})

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(out_dir / "cleaning_report.csv", index=False)
    (out_dir / "cleaning_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log("Cleaning done.")
    log(f"Wrote: {out_dir / 'cleaning_report.csv'}")
    log(f"Wrote: {out_dir / 'cleaning_summary.json'}")


if __name__ == "__main__":
    run()
