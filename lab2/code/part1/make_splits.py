#!/usr/bin/env python3
"""Generate deterministic train/val/test splits for STAT 214 Lab 2.

Artifacts (default output root: ./results/splits):
- by_image/holdout_<image_id>/{train,val,test}.csv
- spatial_within_image/<image_id>_x_gt_qXX/{train,val,test}.csv
- split_manifest.csv
- split_manifest.json
- split_justification.md

Run examples:
- python code/part1/make_splits.py
- python -m code.part1.make_splits --test_frac 0.2 --val_frac 0.2
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

LABELED_IDS = ["O013257", "O013490", "O012791"]
BASE_COLUMNS = ["y", "x", "NDAI", "SD", "CORR", "DF", "CF", "BF", "AF", "AN"]
ALL_COLUMNS = BASE_COLUMNS + ["label"]
LABELLED_VALUES = {-1, 1}


@dataclass
class LoadedImage:
    image_id: str
    df: pd.DataFrame
    source_key: str
    source_shape: Tuple[int, ...]


def log(msg: str) -> None:
    print(f"[SPLIT] {msg}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic splits for MISR labeled data")
    parser.add_argument("--test_frac", type=float, default=0.2, help="Fraction for spatial test holdout by x-coordinate")
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.2,
        help="Fraction for validation holdout from pre-test pool by x-coordinate",
    )
    parser.add_argument("--seed", type=int, default=214, help="Seed kept for reproducibility bookkeeping")
    parser.add_argument("--out_dir", type=str, default="results/splits", help="Output directory (relative to lab2 root unless absolute)")
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


def load_npz_as_df(npz_path: Path) -> LoadedImage:
    with np.load(npz_path, allow_pickle=True) as npz_obj:
        log(f"Inspecting {npz_path.name}: keys={list(npz_obj.files)}")
        key, arr = pick_main_array(npz_obj)

    arr = np.asarray(arr)
    if arr.shape[1] != 11:
        raise ValueError(f"Expected labeled file with 11 columns, got {arr.shape} for {npz_path.name}")

    df = pd.DataFrame(arr, columns=ALL_COLUMNS)
    for col in ALL_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    image_id = npz_path.stem
    df["image_id"] = image_id
    df["row_id"] = np.arange(len(df), dtype=np.int64)
    df["label"] = df["label"].astype("Int64")
    return LoadedImage(image_id=image_id, df=df, source_key=key, source_shape=tuple(arr.shape))


def filter_supervised_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["label"].isin(LABELLED_VALUES)].copy()
    out["label"] = out["label"].astype(int)
    return out


def select_spatial_splits(df: pd.DataFrame, test_frac: float, val_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    if not 0 < test_frac < 1:
        raise ValueError(f"test_frac must be in (0,1), got {test_frac}")
    if not 0 < val_frac < 1:
        raise ValueError(f"val_frac must be in (0,1), got {val_frac}")

    x = df["x"].to_numpy(dtype=float)
    test_cut = float(np.quantile(x, 1.0 - test_frac))

    test = df[df["x"] > test_cut].copy()
    pre_train = df[df["x"] <= test_cut].copy()

    if pre_train.empty:
        raise ValueError("No rows left for train/val after test split; adjust fractions.")

    val_cut = float(np.quantile(pre_train["x"].to_numpy(dtype=float), 1.0 - val_frac))
    val = pre_train[pre_train["x"] > val_cut].copy()
    train = pre_train[pre_train["x"] <= val_cut].copy()

    meta = {
        "test_x_threshold": test_cut,
        "val_x_threshold": val_cut,
        "test_frac_target": test_frac,
        "val_frac_target": val_frac,
    }
    return train, val, test, meta


def select_contiguous_val_only(df: pd.DataFrame, val_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    if not 0 < val_frac < 1:
        raise ValueError(f"val_frac must be in (0,1), got {val_frac}")
    x = df["x"].to_numpy(dtype=float)
    val_cut = float(np.quantile(x, 1.0 - val_frac))
    val = df[df["x"] > val_cut].copy()
    train = df[df["x"] <= val_cut].copy()
    meta = {"val_x_threshold": val_cut, "val_frac_target": val_frac}
    return train, val, meta


def write_split_csv(df: pd.DataFrame, out_path: Path) -> None:
    cols = ["image_id", "row_id", "label", "x", "y", "NDAI", "SD", "CORR", "DF", "CF", "BF", "AF", "AN"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(out_path, index=False)


def summarize_subset(df: pd.DataFrame) -> Dict[str, int]:
    counts = df["label"].value_counts().to_dict()
    return {
        "n_rows": int(len(df)),
        "n_cloud": int(counts.get(1, 0)),
        "n_non_cloud": int(counts.get(-1, 0)),
    }


def write_justification(path: Path, test_frac: float, val_frac: float) -> None:
    pct_test = int(round(100 * test_frac))
    pct_val = int(round(100 * val_frac))
    text = f"""# Split Justification

## Why these split strategies
- Pixel-wise random splitting can leak local spatial structure into both train and test because neighboring pixels are strongly autocorrelated.
- By-image holdout better reflects transfer to a new image/orbit condition.
- Spatial holdout (rightmost contiguous x block) stress-tests within-image spatial generalization.

## Concrete rules in this generator
1. By-image strategy:
- Test: one full labeled image (holdout).
- Train/Val inside the other two images: contiguous x-based split.
- Validation is the rightmost {pct_val}% of the pre-test pool in each training image.

2. Spatial-within-image strategy:
- Test: rightmost {pct_test}% by x in a single image.
- Validation: rightmost {pct_val}% by x of the remaining pre-test region.
- Train: the remainder.

## Label usage
- Supervised splits use only expert-labeled rows (`label` in {{-1, +1}}).
- Unlabeled rows (`label=0`) are excluded from train/val/test supervision files.
"""
    path.write_text(text, encoding="utf-8")


def run() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    root, data_dir, out_dir = resolve_paths(args.out_dir)
    log(f"root={root}")
    log(f"data_dir={data_dir}")
    log(f"out_dir={out_dir}")

    loaded: Dict[str, pd.DataFrame] = {}
    for image_id in LABELED_IDS:
        npz_path = data_dir / f"{image_id}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing labeled file: {npz_path}")
        item = load_npz_as_df(npz_path)
        loaded[image_id] = filter_supervised_rows(item.df)

    manifest_rows: List[Dict[str, object]] = []

    # Strategy 1: By-image holdout with spatial val split inside train images.
    for holdout in LABELED_IDS:
        train_images = [img for img in LABELED_IDS if img != holdout]
        train_parts: List[pd.DataFrame] = []
        val_parts: List[pd.DataFrame] = []
        meta_parts: Dict[str, Dict[str, float]] = {}

        for img in train_images:
            train_i, val_i, meta_i = select_contiguous_val_only(loaded[img], val_frac=args.val_frac)
            train_parts.append(train_i)
            val_parts.append(val_i)
            meta_parts[img] = meta_i

        train_df = pd.concat(train_parts, ignore_index=True)
        val_df = pd.concat(val_parts, ignore_index=True)
        test_df = loaded[holdout].copy()

        split_id = f"holdout_{holdout}"
        split_dir = out_dir / "by_image" / split_id
        write_split_csv(train_df, split_dir / "train.csv")
        write_split_csv(val_df, split_dir / "val.csv")
        write_split_csv(test_df, split_dir / "test.csv")

        row = {
            "strategy": "by_image",
            "split_id": split_id,
            "seed": args.seed,
            "train_images": ",".join(train_images),
            "test_images": holdout,
            **{f"train_{k}": v for k, v in summarize_subset(train_df).items()},
            **{f"val_{k}": v for k, v in summarize_subset(val_df).items()},
            **{f"test_{k}": v for k, v in summarize_subset(test_df).items()},
            "test_frac": args.test_frac,
            "val_frac": args.val_frac,
            "notes": f"Validation is contiguous right block in each train image; thresholds={meta_parts}",
        }
        manifest_rows.append(row)
        log(f"Wrote by-image split: {split_dir}")

    # Strategy 2: Spatial holdout within each image.
    for image_id in LABELED_IDS:
        train_df, val_df, test_df, meta = select_spatial_splits(loaded[image_id], test_frac=args.test_frac, val_frac=args.val_frac)
        split_id = f"{image_id}_x_gt_q{int(round((1.0 - args.test_frac) * 100))}"
        split_dir = out_dir / "spatial_within_image" / split_id
        write_split_csv(train_df, split_dir / "train.csv")
        write_split_csv(val_df, split_dir / "val.csv")
        write_split_csv(test_df, split_dir / "test.csv")

        row = {
            "strategy": "spatial_within_image",
            "split_id": split_id,
            "seed": args.seed,
            "train_images": image_id,
            "test_images": image_id,
            **{f"train_{k}": v for k, v in summarize_subset(train_df).items()},
            **{f"val_{k}": v for k, v in summarize_subset(val_df).items()},
            **{f"test_{k}": v for k, v in summarize_subset(test_df).items()},
            "test_frac": args.test_frac,
            "val_frac": args.val_frac,
            "notes": f"Contiguous x-thresholds: test>{meta['test_x_threshold']:.3f}, val>{meta['val_x_threshold']:.3f} in pre-test region",
        }
        manifest_rows.append(row)
        log(f"Wrote spatial split: {split_dir}")

    manifest_csv = out_dir / "split_manifest.csv"
    fieldnames = list(manifest_rows[0].keys()) if manifest_rows else []
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    manifest_json = out_dir / "split_manifest.json"
    manifest_json.write_text(json.dumps(manifest_rows, indent=2), encoding="utf-8")

    write_justification(out_dir / "split_justification.md", test_frac=args.test_frac, val_frac=args.val_frac)
    log(f"Done. Manifest: {manifest_csv}")


if __name__ == "__main__":
    run()
