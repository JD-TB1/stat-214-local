#!/usr/bin/env python3
"""Merged Part 2 feature engineering for the isolated Lab 2 pipeline.

This version keeps the current pipeline's standardized schema so Part 3 can run
unchanged, while integrating the group-merged additions:
- extra angle-pair / angular-spread features
- reflect-padded local features with edge fallback
- richer screening output and predictor catalog
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List
import warnings

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

LABELED_IDS = ["O013257", "O013490", "O012791"]
BASE_COLUMNS = ["y", "x", "NDAI", "SD", "CORR", "DF", "CF", "BF", "AF", "AN", "label"]
RADIANCE_COLS = ["DF", "CF", "BF", "AF", "AN"]
EXPERT_COLS = ["NDAI", "SD", "CORR"]
LOCAL_BASE_COLS = ["NDAI", "SD", "CORR", "rad_mean", "rad_std"]
ANGLE_PAIR_PREFIXES = (
    "ndai_cf_",
    "ndai_bf_",
    "ndai_af_",
    "ndai_df_",
    "ndai_angular",
    "cov_",
    "sd_proxy_",
    "sd_angular",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build merged Part 2 engineered predictors and screening tables")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../../../data/image_data",
        help="Path to the shared Lab 2 MISR image directory",
    )
    parser.add_argument("--out_dir", type=str, default="../../results/part2")
    parser.add_argument("--docs_dir", type=str, default="../../documents/part2")
    parser.add_argument("--local_window", type=int, default=3)
    return parser.parse_args()


def resolve_path(base: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (base / path).resolve()


def pick_main_array(npz_obj: np.lib.npyio.NpzFile) -> np.ndarray:
    candidates: List[np.ndarray] = []
    for key in npz_obj.files:
        arr = npz_obj[key]
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] in (10, 11):
            candidates.append(arr)
    if not candidates:
        raise ValueError(f"No usable 2D array found in keys={list(npz_obj.files)}")
    return np.asarray(max(candidates, key=lambda arr: arr.shape[0]))


def load_labeled_images(data_dir: Path) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for image_id in LABELED_IDS:
        path = data_dir / f"{image_id}.npz"
        with np.load(path, allow_pickle=True) as npz_obj:
            arr = pick_main_array(npz_obj)
        if arr.shape[1] != 11:
            raise ValueError(f"Expected 11 columns for labeled image {image_id}, got {arr.shape}")
        df = pd.DataFrame(arr, columns=BASE_COLUMNS)
        for col in BASE_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["label"] = df["label"].astype(int)
        df["image_id"] = image_id
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def add_angle_pair_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    eps = 1e-6

    out["ndai_cf_an"] = (out["CF"] - out["AN"]) / (out["CF"] + out["AN"] + eps)
    out["ndai_bf_an"] = (out["BF"] - out["AN"]) / (out["BF"] + out["AN"] + eps)
    out["ndai_af_an"] = (out["AF"] - out["AN"]) / (out["AF"] + out["AN"] + eps)
    out["ndai_df_af"] = (out["DF"] - out["AF"]) / (out["DF"] + out["AF"] + eps)

    ndai_cols = ["NDAI", "ndai_cf_an", "ndai_bf_an", "ndai_af_an"]
    out["ndai_angular_range"] = out[ndai_cols].max(axis=1) - out[ndai_cols].min(axis=1)
    out["ndai_angular_std"] = out[ndai_cols].std(axis=1)

    rad_mean_overall = out[RADIANCE_COLS].mean(axis=1)
    out["cov_df_an"] = (out["DF"] - rad_mean_overall) * (out["AN"] - rad_mean_overall)
    out["cov_cf_af"] = (out["CF"] - rad_mean_overall) * (out["AF"] - rad_mean_overall)
    out["cov_df_bf"] = (out["DF"] - rad_mean_overall) * (out["BF"] - rad_mean_overall)

    for col in RADIANCE_COLS:
        out[f"sd_proxy_{col.lower()}"] = (out[col] - rad_mean_overall).abs()
    sd_proxy_cols = [f"sd_proxy_{col.lower()}" for col in RADIANCE_COLS]
    out["sd_angular_range"] = out[sd_proxy_cols].max(axis=1) - out[sd_proxy_cols].min(axis=1)
    out["sd_angular_cv"] = out[sd_proxy_cols].std(axis=1) / (out[sd_proxy_cols].mean(axis=1).abs() + eps)
    return out


def add_pointwise_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    eps = 1e-6

    out["rad_mean"] = out[RADIANCE_COLS].mean(axis=1)
    out["rad_std"] = out[RADIANCE_COLS].std(axis=1)
    out["rad_range"] = out[RADIANCE_COLS].max(axis=1) - out[RADIANCE_COLS].min(axis=1)
    out["rad_cv"] = out["rad_std"] / (out["rad_mean"].abs() + eps)

    out["front_mean"] = out[["DF", "CF", "BF"]].mean(axis=1)
    out["aft_mean"] = out[["AF", "AN"]].mean(axis=1)
    out["forward_backward_gap"] = out["front_mean"] - out["aft_mean"]
    out["front_back_ratio"] = (out["front_mean"] + eps) / (out["aft_mean"] + eps)

    out["df_an_ratio"] = out["DF"] / (out["AN"] + eps)
    out["af_an_ratio"] = out["AF"] / (out["AN"] + eps)
    out["an_df_gap"] = out["AN"] - out["DF"]
    out["af_df_gap"] = out["AF"] - out["DF"]
    out["bf_cf_gap"] = out["BF"] - out["CF"]
    out["an_af_gap"] = out["AN"] - out["AF"]

    out["ndai_x_sd"] = out["NDAI"] * out["SD"]
    out["corr_x_sd"] = out["CORR"] * out["SD"]
    out["corr_x_ndai"] = out["CORR"] * out["NDAI"]
    return out


def add_local_patch_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    if window % 2 == 0:
        raise ValueError("local_window must be odd")

    radius = window // 2
    out_parts: List[pd.DataFrame] = []

    for image_id, part in df.groupby("image_id", sort=False):
        image = part.copy()
        x = image["x"].to_numpy(dtype=int)
        y = image["y"].to_numpy(dtype=int)
        x0, y0 = x.min(), y.min()
        width = int(x.max() - x0 + 1)
        height = int(y.max() - y0 + 1)
        xr = x - x0
        yr = y - y0

        for col in LOCAL_BASE_COLS:
            grid = np.full((height, width), np.nan, dtype=float)
            grid[yr, xr] = image[col].to_numpy(dtype=float)

            padded = np.pad(grid, radius, mode="reflect")
            windows = sliding_window_view(padded, (window, window))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Mean of empty slice")
                warnings.filterwarnings("ignore", message="Degrees of freedom*")
                local_mean = np.nanmean(windows, axis=(-2, -1))
                local_std = np.nanstd(windows, axis=(-2, -1))

            center_vals = grid[yr, xr]
            mean_vals = local_mean[yr, xr]
            std_vals = local_std[yr, xr]

            mean_vals = np.where(np.isfinite(mean_vals), mean_vals, np.where(np.isfinite(center_vals), center_vals, 0.0))
            std_vals = np.where(np.isfinite(std_vals), std_vals, 0.0)

            image[f"local_{col}_mean{window}"] = mean_vals
            image[f"local_{col}_std{window}"] = std_vals
            image[f"local_{col}_centered{window}"] = center_vals - mean_vals

        out_parts.append(image)

    return pd.concat(out_parts, ignore_index=True)


def cohen_d(x1: np.ndarray, x0: np.ndarray) -> float:
    x1 = x1[np.isfinite(x1)]
    x0 = x0[np.isfinite(x0)]
    if len(x1) < 2 or len(x0) < 2:
        return np.nan
    v1 = np.var(x1, ddof=1)
    v0 = np.var(x0, ddof=1)
    pooled = ((len(x1) - 1) * v1 + (len(x0) - 1) * v0) / (len(x1) + len(x0) - 2)
    if pooled <= 0 or not np.isfinite(pooled):
        return np.nan
    return (np.mean(x1) - np.mean(x0)) / np.sqrt(pooled)


def classify_family(feature: str) -> str:
    if feature in EXPERT_COLS:
        return "expert"
    if feature in RADIANCE_COLS:
        return "radiance"
    if feature.startswith("local_"):
        return "patch_local"
    if feature.startswith(ANGLE_PAIR_PREFIXES):
        return "angle_pair"
    if feature.startswith("ae"):
        return "autoencoder"
    return "engineered_scalar"


def screen_features(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    supervised = df[df["label"].isin([-1, 1])].copy()
    supervised["target"] = (supervised["label"] == 1).astype(int)

    rows: List[Dict[str, float | str]] = []
    for feature in features:
        sub = supervised[[feature, "target", "label", "image_id"]].replace([np.inf, -np.inf], np.nan).dropna()
        if sub.empty or sub[feature].nunique() < 2:
            continue

        values = sub[feature].to_numpy(dtype=float)
        target = sub["target"].to_numpy(dtype=int)
        auc_raw = roc_auc_score(target, values)
        auc_symmetric = max(auc_raw, 1.0 - auc_raw)
        mi = float(mutual_info_classif(values.reshape(-1, 1), target, discrete_features=False, random_state=214)[0])
        d = cohen_d(sub.loc[sub["label"] == 1, feature].to_numpy(), sub.loc[sub["label"] == -1, feature].to_numpy())

        row: Dict[str, float | str] = {
            "feature": feature,
            "family": classify_family(feature),
            "n": int(len(sub)),
            "auc_raw": float(auc_raw),
            "auc_symmetric": float(auc_symmetric),
            "mutual_info": mi,
            "cohen_d": float(d) if np.isfinite(d) else np.nan,
            "abs_cohen_d": float(abs(d)) if np.isfinite(d) else np.nan,
        }

        for image_id in LABELED_IDS:
            image_sub = sub[sub["image_id"] == image_id]
            if image_sub["target"].nunique() < 2 or image_sub[feature].nunique() < 2:
                row[f"auc_{image_id}"] = np.nan
            else:
                auc_i = roc_auc_score(image_sub["target"], image_sub[feature])
                row[f"auc_{image_id}"] = float(max(auc_i, 1.0 - auc_i))
        rows.append(row)

    out = pd.DataFrame(rows)
    out["rank_score"] = 0.5 * out["auc_symmetric"].rank(ascending=False, method="average") + 0.5 * out[
        "mutual_info"
    ].rank(ascending=False, method="average")
    return out.sort_values(["rank_score", "auc_symmetric", "mutual_info"], ascending=[True, False, False]).reset_index(drop=True)


def write_catalog(path: Path, ranked: pd.DataFrame) -> None:
    top = ranked.head(15)
    groups = {
        "Expert features": ranked[ranked["family"] == "expert"].head(5),
        "Radiance features": ranked[ranked["family"] == "radiance"].head(5),
        "Engineered scalar features": ranked[ranked["family"] == "engineered_scalar"].head(5),
        "Angle-pair features": ranked[ranked["family"] == "angle_pair"].head(5),
        "Patch-local features": ranked[ranked["family"] == "patch_local"].head(5),
    }

    lines: List[str] = []
    lines.append("# Part 2 Predictor Catalog\n")
    lines.append("## Meeting-ready shortlist\n")
    lines.append("Recommended predictors to discuss first:")
    for _, row in top.head(8).iterrows():
        lines.append(
            f"- `{row['feature']}` ({row['family']}): symmetric AUC={row['auc_symmetric']:.3f}, MI={row['mutual_info']:.3f}, |d|={row['abs_cohen_d']:.3f}"
        )

    lines.append("\n## Predictor families and why they matter\n")
    lines.append("- `expert`: paper-style features such as NDAI, SD, and CORR.")
    lines.append("- `radiance`: raw multi-angle radiance channels.")
    lines.append("- `engineered_scalar`: simple gaps, ratios, and interactions built from expert and radiance features.")
    lines.append("- `angle_pair`: extended angular relationships, covariance-style proxies, and angular spread measures from the merged group script.")
    lines.append("- `patch_local`: 3x3 neighborhood summaries that capture local texture and context.")

    lines.append("\n## Top candidates by family\n")
    for title, block in groups.items():
        lines.append(f"### {title}")
        if block.empty:
            lines.append("- none")
            continue
        for _, row in block.iterrows():
            lines.append(
                f"- `{row['feature']}`: symmetric AUC={row['auc_symmetric']:.3f}, MI={row['mutual_info']:.3f}, |d|={row['abs_cohen_d']:.3f}"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    data_dir = resolve_path(script_dir, args.data_dir)
    out_dir = resolve_path(script_dir, args.out_dir)
    docs_dir = resolve_path(script_dir, args.docs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    df = load_labeled_images(data_dir)
    df = add_pointwise_features(df)
    df = add_angle_pair_features(df)
    df = add_local_patch_features(df, window=args.local_window)

    feature_cols = [c for c in df.columns if c not in {"image_id", "label", "x", "y"}]
    ranked = screen_features(df, feature_cols)

    df.to_csv(out_dir / "labeled_engineered_features.csv", index=False)
    ranked.to_csv(out_dir / "feature_screening.csv", index=False)
    write_catalog(docs_dir / "predictor_catalog.md", ranked)

    print(f"[merged-part2] wrote {out_dir / 'labeled_engineered_features.csv'}")
    print(f"[merged-part2] wrote {out_dir / 'feature_screening.csv'}")
    print(f"[merged-part2] wrote {docs_dir / 'predictor_catalog.md'}")


if __name__ == "__main__":
    main()
