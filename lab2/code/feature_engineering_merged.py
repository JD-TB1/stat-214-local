#!/usr/bin/env python3
"""
Part 2 Feature Engineering for STAT 214 Lab 2 (MISR cloud detection).

This script:
1) Loads the 3 labeled images (expert labels).
2) Builds a compact set of engineered predictors:
   - radiance aggregates / contrasts
   - simple ratios
   - angle-pair extensions
   - 3x3 local neighborhood summaries
3) Screens predictors with multiple univariate ranking metrics.
4) Writes:
   - results/part2/labeled_engineered_features.csv
   - results/part2/feature_screening.csv
   - documents/part2/predictor_catalog.md

Run:
  python feature_engineering.py
"""

from __future__ import annotations

# [ADDED] Borrowed the argument-parsing idea so output paths and window size
# are no longer hard-coded inside the script.
import argparse

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# [ADDED] Keep roc_auc_score, and add mutual_info_classif to make screening richer.
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif

from utils_npz import NPZUtils

LABELED_FILES = ["O013257.npz", "O013490.npz", "O012791.npz"]

# [CHANGED] Removed hard-coded RESULTS_DIR / DOCS_DIR creation here.
# We will build them inside main() using command-line arguments.

RAD_COLS = ["ra_df", "ra_cf", "ra_bf", "ra_af", "ra_an"]

# [ADDED] Explicit list of labeled image IDs for per-image AUC reporting.
LABELED_IDS = ["O013257", "O013490", "O012791"]

# [ADDED] Prefixes used to identify the special angle-pair family
# that is unique to your version.
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


# [ADDED] Argument parser borrowed from Selina's engineering style.
# This makes the script easier to reuse without editing source code.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Part 2 engineered predictors and screening tables"
    )
    parser.add_argument("--out_dir", type=str, default="../data/features", help="Directory for engineered features and screening results")
    parser.add_argument("--docs_dir", type=str, default="../data/features/docs", help="Directory for documentation files")
    parser.add_argument("--local_window", type=int, default=3)
    return parser.parse_args()


# [ADDED] Helper to resolve relative paths from the script directory.
def resolve_path(base: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (base / path).resolve()


def load_labeled_pixels(utils: NPZUtils, files: list[str]) -> pd.DataFrame:
    dfs = []
    for f in files:
        df = utils.load_img_to_df(f).copy()
        df["image_id"] = f.replace(".npz", "")
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)

    labeled_df = all_df[all_df["label"].isin([-1, 1])].copy()
    for c in ["x", "y"]:
        labeled_df[c] = labeled_df[c].astype(int)
    labeled_df["label"] = labeled_df["label"].astype(int)
    return labeled_df


# Akansha's unique function: add_angle_pair_features
def add_angle_pair_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add angle pair features. The paper identifies three features
    (NDAI, SD, CORR), this function extends those ideas.
    """
    out = df.copy()
    eps = 1e-6

    # the original ndai in paper uses df vs an angles,
    # so we are extending to cf vs an, bf vs an, af vs an and df vs af
    out["ndai_cf_an"] = (out["ra_cf"] - out["ra_an"]) / (out["ra_cf"] + out["ra_an"] + eps)
    out["ndai_bf_an"] = (out["ra_bf"] - out["ra_an"]) / (out["ra_bf"] + out["ra_an"] + eps)
    out["ndai_af_an"] = (out["ra_af"] - out["ra_an"]) / (out["ra_af"] + out["ra_an"] + eps)
    out["ndai_df_af"] = (out["ra_df"] - out["ra_af"]) / (out["ra_df"] + out["ra_af"] + eps)

    # high spread means angular asymmetry is inconsistent across zenith angles,
    # suggesting complex cloud structure or mixed scenes.
    ndai_cols = ["ndai", "ndai_cf_an", "ndai_bf_an", "ndai_af_an"]
    out["ndai_angular_range"] = out[ndai_cols].max(axis=1) - out[ndai_cols].min(axis=1)
    out["ndai_angular_std"] = out[ndai_cols].std(axis=1)

    # cross angle covariance proxies
    # af/an and bf/an already captured by CORR
    # so choosing df/an, cf/af and df/bf
    rad_mean_overall = out[RAD_COLS].mean(axis=1)
    out["cov_df_an"] = (out["ra_df"] - rad_mean_overall) * (out["ra_an"] - rad_mean_overall)
    out["cov_cf_af"] = (out["ra_cf"] - rad_mean_overall) * (out["ra_af"] - rad_mean_overall)
    out["cov_df_bf"] = (out["ra_df"] - rad_mean_overall) * (out["ra_bf"] - rad_mean_overall)

    # per angle SD proxies and their angular spread
    for col in RAD_COLS:
        angle = col.replace("ra_", "")
        out[f"sd_proxy_{angle}"] = (out[col] - rad_mean_overall).abs()
    sd_proxy_cols = [f"sd_proxy_{c.replace('ra_', '')}" for c in RAD_COLS]
    out["sd_angular_range"] = out[sd_proxy_cols].max(axis=1) - out[sd_proxy_cols].min(axis=1)
    out["sd_angular_cv"] = (
        out[sd_proxy_cols].std(axis=1) / (out[sd_proxy_cols].mean(axis=1).abs() + eps)
    )
    return out


def add_pointwise_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    eps = 1e-6

    # radiance aggregates
    out["rad_mean"] = out[RAD_COLS].mean(axis=1)
    out["rad_std"] = out[RAD_COLS].std(axis=1)
    out["rad_range"] = out[RAD_COLS].max(axis=1) - out[RAD_COLS].min(axis=1)
    out["rad_cv"] = out["rad_std"] / (out["rad_mean"].abs() + eps)

    # front vs aft blocks (DF/CF/BF vs AF/AN)
    out["front_mean"] = out[["ra_df", "ra_cf", "ra_bf"]].mean(axis=1)
    out["aft_mean"] = out[["ra_af", "ra_an"]].mean(axis=1)
    out["forward_backward_gap"] = out["front_mean"] - out["aft_mean"]
    out["front_back_ratio"] = (out["front_mean"] + eps) / (out["aft_mean"] + eps)

    # simple gaps / ratios
    out["df_an_ratio"] = out["ra_df"] / (out["ra_an"] + eps)
    out["af_an_ratio"] = out["ra_af"] / (out["ra_an"] + eps)
    out["an_df_gap"] = out["ra_an"] - out["ra_df"]
    out["af_df_gap"] = out["ra_af"] - out["ra_df"]

    # [ADDED] Borrowed two simple gap features from Selina's version.
    # These are cheap to compute and consistent with your current design.
    out["bf_cf_gap"] = out["ra_bf"] - out["ra_cf"]
    out["an_af_gap"] = out["ra_an"] - out["ra_af"]

    # simple interactions
    out["ndai_x_sd"] = out["ndai"] * out["sd"]
    out["corr_x_sd"] = out["corr"] * out["sd"]
    out["corr_x_ndai"] = out["corr"] * out["ndai"]

    return out


def add_local_3x3_features(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Add local neighborhood summaries for a small set of base columns.
    Uses reflect padding and a safe fallback when a window has no finite values.

    For each base column, adds:
      - local_<col>_mean3
      - local_<col>_std3
      - local_<col>_centered3  (center - local_mean)
    """
    if window % 2 == 0:
        raise ValueError("window must be odd (e.g., 3)")

    radius = window // 2
    base_local_cols = ["ndai", "sd", "corr", "rad_mean", "rad_std"]
    out_parts = []

    for image_id, part in df.groupby("image_id", sort=False):
        img = part.copy()
        x = img["x"].to_numpy(dtype=int)
        y = img["y"].to_numpy(dtype=int)

        x0, y0 = x.min(), y.min()
        width = int(x.max() - x0 + 1)
        height = int(y.max() - y0 + 1)
        xr = x - x0
        yr = y - y0

        for col in base_local_cols:
            grid = np.full((height, width), np.nan, dtype=float)
            grid[yr, xr] = img[col].to_numpy(dtype=float)

            # [KEPT] We intentionally keep your reflect padding because it is
            # more stable at image edges than constant-NaN padding.
            padded = np.pad(grid, radius, mode="reflect")
            windows = sliding_window_view(padded, (window, window))

            # Suppress warnings from nanmean/nanstd on all-NaN windows
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Mean of empty slice")
                warnings.filterwarnings("ignore", message="Degrees of freedom*")
                local_mean = np.nanmean(windows, axis=(-2, -1))
                local_std = np.nanstd(windows, axis=(-2, -1))

            # gather per-pixel stats
            center_vals = grid[yr, xr]
            m = local_mean[yr, xr]
            s = local_std[yr, xr]

            # [KEPT] Your fallback logic is preserved because it makes local
            # features more robust when a neighborhood has no finite values.
            m = np.where(np.isfinite(m), m, np.where(np.isfinite(center_vals), center_vals, 0.0))
            s = np.where(np.isfinite(s), s, 0.0)

            img[f"local_{col}_mean{window}"] = m
            img[f"local_{col}_std{window}"] = s
            img[f"local_{col}_centered{window}"] = center_vals - m

        out_parts.append(img)

    return pd.concat(out_parts, ignore_index=True)


# [ADDED] Effect-size helper borrowed from Selina's screening logic.
# This gives an extra measure of class separation beyond AUC.
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


# [ADDED] Feature-family classification borrowed from Selina's reporting style.
# I added a separate "angle_pair" family so your unique contribution stays visible.
def classify_family(feature: str) -> str:
    if feature in {"ndai", "sd", "corr"}:
        return "expert"
    if feature in RAD_COLS:
        return "radiance"
    if feature.startswith("local_"):
        return "patch_local"
    if feature.startswith(ANGLE_PAIR_PREFIXES):
        return "angle_pair"
    return "engineered_scalar"


# [CHANGED] Upgraded screening from only AUC to a richer table:
# - family
# - n
# - auc_raw
# - auc_symmetric
# - mutual_info
# - cohen_d
# - abs_cohen_d
# - per-image AUC
# - rank_score
def screen_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []

    for f in feature_cols:
        sub = df[[f, "label", "image_id"]].replace([np.inf, -np.inf], np.nan).dropna()
        sub = sub[sub["label"].isin([-1, 1])].copy()

        if sub.empty or sub[f].nunique() < 2:
            continue

        sub["target"] = (sub["label"] == 1).astype(int)
        x = sub[f].to_numpy(dtype=float)
        t = sub["target"].to_numpy(dtype=int)

        if len(np.unique(t)) < 2 or len(np.unique(x)) < 2:
            continue

        auc = roc_auc_score(t, x)
        auc_sym = max(auc, 1 - auc)
        mi = float(
            mutual_info_classif(
                x.reshape(-1, 1),
                t,
                discrete_features=False,
                random_state=214,
            )[0]
        )
        d = cohen_d(
            sub.loc[sub["label"] == 1, f].to_numpy(),
            sub.loc[sub["label"] == -1, f].to_numpy(),
        )

        row = {
            "feature": f,
            "family": classify_family(f),
            "n": int(len(sub)),
            "auc_raw": float(auc),
            "auc_symmetric": float(auc_sym),
            "mutual_info": mi,
            "cohen_d": float(d) if np.isfinite(d) else np.nan,
            "abs_cohen_d": float(abs(d)) if np.isfinite(d) else np.nan,
        }

        # [ADDED] Per-image AUC helps check whether a feature is stable
        # across the three labeled images instead of only looking good overall.
        for image_id in LABELED_IDS:
            image_sub = sub[sub["image_id"] == image_id]
            if image_sub["target"].nunique() < 2 or image_sub[f].nunique() < 2:
                row[f"auc_{image_id}"] = np.nan
            else:
                auc_i = roc_auc_score(image_sub["target"], image_sub[f])
                row[f"auc_{image_id}"] = float(max(auc_i, 1 - auc_i))

        rows.append(row)

    out = pd.DataFrame(rows)

    # [ADDED] Combined rank score:
    # smaller rank_score means better overall according to both AUC and MI.
    out["rank_score"] = (
        0.5 * out["auc_symmetric"].rank(ascending=False, method="average")
        + 0.5 * out["mutual_info"].rank(ascending=False, method="average")
    )

    return out.sort_values(
        ["rank_score", "auc_symmetric", "mutual_info"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


# [CHANGED] Replaced the short summary markdown with a richer predictor catalog.
# This is more meeting-friendly and preserves your new angle-pair family.
def write_catalog(path: Path, ranked: pd.DataFrame) -> None:
    top = ranked.head(15)

    groups = {
        "Expert features": ranked[ranked["family"] == "expert"].head(5),
        "Radiance features": ranked[ranked["family"] == "radiance"].head(5),
        "Engineered scalar features": ranked[ranked["family"] == "engineered_scalar"].head(5),
        "Angle-pair features": ranked[ranked["family"] == "angle_pair"].head(5),
        "Patch-local features": ranked[ranked["family"] == "patch_local"].head(5),
    }

    lines = []
    lines.append("# Part 2 Predictor Catalog\n")
    lines.append("## Meeting-ready shortlist\n")
    lines.append("Recommended predictors to discuss first:")

    for _, row in top.head(8).iterrows():
        lines.append(
            f"- `{row['feature']}` ({row['family']}): "
            f"symmetric AUC={row['auc_symmetric']:.3f}, "
            f"MI={row['mutual_info']:.3f}, "
            f"|d|={row['abs_cohen_d']:.3f}"
        )

    lines.append("\n## Predictor families and why they matter\n")
    lines.append("- `expert`: original paper-style predictors such as NDAI, SD, and CORR.")
    lines.append("- `radiance`: raw multi-angle radiance channels.")
    lines.append("- `engineered_scalar`: simple gaps, ratios, and interactions built from expert and radiance features.")
    lines.append("- `angle_pair`: extended angular relationships, covariance-style proxies, and angular spread measures.")
    lines.append("- `patch_local`: 3x3 neighborhood summaries that capture local texture and context.")

    lines.append("\n## Top candidates by family\n")
    for title, block in groups.items():
        lines.append(f"### {title}")
        if block.empty:
            lines.append("- none")
            continue
        for _, row in block.iterrows():
            lines.append(
                f"- `{row['feature']}`: "
                f"symmetric AUC={row['auc_symmetric']:.3f}, "
                f"MI={row['mutual_info']:.3f}, "
                f"|d|={row['abs_cohen_d']:.3f}"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    # [ADDED] Parse runtime arguments here.
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    # [ADDED] Build output/document directories dynamically.
    out_dir = resolve_path(script_dir, args.out_dir)
    docs_dir = resolve_path(script_dir, args.docs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    utils = NPZUtils()
    labeled_df = load_labeled_pixels(utils, LABELED_FILES)

    # [KEPT] Your feature-generation order stays the same:
    # pointwise -> angle-pair -> local summaries.
    feat_df = add_pointwise_features(labeled_df)
    feat_df = add_angle_pair_features(feat_df)  # Akansha's function
    feat_df = add_local_3x3_features(feat_df, window=args.local_window)

    exclude = {"label", "image_id", "x", "y"}
    feature_cols = [c for c in feat_df.columns if c not in exclude]
    ranked = screen_features(feat_df, feature_cols)

    out_features = out_dir / "labeled_engineered_features.csv"
    out_rank = out_dir / "feature_screening.csv"
    out_md = docs_dir / "predictor_catalog.md"

    feat_df.to_csv(out_features, index=False)
    ranked.to_csv(out_rank, index=False)
    write_catalog(out_md, ranked)

    print(f"Wrote: {out_features}")
    print(f"Wrote: {out_rank}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()