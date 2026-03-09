#!/usr/bin/env python3
"""STAT 214 Lab 2 - Part 1 EDA for MISR cloud detection.

Generated artifacts (results in `results/part1/eda`, notes in `documents/part1` by default):
- label_map_<image_id>.png
- label_map_labeled_only_<image_id>.png
- radiance_corr_<scope>.png
- radiance_pairs_<scope>.png
- feature_dist_<feature>_<scope>.png
- feature_ranking_<scope>.csv
- feature_ranking.csv  (pooled alias)
- split_diagnostics.csv
- data_quality_report.csv
- summary.json

Run examples:
- python code/part1/eda.py
- python code/part1/eda.py --max_points 50000 --seed 214 --out_dir results/part1/eda --docs_dir documents/part1
- python -m code.part1.eda --max_points 30000
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns

    HAVE_SEABORN = True
except Exception:
    sns = None
    HAVE_SEABORN = False

try:
    from sklearn.feature_selection import mutual_info_classif

    HAVE_SKLEARN = True
except Exception:
    mutual_info_classif = None
    HAVE_SKLEARN = False

LABELED_IDS = ["O013257", "O013490", "O012791"]
BASE_COLUMNS = ["y", "x", "NDAI", "SD", "CORR", "DF", "CF", "BF", "AF", "AN"]
ALL_COLUMNS = BASE_COLUMNS + ["label"]
RADIANCE_FEATURES = ["DF", "CF", "BF", "AF", "AN"]
CLASS_FEATURES = ["NDAI", "SD", "CORR", "DF", "CF", "BF", "AF", "AN"]
LABEL_MAP = {1: "cloud", -1: "non_cloud", 0: "unlabeled"}
LABEL_COLORS = {1: "#1f77b4", -1: "#d62728", 0: "#bdbdbd"}


@dataclass
class LoadedImage:
    image_id: str
    df: pd.DataFrame
    source_key: str
    source_shape: Tuple[int, ...]


def log(msg: str) -> None:
    print(f"[EDA] {msg}")


def resolve_paths(out_dir_arg: str) -> Tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parents[2]
    data_dir = root / "data" / "image_data"
    out_dir = Path(out_dir_arg)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return root, data_dir, out_dir


def discover_helper_loader() -> Optional[str]:
    """Check whether code/data.py exposes a canonical tabular loader for NPZ images."""
    try:
        import importlib.util

        data_path = Path(__file__).resolve().parent / "data.py"
        if not data_path.exists():
            return None
        spec = importlib.util.spec_from_file_location("lab2_data_helper", str(data_path))
        if spec is None or spec.loader is None:
            return None
        helper_data = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper_data)

        if hasattr(helper_data, "load_image"):
            return "load_image"
        if hasattr(helper_data, "load_data"):
            return "load_data"
        if hasattr(helper_data, "make_data"):
            return "make_data"
    except Exception:
        return None
    return None


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
            f"No 2D array with 10/11 columns found. keys={npz_obj.files}, first_shape={getattr(arr, 'shape', None)}"
        )
    key, arr = max(candidates, key=lambda kv: kv[1].shape[0])
    return key, arr


def load_npz_as_df(npz_path: Path, force_labeled: bool = False) -> LoadedImage:
    with np.load(npz_path, allow_pickle=True) as npz_obj:
        log(f"Inspecting {npz_path.name}: keys={list(npz_obj.files)}")
        key, arr = pick_main_array(npz_obj)

    image_id = npz_path.stem
    arr = np.asarray(arr)
    if arr.shape[1] == 11:
        cols = ALL_COLUMNS
    elif arr.shape[1] == 10 and not force_labeled:
        cols = BASE_COLUMNS
    else:
        raise ValueError(f"Unexpected shape for {npz_path.name}: {arr.shape}")

    df = pd.DataFrame(arr, columns=cols)
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "label" not in df.columns:
        df["label"] = 0

    df["image_id"] = image_id
    df["label"] = df["label"].fillna(0).astype(int)
    return LoadedImage(image_id=image_id, df=df, source_key=key, source_shape=tuple(arr.shape))


def stratified_sample(df: pd.DataFrame, max_points: int, seed: int, label_col: str = "label") -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    rng = np.random.default_rng(seed)
    if label_col not in df.columns:
        idx = rng.choice(df.index.to_numpy(), size=max_points, replace=False)
        return df.loc[idx]

    counts = df[label_col].value_counts(dropna=False)
    allocations: Dict[int, int] = {}
    for label_value, count in counts.items():
        allocations[label_value] = max(1, int(round(max_points * (count / len(df)))))

    total = sum(allocations.values())
    while total > max_points:
        k = max(allocations, key=allocations.get)
        if allocations[k] > 1:
            allocations[k] -= 1
            total -= 1
        else:
            break
    while total < max_points:
        k = max(counts.index, key=lambda x: counts[x] - allocations.get(x, 0))
        allocations[k] = allocations.get(k, 0) + 1
        total += 1

    sampled_idx: List[int] = []
    for label_value, n in allocations.items():
        subset = df[df[label_col] == label_value]
        n = min(n, len(subset))
        if n == 0:
            continue
        sampled_idx.extend(rng.choice(subset.index.to_numpy(), size=n, replace=False).tolist())

    return df.loc[sampled_idx]


def is_integer_grid(df: pd.DataFrame, tol: float = 1e-6) -> bool:
    return (
        np.all(np.isfinite(df["x"]))
        and np.all(np.isfinite(df["y"]))
        and np.max(np.abs(df["x"] - np.round(df["x"]))) < tol
        and np.max(np.abs(df["y"] - np.round(df["y"]))) < tol
    )


def _render_label_scatter(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    for label_val in [0, -1, 1]:
        part = df[df["label"] == label_val]
        if len(part) == 0:
            continue
        ax.scatter(part["x"], part["y"], s=1.0, c=LABEL_COLORS[label_val], alpha=0.75, label=LABEL_MAP[label_val], rasterized=True)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.invert_yaxis()
    ax.legend(markerscale=6, fontsize=8, loc="upper right")


def save_label_maps(df: pd.DataFrame, out_dir: Path, image_id: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    _render_label_scatter(axes[0], df, f"Label Scatter: {image_id}")

    if is_integer_grid(df):
        x = np.round(df["x"]).astype(int)
        y = np.round(df["y"]).astype(int)
        x0, y0 = int(x.min()), int(y.min())
        grid = np.full((int(y.max() - y0 + 1), int(x.max() - x0 + 1)), np.nan)
        label_code = df["label"].map({-1: 0, 0: 1, 1: 2}).to_numpy()
        grid[(y - y0).to_numpy(), (x - x0).to_numpy()] = label_code
        cmap = matplotlib.colors.ListedColormap([LABEL_COLORS[-1], LABEL_COLORS[0], LABEL_COLORS[1]])
        axes[1].imshow(grid, cmap=cmap, interpolation="nearest", aspect="auto")
        axes[1].set_title(f"Label Grid Attempt: {image_id}")
        axes[1].set_xlabel("x (relative)")
        axes[1].set_ylabel("y (relative)")
    else:
        axes[1].text(0.5, 0.5, "Grid reconstruction skipped\n(non-integer x/y)", ha="center", va="center")
        axes[1].set_axis_off()

    fig.savefig(out_dir / f"label_map_{image_id}.png", dpi=200)
    plt.close(fig)

    labeled_only = df[df["label"].isin([-1, 1])]
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    _render_label_scatter(ax, labeled_only, f"Labeled Only: {image_id}")
    fig.savefig(out_dir / f"label_map_labeled_only_{image_id}.png", dpi=200)
    plt.close(fig)


def plot_corr_heatmaps(df: pd.DataFrame, features: Sequence[str], out_path: Path) -> None:
    subsets = {
        "overall": df,
        "cloud": df[df["label"] == 1],
        "non_cloud": df[df["label"] == -1],
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, (name, sub) in zip(axes, subsets.items()):
        corr = sub[list(features)].corr()
        im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title(f"{name} (n={len(sub)})")
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels(features, rotation=45, ha="right")
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_radiance_pairs(df: pd.DataFrame, features: Sequence[str], out_path: Path, max_points: int, seed: int) -> None:
    sample = stratified_sample(df[[*features, "label"]].dropna(), max_points=max_points, seed=seed)
    k = len(features)
    fig, axes = plt.subplots(k, k, figsize=(2.7 * k, 2.7 * k), constrained_layout=True)

    for i, f1 in enumerate(features):
        for j, f2 in enumerate(features):
            ax = axes[i, j]
            if i == j:
                vals = sample[f1].to_numpy()
                ax.hist(vals, bins=40, color="#4c78a8", alpha=0.8)
                ax.set_ylabel("")
            else:
                ax.hexbin(sample[f2], sample[f1], gridsize=35, mincnt=1, cmap="viridis")
            if i == k - 1:
                ax.set_xlabel(f2)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(f1)
            else:
                ax.set_yticklabels([])
    fig.suptitle(f"Radiance Pair Relationships (sample n={len(sample)})")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def cohen_d(x_pos: np.ndarray, x_neg: np.ndarray) -> float:
    x_pos = x_pos[np.isfinite(x_pos)]
    x_neg = x_neg[np.isfinite(x_neg)]
    if len(x_pos) < 2 or len(x_neg) < 2:
        return float("nan")
    m1, m0 = x_pos.mean(), x_neg.mean()
    s1, s0 = x_pos.std(ddof=1), x_neg.std(ddof=1)
    pooled = np.sqrt(((len(x_pos) - 1) * s1**2 + (len(x_neg) - 1) * s0**2) / (len(x_pos) + len(x_neg) - 2))
    if pooled == 0:
        return 0.0
    return float((m1 - m0) / pooled)


def binary_auc(y_true01: np.ndarray, scores: np.ndarray) -> float:
    mask = np.isfinite(scores) & np.isfinite(y_true01)
    y = y_true01[mask].astype(int)
    s = scores[mask]
    if len(y) == 0:
        return float("nan")
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    sum_pos = ranks[y == 1].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def plot_feature_dist(df_labeled: pd.DataFrame, feature: str, out_path: Path) -> None:
    clouds = df_labeled[df_labeled["label"] == 1][feature].to_numpy()
    non_clouds = df_labeled[df_labeled["label"] == -1][feature].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    bins = 50
    axes[0].hist(non_clouds[np.isfinite(non_clouds)], bins=bins, density=True, alpha=0.5, label="non_cloud", color=LABEL_COLORS[-1])
    axes[0].hist(clouds[np.isfinite(clouds)], bins=bins, density=True, alpha=0.5, label="cloud", color=LABEL_COLORS[1])
    axes[0].set_title(f"Histogram by class: {feature}")
    axes[0].set_xlabel(feature)
    axes[0].set_ylabel("density")
    axes[0].legend()

    if HAVE_SEABORN:
        sns.violinplot(data=df_labeled[["label", feature]].replace({"label": {-1: "non_cloud", 1: "cloud"}}), x="label", y=feature, ax=axes[1], inner="box")
    else:
        axes[1].boxplot(
            [non_clouds[np.isfinite(non_clouds)], clouds[np.isfinite(clouds)]],
            labels=["non_cloud", "cloud"],
            showfliers=False,
        )
    axes[1].set_title(f"Class spread: {feature}")
    axes[1].set_xlabel("label")
    axes[1].set_ylabel(feature)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def feature_metrics(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    use = df[df["label"].isin([-1, 1])].copy()
    if use.empty:
        return pd.DataFrame(columns=["feature", "mean_cloud", "mean_non_cloud", "cohen_d", "auc", "mutual_info"])
    y01 = (use["label"] == 1).astype(int).to_numpy()
    rows = []
    for feat in features:
        vals = use[feat].to_numpy()
        cloud_vals = use.loc[use["label"] == 1, feat].to_numpy()
        non_vals = use.loc[use["label"] == -1, feat].to_numpy()
        mean_cloud = float(np.nanmean(cloud_vals))
        mean_non = float(np.nanmean(non_vals))
        d = cohen_d(cloud_vals, non_vals)
        direction = 1.0 if mean_cloud >= mean_non else -1.0
        auc = binary_auc(y01, direction * vals)

        if HAVE_SKLEARN:
            valid = np.isfinite(vals)
            try:
                mi = float(mutual_info_classif(vals[valid].reshape(-1, 1), y01[valid], random_state=0, discrete_features=False)[0])
            except Exception:
                mi = float("nan")
        else:
            mi = float("nan")

        rows.append(
            {
                "feature": feat,
                "mean_cloud": mean_cloud,
                "mean_non_cloud": mean_non,
                "cohen_d": d,
                "abs_cohen_d": abs(d) if np.isfinite(d) else np.nan,
                "auc": auc,
                "auc_distance_from_0_5": abs(auc - 0.5) if np.isfinite(auc) else np.nan,
                "mutual_info": mi,
            }
        )
    ranked = pd.DataFrame(rows).sort_values(["abs_cohen_d", "auc_distance_from_0_5"], ascending=False)
    return ranked


def summarize_labels(df: pd.DataFrame) -> Dict[str, float]:
    counts = df["label"].value_counts().to_dict()
    total = len(df)
    cloud = int(counts.get(1, 0))
    non_cloud = int(counts.get(-1, 0))
    unlabeled = int(counts.get(0, 0))
    return {
        "n_total": total,
        "n_cloud": cloud,
        "n_non_cloud": non_cloud,
        "n_unlabeled": unlabeled,
        "p_cloud": cloud / total if total else 0.0,
        "p_non_cloud": non_cloud / total if total else 0.0,
        "p_unlabeled": unlabeled / total if total else 0.0,
    }


def quality_checks(df: pd.DataFrame, scope_name: str) -> Dict[str, float]:
    feature_df = df[CLASS_FEATURES]
    missing_cells = int(feature_df.isna().sum().sum())
    finite_mask = np.isfinite(feature_df.to_numpy(dtype=float))
    non_finite_cells = int((~finite_mask).sum())
    non_finite_rows = int((~finite_mask).any(axis=1).sum())

    dup_xy = int(df.duplicated(subset=["x", "y"]).sum())

    ndai_out = int(((df["NDAI"] < -1) | (df["NDAI"] > 1)).fillna(False).sum())
    corr_out = int(((df["CORR"] < 0) | (df["CORR"] > 1)).fillna(False).sum())
    rad_neg = int((df[RADIANCE_FEATURES] < 0).fillna(False).sum().sum())

    outlier_counts: Dict[str, int] = {}
    for feat in CLASS_FEATURES:
        vals = df[feat].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            outlier_counts[feat] = 0
            continue
        q_lo, q_hi = np.quantile(vals, [0.005, 0.995])
        count = int(((df[feat] < q_lo) | (df[feat] > q_hi)).fillna(False).sum())
        outlier_counts[feat] = count

    total_outlier_cells = int(sum(outlier_counts.values()))
    result: Dict[str, float] = {
        "scope": scope_name,
        "n_rows": int(len(df)),
        "missing_cells": missing_cells,
        "non_finite_cells": non_finite_cells,
        "non_finite_rows": non_finite_rows,
        "duplicate_xy_pairs": dup_xy,
        "ndai_out_of_range": ndai_out,
        "corr_out_of_range": corr_out,
        "radiance_negative_count": rad_neg,
        "total_outlier_cells_q005_q995": total_outlier_cells,
    }
    for feat, c in outlier_counts.items():
        result[f"outlier_{feat}"] = c
    return result


def compare_splits(train: pd.DataFrame, test: pd.DataFrame, strategy: str, split_id: str) -> pd.DataFrame:
    rows = []
    for feat in CLASS_FEATURES:
        tr = train[feat].to_numpy(dtype=float)
        te = test[feat].to_numpy(dtype=float)
        tr = tr[np.isfinite(tr)]
        te = te[np.isfinite(te)]
        tr_mean = float(np.mean(tr)) if len(tr) else np.nan
        te_mean = float(np.mean(te)) if len(te) else np.nan
        tr_std = float(np.std(tr, ddof=1)) if len(tr) > 1 else np.nan
        te_std = float(np.std(te, ddof=1)) if len(te) > 1 else np.nan
        pooled_std = np.sqrt(np.nanmean([tr_std**2, te_std**2])) if np.isfinite(tr_std) and np.isfinite(te_std) else np.nan
        smd = float((te_mean - tr_mean) / pooled_std) if np.isfinite(pooled_std) and pooled_std > 0 else np.nan
        rows.append(
            {
                "strategy": strategy,
                "split_id": split_id,
                "feature": feat,
                "train_n": len(train),
                "test_n": len(test),
                "train_mean": tr_mean,
                "test_mean": te_mean,
                "train_std": tr_std,
                "test_std": te_std,
                "standardized_mean_diff": smd,
                "abs_standardized_mean_diff": abs(smd) if np.isfinite(smd) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def split_diagnostics(labeled_by_image: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    results: List[pd.DataFrame] = []

    ids = list(labeled_by_image.keys())
    for holdout in ids:
        test = labeled_by_image[holdout].copy()
        train = pd.concat([labeled_by_image[i] for i in ids if i != holdout], ignore_index=True)
        results.append(compare_splits(train, test, "by_image", f"holdout_{holdout}"))

    for image_id, df in labeled_by_image.items():
        x_cut = float(np.quantile(df["x"], 0.8))
        train = df[df["x"] <= x_cut].copy()
        test = df[df["x"] > x_cut].copy()
        results.append(compare_splits(train, test, "spatial_within_image", f"{image_id}_x_gt_q80"))

    return pd.concat(results, ignore_index=True)


def write_split_notes(out_path: Path) -> None:
    text = """# Split Diagnostics Notes

## Why pixel-wise random split is risky
A random pixel split leaks spatially adjacent pixels from the same cloud structures into both train and test. Because nearby pixels are highly autocorrelated, this inflates test performance and understates generalization error.

## Recommended split strategies
1. By-image split: train on two labeled images and test on the held-out image.
2. Within-image spatial split: hold out a contiguous spatial block (for example, rightmost 20% of x).

## Cleaning recommendations (analysis-only stage)
- Keep all rows for core accounting, but exclude non-finite rows in plots/statistics.
- Track and investigate out-of-range NDAI/CORR values before modeling.
- Preserve duplicate (x, y) findings for QA and dedup policy in the modeling phase.
"""
    out_path.write_text(text, encoding="utf-8")


def run_eda(max_points: int, seed: int, out_dir_arg: str, docs_dir_arg: str) -> None:
    np.random.seed(seed)
    root, data_dir, out_dir = resolve_paths(out_dir_arg)
    docs_dir = Path(docs_dir_arg)
    if not docs_dir.is_absolute():
        docs_dir = root / docs_dir
    docs_dir.mkdir(parents=True, exist_ok=True)

    helper_name = discover_helper_loader()
    if helper_name == "make_data":
        log("Detected code/data.py::make_data, but it is patch-oriented and not suitable for labeled tabular EDA. Using robust NPZ loader.")
    elif helper_name:
        log(f"Detected helper loader '{helper_name}', but robust NPZ loader is used for consistent schema handling.")
    else:
        log("No canonical helper loader found in code/data.py. Using robust NPZ loader.")

    labeled_loaded: List[LoadedImage] = []
    for image_id in LABELED_IDS:
        npz_path = data_dir / f"{image_id}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing labeled file: {npz_path}")
        labeled_loaded.append(load_npz_as_df(npz_path, force_labeled=True))

    labeled_map: Dict[str, pd.DataFrame] = {}
    for item in labeled_loaded:
        df = item.df.copy()
        for col in ALL_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        labeled_map[item.image_id] = df

    pooled = pd.concat([labeled_map[i] for i in LABELED_IDS], ignore_index=True)

    summary: Dict[str, object] = {
        "seed": seed,
        "max_points": max_points,
        "paths": {
            "root": str(root),
            "data_dir": str(data_dir),
            "out_dir": str(out_dir),
        },
        "label_stats": {},
        "feature_rankings": {},
        "data_quality": {},
    }

    log("Step B: generating label maps and label counts")
    for image_id, df in labeled_map.items():
        save_label_maps(df, out_dir, image_id)
        summary["label_stats"][image_id] = summarize_labels(df)

    scopes: Dict[str, pd.DataFrame] = {**labeled_map, "pooled": pooled}

    log("Step C: radiance relationships, feature distributions, and separation metrics")
    for scope, df in scopes.items():
        finite = df.replace([np.inf, -np.inf], np.nan)
        finite = finite.dropna(subset=RADIANCE_FEATURES)

        plot_corr_heatmaps(finite, RADIANCE_FEATURES, out_dir / f"radiance_corr_{scope}.png")
        plot_radiance_pairs(finite, RADIANCE_FEATURES, out_dir / f"radiance_pairs_{scope}.png", max_points=max_points, seed=seed)

        labeled_only = df[df["label"].isin([-1, 1])].replace([np.inf, -np.inf], np.nan)
        for feat in CLASS_FEATURES:
            plot_feature_dist(
                labeled_only[["label", feat]].dropna(),
                feat,
                out_dir / f"feature_dist_{feat}_{scope}.png",
            )

        ranking = feature_metrics(df, CLASS_FEATURES)
        ranking.to_csv(out_dir / f"feature_ranking_{scope}.csv", index=False)
        if scope == "pooled":
            ranking.to_csv(out_dir / "feature_ranking.csv", index=False)
            summary["feature_rankings"]["pooled_top3_by_abs_cohen_d"] = ranking.head(3)[
                ["feature", "cohen_d", "auc", "mutual_info"]
            ].to_dict(orient="records")

    log("Step D: split diagnostics and notes")
    labeled_for_split = {k: v[v["label"].isin([-1, 1])].copy() for k, v in labeled_map.items()}
    split_df = split_diagnostics(labeled_for_split)
    split_df.to_csv(out_dir / "split_diagnostics.csv", index=False)
    write_split_notes(docs_dir / "split_notes.md")

    split_summary = (
        split_df.groupby(["strategy", "split_id"], as_index=False)["abs_standardized_mean_diff"].mean().sort_values("abs_standardized_mean_diff", ascending=False)
    )
    summary["split_shift_summary_top"] = split_summary.head(6).to_dict(orient="records")

    log("Step E: data quality checks")
    quality_rows = []
    for scope, df in scopes.items():
        q = quality_checks(df, scope)
        quality_rows.append(q)
        summary["data_quality"][scope] = q
    quality_df = pd.DataFrame(quality_rows)
    quality_df.to_csv(out_dir / "data_quality_report.csv", index=False)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log(f"EDA complete. Artifacts saved in: {out_dir}")
    log(f"Summary JSON: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STAT 214 Lab 2 Part 1 EDA (MISR cloud detection)")
    parser.add_argument("--max_points", type=int, default=50000, help="Max sampled points for heavy plots")
    parser.add_argument("--seed", type=int, default=214, help="Random seed for deterministic sampling")
    parser.add_argument("--out_dir", type=str, default="results/part1/eda", help="Output directory relative to lab2 root unless absolute")
    parser.add_argument("--docs_dir", type=str, default="documents/part1", help="Documentation directory relative to lab2 root unless absolute")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eda(max_points=args.max_points, seed=args.seed, out_dir_arg=args.out_dir, docs_dir_arg=args.docs_dir)
