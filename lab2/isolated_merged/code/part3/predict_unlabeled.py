#!/usr/bin/env python3
"""Run the selected Part 3 model on fixed unlabeled images."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Sequence

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import (
    KEY_COLUMNS,
    UNLABELED_SANITY_IDS,
    build_unlabeled_feature_table,
    resolve_path,
    summarize_cloud_fraction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score fixed unlabeled images with the selected Part 3 model")
    parser.add_argument("--model_path", type=str, default="results/part3/models/final_model.joblib")
    parser.add_argument("--out_dir", type=str, default="results/part3/unlabeled_predictions")
    parser.add_argument("--image_ids", nargs="*", default=UNLABELED_SANITY_IDS)
    return parser.parse_args()


def predict_prob(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    raise TypeError(f"Model {type(model).__name__} does not expose probabilities or scores")


def make_grid(df: pd.DataFrame, value_col: str) -> np.ndarray:
    x = df["x"].to_numpy(dtype=int)
    y = df["y"].to_numpy(dtype=int)
    x0, y0 = x.min(), y.min()
    width = int(x.max() - x0 + 1)
    height = int(y.max() - y0 + 1)
    grid = np.full((height, width), np.nan, dtype=float)
    grid[y - y0, x - x0] = df[value_col].to_numpy(dtype=float)
    return grid


def save_heatmap(grid: np.ndarray, out_path: Path, title: str, cmap: str, vmin: float | None = None, vmax: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(grid, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    model_path = resolve_path(args.model_path)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle: Dict[str, object] = joblib.load(model_path)
    model = bundle["model"]
    feature_names = list(bundle["feature_names"])
    threshold = float(bundle["threshold"])

    feature_df = build_unlabeled_feature_table(image_ids=args.image_ids)
    X = feature_df.loc[:, feature_names].to_numpy(dtype=float)
    prob = predict_prob(model, X)
    feature_df["prob_cloud"] = prob
    feature_df["pred_label"] = (prob >= threshold).astype(int)

    for image_id, part in feature_df.groupby("image_id", sort=False):
        image_dir = out_dir / image_id
        image_dir.mkdir(parents=True, exist_ok=True)
        part.to_csv(image_dir / "pixel_predictions.csv", index=False)

        prob_grid = make_grid(part, "prob_cloud")
        mask_grid = make_grid(part, "pred_label")
        save_heatmap(
            prob_grid,
            image_dir / "cloud_probability_map.png",
            f"{image_id} cloud probability",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        save_heatmap(
            mask_grid,
            image_dir / "cloud_mask.png",
            f"{image_id} binary cloud mask",
            cmap="Greys",
            vmin=0.0,
            vmax=1.0,
        )

    summary = summarize_cloud_fraction(feature_df)
    summary.to_csv(out_dir / "cloud_fraction_summary.csv", index=False)
    print(f"[part3-unlabeled] wrote {out_dir}")


if __name__ == "__main__":
    main()
