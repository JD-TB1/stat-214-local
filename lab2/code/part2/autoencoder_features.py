#!/usr/bin/env python3
"""Part 2 autoencoder-derived feature extraction for STAT 214 Lab 2.

This script uses code/original/autoencoder.py as the starting point for patch-based
feature engineering. It builds 9x9 patches around supervised pixels, loads the
existing checkpoint if available, extracts latent coordinates, and screens the
embedding dimensions as candidate predictors.

Outputs:
- results (default: `../../results/part2`)
- autoencoder_embeddings_supervised.csv
- autoencoder_feature_screening.csv
- documentation (default: `../../documents/part2`)
- autoencoder_feature_notes.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

import importlib.util


def load_autoencoder_class():
    autoencoder_path = Path(__file__).resolve().parents[1] / "original" / "autoencoder.py"
    spec = importlib.util.spec_from_file_location("lab2_original_autoencoder", autoencoder_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load autoencoder from {autoencoder_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Autoencoder


Autoencoder = load_autoencoder_class()

LABELED_IDS = ["O013257", "O013490", "O012791"]
BASE_COLUMNS = ["y", "x", "NDAI", "SD", "CORR", "DF", "CF", "BF", "AF", "AN", "label"]
FEATURE_COLS = ["NDAI", "SD", "CORR", "DF", "CF", "BF", "AF", "AN"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract autoencoder features for supervised rows")
    parser.add_argument("--data_dir", type=str, default="../../data/image_data")
    parser.add_argument("--out_dir", type=str, default="../../results/part2")
    parser.add_argument("--docs_dir", type=str, default="../../documents/part2")
    parser.add_argument("--checkpoint", type=str, default="../original/checkpoints/gsi-model.ckpt")
    parser.add_argument("--patch_size", type=int, default=9)
    parser.add_argument("--embedding_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=2048)
    return parser.parse_args()


def resolve_path(base: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (base / raw).resolve()


def pick_main_array(npz_obj: np.lib.npyio.NpzFile) -> np.ndarray:
    candidates: List[np.ndarray] = []
    for key in npz_obj.files:
        arr = npz_obj[key]
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] in (10, 11):
            candidates.append(arr)
    if not candidates:
        raise ValueError(f"No usable 2D array found in keys={list(npz_obj.files)}")
    return max(candidates, key=lambda arr: arr.shape[0])


def load_images(data_dir: Path) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for image_id in LABELED_IDS:
        path = data_dir / f"{image_id}.npz"
        with np.load(path, allow_pickle=True) as npz_obj:
            arr = np.asarray(pick_main_array(npz_obj))
        if arr.shape[1] != 11:
            raise ValueError(f"Expected 11 columns for labeled image {image_id}, got {arr.shape}")
        df = pd.DataFrame(arr, columns=BASE_COLUMNS)
        for col in BASE_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["label"] = df["label"].astype(int)
        df["image_id"] = image_id
        out[image_id] = df
    return out


def compute_channel_norm(images: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate([df[FEATURE_COLS].to_numpy(dtype=np.float32) for df in images.values()], axis=0)
    means = stacked.mean(axis=0)
    stds = stacked.std(axis=0)
    stds[stds == 0] = 1.0
    return means.astype(np.float32), stds.astype(np.float32)


def build_normalized_grid(df: pd.DataFrame, means: np.ndarray, stds: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = df["x"].to_numpy(dtype=int)
    y = df["y"].to_numpy(dtype=int)
    x0, y0 = x.min(), y.min()
    width = int(x.max() - x0 + 1)
    height = int(y.max() - y0 + 1)
    xr = x - x0
    yr = y - y0

    grid = np.zeros((len(FEATURE_COLS), height, width), dtype=np.float32)
    raw = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    norm = (raw - means[None, :]) / stds[None, :]
    for channel in range(len(FEATURE_COLS)):
        grid[channel, yr, xr] = norm[:, channel]
    return grid, yr, xr


def extract_patches_for_supervised(
    image_id: str,
    df: pd.DataFrame,
    means: np.ndarray,
    stds: np.ndarray,
    patch_size: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    radius = patch_size // 2
    grid, yr, xr = build_normalized_grid(df, means, stds)
    padded = np.pad(grid, ((0, 0), (radius, radius), (radius, radius)), mode="reflect")

    supervised = df[df["label"].isin([-1, 1])].copy().reset_index(drop=True)
    sx = supervised["x"].to_numpy(dtype=int)
    sy = supervised["y"].to_numpy(dtype=int)
    x0, y0 = int(df["x"].min()), int(df["y"].min())
    sxr = sx - x0 + radius
    syr = sy - y0 + radius

    patches = np.empty((len(supervised), len(FEATURE_COLS), patch_size, patch_size), dtype=np.float32)
    for idx, (yy, xx) in enumerate(zip(syr, sxr, strict=False)):
        patches[idx] = padded[:, yy - radius : yy + radius + 1, xx - radius : xx + radius + 1]
    supervised["image_id"] = image_id
    return patches, supervised


def cohen_d(x1: np.ndarray, x0: np.ndarray) -> float:
    if len(x1) < 2 or len(x0) < 2:
        return np.nan
    v1 = np.var(x1, ddof=1)
    v0 = np.var(x0, ddof=1)
    pooled = ((len(x1) - 1) * v1 + (len(x0) - 1) * v0) / (len(x1) + len(x0) - 2)
    if pooled <= 0 or not np.isfinite(pooled):
        return np.nan
    return (np.mean(x1) - np.mean(x0)) / np.sqrt(pooled)


def rank_embeddings(df: pd.DataFrame, embed_cols: Sequence[str]) -> pd.DataFrame:
    target = (df["label"] == 1).astype(int).to_numpy(dtype=int)
    rows: List[Dict[str, float | str]] = []
    for col in embed_cols:
        values = df[col].to_numpy(dtype=float)
        auc = roc_auc_score(target, values)
        auc_sep = max(auc, 1.0 - auc)
        mi = float(mutual_info_classif(values.reshape(-1, 1), target, discrete_features=False, random_state=214)[0])
        d = cohen_d(df.loc[df["label"] == 1, col].to_numpy(), df.loc[df["label"] == -1, col].to_numpy())
        rows.append(
            {
                "feature": col,
                "auc": float(auc),
                "auc_separation": float(auc_sep),
                "mutual_info": mi,
                "cohen_d": float(d) if np.isfinite(d) else np.nan,
                "abs_cohen_d": float(abs(d)) if np.isfinite(d) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out["rank_score"] = 0.5 * out["auc_separation"].rank(ascending=False) + 0.5 * out["mutual_info"].rank(ascending=False)
    return out.sort_values(["rank_score", "auc_separation", "mutual_info"], ascending=[True, False, False]).reset_index(drop=True)


def write_notes(path: Path, ranked: pd.DataFrame, checkpoint: Path) -> None:
    top = ranked.head(5)
    lines: List[str] = []
    lines.append("# Autoencoder Feature Notes\n")
    lines.append(f"Checkpoint used: `{checkpoint}`")
    lines.append("\n## What this script does")
    lines.append("- Reuses the patch-based autoencoder architecture from `code/autoencoder.py`.")
    lines.append("- Builds 9x9 patches around supervised pixels from the eight raw MISR channels/features.")
    lines.append("- Screens each latent coordinate as a candidate Part 2 predictor.")
    lines.append("\n## Meeting-ready autoencoder candidates")
    for _, row in top.iterrows():
        lines.append(
            f"- `{row['feature']}`: separation AUC={row['auc_separation']:.3f}, MI={row['mutual_info']:.3f}, |d|={row['abs_cohen_d']:.3f}"
        )
    lines.append("\n## How to use these in Part 2/3")
    lines.append("- Treat the embedding dimensions as complementary predictors alongside interpretable expert and radiance features.")
    lines.append("- Keep only the strongest latent dimensions instead of all eight if you want lower-variance downstream models.")
    lines.append("- Prefer combining embeddings with NDAI, SD, and a small radiance contrast set rather than relying on embeddings alone.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    data_dir = resolve_path(script_dir, args.data_dir)
    out_dir = resolve_path(script_dir, args.out_dir)
    docs_dir = resolve_path(script_dir, args.docs_dir)
    checkpoint = resolve_path(script_dir, args.checkpoint)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    images = load_images(data_dir)
    means, stds = compute_channel_norm(images)

    patches_list: List[np.ndarray] = []
    meta_list: List[pd.DataFrame] = []
    for image_id in LABELED_IDS:
        patches, meta = extract_patches_for_supervised(image_id, images[image_id], means, stds, args.patch_size)
        patches_list.append(patches)
        meta_list.append(meta)

    patches_all = np.concatenate(patches_list, axis=0)
    meta_all = pd.concat(meta_list, ignore_index=True)

    model = Autoencoder(n_input_channels=len(FEATURE_COLS), patch_size=args.patch_size, embedding_size=args.embedding_size)
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    batches: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(patches_all), args.batch_size):
            end = start + args.batch_size
            batch = torch.from_numpy(patches_all[start:end])
            emb = model.embed(batch).detach().cpu().numpy()
            batches.append(emb)

    embeddings = np.concatenate(batches, axis=0)
    embed_cols = [f"ae{i}" for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=embed_cols)
    out_df = pd.concat([meta_all[["image_id", "x", "y", "label"]].reset_index(drop=True), emb_df], axis=1)
    ranked = rank_embeddings(out_df, embed_cols)

    out_df.to_csv(out_dir / "autoencoder_embeddings_supervised.csv", index=False)
    ranked.to_csv(out_dir / "autoencoder_feature_screening.csv", index=False)
    write_notes(docs_dir / "autoencoder_feature_notes.md", ranked, checkpoint)

    print(f"[part2-ae] wrote {out_dir / 'autoencoder_embeddings_supervised.csv'}")
    print(f"[part2-ae] wrote {out_dir / 'autoencoder_feature_screening.csv'}")
    print(f"[part2-ae] wrote {docs_dir / 'autoencoder_feature_notes.md'}")


if __name__ == "__main__":
    main()
