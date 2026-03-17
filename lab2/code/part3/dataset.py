#!/usr/bin/env python3
"""Dataset and feature helpers for Lab 2 Part 3 predictive modeling."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PART2_FEATURE = _load_module(
    "lab2_part2_feature_engineering",
    Path(__file__).resolve().parents[1] / "part2" / "feature_engineering.py",
)
_PART2_AE = _load_module(
    "lab2_part2_autoencoder_features",
    Path(__file__).resolve().parents[1] / "part2" / "autoencoder_features.py",
)

Autoencoder = _PART2_AE.Autoencoder
build_normalized_grid = _PART2_AE.build_normalized_grid
compute_channel_norm = _PART2_AE.compute_channel_norm
load_labeled_images_for_norm = _PART2_AE.load_images
add_local_patch_features = _PART2_FEATURE.add_local_patch_features
add_pointwise_features = _PART2_FEATURE.add_pointwise_features
pick_main_array = _PART2_FEATURE.pick_main_array

LABELED_IDS = ["O013257", "O013490", "O012791"]
UNLABELED_SANITY_IDS = ["O002539", "O045178", "O119738"]
BASE_COLUMNS = ["y", "x", "NDAI", "SD", "CORR", "DF", "CF", "BF", "AF", "AN"]
LABELED_COLUMNS = BASE_COLUMNS + ["label"]
KEY_COLUMNS = ["image_id", "x", "y", "label"]
AE_COLUMNS = [f"ae{i}" for i in range(8)]
AE_INPUT_COLUMNS = ["NDAI", "SD", "CORR", "DF", "CF", "BF", "AF", "AN"]

FEATURE_BLOCKS: Dict[str, List[str]] = {
    "B0_base": ["SD", "NDAI", "AF", "AN", "BF"],
    "B1_engineered": [
        "SD",
        "NDAI",
        "AF",
        "AN",
        "BF",
        "ndai_x_sd",
        "af_df_gap",
        "front_back_ratio",
        "rad_cv",
        "rad_range",
    ],
    "B2_context": [
        "SD",
        "NDAI",
        "AF",
        "AN",
        "BF",
        "ndai_x_sd",
        "af_df_gap",
        "front_back_ratio",
        "rad_cv",
        "rad_range",
        "local_SD_mean3",
        "local_NDAI_std3",
        "local_SD_std3",
        "local_rad_std_std3",
        "local_rad_mean_std3",
    ],
    "B3_context_ae": [
        "SD",
        "NDAI",
        "AF",
        "AN",
        "BF",
        "ndai_x_sd",
        "af_df_gap",
        "front_back_ratio",
        "rad_cv",
        "rad_range",
        "local_SD_mean3",
        "local_NDAI_std3",
        "local_SD_std3",
        "local_rad_std_std3",
        "local_rad_mean_std3",
        "ae4",
        "ae5",
        "ae0",
    ],
}


@dataclass(frozen=True)
class SplitTables:
    split_id: str
    strategy: str
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def lab_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (lab_root() / path).resolve()


def default_engineered_path() -> Path:
    return lab_root() / "results" / "part2" / "labeled_engineered_features.csv"


def default_embedding_path() -> Path:
    return lab_root() / "results" / "part2" / "autoencoder_embeddings_supervised.csv"


def default_split_root() -> Path:
    return lab_root() / "results" / "part1" / "splits"


def default_data_dir() -> Path:
    return lab_root() / "data" / "image_data"


def default_checkpoint_path() -> Path:
    return lab_root() / "code" / "original" / "checkpoints" / "gsi-model.ckpt"


def _coerce_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["image_id"] = out["image_id"].astype(str)
    out["x"] = pd.to_numeric(out["x"], errors="raise").astype(float)
    out["y"] = pd.to_numeric(out["y"], errors="raise").astype(float)
    out["label"] = pd.to_numeric(out["label"], errors="raise").astype(int)
    return out


def load_supervised_feature_table(
    engineered_path: str | Path | None = None,
    embedding_path: str | Path | None = None,
) -> pd.DataFrame:
    engineered_path = resolve_path(engineered_path or default_engineered_path())
    embedding_path = resolve_path(embedding_path or default_embedding_path())

    engineered = pd.read_csv(engineered_path)
    embeddings = pd.read_csv(embedding_path)

    engineered = _coerce_key_columns(engineered)
    embeddings = _coerce_key_columns(embeddings)
    engineered = engineered[engineered["label"].isin([-1, 1])].copy()

    if engineered.duplicated(KEY_COLUMNS).any():
        dupes = engineered.loc[engineered.duplicated(KEY_COLUMNS, keep=False), KEY_COLUMNS]
        raise ValueError(f"Duplicate supervised keys in engineered table:\n{dupes.head()}")
    if embeddings.duplicated(KEY_COLUMNS).any():
        dupes = embeddings.loc[embeddings.duplicated(KEY_COLUMNS, keep=False), KEY_COLUMNS]
        raise ValueError(f"Duplicate supervised keys in embedding table:\n{dupes.head()}")

    merged = engineered.merge(embeddings, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if len(merged) != len(engineered):
        raise ValueError(
            f"Supervised merge dropped rows: engineered={len(engineered)} merged={len(merged)}"
        )

    merged["target"] = (merged["label"] == 1).astype(int)
    merged = merged.sort_values(["image_id", "y", "x"]).reset_index(drop=True)

    missing = set(FEATURE_BLOCKS["B3_context_ae"]) - set(merged.columns)
    if missing:
        raise ValueError(f"Missing expected modeling features: {sorted(missing)}")

    return merged


def check_supervised_feature_table(df: pd.DataFrame) -> Dict[str, int]:
    duplicate_keys = int(df.duplicated(KEY_COLUMNS).sum())
    missing_selected = int(df[FEATURE_BLOCKS["B3_context_ae"]].isna().any(axis=1).sum())
    return {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "duplicate_key_rows": duplicate_keys,
        "rows_with_missing_selected_features": missing_selected,
        "n_cloud": int((df["target"] == 1).sum()),
        "n_non_cloud": int((df["target"] == 0).sum()),
    }


def load_split_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(resolve_path(path))
    df["image_id"] = df["image_id"].astype(str)
    df["x"] = pd.to_numeric(df["x"], errors="raise").astype(float)
    df["y"] = pd.to_numeric(df["y"], errors="raise").astype(float)
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    df["target"] = (df["label"] == 1).astype(int)
    return df


def align_split_with_features(split_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    split_keys = split_df.loc[:, KEY_COLUMNS + ["target"]].copy()
    aligned = split_keys.merge(features_df, on=KEY_COLUMNS + ["target"], how="left", validate="one_to_one")
    if len(aligned) != len(split_df):
        raise ValueError(f"Aligned split row count mismatch: split={len(split_df)} aligned={len(aligned)}")
    feature_na = aligned[FEATURE_BLOCKS["B3_context_ae"]].isna().any(axis=1)
    if feature_na.any():
        raise ValueError(
            f"Split alignment left missing selected features for {int(feature_na.sum())} rows"
        )
    return aligned


def load_split_tables(
    split_id: str,
    strategy: str,
    features_df: pd.DataFrame,
    split_root: str | Path | None = None,
) -> SplitTables:
    split_root = resolve_path(split_root or default_split_root())
    split_dir = split_root / strategy / split_id
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    train = align_split_with_features(load_split_csv(split_dir / "train.csv"), features_df)
    val = align_split_with_features(load_split_csv(split_dir / "val.csv"), features_df)
    test = align_split_with_features(load_split_csv(split_dir / "test.csv"), features_df)
    return SplitTables(split_id=split_id, strategy=strategy, train=train, val=val, test=test)


def list_split_ids(strategy: str, split_root: str | Path | None = None) -> List[str]:
    split_root = resolve_path(split_root or default_split_root())
    strategy_dir = split_root / strategy
    if not strategy_dir.exists():
        return []
    return sorted(path.name for path in strategy_dir.iterdir() if path.is_dir())


def get_primary_by_image_split_ids(split_root: str | Path | None = None) -> List[str]:
    return list_split_ids("by_image", split_root=split_root)


def get_secondary_spatial_split_ids(split_root: str | Path | None = None) -> List[str]:
    split_root = resolve_path(split_root or default_split_root())
    candidates = list_split_ids("spatial_within_image", split_root=split_root)
    usable: List[str] = []
    for split_id in candidates:
        test_df = load_split_csv(split_root / "spatial_within_image" / split_id / "test.csv")
        if test_df["target"].nunique() == 2:
            usable.append(split_id)
    return usable


def collect_split_integrity(
    features_df: pd.DataFrame,
    split_root: str | Path | None = None,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    split_root = resolve_path(split_root or default_split_root())
    for strategy in ["by_image", "spatial_within_image"]:
        for split_id in list_split_ids(strategy, split_root=split_root):
            split_dir = split_root / strategy / split_id
            for subset in ["train", "val", "test"]:
                subset_df = align_split_with_features(load_split_csv(split_dir / f"{subset}.csv"), features_df)
                rows.append(
                    {
                        "strategy": strategy,
                        "split_id": split_id,
                        "subset": subset,
                        "n_rows": int(len(subset_df)),
                        "n_cloud": int(subset_df["target"].sum()),
                        "n_non_cloud": int((1 - subset_df["target"]).sum()),
                        "n_missing_selected": int(
                            subset_df[FEATURE_BLOCKS["B3_context_ae"]].isna().any(axis=1).sum()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def prepare_feature_matrices(
    split_tables: SplitTables,
    feature_names: Sequence[str],
) -> Dict[str, np.ndarray]:
    missing = set(feature_names) - set(split_tables.train.columns)
    if missing:
        raise ValueError(f"Missing requested features in split tables: {sorted(missing)}")

    out: Dict[str, np.ndarray] = {}
    for subset_name in ["train", "val", "test"]:
        subset = getattr(split_tables, subset_name)
        out[f"X_{subset_name}"] = subset.loc[:, feature_names].to_numpy(dtype=float)
        out[f"y_{subset_name}"] = subset["target"].to_numpy(dtype=int)
    return out


def load_npz_frame(image_id: str, data_dir: str | Path | None = None) -> pd.DataFrame:
    data_dir = resolve_path(data_dir or default_data_dir())
    npz_path = data_dir / f"{image_id}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing image file: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as npz_obj:
        arr = np.asarray(pick_main_array(npz_obj))

    if arr.shape[1] == 11:
        columns = LABELED_COLUMNS
    elif arr.shape[1] == 10:
        columns = BASE_COLUMNS
    else:
        raise ValueError(f"Unexpected array shape for {image_id}: {arr.shape}")

    df = pd.DataFrame(arr, columns=columns)
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "label" not in df.columns:
        df["label"] = 0
    df["image_id"] = image_id
    return df


def load_unlabeled_images(
    image_ids: Sequence[str],
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    parts = [load_npz_frame(image_id, data_dir=data_dir) for image_id in image_ids]
    return pd.concat(parts, ignore_index=True)


def _load_autoencoder_model(
    checkpoint_path: str | Path | None = None,
    embedding_size: int = 8,
    patch_size: int = 9,
) -> Autoencoder:
    checkpoint_path = resolve_path(checkpoint_path or default_checkpoint_path())
    model = Autoencoder(
        n_input_channels=len(AE_INPUT_COLUMNS),
        patch_size=patch_size,
        embedding_size=embedding_size,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _extract_embeddings_for_frame(
    df: pd.DataFrame,
    means: np.ndarray,
    stds: np.ndarray,
    checkpoint_path: str | Path | None = None,
    patch_size: int = 9,
    embedding_size: int = 8,
    batch_size: int = 2048,
) -> pd.DataFrame:
    model = _load_autoencoder_model(
        checkpoint_path=checkpoint_path,
        embedding_size=embedding_size,
        patch_size=patch_size,
    )

    radius = patch_size // 2
    out_parts: List[pd.DataFrame] = []

    with torch.no_grad():
        for image_id, part in df.groupby("image_id", sort=False):
            work = part.copy().reset_index(drop=True)
            grid, _, _ = build_normalized_grid(work, means, stds)
            padded = np.pad(grid, ((0, 0), (radius, radius), (radius, radius)), mode="reflect")

            x = work["x"].to_numpy(dtype=int)
            y = work["y"].to_numpy(dtype=int)
            x0 = int(work["x"].min())
            y0 = int(work["y"].min())
            xr = x - x0 + radius
            yr = y - y0 + radius

            patches = np.empty((len(work), len(AE_INPUT_COLUMNS), patch_size, patch_size), dtype=np.float32)
            for idx, (yy, xx) in enumerate(zip(yr, xr, strict=False)):
                patches[idx] = padded[:, yy - radius : yy + radius + 1, xx - radius : xx + radius + 1]

            batches: List[np.ndarray] = []
            for start in range(0, len(work), batch_size):
                batch = torch.from_numpy(patches[start : start + batch_size])
                emb = model.embed(batch).detach().cpu().numpy()
                batches.append(emb)

            embedding = np.concatenate(batches, axis=0)
            emb_df = pd.DataFrame(embedding, columns=AE_COLUMNS)
            meta = work[KEY_COLUMNS].reset_index(drop=True)
            out_parts.append(pd.concat([meta, emb_df], axis=1))

    return pd.concat(out_parts, ignore_index=True)


def build_unlabeled_feature_table(
    image_ids: Sequence[str] | None = None,
    data_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    local_window: int = 3,
    patch_size: int = 9,
    embedding_size: int = 8,
    batch_size: int = 2048,
) -> pd.DataFrame:
    image_ids = list(image_ids or UNLABELED_SANITY_IDS)
    if not image_ids:
        raise ValueError("No image ids provided for unlabeled feature generation")

    unlabeled = load_unlabeled_images(image_ids=image_ids, data_dir=data_dir)
    unlabeled = add_pointwise_features(unlabeled)
    unlabeled = add_local_patch_features(unlabeled, window=local_window)

    labeled_images = load_labeled_images_for_norm(resolve_path(data_dir or default_data_dir()))
    means, stds = compute_channel_norm(labeled_images)
    ae_df = _extract_embeddings_for_frame(
        unlabeled[KEY_COLUMNS + AE_INPUT_COLUMNS],
        means=means,
        stds=stds,
        checkpoint_path=checkpoint_path,
        patch_size=patch_size,
        embedding_size=embedding_size,
        batch_size=batch_size,
    )

    merged = unlabeled.merge(ae_df, on=KEY_COLUMNS, how="left", validate="one_to_one")
    missing = merged[FEATURE_BLOCKS["B3_context_ae"]].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"Unlabeled feature build left missing selected features for {int(missing.sum())} rows")
    return merged.sort_values(["image_id", "y", "x"]).reset_index(drop=True)


def summarize_cloud_fraction(df: pd.DataFrame, prob_col: str = "prob_cloud", pred_col: str = "pred_label") -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []
    for image_id, part in df.groupby("image_id", sort=False):
        rows.append(
            {
                "image_id": image_id,
                "n_rows": int(len(part)),
                "mean_prob_cloud": float(part[prob_col].mean()),
                "predicted_cloud_fraction": float((part[pred_col] == 1).mean()),
            }
        )
    return pd.DataFrame(rows)


def feature_block_names() -> List[str]:
    return list(FEATURE_BLOCKS)


def feature_names_for_block(block_name: str) -> List[str]:
    if block_name not in FEATURE_BLOCKS:
        raise KeyError(f"Unknown feature block: {block_name}")
    return FEATURE_BLOCKS[block_name]


def blocks_for_ablation() -> List[str]:
    return ["B0_base", "B2_context", "B3_context_ae"]
