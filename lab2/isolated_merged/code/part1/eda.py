#!/usr/bin/env python3
"""Merged Part 1 EDA visuals for the isolated Lab 2 pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from utils_npz import NPZUtils

DEFAULT_FILES = ["O013257.npz", "O013490.npz", "O012791.npz"]
DEFAULT_PAIRS = [
    ("corr", "ndai"),
    ("corr", "sd"),
    ("ra_df", "ra_an"),
    ("ra_bf", "ra_an"),
    ("ra_cf", "ra_an"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run merged Part 1 EDA visuals")
    parser.add_argument("--out_dir", type=str, default="../../results/part1/eda/merged_visuals")
    parser.add_argument("--angles", nargs="*", type=int, default=[5, 6, 7, 8, 9])
    return parser.parse_args()


def resolve_path(base: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (base / path).resolve()


def visualize_radiance_angles(utils: NPZUtils, files_to_visualize: list[str], angles: list[int], out_dir: Path) -> None:
    dict_radiance_angles = {k: v for k, v in utils.dict_radiance_angles.items() if k in angles}

    for filename in files_to_visualize:
        df = utils.load_img_to_df(filename)
        fig, axes = plt.subplots(
            1,
            len(dict_radiance_angles),
            figsize=(5 * len(dict_radiance_angles), 4),
            constrained_layout=True,
        )

        if len(dict_radiance_angles) == 1:
            axes = [axes]

        for idx, (radiance_angle, angle_name) in enumerate(dict_radiance_angles.items()):
            ax = axes[idx]
            img = utils.prepare_image_from_df(df, radiance_angle)
            ax.imshow(img, cmap="gray")
            ax.set_title(angle_name, fontsize=10)
            ax.axis("off")

        fig.suptitle(f"Radiance Angles for {filename}", fontsize=16)
        fig.savefig(out_dir / f"radiance_angles_{filename}.png", dpi=180)
        plt.close(fig)


def visualise_comparison(utils: NPZUtils, files_to_visualize: list[str], feature1: str, feature2: str, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(files_to_visualize), figsize=(6 * len(files_to_visualize), 5), constrained_layout=True)
    if len(files_to_visualize) == 1:
        axes = [axes]

    for ax, img in zip(axes, files_to_visualize, strict=False):
        df = utils.load_img_to_df(img)
        labeled = df[df["label"] != 0].copy()
        sns.scatterplot(
            data=labeled,
            x=feature1,
            y=feature2,
            hue="label",
            alpha=0.5,
            ax=ax,
        )
        ax.set_title(img)
        ax.set_xlabel(utils.column_map[feature1])
        ax.set_ylabel(utils.column_map[feature2])
        ax.legend(title="label")

    fig.savefig(out_dir / f"{feature1}_vs_{feature2}.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    out_dir = resolve_path(script_dir, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_utils = NPZUtils()
    visualize_radiance_angles(npz_utils, DEFAULT_FILES, angles=args.angles, out_dir=out_dir)
    for feature1, feature2 in DEFAULT_PAIRS:
        visualise_comparison(npz_utils, DEFAULT_FILES, feature1, feature2, out_dir=out_dir)

    print(f"[merged-part1] wrote visuals to {out_dir}")


if __name__ == "__main__":
    main()
