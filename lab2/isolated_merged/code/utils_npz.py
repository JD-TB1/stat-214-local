from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def pick_main_array(npz_obj: np.lib.npyio.NpzFile) -> np.ndarray:
    candidates = []
    for key in npz_obj.files:
        arr = npz_obj[key]
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] in (10, 11):
            candidates.append(arr)
    if not candidates:
        raise ValueError(f"No usable 2D array found in keys={list(npz_obj.files)}")
    return np.asarray(max(candidates, key=lambda arr: arr.shape[0]))


class NPZUtils:
    def __init__(self, data_path: Path | None = None):
        lab2_root = Path(__file__).resolve().parents[2]
        self.data_path = Path(data_path) if data_path is not None else lab2_root / "data" / "image_data"
        self.dict_radiance_angles = {
            5: "DF",
            6: "CF",
            7: "BF",
            8: "AF",
            9: "AN",
            10: "Expert Labels",
        }
        self.columns = [
            "y",
            "x",
            "ndai",
            "sd",
            "corr",
            "ra_df",
            "ra_cf",
            "ra_bf",
            "ra_af",
            "ra_an",
            "label",
        ]
        self.column_map = {
            "y": "Y",
            "x": "X",
            "ndai": "NDAI",
            "sd": "SD",
            "corr": "CORR",
            "ra_df": "Radiance angle DF",
            "ra_cf": "Radiance angle CF",
            "ra_bf": "Radiance angle BF",
            "ra_af": "Radiance angle AF",
            "ra_an": "Radiance angle AN",
            "label": "Expert Label",
        }

    def angle_to_column(self, radiance_angle: int) -> str:
        mapping = {
            5: "ra_df",
            6: "ra_cf",
            7: "ra_bf",
            8: "ra_af",
            9: "ra_an",
            10: "label",
        }
        return mapping[radiance_angle]

    def prepare_image_from_df(self, df: pd.DataFrame, radiance_angle: int) -> np.ndarray:
        value_column = self.angle_to_column(radiance_angle)
        x_coords = df["x"].astype(int).to_numpy()
        y_coords = df["y"].astype(int).to_numpy()
        values = df[value_column].to_numpy(dtype=float)

        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        width = int(x_max - x_min + 1)
        height = int(y_max - y_min + 1)
        img = np.zeros((height, width), dtype=float)
        img[y_coords - y_min, x_coords - x_min] = values

        img_min = float(np.nanmin(img))
        img_max = float(np.nanmax(img))
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        return img

    def load_img_to_df(self, img: str) -> pd.DataFrame:
        path = self.data_path / img
        with np.load(path, allow_pickle=True) as npz_obj:
            arr = pick_main_array(npz_obj)
        if arr.shape[1] == 10:
            label_col = np.zeros((arr.shape[0], 1))
            arr = np.hstack([arr, label_col])
        df = pd.DataFrame(arr, columns=self.columns)
        for col in self.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["label"] = df["label"].fillna(0).astype(int)
        return df
