#!/usr/bin/env python3
"""
model_figures.py  (Lab 1)

Generates model figures for the report using outputs from models.py.

Inputs (expected in lab1/output/)
- predictions_pecarn.csv
- predictions_logreg_baseline.csv
- predictions_logreg_bin6.csv
- predictions_logreg_bin12.csv
- predictions_rf_baseline.csv
- predictions_rf_bin6.csv
- predictions_rf_bin12.csv
- logreg_coefficients_*.csv
- rf_feature_importances_*.csv

Outputs
- Saves figures into: lab1/report/figs/model/
  (created if missing)

Run
- From repo root (stat-214-local):
  python lab1/code/model_figures.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    brier_score_loss,
)


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class FigPaths:
    lab1_dir: Path
    out_dirname: str = "output"
    fig_dir_rel: str = "report/figs/model"

    def out_dir(self) -> Path:
        return self.lab1_dir / self.out_dirname

    def fig_dir(self) -> Path:
        return self.lab1_dir / self.fig_dir_rel


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------
def _read_pred(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"y_true", "y_pred"}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"{path.name} missing required columns: {sorted(missing)}")
    if "y_prob" not in df.columns:
        df["y_prob"] = np.nan
    if "split" not in df.columns:
        df["split"] = "all"
    return df


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Plot helpers (publication-style, matplotlib only)
# -----------------------------------------------------------------------------
def _savefig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _roc_plot(models: Dict[str, pd.DataFrame], split: str, title: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(7.2, 5.2))
    ax = plt.gca()

    for name, df in models.items():
        sub = df[df["split"] == split].copy() if split != "all" else df.copy()
        if sub["y_prob"].isna().all():
            continue  # cannot do ROC without probabilities
        y = sub["y_true"].to_numpy()
        p = sub["y_prob"].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.25)
    _savefig(fig, out_path)


def _pr_plot(models: Dict[str, pd.DataFrame], split: str, title: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(7.2, 5.2))
    ax = plt.gca()

    for name, df in models.items():
        sub = df[df["split"] == split].copy() if split != "all" else df.copy()
        if sub["y_prob"].isna().all():
            continue
        y = sub["y_true"].to_numpy()
        p = sub["y_prob"].to_numpy()
        prec, rec, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})")

    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.25)
    _savefig(fig, out_path)


def _confmat_plot(df: pd.DataFrame, split: str, title: str, out_path: Path, normalize: Optional[str] = None) -> None:
    sub = df[df["split"] == split].copy() if split != "all" else df.copy()
    y = sub["y_true"].astype(int).to_numpy()
    yp = sub["y_pred"].astype(int).to_numpy()
    cm = confusion_matrix(y, yp, labels=[0, 1], normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No ciTBI", "ciTBI"])

    fig = plt.figure(figsize=(5.8, 5.2))
    ax = plt.gca()
    disp.plot(ax=ax, colorbar=False, values_format=".2f" if normalize else "d")
    ax.set_title(title)
    _savefig(fig, out_path)


def _calibration_plot(models: Dict[str, pd.DataFrame], split: str, title: str, out_path: Path) -> None:
    """
    Reliability curve using binned predicted probabilities.
    """
    fig = plt.figure(figsize=(7.2, 5.2))
    ax = plt.gca()

    for name, df in models.items():
        sub = df[df["split"] == split].copy() if split != "all" else df.copy()
        if sub["y_prob"].isna().all():
            continue

        y = sub["y_true"].to_numpy()
        p = sub["y_prob"].to_numpy()

        # clip for safety
        p = np.clip(p, 0.0, 1.0)

        # binning
        bins = np.linspace(0, 1, 11)
        bin_id = np.digitize(p, bins) - 1
        xs, ys = [], []
        for b in range(len(bins) - 1):
            mask = bin_id == b
            if mask.sum() < 50:
                continue
            xs.append(p[mask].mean())
            ys.append(y[mask].mean())
        if len(xs) == 0:
            continue

        brier = brier_score_loss(y, p)
        ax.plot(xs, ys, marker="o", linewidth=1.5, label=f"{name} (Brier={brier:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.25)
    _savefig(fig, out_path)


def _threshold_tradeoff_plot(df: pd.DataFrame, split: str, title: str, out_path: Path) -> None:
    """
    Sensitivity/specificity vs threshold curve.
    Uses y_prob and y_true.
    """
    sub = df[df["split"] == split].copy() if split != "all" else df.copy()
    if sub["y_prob"].isna().all():
        return

    y = sub["y_true"].astype(int).to_numpy()
    p = np.clip(sub["y_prob"].to_numpy(), 0.0, 1.0)

    thresholds = np.linspace(0.0, 1.0, 201)
    sens, spec = [], []

    for t in thresholds:
        yp = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
        se = tp / (tp + fn) if (tp + fn) else np.nan
        sp = tn / (tn + fp) if (tn + fp) else np.nan
        sens.append(se)
        spec.append(sp)

    fig = plt.figure(figsize=(7.2, 5.2))
    ax = plt.gca()
    ax.plot(thresholds, sens, label="Sensitivity", linewidth=1.6)
    ax.plot(thresholds, spec, label="Specificity", linewidth=1.6)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric value")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    _savefig(fig, out_path)


def _risk_score_boxplot(
    df_pecarn: pd.DataFrame,
    split: str,
    title: str,
    out_path: Path,
) -> None:
    """
    PECARN-only: show distribution of risk score separated by outcome (ciTBI vs no ciTBI).
    """
    sub = df_pecarn[df_pecarn["split"] == split].copy() if split != "all" else df_pecarn.copy()
    if "y_prob" not in sub.columns or sub["y_prob"].isna().all():
        return

    y0 = sub.loc[sub["y_true"] == 0, "y_prob"].dropna().to_numpy()
    y1 = sub.loc[sub["y_true"] == 1, "y_prob"].dropna().to_numpy()

    fig = plt.figure(figsize=(6.8, 5.0))
    ax = plt.gca()
    ax.boxplot([y0, y1], labels=["No ciTBI", "ciTBI"], showfliers=False)
    ax.set_ylabel("PECARN risk score (count of non-absent criteria)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    _savefig(fig, out_path)


def _topk_bar_from_csv(csv_path: Path, value_col: str, k: int, title: str, out_path: Path) -> None:
    """
    Generic top-k horizontal bar plot for interpretability artifacts.
    """
    df = pd.read_csv(csv_path)
    if "feature" not in df.columns or value_col not in df.columns:
        raise KeyError(f"{csv_path.name} must have columns ['feature','{value_col}']")

    top = df.sort_values(value_col, ascending=False).head(k).iloc[::-1]

    fig = plt.figure(figsize=(7.6, 5.8))
    ax = plt.gca()
    ax.barh(top["feature"].astype(str), top[value_col].to_numpy())
    ax.set_title(title)
    ax.set_xlabel(value_col)
    ax.grid(True, axis="x", alpha=0.25)
    _savefig(fig, out_path)


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------
def main() -> None:
    lab1_dir = Path(__file__).resolve().parents[1]
    paths = FigPaths(lab1_dir=lab1_dir)
    out_dir = paths.out_dir()
    fig_dir = paths.fig_dir()
    _ensure_dir(fig_dir)

    # Load predictions
    pred_paths = {
        "PECARN": out_dir / "predictions_pecarn.csv",
        "LogReg (baseline)": out_dir / "predictions_logreg_baseline.csv",
        "LogReg (bin6)": out_dir / "predictions_logreg_bin6.csv",
        "LogReg (bin12)": out_dir / "predictions_logreg_bin12.csv",
        "RF (baseline)": out_dir / "predictions_rf_baseline.csv",
        "RF (bin6)": out_dir / "predictions_rf_bin6.csv",
        "RF (bin12)": out_dir / "predictions_rf_bin12.csv",
    }
    preds = {k: _read_pred(v) for k, v in pred_paths.items() if v.exists()}

    # Separate probability-capable sets (exclude PECARN if you used risk score as y_prob it is fine)
    prob_models = {k: v for k, v in preds.items() if not v["y_prob"].isna().all()}

    # Determine split naming
    # - PECARN file is split="all"
    # - ML files are split in {"train","test"}
    splits_present = sorted({s for df in preds.values() for s in df["split"].unique()})
    has_test = "test" in splits_present

    # 1) ROC + PR on test split for ML models (and PECARN risk score if present)
    if has_test:
        _roc_plot(
            prob_models,
            split="test",
            title="ROC curves (test split)",
            out_path=fig_dir / "roc_test.png",
        )
        _pr_plot(
            prob_models,
            split="test",
            title="Precision–Recall curves (test split)",
            out_path=fig_dir / "pr_test.png",
        )

        # 2) Calibration on test split
        _calibration_plot(
            prob_models,
            split="test",
            title="Calibration / reliability (test split)",
            out_path=fig_dir / "calibration_test.png",
        )

    # 3) Confusion matrices (test) for key models
    if has_test:
        for key in ["LogReg (baseline)", "LogReg (bin6)", "LogReg (bin12)", "RF (baseline)", "RF (bin6)", "RF (bin12)"]:
            if key in preds:
                _confmat_plot(
                    preds[key],
                    split="test",
                    title=f"Confusion matrix (test): {key}",
                    out_path=fig_dir / f"cm_{key.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '-')}_test.png",
                    normalize=None,
                )
                _confmat_plot(
                    preds[key],
                    split="test",
                    title=f"Confusion matrix (test, normalized): {key}",
                    out_path=fig_dir / f"cm_{key.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '-')}_test_norm.png",
                    normalize="true",
                )

    # 4) Threshold tradeoff curves (test): logistic baseline + RF baseline
    if has_test and "LogReg (baseline)" in preds:
        _threshold_tradeoff_plot(
            preds["LogReg (baseline)"],
            split="test",
            title="Threshold tradeoff (test): Logistic regression baseline",
            out_path=fig_dir / "threshold_tradeoff_logreg_baseline_test.png",
        )
    if has_test and "RF (baseline)" in preds:
        _threshold_tradeoff_plot(
            preds["RF (baseline)"],
            split="test",
            title="Threshold tradeoff (test): Random forest baseline",
            out_path=fig_dir / "threshold_tradeoff_rf_baseline_test.png",
        )

    # 5) Stability figure: compare predictions across perturbations (binning)
    #    Plot: fraction predicted positive (CT recommended) on test across variants
    if has_test:
        variants = [
            ("LogReg", ["LogReg (baseline)", "LogReg (bin6)", "LogReg (bin12)"]),
            ("RF", ["RF (baseline)", "RF (bin6)", "RF (bin12)"]),
        ]
        for family, keys in variants:
            xs, ys = [], []
            for k in keys:
                if k not in preds:
                    continue
                sub = preds[k][preds[k]["split"] == "test"].copy()
                rate_pos = float(sub["y_pred"].mean())
                xs.append(k.replace(f"{family} ", ""))
                ys.append(rate_pos)

            if xs:
                fig = plt.figure(figsize=(6.8, 5.0))
                ax = plt.gca()
                ax.plot(xs, ys, marker="o", linewidth=1.8)
                ax.set_ylim(0, 1)
                ax.set_ylabel("Predicted positive rate (CT recommended)")
                ax.set_title(f"Stability across age-binning perturbations (test): {family}")
                ax.grid(True, alpha=0.25)
                _savefig(fig, fig_dir / f"stability_posrate_{family.lower()}_test.png")

    # 6) PECARN risk score vs outcome (all)
    if "PECARN" in preds:
        _risk_score_boxplot(
            preds["PECARN"],
            split="all",
            title="PECARN risk score by outcome (all rule-cohort rows)",
            out_path=fig_dir / "pecarn_riskscore_boxplot.png",
        )

    # 7) Interpretability plots: top coefficients / top importances
    coef_files = [
        ("baseline", out_dir / "logreg_coefficients_baseline.csv"),
        ("bin6", out_dir / "logreg_coefficients_bin6.csv"),
        ("bin12", out_dir / "logreg_coefficients_bin12.csv"),
    ]
    for tag, p in coef_files:
        if p.exists():
            _topk_bar_from_csv(
                p,
                value_col="odds_ratio",
                k=20,
                title=f"Logistic regression: top 20 odds ratios ({tag})",
                out_path=fig_dir / f"logreg_top_oddsratio_{tag}.png",
            )

    imp_files = [
        ("baseline", out_dir / "rf_feature_importances_baseline.csv"),
        ("bin6", out_dir / "rf_feature_importances_bin6.csv"),
        ("bin12", out_dir / "rf_feature_importances_bin12.csv"),
    ]
    for tag, p in imp_files:
        if p.exists():
            _topk_bar_from_csv(
                p,
                value_col="importance",
                k=20,
                title=f"Random forest: top 20 feature importances ({tag})",
                out_path=fig_dir / f"rf_top_importances_{tag}.png",
            )

    print(f"Saved model figures to: {fig_dir.resolve()}")


if __name__ == "__main__":
    main()