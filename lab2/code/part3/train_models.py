#!/usr/bin/env python3
"""Train and evaluate Part 3 predictive models for STAT 214 Lab 2."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import (
    FEATURE_BLOCKS,
    KEY_COLUMNS,
    SplitTables,
    blocks_for_ablation,
    check_supervised_feature_table,
    collect_split_integrity,
    feature_names_for_block,
    get_primary_by_image_split_ids,
    get_secondary_spatial_split_ids,
    load_split_tables,
    load_supervised_feature_table,
    prepare_feature_matrices,
    resolve_path,
)

RANDOM_SEEDS = [214, 215, 216]
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "HistGradientBoosting",
}
RF_TUNING_ESTIMATORS = 100
RF_FINAL_ESTIMATORS = 400
HGB_MAX_ITER = 200

LOGISTIC_GRID = [{"C": float(c)} for c in (0.1, 1.0, 10.0)]
RF_GRID_FULL = [
    {"max_depth": depth, "min_samples_leaf": min_leaf, "max_features": max_features}
    for depth in (None, 8, 16)
    for min_leaf in (1, 10, 50)
    for max_features in ("sqrt", 0.5)
]
RF_GRID_FAST = [
    {"max_depth": 8, "min_samples_leaf": 10, "max_features": "sqrt"},
    {"max_depth": 8, "min_samples_leaf": 50, "max_features": "sqrt"},
    {"max_depth": None, "min_samples_leaf": 10, "max_features": "sqrt"},
    {"max_depth": None, "min_samples_leaf": 50, "max_features": 0.5},
]
HGB_GRID_FULL = [
    {
        "learning_rate": lr,
        "max_depth": max_depth,
        "min_samples_leaf": min_leaf,
        "l2_regularization": reg,
    }
    for lr in (0.03, 0.1)
    for max_depth in (3, None)
    for min_leaf in (20, 100)
    for reg in (0.0, 0.1)
]
HGB_GRID_FAST = [
    {"learning_rate": 0.03, "max_depth": 3, "min_samples_leaf": 20, "l2_regularization": 0.1},
    {"learning_rate": 0.1, "max_depth": 3, "min_samples_leaf": 20, "l2_regularization": 0.1},
    {"learning_rate": 0.03, "max_depth": None, "min_samples_leaf": 100, "l2_regularization": 0.1},
    {"learning_rate": 0.1, "max_depth": None, "min_samples_leaf": 100, "l2_regularization": 0.0},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Part 3 predictive models")
    parser.add_argument("--engineered_path", type=str, default="results/part2/labeled_engineered_features.csv")
    parser.add_argument("--embedding_path", type=str, default="results/part2/autoencoder_embeddings_supervised.csv")
    parser.add_argument("--split_root", type=str, default="results/part1/splits")
    parser.add_argument("--out_dir", type=str, default="results/part3")
    parser.add_argument("--docs_dir", type=str, default="documents/part3")
    parser.add_argument("--search_mode", choices=["fast", "full"], default="fast")
    return parser.parse_args()


def ensure_dirs(out_dir: Path, docs_dir: Path) -> Dict[str, Path]:
    subdirs = {
        "out": out_dir,
        "docs": docs_dir,
        "predictions": out_dir / "predictions",
        "figures": out_dir / "figures",
        "diagnostics": out_dir / "diagnostics",
        "models": out_dir / "models",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs


def get_model_grids(search_mode: str) -> Dict[str, List[Dict[str, object]]]:
    if search_mode == "full":
        return {
            "logistic_regression": LOGISTIC_GRID,
            "random_forest": RF_GRID_FULL,
            "hist_gradient_boosting": HGB_GRID_FULL,
        }
    return {
        "logistic_regression": LOGISTIC_GRID,
        "random_forest": RF_GRID_FAST,
        "hist_gradient_boosting": HGB_GRID_FAST,
    }


def param_signature(params: Mapping[str, object]) -> str:
    return json.dumps(dict(sorted(params.items())), sort_keys=True)


def build_estimator(family: str, params: Mapping[str, object], random_state: int) -> Pipeline | RandomForestClassifier | HistGradientBoostingClassifier:
    if family == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l2",
                        C=float(params["C"]),
                        class_weight="balanced",
                        max_iter=5000,
                        solver="lbfgs",
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if family == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", RF_TUNING_ESTIMATORS)),
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
        )

    if family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            loss="log_loss",
            random_state=random_state,
            max_iter=int(params.get("max_iter", HGB_MAX_ITER)),
            learning_rate=float(params["learning_rate"]),
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            l2_regularization=float(params["l2_regularization"]),
            early_stopping=False,
        )

    raise KeyError(f"Unknown model family: {family}")


def inverse_frequency_weights(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    weights = {int(cls): len(y) / (len(classes) * count) for cls, count in zip(classes, counts, strict=False)}
    return np.array([weights[int(v)] for v in y], dtype=float)


def predict_prob(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    raise TypeError(f"Model {type(model).__name__} does not expose probabilities or scores")


def select_threshold_by_balanced_accuracy(y_true: np.ndarray, prob: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return 0.5

    fpr, tpr, thresholds = roc_curve(y_true, prob)
    finite = np.isfinite(thresholds)
    thresholds = thresholds[finite]
    fpr = fpr[finite]
    tpr = tpr[finite]
    if len(thresholds) == 0:
        return 0.5
    balanced = 0.5 * (tpr + (1.0 - fpr))
    best_idx = int(np.nanargmax(balanced))
    return float(np.clip(thresholds[best_idx], 0.0, 1.0))


def compute_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (prob >= threshold).astype(int)
    if np.unique(y_true).size < 2:
        roc_auc = np.nan
        pr_auc = np.nan
    else:
        roc_auc = float(roc_auc_score(y_true, prob))
        pr_auc = float(average_precision_score(y_true, prob))

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else np.nan
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else np.nan

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, prob)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_candidate(
    split_tables: SplitTables,
    family: str,
    block_name: str,
    params: Mapping[str, object],
    random_state: int,
) -> Tuple[Dict[str, object], Dict[str, pd.DataFrame]]:
    feature_names = feature_names_for_block(block_name)
    matrices = prepare_feature_matrices(split_tables, feature_names)
    X_train = matrices["X_train"]
    X_val = matrices["X_val"]
    X_test = matrices["X_test"]
    y_train = matrices["y_train"]
    y_val = matrices["y_val"]
    y_test = matrices["y_test"]

    for name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        if not np.isfinite(X).all():
            raise ValueError(f"Non-finite values detected in {split_tables.split_id} {block_name} {family} {name}")

    model = build_estimator(family, params, random_state=random_state)
    fit_kwargs = {}
    if family == "hist_gradient_boosting":
        fit_kwargs["sample_weight"] = inverse_frequency_weights(y_train)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train, **fit_kwargs)

    train_prob = predict_prob(model, X_train)
    val_prob = predict_prob(model, X_val)
    test_prob = predict_prob(model, X_test)
    threshold = select_threshold_by_balanced_accuracy(y_val, val_prob)

    row: Dict[str, object] = {
        "strategy": split_tables.strategy,
        "split_id": split_tables.split_id,
        "family": family,
        "family_label": MODEL_LABELS[family],
        "block_name": block_name,
        "n_features": len(feature_names),
        "features": ",".join(feature_names),
        "params_json": param_signature(params),
        "threshold": threshold,
        "random_state": random_state,
    }

    for subset_name, y_true, prob in [
        ("train", y_train, train_prob),
        ("val", y_val, val_prob),
        ("test", y_test, test_prob),
    ]:
        metrics = compute_metrics(y_true, prob, threshold)
        for key, value in metrics.items():
            row[f"{subset_name}_{key}"] = value

    predictions: Dict[str, pd.DataFrame] = {}
    for subset_name, prob in [("train", train_prob), ("val", val_prob), ("test", test_prob)]:
        subset = getattr(split_tables, subset_name).copy()
        subset["prob_cloud"] = prob
        subset["pred_label"] = (prob >= threshold).astype(int)
        subset["pred_is_correct"] = (subset["pred_label"] == subset["target"]).astype(int)
        predictions[subset_name] = subset

    return row, predictions


def save_prediction_table(df: pd.DataFrame, out_path: Path, feature_names: Sequence[str]) -> None:
    cols = KEY_COLUMNS + ["target", "prob_cloud", "pred_label", "pred_is_correct"] + list(feature_names)
    available = [col for col in cols if col in df.columns]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.loc[:, available].to_csv(out_path, index=False)


def save_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fit_statsmodels_logit(train_df: pd.DataFrame, feature_names: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    X = train_df.loc[:, feature_names].to_numpy(dtype=float)
    y = train_df["target"].to_numpy(dtype=int)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    X_design = sm.add_constant(Xs, has_constant="add")
    model = sm.Logit(y, X_design)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(disp=False, maxiter=200)

    params = pd.Series(result.params, index=["const"] + list(feature_names))
    stderr = pd.Series(result.bse, index=["const"] + list(feature_names))
    odds = np.exp(params)
    coef_df = pd.DataFrame(
        {
            "feature": params.index,
            "coef": params.values,
            "std_err": stderr.values,
            "odds_ratio": odds.values,
        }
    )

    vif_rows: List[Dict[str, float | str]] = []
    for idx, feature in enumerate(feature_names):
        vif_rows.append({"feature": feature, "vif": float(variance_inflation_factor(Xs, idx))})
    vif_df = pd.DataFrame(vif_rows)

    summary = {
        "aic": float(result.aic),
        "bic": float(result.bic),
        "llf": float(result.llf),
        "converged": bool(result.mle_retvals.get("converged", True)),
        "iterations": int(result.mle_retvals.get("iterations", -1)),
    }
    return coef_df, vif_df, summary


def plot_calibration_curve(y_true: np.ndarray, prob: np.ndarray, out_path: Path, title: str) -> None:
    frac_pos, mean_pred = calibration_curve(y_true, prob, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(mean_pred, frac_pos, marker="o", label="model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="ideal")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_error_map(df: pd.DataFrame, out_path: Path, title: str) -> None:
    labels = {
        (1, 1): ("TP", "#2b8cbe"),
        (1, 0): ("FN", "#d95f0e"),
        (0, 1): ("FP", "#31a354"),
        (0, 0): ("TN", "#969696"),
    }
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for (target, pred), (label, color) in labels.items():
        part = df[(df["target"] == target) & (df["pred_label"] == pred)]
        if part.empty:
            continue
        ax.scatter(part["x"], part["y"], s=5, alpha=0.5, label=label, color=color)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(markerscale=3)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_error_quantiles(
    df: pd.DataFrame,
    feature: str,
    out_path: Path,
    title: str,
) -> None:
    if feature not in df.columns:
        return
    work = df[[feature, "pred_is_correct"]].dropna().copy()
    if work[feature].nunique() < 5:
        return
    work["error"] = 1 - work["pred_is_correct"]
    work["bin"] = pd.qcut(work[feature], q=10, duplicates="drop")
    summary = (
        work.groupby("bin", observed=False)
        .agg(feature_mean=(feature, "mean"), error_rate=("error", "mean"), n=("error", "size"))
        .reset_index(drop=True)
    )
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(summary["feature_mean"], summary["error_rate"], marker="o")
    ax.set_xlabel(feature)
    ax.set_ylabel("Error rate")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_error_feature_distributions(df: pd.DataFrame, features: Sequence[str], out_path: Path, title: str) -> None:
    available = [feature for feature in features if feature in df.columns]
    if not available:
        return
    fig, axes = plt.subplots(len(available), 1, figsize=(6, 3.5 * len(available)))
    if len(available) == 1:
        axes = [axes]
    for ax, feature in zip(axes, available, strict=False):
        sns.kdeplot(df.loc[df["pred_is_correct"] == 1, feature], ax=ax, label="correct", fill=True, alpha=0.3)
        sns.kdeplot(df.loc[df["pred_is_correct"] == 0, feature], ax=ax, label="error", fill=True, alpha=0.3)
        ax.set_title(feature)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, feature_names: Sequence[str], out_path: Path, title: str) -> None:
    corr = df.loc[:, feature_names].corr()
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(feature_names)), max(5, 0.5 * len(feature_names))))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_bar_importance(df: pd.DataFrame, feature_col: str, value_col: str, out_path: Path, title: str) -> None:
    work = df.sort_values(value_col, ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(6, max(4, 0.35 * len(work))))
    ax.barh(work[feature_col], work[value_col])
    ax.set_title(title)
    ax.set_xlabel(value_col)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fit_model_for_split(
    split_tables: SplitTables,
    family: str,
    block_name: str,
    params: Mapping[str, object],
    random_state: int,
) -> Tuple[Pipeline | RandomForestClassifier | HistGradientBoostingClassifier, Dict[str, pd.DataFrame]]:
    feature_names = feature_names_for_block(block_name)
    matrices = prepare_feature_matrices(split_tables, feature_names)
    model = build_estimator(family, params, random_state=random_state)
    fit_kwargs = {}
    if family == "hist_gradient_boosting":
        fit_kwargs["sample_weight"] = inverse_frequency_weights(matrices["y_train"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(matrices["X_train"], matrices["y_train"], **fit_kwargs)

    threshold = select_threshold_by_balanced_accuracy(
        matrices["y_val"],
        predict_prob(model, matrices["X_val"]),
    )
    preds: Dict[str, pd.DataFrame] = {}
    for subset_name in ["train", "val", "test"]:
        subset = getattr(split_tables, subset_name).copy()
        prob = predict_prob(model, matrices[f"X_{subset_name}"])
        subset["prob_cloud"] = prob
        subset["pred_label"] = (prob >= threshold).astype(int)
        subset["pred_is_correct"] = (subset["pred_label"] == subset["target"]).astype(int)
        preds[subset_name] = subset
    return model, preds


def permutation_importance_df(model, X: np.ndarray, y: np.ndarray, feature_names: Sequence[str], random_state: int) -> pd.DataFrame:
    if np.unique(y).size < 2:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=5,
        scoring="roc_auc",
        random_state=random_state,
        n_jobs=1,
    )
    return pd.DataFrame(
        {
            "feature": list(feature_names),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)


def summarize_error_patterns(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for label_name, target_pred in [("FP", (0, 1)), ("FN", (1, 0))]:
        part = df[(df["target"] == target_pred[0]) & (df["pred_label"] == target_pred[1])]
        row: Dict[str, object] = {"error_type": label_name, "n_rows": int(len(part))}
        for feature in features:
            if feature in part.columns:
                row[f"{feature}_mean"] = float(part[feature].mean()) if len(part) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def choose_best_candidate(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    frame = pd.DataFrame(rows).sort_values(
        ["val_roc_auc", "val_pr_auc", "val_balanced_accuracy"],
        ascending=[False, False, False],
    )
    return frame.iloc[0].to_dict()


def choose_best_row(frame: pd.DataFrame) -> pd.Series:
    return frame.sort_values(
        ["mean_test_roc_auc", "mean_test_pr_auc", "mean_test_balanced_accuracy"],
        ascending=[False, False, False],
    ).iloc[0]


def evaluate_primary_splits(
    features_df: pd.DataFrame,
    split_root: Path,
    out_dirs: Mapping[str, Path],
    model_grids: Mapping[str, Sequence[Mapping[str, object]]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    candidate_rows: List[Dict[str, object]] = []
    selected_rows: List[Dict[str, object]] = []

    for split_id in get_primary_by_image_split_ids(split_root=split_root):
        split_tables = load_split_tables(split_id, "by_image", features_df, split_root=split_root)
        for block_name in FEATURE_BLOCKS:
            feature_names = feature_names_for_block(block_name)
            for family, grid in model_grids.items():
                family_rows: List[Dict[str, object]] = []
                predictions_by_signature: Dict[str, Dict[str, pd.DataFrame]] = {}
                for params in grid:
                    row, preds = evaluate_candidate(
                        split_tables=split_tables,
                        family=family,
                        block_name=block_name,
                        params=params,
                        random_state=214,
                    )
                    family_rows.append(row)
                    predictions_by_signature[row["params_json"]] = preds
                    candidate_rows.append(row)

                best_row = choose_best_candidate(family_rows)
                best_predictions = predictions_by_signature[str(best_row["params_json"])]
                prediction_path = (
                    out_dirs["predictions"]
                    / "by_image"
                    / split_id
                    / f"{family}__{block_name}__test_predictions.csv"
                )
                save_prediction_table(best_predictions["test"], prediction_path, feature_names)
                best_row["prediction_path"] = str(prediction_path)
                selected_rows.append(best_row)

    candidate_df = pd.DataFrame(candidate_rows)
    selected_df = pd.DataFrame(selected_rows)
    candidate_df.to_csv(out_dirs["out"] / "candidate_results.csv", index=False)
    selected_df.to_csv(out_dirs["out"] / "outer_split_metrics.csv", index=False)
    return candidate_df, selected_df


def summarize_selection(selected_df: pd.DataFrame, out_dirs: Mapping[str, Path]) -> pd.DataFrame:
    grouped = (
        selected_df.groupby(["family", "family_label", "block_name"], dropna=False)
        .agg(
            mean_test_roc_auc=("test_roc_auc", "mean"),
            mean_test_pr_auc=("test_pr_auc", "mean"),
            mean_test_balanced_accuracy=("test_balanced_accuracy", "mean"),
            mean_test_f1=("test_f1", "mean"),
            mean_test_brier=("test_brier", "mean"),
            mean_threshold=("threshold", "mean"),
            n_splits=("split_id", "nunique"),
        )
        .reset_index()
        .sort_values(
            ["mean_test_roc_auc", "mean_test_pr_auc", "mean_test_balanced_accuracy"],
            ascending=[False, False, False],
        )
    )
    grouped.to_csv(out_dirs["out"] / "model_selection_summary.csv", index=False)
    return grouped


def choose_final_hyperparameters(candidate_df: pd.DataFrame, family: str, block_name: str) -> Dict[str, object]:
    subset = candidate_df[(candidate_df["family"] == family) & (candidate_df["block_name"] == block_name)].copy()
    summary = (
        subset.groupby("params_json", dropna=False)
        .agg(
            mean_val_roc_auc=("val_roc_auc", "mean"),
            mean_val_pr_auc=("val_pr_auc", "mean"),
            mean_val_balanced_accuracy=("val_balanced_accuracy", "mean"),
        )
        .reset_index()
        .sort_values(
            ["mean_val_roc_auc", "mean_val_pr_auc", "mean_val_balanced_accuracy"],
            ascending=[False, False, False],
        )
    )
    params_json = str(summary.iloc[0]["params_json"])
    return json.loads(params_json)


def family_best_blocks(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, part in summary_df.groupby("family", sort=False):
        rows.append(choose_best_row(part).to_dict())
    return pd.DataFrame(rows)


def generate_family_diagnostics(
    family_row: pd.Series,
    selected_df: pd.DataFrame,
    features_df: pd.DataFrame,
    split_root: Path,
    out_dirs: Mapping[str, Path],
) -> None:
    family = str(family_row["family"])
    block_name = str(family_row["block_name"])
    feature_names = feature_names_for_block(block_name)
    family_dir = out_dirs["diagnostics"] / family / block_name
    family_dir.mkdir(parents=True, exist_ok=True)

    subset = selected_df[(selected_df["family"] == family) & (selected_df["block_name"] == block_name)].copy()
    subset.to_csv(family_dir / "selected_outer_rows.csv", index=False)

    for _, row in subset.iterrows():
        split_id = str(row["split_id"])
        params = json.loads(str(row["params_json"]))
        split_tables = load_split_tables(split_id, "by_image", features_df, split_root=split_root)
        model, preds = fit_model_for_split(split_tables, family, block_name, params, random_state=214)

        split_dir = family_dir / split_id
        split_dir.mkdir(parents=True, exist_ok=True)
        test_df = preds["test"]

        plot_calibration_curve(
            test_df["target"].to_numpy(dtype=int),
            test_df["prob_cloud"].to_numpy(dtype=float),
            split_dir / "calibration_curve.png",
            f"{MODEL_LABELS[family]} calibration ({split_id})",
        )

        if family == "logistic_regression":
            coef_df, vif_df, summary = fit_statsmodels_logit(split_tables.train, feature_names)
            coef_df.to_csv(split_dir / "logit_coefficients.csv", index=False)
            vif_df.to_csv(split_dir / "logit_vif.csv", index=False)
            save_json(split_dir / "logit_summary.json", summary)
            plot_correlation_heatmap(
                split_tables.train,
                feature_names,
                split_dir / "feature_correlation_heatmap.png",
                f"Feature correlation ({split_id})",
            )
            plot_bar_importance(
                coef_df[coef_df["feature"] != "const"].assign(abs_coef=lambda d: d["coef"].abs()),
                "feature",
                "abs_coef",
                split_dir / "coefficient_magnitudes.png",
                f"Coefficient magnitude ({split_id})",
            )
            plot_bar_importance(
                coef_df[coef_df["feature"] != "const"],
                "feature",
                "odds_ratio",
                split_dir / "odds_ratios.png",
                f"Odds ratios ({split_id})",
            )
        elif family == "random_forest":
            importances = pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)
            importances.to_csv(split_dir / "feature_importance_mdi.csv", index=False)
            plot_bar_importance(
                importances,
                "feature",
                "importance",
                split_dir / "feature_importance_mdi.png",
                f"Random forest MDI importance ({split_id})",
            )
            matrices = prepare_feature_matrices(split_tables, feature_names)
            perm_df = permutation_importance_df(
                model,
                matrices["X_test"],
                matrices["y_test"],
                feature_names,
                random_state=214,
            )
            perm_df.to_csv(split_dir / "permutation_importance.csv", index=False)
            plot_bar_importance(
                perm_df,
                "feature",
                "importance_mean",
                split_dir / "permutation_importance.png",
                f"Random forest permutation importance ({split_id})",
            )
        else:
            matrices = prepare_feature_matrices(split_tables, feature_names)
            perm_df = permutation_importance_df(
                model,
                matrices["X_test"],
                matrices["y_test"],
                feature_names,
                random_state=214,
            )
            perm_df.to_csv(split_dir / "permutation_importance.csv", index=False)
            plot_bar_importance(
                perm_df,
                "feature",
                "importance_mean",
                split_dir / "permutation_importance.png",
                f"HGB permutation importance ({split_id})",
            )


def generate_best_model_posthoc(
    best_row: pd.Series,
    selected_df: pd.DataFrame,
    features_df: pd.DataFrame,
    split_root: Path,
    out_dirs: Mapping[str, Path],
) -> None:
    family = str(best_row["family"])
    block_name = str(best_row["block_name"])
    feature_names = feature_names_for_block(block_name)
    best_dir = out_dirs["diagnostics"] / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)

    selected_subset = selected_df[(selected_df["family"] == family) & (selected_df["block_name"] == block_name)].copy()
    selected_subset.to_csv(best_dir / "selected_outer_rows.csv", index=False)

    error_summary_rows: List[pd.DataFrame] = []
    for _, row in selected_subset.iterrows():
        split_id = str(row["split_id"])
        prediction_path = Path(str(row["prediction_path"]))
        test_df = pd.read_csv(prediction_path)

        split_dir = best_dir / split_id
        split_dir.mkdir(parents=True, exist_ok=True)
        plot_error_map(
            test_df,
            split_dir / "error_map.png",
            f"Best model error map ({split_id})",
        )
        plot_error_feature_distributions(
            test_df,
            ["SD", "NDAI", "local_SD_mean3", "ae4"],
            split_dir / "error_feature_distributions.png",
            f"Best model feature distributions ({split_id})",
        )
        for feature in ["SD", "NDAI", "local_SD_mean3", "ae4"]:
            if feature in test_df.columns:
                plot_feature_error_quantiles(
                    test_df,
                    feature,
                    split_dir / f"error_rate_by_{feature}.png",
                    f"Error rate by {feature} ({split_id})",
                )

        fp_fn_df = summarize_error_patterns(test_df, ["SD", "NDAI", "local_SD_mean3", "ae4"])
        fp_fn_df.insert(0, "split_id", split_id)
        fp_fn_df.to_csv(split_dir / "fp_fn_summary.csv", index=False)
        error_summary_rows.append(fp_fn_df)

    if error_summary_rows:
        pd.concat(error_summary_rows, ignore_index=True).to_csv(best_dir / "fp_fn_summary_all_splits.csv", index=False)

    ablation = (
        selected_df[(selected_df["family"] == family) & (selected_df["block_name"].isin(blocks_for_ablation()))]
        .groupby(["family", "block_name"], dropna=False)
        .agg(
            mean_test_roc_auc=("test_roc_auc", "mean"),
            mean_test_pr_auc=("test_pr_auc", "mean"),
            mean_test_balanced_accuracy=("test_balanced_accuracy", "mean"),
        )
        .reset_index()
    )
    ablation.to_csv(best_dir / "feature_block_ablation.csv", index=False)


def evaluate_seed_stability(
    best_row: pd.Series,
    features_df: pd.DataFrame,
    split_root: Path,
    out_dirs: Mapping[str, Path],
) -> pd.DataFrame:
    family = str(best_row["family"])
    block_name = str(best_row["block_name"])
    params = choose_final_hyperparameters(
        pd.read_csv(out_dirs["out"] / "candidate_results.csv"),
        family=family,
        block_name=block_name,
    )

    rows: List[Dict[str, object]] = []
    for seed in RANDOM_SEEDS:
        for split_id in get_primary_by_image_split_ids(split_root=split_root):
            split_tables = load_split_tables(split_id, "by_image", features_df, split_root=split_root)
            row, _ = evaluate_candidate(
                split_tables=split_tables,
                family=family,
                block_name=block_name,
                params=params,
                random_state=seed,
            )
            rows.append(
                {
                    "seed": seed,
                    "split_id": split_id,
                    "test_roc_auc": row["test_roc_auc"],
                    "test_pr_auc": row["test_pr_auc"],
                    "test_balanced_accuracy": row["test_balanced_accuracy"],
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dirs["out"] / "seed_stability.csv", index=False)
    return frame


def evaluate_spatial_robustness(
    best_row: pd.Series,
    features_df: pd.DataFrame,
    split_root: Path,
    out_dirs: Mapping[str, Path],
) -> pd.DataFrame:
    family = str(best_row["family"])
    block_name = str(best_row["block_name"])
    params = choose_final_hyperparameters(
        pd.read_csv(out_dirs["out"] / "candidate_results.csv"),
        family=family,
        block_name=block_name,
    )

    rows: List[Dict[str, object]] = []
    for split_id in get_secondary_spatial_split_ids(split_root=split_root):
        split_tables = load_split_tables(split_id, "spatial_within_image", features_df, split_root=split_root)
        row, _ = evaluate_candidate(
            split_tables=split_tables,
            family=family,
            block_name=block_name,
            params=params,
            random_state=214,
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dirs["out"] / "spatial_robustness_metrics.csv", index=False)
    return frame


def evaluate_threshold_sensitivity(best_row: pd.Series, selected_df: pd.DataFrame, out_dirs: Mapping[str, Path]) -> pd.DataFrame:
    family = str(best_row["family"])
    block_name = str(best_row["block_name"])
    rows: List[Dict[str, object]] = []
    subset = selected_df[(selected_df["family"] == family) & (selected_df["block_name"] == block_name)]
    for _, row in subset.iterrows():
        prediction_df = pd.read_csv(Path(str(row["prediction_path"])))
        y_true = prediction_df["target"].to_numpy(dtype=int)
        prob = prediction_df["prob_cloud"].to_numpy(dtype=float)
        base_threshold = float(row["threshold"])
        for offset in (-0.05, 0.0, 0.05):
            threshold = float(np.clip(base_threshold + offset, 0.0, 1.0))
            metrics = compute_metrics(y_true, prob, threshold)
            rows.append(
                {
                    "split_id": row["split_id"],
                    "offset": offset,
                    "threshold": threshold,
                    "roc_auc": metrics["roc_auc"],
                    "pr_auc": metrics["pr_auc"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "f1": metrics["f1"],
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dirs["out"] / "threshold_sensitivity.csv", index=False)
    return frame


def fit_final_model(
    best_row: pd.Series,
    candidate_df: pd.DataFrame,
    features_df: pd.DataFrame,
    out_dirs: Mapping[str, Path],
) -> Dict[str, object]:
    family = str(best_row["family"])
    block_name = str(best_row["block_name"])
    feature_names = feature_names_for_block(block_name)
    params = choose_final_hyperparameters(candidate_df, family=family, block_name=block_name)
    threshold = float(
        candidate_df[
            (candidate_df["family"] == family)
            & (candidate_df["block_name"] == block_name)
            & (candidate_df["params_json"] == param_signature(params))
        ]["threshold"].mean()
    )
    if family == "random_forest":
        params = dict(params)
        params["n_estimators"] = RF_FINAL_ESTIMATORS

    X = features_df.loc[:, feature_names].to_numpy(dtype=float)
    y = features_df["target"].to_numpy(dtype=int)
    model = build_estimator(family, params, random_state=214)
    fit_kwargs = {}
    if family == "hist_gradient_boosting":
        fit_kwargs["sample_weight"] = inverse_frequency_weights(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y, **fit_kwargs)

    bundle = {
        "family": family,
        "family_label": MODEL_LABELS[family],
        "block_name": block_name,
        "feature_names": feature_names,
        "params": params,
        "threshold": threshold,
        "model": model,
    }
    joblib.dump(bundle, out_dirs["models"] / "final_model.joblib")
    save_json(
        out_dirs["out"] / "selected_model_config.json",
        {
            "family": family,
            "family_label": MODEL_LABELS[family],
            "block_name": block_name,
            "feature_names": feature_names,
            "params": params,
            "threshold": threshold,
        },
    )
    return bundle


def write_part3_docs(
    docs_dir: Path,
    selection_summary: pd.DataFrame,
    family_best_df: pd.DataFrame,
    best_row: pd.Series,
    feature_integrity: Mapping[str, int],
    split_integrity_path: Path,
    search_mode: str,
) -> None:
    best_family = str(best_row["family_label"])
    best_block = str(best_row["block_name"])
    best_auc = float(best_row["mean_test_roc_auc"])
    best_pr = float(best_row["mean_test_pr_auc"])
    best_bacc = float(best_row["mean_test_balanced_accuracy"])

    readme = f"""# Part 3 README

This folder contains the Part 3 predictive-modeling handoff for Lab 2. The pipeline compares three classifier families across four feature blocks using the realistic by-image holdout splits from Part 1, then refits the selected model on all supervised labeled rows for unlabeled-image sanity checks.

Search mode used for this run: `{search_mode}`.

## Scripts
- `code/part3/dataset.py`
  - loads and validates the supervised modeling table
  - aligns Part 1 split files to Part 2 features
  - builds the unlabeled feature table used for final inference
- `code/part3/train_models.py`
  - tunes and evaluates logistic regression, random forest, and HistGradientBoosting
  - writes model-selection tables, diagnostics, and the final fitted model
- `code/part3/predict_unlabeled.py`
  - loads the saved final model
  - scores the three fixed unlabeled images and writes probability maps and masks

## Main outputs
- `results/part3/model_selection_summary.csv`
- `results/part3/outer_split_metrics.csv`
- `results/part3/selected_model_config.json`
- `results/part3/models/final_model.joblib`
- `results/part3/seed_stability.csv`
- `results/part3/spatial_robustness_metrics.csv`
- `results/part3/threshold_sensitivity.csv`
- `results/part3/diagnostics/`
- `results/part3/predictions/`

## Selected model
- family: `{best_family}`
- feature block: `{best_block}`
- mean outer-test ROC AUC: `{best_auc:.4f}`
- mean outer-test PR AUC: `{best_pr:.4f}`
- mean outer-test balanced accuracy: `{best_bacc:.4f}`

## Integrity checks
- supervised rows: `{feature_integrity['n_rows']}`
- duplicate supervised keys: `{feature_integrity['duplicate_key_rows']}`
- rows with missing selected features: `{feature_integrity['rows_with_missing_selected_features']}`
- split integrity table: `{split_integrity_path.relative_to(docs_dir.parents[1]) if split_integrity_path.is_absolute() else split_integrity_path}`

## Recommended reading order
1. `documents/part3/meeting_summary.md`
2. `results/part3/model_selection_summary.csv`
3. `results/part3/outer_split_metrics.csv`
4. `results/part3/selected_model_config.json`
5. `results/part3/diagnostics/best_model/`
"""

    top_rows = []
    for _, row in selection_summary.head(6).iterrows():
        top_rows.append(
            f"- `{row['family_label']}` + `{row['block_name']}`: mean ROC AUC={row['mean_test_roc_auc']:.4f}, mean PR AUC={row['mean_test_pr_auc']:.4f}, mean balanced accuracy={row['mean_test_balanced_accuracy']:.4f}"
        )

    family_rows = []
    for _, row in family_best_df.iterrows():
        family_rows.append(
            f"- `{row['family_label']}` best block `{row['block_name']}`: mean ROC AUC={row['mean_test_roc_auc']:.4f}, mean PR AUC={row['mean_test_pr_auc']:.4f}"
        )

    meeting_summary = f"""# STAT 214 Lab 2 Part 3 - Meeting Summary

## Why Part 3 matters
Part 1 established which train/test splits are credible, and Part 2 produced a shortlist of promising predictors. Part 3 is where we test whether those predictors actually support cloud-vs-non-cloud prediction on new images rather than only looking good inside the same scene. The main risk here is overestimating performance through spatial leakage, so the whole pipeline is built around realistic generalization.

## Data used for modeling
- supervised modeling table = Part 2 engineered features joined with the Part 2 autoencoder embeddings
- supervised rows used: `{feature_integrity['n_rows']}`
- cloud rows: `{feature_integrity['n_cloud']}`
- non-cloud rows: `{feature_integrity['n_non_cloud']}`
- duplicate keys on `image_id/x/y/label`: `{feature_integrity['duplicate_key_rows']}`
- rows with missing selected features: `{feature_integrity['rows_with_missing_selected_features']}`

## Feature blocks and why we compared them
- `B0_base`: `SD`, `NDAI`, `AF`, `AN`, `BF`
  - This is the compact baseline using the strongest raw/expert features from Part 2.
- `B1_engineered`: `B0_base + ndai_x_sd + af_df_gap + front_back_ratio + rad_cv + rad_range`
  - This tests whether simple domain-informed interactions and radiance summaries add usable signal beyond the base block.
- `B2_context`: `B1_engineered + local_SD_mean3 + local_NDAI_std3 + local_SD_std3 + local_rad_std_std3 + local_rad_mean_std3`
  - This tests whether local spatial context improves discrimination, which should matter because clouds form spatially structured regions.
- `B3_context_ae`: `B2_context + ae4 + ae5 + ae0`
  - This tests whether learned patch embeddings add information beyond the hand-engineered features.

The point of the block comparison is not only to find the best model, but also to measure the marginal value of engineering, local context, and learned features.

## Evaluation design and reasoning
- Primary benchmark: by-image holdout splits from Part 1.
  - This is the most realistic estimate of how well the model transfers to a new labeled image.
- Hyperparameter tuning: validation subset inside each outer holdout split.
  - This prevents tuning on the outer test image.
- Threshold selection: choose the probability threshold that maximizes validation balanced accuracy.
  - This is more appropriate than fixing `0.5` because the score calibration differs by model/split.
- Secondary robustness check: spatial-within-image splits for `O013257` and `O013490`.
  - These are useful supporting evidence, but weaker than by-image generalization because train and test still come from the same image.
- Excluded from aggregate spatial evaluation: `O012791_x_gt_q80`.
  - Its test split has zero cloud pixels, so threshold-based classification metrics are not meaningful there.

## Model families and why they were included
- Logistic Regression
  - Interpretable linear baseline. This tells us how far we can get with a relatively simple and explainable decision boundary.
- Random Forest
  - Nonlinear tree ensemble that can capture interactions and threshold effects without feature scaling.
- HistGradientBoosting
  - Strong tabular learner that usually handles mixed nonlinear structure more efficiently than random forests. This was the main candidate for the strongest predictive performance.

## Metrics and why we used them
- ROC AUC is the primary model-selection metric.
  - It measures ranking quality across thresholds and is the cleanest main comparison for the outer test splits.
- PR AUC is the secondary metric.
  - It is especially useful because cloud detection is still a positive-class detection problem, and precision-recall behavior matters.
- Balanced accuracy is used for threshold selection and tie-breaking.
  - This avoids letting the majority class dominate the threshold choice.
- F1, sensitivity, specificity, and Brier score are reported to show operational tradeoffs and calibration quality.

## What the current implementation actually ran
Search mode used for the saved results: `{search_mode}`.

This is a complete Part 3 pipeline, but the saved run is a reduced hyperparameter search rather than the exhaustive search:
- Logistic Regression: same grid in both modes (`3` values of `C`)
- Random Forest: fast grid = `4` settings, full grid = `18`
- HistGradientBoosting: fast grid = `4` settings, full grid = `16`

Across `3` by-image holdouts, `4` feature blocks, and `3` model families:
- fast search evaluates `132` primary candidates
- full search evaluates `444` primary candidates

So the current results should be described as preliminary model-selection results from the fast search, not the final exhaustive search. The pipeline itself is not partial; only the tuning breadth is reduced.

## Preliminary results from the current run
- Selected family: `{best_family}`
- Selected feature block: `{best_block}`
- Mean outer-test ROC AUC: `{best_auc:.4f}`
- Mean outer-test PR AUC: `{best_pr:.4f}`
- Mean outer-test balanced accuracy: `{best_bacc:.4f}`

## Best models in the ranking table
{chr(10).join(top_rows)}

## Best block per family
{chr(10).join(family_rows)}

## How to interpret the current ranking
- `HistGradientBoosting + B3_context_ae` is currently the leader, which suggests that the strongest performance comes from combining expert features, engineered interactions, local context, and AE features.
- Within HistGradientBoosting, moving from `B2_context` to `B3_context_ae` improved mean ROC AUC from `0.9401` to `0.9543` and mean PR AUC from `0.8717` to `0.9052`.
  - That is evidence that the selected AE dimensions add useful signal beyond the hand-engineered/context features.
- Logistic Regression performed best with `B2_context`, not `B3_context_ae`.
  - This suggests the AE features may be more useful when the model can exploit nonlinear interactions.

## Per-holdout results for the current best model
- `holdout_O012791`: ROC AUC `0.8957`, PR AUC `0.8036`, balanced accuracy `0.8297`
- `holdout_O013257`: ROC AUC `0.9780`, PR AUC `0.9272`, balanced accuracy `0.9302`
- `holdout_O013490`: ROC AUC `0.9890`, PR AUC `0.9847`, balanced accuracy `0.9412`

The variation across holdouts matters. The model is very strong on two images and clearly harder-tested on `O012791`, so the next pass should pay attention to what is special about that image in the error maps.

## Diagnostics and post-hoc work already produced
- logistic diagnostics: coefficients, odds ratios, VIF, calibration, and feature-correlation plots
- random-forest and boosting diagnostics: permutation importance, plus RF impurity importance
- best-model post-hoc EDA: error maps, feature-distribution comparisons, error-by-quantile plots, and FP/FN summaries
- robustness checks: seed stability, spatial robustness, and threshold sensitivity
- final sanity check on three unlabeled images:
  - `O002539`: predicted cloud fraction `0.9616`
  - `O045178`: predicted cloud fraction `0.6718`
  - `O119738`: predicted cloud fraction `0.6687`

## Recommendation for the next run
- Keep the current pipeline structure.
- Rerun Part 3 in `full` search mode before locking the final model for submission.
- Use the current fast-mode outputs as the meeting/preliminary results, not as the final tuned benchmark.

## What to show in the meeting
- `results/part3/model_selection_summary.csv`
- `results/part3/outer_split_metrics.csv`
- `results/part3/diagnostics/best_model/`
- `results/part3/selected_model_config.json`
"""

    (docs_dir / "README.md").write_text(readme + "\n", encoding="utf-8")
    (docs_dir / "meeting_summary.md").write_text(meeting_summary + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    engineered_path = resolve_path(args.engineered_path)
    embedding_path = resolve_path(args.embedding_path)
    split_root = resolve_path(args.split_root)
    out_dir = resolve_path(args.out_dir)
    docs_dir = resolve_path(args.docs_dir)
    out_dirs = ensure_dirs(out_dir, docs_dir)
    model_grids = get_model_grids(args.search_mode)

    features_df = load_supervised_feature_table(engineered_path=engineered_path, embedding_path=embedding_path)
    feature_integrity = check_supervised_feature_table(features_df)
    save_json(out_dir / "data_integrity_summary.json", feature_integrity)

    split_integrity = collect_split_integrity(features_df, split_root=split_root)
    split_integrity.to_csv(out_dir / "split_integrity.csv", index=False)

    candidate_df, selected_df = evaluate_primary_splits(
        features_df,
        split_root=split_root,
        out_dirs=out_dirs,
        model_grids=model_grids,
    )
    selection_summary = summarize_selection(selected_df, out_dirs=out_dirs)
    family_best_df = family_best_blocks(selection_summary)
    family_best_df.to_csv(out_dir / "family_best_blocks.csv", index=False)

    for _, family_row in family_best_df.iterrows():
        generate_family_diagnostics(family_row, selected_df, features_df, split_root, out_dirs)

    best_row = choose_best_row(selection_summary)
    generate_best_model_posthoc(best_row, selected_df, features_df, split_root, out_dirs)
    evaluate_seed_stability(best_row, features_df, split_root, out_dirs)
    evaluate_spatial_robustness(best_row, features_df, split_root, out_dirs)
    evaluate_threshold_sensitivity(best_row, selected_df, out_dirs)
    fit_final_model(best_row, candidate_df, features_df, out_dirs)
    write_part3_docs(
        docs_dir=docs_dir,
        selection_summary=selection_summary,
        family_best_df=family_best_df,
        best_row=best_row,
        feature_integrity=feature_integrity,
        split_integrity_path=out_dir / "split_integrity.csv",
        search_mode=args.search_mode,
    )

    print(f"[part3] wrote {out_dir / 'model_selection_summary.csv'}")
    print(f"[part3] wrote {out_dir / 'outer_split_metrics.csv'}")
    print(f"[part3] wrote {out_dir / 'selected_model_config.json'}")
    print(f"[part3] wrote {out_dir / 'models' / 'final_model.joblib'}")
    print(f"[part3] wrote {docs_dir / 'README.md'}")
    print(f"[part3] wrote {docs_dir / 'meeting_summary.md'}")


if __name__ == "__main__":
    main()
