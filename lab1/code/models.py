#!/usr/bin/env python3
"""
models.py  (Lab 1)  -- FIXED (pandas>=2.0)

Fix in this version
-------------------
Your error comes from:
  pd.to_numeric(..., errors="ignore")
In newer pandas, valid errors are {"raise","coerce"} (and "ignore" can error depending on version).

This version:
- removes errors="ignore" usage
- safely coerces categoricals with try/except
- keeps everything else from the prior "fixed/upgraded" models.py

Outputs (lab1/output/)
----------------------
Core:
- model_report.txt
- impact_summary.txt
- false_negatives.csv

Predictions:
- predictions_pecarn.csv
- predictions_logreg_baseline.csv
- predictions_logreg_bin6.csv
- predictions_logreg_bin12.csv
- predictions_rf_baseline.csv
- predictions_rf_bin6.csv
- predictions_rf_bin12.csv

Interpretability:
- logreg_coefficients_baseline.csv
- logreg_coefficients_bin6.csv
- logreg_coefficients_bin12.csv
- rf_feature_importances_baseline.csv
- rf_feature_importances_bin6.csv
- rf_feature_importances_bin12.csv

Owner: Selina Yu
Last modified: 2026-02-20
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Paths:
    lab1_dir: Path
    output_dirname: str = "output"
    cleaned_full_name: str = "cleaned_full.csv"
    cleaned_rule_name: str = "cleaned_rule_cohort.csv"

    model_report_name: str = "model_report.txt"
    impact_summary_name: str = "impact_summary.txt"
    false_negatives_name: str = "false_negatives.csv"

    pred_pecarn_name: str = "predictions_pecarn.csv"
    pred_logreg_baseline_name: str = "predictions_logreg_baseline.csv"
    pred_logreg_bin6_name: str = "predictions_logreg_bin6.csv"
    pred_logreg_bin12_name: str = "predictions_logreg_bin12.csv"

    pred_rf_baseline_name: str = "predictions_rf_baseline.csv"
    pred_rf_bin6_name: str = "predictions_rf_bin6.csv"
    pred_rf_bin12_name: str = "predictions_rf_bin12.csv"

    coef_logreg_baseline_name: str = "logreg_coefficients_baseline.csv"
    coef_logreg_bin6_name: str = "logreg_coefficients_bin6.csv"
    coef_logreg_bin12_name: str = "logreg_coefficients_bin12.csv"

    imp_rf_baseline_name: str = "rf_feature_importances_baseline.csv"
    imp_rf_bin6_name: str = "rf_feature_importances_bin6.csv"
    imp_rf_bin12_name: str = "rf_feature_importances_bin12.csv"


# CSV loses pandas "category". Recreate for these (if present).
FORCE_CATEGORICAL = [
    "InjuryMech",
    "High_impact_InjSev",
    "HemaLoc",
    "HemaSize",
    "HASeverity",
    "LocLen",
]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _safe_bool_absent(series: pd.Series) -> pd.Series:
    """
    Conservative 'absent' check: True iff value is explicitly 0.
    NA/unknown -> False (cannot claim absent).
    """
    return (series == 0).fillna(False)


def _choose_threshold_for_sensitivity(y_true: np.ndarray, y_prob: np.ndarray, target_sens: float) -> float:
    """
    Choose the smallest threshold that achieves at least target sensitivity (recall on positives)
    on the provided labels/probabilities. If impossible, returns 1.0.
    """
    order = np.argsort(-y_prob)
    y_true_sorted = y_true[order]
    y_prob_sorted = y_prob[order]

    n_pos = int((y_true_sorted == 1).sum())
    if n_pos == 0:
        return 1.0

    tp = 0
    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            tp += 1
        sens = tp / n_pos
        if sens >= target_sens:
            return float(y_prob_sorted[i])
    return 1.0


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    acc = accuracy_score(y_true, y_pred)

    out: dict[str, Any] = {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
        "accuracy": float(acc),
    }
    if y_prob is not None:
        try:
            out["auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            out["auc"] = np.nan
    return out


def _write_dict_block(f, title: str, d: dict[str, Any]) -> None:
    f.write(title + "\n")
    f.write("-" * len(title) + "\n")
    for k, v in d.items():
        f.write(f"{k}: {v}\n")
    f.write("\n")


def _make_split_indices(y: np.ndarray, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(np.zeros_like(y), y))
    return train_idx, test_idx


def _coerce_categoricals_after_read(df: pd.DataFrame) -> pd.DataFrame:
    """
    CSV roundtrip loses pandas 'category'. This recreates categories for a small set
    of known coded categoricals if they exist.

    IMPORTANT: Avoid pd.to_numeric(..., errors="ignore") due to pandas version behavior.
    """
    out = df.copy()
    for c in FORCE_CATEGORICAL:
        if c not in out.columns:
            continue
        # Try numeric coercion; if it doesn't change meaning, keep result.
        # Otherwise keep original values.
        try:
            coerced = pd.to_numeric(out[c], errors="coerce")
            # If coercion yields at least some non-NA, use it; else keep original.
            if coerced.notna().any():
                out[c] = coerced
        except Exception:
            pass
        out[c] = out[c].astype("category")
    return out


# -----------------------------------------------------------------------------
# Model 1: PECARN clinical decision rule (CT recommendation)
# -----------------------------------------------------------------------------
def pecarn_predict_ct(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces PECARN-based CT recommendation on the rule cohort.

    Required columns (derived by clean.py):
      age_lt_2, age_ge_2,
      ams,
      palpable_or_unclear_skull_fx,
      scalp_hematoma_opt,
      loc_ge_5s,
      severe_mechanism,
      not_acting_normally,
      basilar_skull_fx_signs,
      loc_any,
      vomiting,
      severe_headache
    """
    needed = [
        "age_lt_2",
        "age_ge_2",
        "ams",
        "palpable_or_unclear_skull_fx",
        "scalp_hematoma_opt",
        "loc_ge_5s",
        "severe_mechanism",
        "not_acting_normally",
        "basilar_skull_fx_signs",
        "loc_any",
        "vomiting",
        "severe_headache",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required PECARN feature columns: {missing}")

    u2_absent = (
        _safe_bool_absent(df["ams"])
        & _safe_bool_absent(df["palpable_or_unclear_skull_fx"])
        & _safe_bool_absent(df["scalp_hematoma_opt"])
        & _safe_bool_absent(df["loc_ge_5s"])
        & _safe_bool_absent(df["severe_mechanism"])
        & _safe_bool_absent(df["not_acting_normally"])
    )
    o2_absent = (
        _safe_bool_absent(df["ams"])
        & _safe_bool_absent(df["basilar_skull_fx_signs"])
        & _safe_bool_absent(df["loc_any"])
        & _safe_bool_absent(df["vomiting"])
        & _safe_bool_absent(df["severe_mechanism"])
        & _safe_bool_absent(df["severe_headache"])
    )

    age_lt_2 = df["age_lt_2"].fillna(False).astype(bool)
    age_ge_2 = df["age_ge_2"].fillna(False).astype(bool)

    low_risk = (age_lt_2 & u2_absent) | (age_ge_2 & o2_absent)
    ct_recommend = (~low_risk).astype(int)

    # Optional: simple "risk score" = count of criteria that are not explicitly absent
    u2_terms = [
        df["ams"],
        df["palpable_or_unclear_skull_fx"],
        df["scalp_hematoma_opt"],
        df["loc_ge_5s"],
        df["severe_mechanism"],
        df["not_acting_normally"],
    ]
    o2_terms = [
        df["ams"],
        df["basilar_skull_fx_signs"],
        df["loc_any"],
        df["vomiting"],
        df["severe_mechanism"],
        df["severe_headache"],
    ]
    u2_score = sum((t != 0).fillna(True).astype(int) for t in u2_terms)
    o2_score = sum((t != 0).fillna(True).astype(int) for t in o2_terms)
    score = np.where(age_lt_2, u2_score, np.where(age_ge_2, o2_score, np.nan))

    out = df.copy()
    out["pred_low_risk"] = low_risk.astype(int)
    out["pred_ct_recommend"] = ct_recommend
    out["pecarn_risk_score"] = score
    return out


# -----------------------------------------------------------------------------
# ML preprocessor + feature name extraction
# -----------------------------------------------------------------------------
def _build_preprocessor(categorical_cols: Iterable[str], numeric_cols: Iterable[str]) -> ColumnTransformer:
    cat = list(categorical_cols)
    num = list(numeric_cols)

    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num),
            ("cat", cat_pipe, cat),
        ],
        remainder="drop",
    )


def _get_expanded_feature_names(pipe: Pipeline) -> list[str]:
    """
    Extract expanded feature names after preprocessing, including one-hot categories.
    Assumes pipeline: ("pre" -> ColumnTransformer) + ("clf" -> estimator)
    """
    pre: ColumnTransformer = pipe.named_steps["pre"]
    names: list[str] = []

    num_cols = pre.transformers_[0][2]
    names.extend(list(num_cols))

    cat_cols = pre.transformers_[1][2]
    cat_pipe: Pipeline = pre.named_transformers_["cat"]
    ohe: OneHotEncoder = cat_pipe.named_steps["onehot"]
    names.extend(list(ohe.get_feature_names_out(cat_cols)))

    return names


# -----------------------------------------------------------------------------
# Training: logistic regression + random forest
# -----------------------------------------------------------------------------
def train_logistic_regression(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    target_sensitivity: float | None = 0.95,
) -> tuple[Pipeline, dict[str, Any], dict[str, Any]]:
    X = df[feature_cols].copy()
    y = df[target_col].astype(int).to_numpy()

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    cat_cols = [c for c in feature_cols if str(X[c].dtype) == "category" or X[c].dtype == object]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    pre = _build_preprocessor(cat_cols, num_cols)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )

    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    p_train = pipe.predict_proba(X_train)[:, 1]
    p_test = pipe.predict_proba(X_test)[:, 1]

    thr = 0.5 if target_sensitivity is None else _choose_threshold_for_sensitivity(y_train, p_train, target_sensitivity)
    pred_test = (p_test >= thr).astype(int)

    metrics = _binary_metrics(y_test, pred_test, p_test)
    metrics["threshold"] = float(thr)
    metrics["n_train"] = int(len(train_idx))
    metrics["n_test"] = int(len(test_idx))
    metrics["target_sensitivity_train"] = target_sensitivity

    artifacts = {
        "p_train": p_train,
        "p_test": p_test,
        "y_train": y_train,
        "y_test": y_test,
        "thr": float(thr),
        "pred_train": (p_train >= thr).astype(int),
        "pred_test": pred_test,
    }
    return pipe, metrics, artifacts


def train_random_forest(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int = 214,
) -> tuple[Pipeline, dict[str, Any], dict[str, Any]]:
    X = df[feature_cols].copy()
    y = df[target_col].astype(int).to_numpy()

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    cat_cols = [c for c in feature_cols if str(X[c].dtype) == "category" or X[c].dtype == object]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    pre = _build_preprocessor(cat_cols, num_cols)

    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=seed,
        class_weight="balanced_subsample",
        min_samples_leaf=5,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[("pre", pre), ("clf", rf)])
    pipe.fit(X_train, y_train)

    p_train = pipe.predict_proba(X_train)[:, 1]
    p_test = pipe.predict_proba(X_test)[:, 1]

    thr = 0.5
    pred_test = (p_test >= thr).astype(int)

    metrics = _binary_metrics(y_test, pred_test, p_test)
    metrics["threshold"] = float(thr)
    metrics["n_train"] = int(len(train_idx))
    metrics["n_test"] = int(len(test_idx))

    artifacts = {
        "p_train": p_train,
        "p_test": p_test,
        "y_train": y_train,
        "y_test": y_test,
        "thr": float(thr),
        "pred_train": (p_train >= thr).astype(int),
        "pred_test": pred_test,
    }
    return pipe, metrics, artifacts


# -----------------------------------------------------------------------------
# Predictions + artifacts writers
# -----------------------------------------------------------------------------
def _save_predictions(
    out_path: Path,
    df: pd.DataFrame,
    model_name: str,
    y_true_col: str,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    indices: np.ndarray,
    split: np.ndarray,
) -> None:
    base = df.iloc[indices].copy()

    out = pd.DataFrame(
        {
            "PatNum": base["PatNum"] if "PatNum" in base.columns else indices,
            "y_true": base[y_true_col].astype(int).to_numpy(),
            "y_pred": y_pred.astype(int),
            "model": model_name,
            "split": split,
        }
    )
    if y_prob is not None:
        out["y_prob"] = y_prob
    out.to_csv(out_path, index=False)


def _collect_false_negatives(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred: np.ndarray,
    indices: np.ndarray,
    model_name: str,
    split: np.ndarray,
) -> pd.DataFrame:
    base = df.iloc[indices].copy()
    y_true = base[y_true_col].astype(int).to_numpy()
    mask_fn = (y_true == 1) & (y_pred == 0)

    cols = ["PatNum", "AgeInMonth", "AgeInYears", "AgeTwoPlus", "PosIntFinal", "GCSGroup", "GCSTotal", "CTDone"]
    keep = [c for c in cols if c in base.columns]

    fn = base.loc[mask_fn, keep].copy()
    fn["model"] = model_name
    fn["split"] = split[mask_fn]
    return fn


def _save_logreg_coefficients(out_path: Path, pipe: Pipeline) -> None:
    feat_names = _get_expanded_feature_names(pipe)
    coef = pipe.named_steps["clf"].coef_.ravel()
    or_ = np.exp(coef)
    coef_df = pd.DataFrame({"feature": feat_names, "coef": coef, "odds_ratio": or_})
    coef_df = coef_df.reindex(coef_df["odds_ratio"].abs().sort_values(ascending=False).index)
    coef_df.to_csv(out_path, index=False)


def _save_rf_importances(out_path: Path, pipe: Pipeline) -> None:
    feat_names = _get_expanded_feature_names(pipe)
    imp = pipe.named_steps["clf"].feature_importances_
    imp_df = pd.DataFrame({"feature": feat_names, "importance": imp}).sort_values("importance", ascending=False)
    imp_df.to_csv(out_path, index=False)


# -----------------------------------------------------------------------------
# Stability variants (age binning)
# -----------------------------------------------------------------------------
def add_age_binned(df: pd.DataFrame, bin_months: int) -> pd.DataFrame:
    out = df.copy()
    if "AgeInMonth" not in out.columns:
        raise KeyError("AgeInMonth is required to create binned age features.")
    out[f"AgeBin{bin_months}m"] = (out["AgeInMonth"] // bin_months).astype("Int64")
    out[f"AgeBin{bin_months}m"] = out[f"AgeBin{bin_months}m"].astype("category")
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    lab1_dir = Path(__file__).resolve().parents[1]
    paths = Paths(lab1_dir=lab1_dir)
    out_dir = paths.lab1_dir / paths.output_dirname
    out_dir.mkdir(parents=True, exist_ok=True)

    full_path = out_dir / paths.cleaned_full_name
    rule_path = out_dir / paths.cleaned_rule_name

    if not full_path.exists() or not rule_path.exists():
        raise FileNotFoundError(
            "Missing cleaned files. Expected:\n"
            f"- {full_path}\n"
            f"- {rule_path}\n"
            "Run clean.py first."
        )

    df_full = pd.read_csv(full_path)
    df_rule = pd.read_csv(rule_path)

    df_full = _coerce_categoricals_after_read(df_full)
    df_rule = _coerce_categoricals_after_read(df_rule)

    target = "PosIntFinal"
    if target not in df_rule.columns:
        raise KeyError(f"Missing target column in rule cohort: {target}")

    # One fixed split for all ML variants
    y_all = df_rule[target].astype(int).to_numpy()
    seed = 214
    test_size = 0.30
    train_idx, test_idx = _make_split_indices(y_all, test_size=test_size, seed=seed)

    idx_all = np.arange(len(df_rule))
    split_all = np.array(["all"] * len(df_rule))
    idx_trte = np.concatenate([train_idx, test_idx])
    split_trte = np.array(["train"] * len(train_idx) + ["test"] * len(test_idx))

    # -------------------------------------------------------------------------
    # Model 1: PECARN CDR
    # -------------------------------------------------------------------------
    df_rule_pec = pecarn_predict_ct(df_rule)
    y_true_all = df_rule_pec[target].astype(int).to_numpy()
    y_pred_pec = df_rule_pec["pred_ct_recommend"].astype(int).to_numpy()
    pec_metrics = _binary_metrics(y_true_all, y_pred_pec, None)

    _save_predictions(
        out_dir / paths.pred_pecarn_name,
        df_rule_pec,
        model_name="pecarn_cdr",
        y_true_col=target,
        y_pred=y_pred_pec,
        y_prob=df_rule_pec["pecarn_risk_score"].to_numpy(),
        indices=idx_all,
        split=split_all,
    )

    # -------------------------------------------------------------------------
    # Features
    # -------------------------------------------------------------------------
    base_candidate_features = [
        "AgeYearsDerived",
        "AgeTwoPlus",
        "Gender",
        "InjuryMech",
        "High_impact_InjSev",
        "ams",
        "palpable_or_unclear_skull_fx",
        "scalp_hematoma_opt",
        "loc_ge_5s",
        "severe_mechanism",
        "not_acting_normally",
        "basilar_skull_fx_signs",
        "loc_any",
        "vomiting",
        "severe_headache",
        "headache_assessable",
        "seizure",
        "GCSTotal",
    ]
    base_features = [c for c in base_candidate_features if c in df_rule.columns]

    df_bin6 = add_age_binned(df_rule, 6)
    df_bin12 = add_age_binned(df_rule, 12)
    feat_bin6 = [c for c in base_features if c != "AgeYearsDerived"] + ["AgeBin6m"]
    feat_bin12 = [c for c in base_features if c != "AgeYearsDerived"] + ["AgeBin12m"]

    # -------------------------------------------------------------------------
    # Logistic regression variants
    # -------------------------------------------------------------------------
    logreg_base_pipe, logreg_base_metrics, logreg_base_art = train_logistic_regression(
        df_rule, feature_cols=base_features, target_col=target,
        train_idx=train_idx, test_idx=test_idx, target_sensitivity=0.95
    )
    logreg_bin6_pipe, logreg_bin6_metrics, logreg_bin6_art = train_logistic_regression(
        df_bin6, feature_cols=feat_bin6, target_col=target,
        train_idx=train_idx, test_idx=test_idx, target_sensitivity=0.95
    )
    logreg_bin12_pipe, logreg_bin12_metrics, logreg_bin12_art = train_logistic_regression(
        df_bin12, feature_cols=feat_bin12, target_col=target,
        train_idx=train_idx, test_idx=test_idx, target_sensitivity=0.95
    )

    def _save_ml_preds(filename: str, df_src: pd.DataFrame, art: dict[str, Any], model_name: str) -> None:
        y_pred = np.concatenate([art["pred_train"], art["pred_test"]])
        y_prob = np.concatenate([art["p_train"], art["p_test"]])
        _save_predictions(
            out_dir / filename,
            df_src,
            model_name=model_name,
            y_true_col=target,
            y_pred=y_pred,
            y_prob=y_prob,
            indices=idx_trte,
            split=split_trte,
        )

    _save_ml_preds(paths.pred_logreg_baseline_name, df_rule, logreg_base_art, "logistic_regression_baseline")
    _save_ml_preds(paths.pred_logreg_bin6_name, df_bin6, logreg_bin6_art, "logistic_regression_bin6")
    _save_ml_preds(paths.pred_logreg_bin12_name, df_bin12, logreg_bin12_art, "logistic_regression_bin12")

    _save_logreg_coefficients(out_dir / paths.coef_logreg_baseline_name, logreg_base_pipe)
    _save_logreg_coefficients(out_dir / paths.coef_logreg_bin6_name, logreg_bin6_pipe)
    _save_logreg_coefficients(out_dir / paths.coef_logreg_bin12_name, logreg_bin12_pipe)

    # -------------------------------------------------------------------------
    # Random forest variants
    # -------------------------------------------------------------------------
    rf_base_pipe, rf_base_metrics, rf_base_art = train_random_forest(
        df_rule, feature_cols=base_features, target_col=target,
        train_idx=train_idx, test_idx=test_idx, seed=seed
    )
    rf_bin6_pipe, rf_bin6_metrics, rf_bin6_art = train_random_forest(
        df_bin6, feature_cols=feat_bin6, target_col=target,
        train_idx=train_idx, test_idx=test_idx, seed=seed
    )
    rf_bin12_pipe, rf_bin12_metrics, rf_bin12_art = train_random_forest(
        df_bin12, feature_cols=feat_bin12, target_col=target,
        train_idx=train_idx, test_idx=test_idx, seed=seed
    )

    _save_ml_preds(paths.pred_rf_baseline_name, df_rule, rf_base_art, "random_forest_baseline")
    _save_ml_preds(paths.pred_rf_bin6_name, df_bin6, rf_bin6_art, "random_forest_bin6")
    _save_ml_preds(paths.pred_rf_bin12_name, df_bin12, rf_bin12_art, "random_forest_bin12")

    _save_rf_importances(out_dir / paths.imp_rf_baseline_name, rf_base_pipe)
    _save_rf_importances(out_dir / paths.imp_rf_bin6_name, rf_bin6_pipe)
    _save_rf_importances(out_dir / paths.imp_rf_bin12_name, rf_bin12_pipe)

    # -------------------------------------------------------------------------
    # False negatives
    # -------------------------------------------------------------------------
    fn_frames = []

    fn_frames.append(
        _collect_false_negatives(
            df_rule_pec, target, y_pred_pec,
            indices=idx_all, model_name="pecarn_cdr", split=split_all
        )
    )

    def _fn_test_only(df_src: pd.DataFrame, art: dict[str, Any], model_name: str) -> pd.DataFrame:
        return _collect_false_negatives(
            df_src,
            y_true_col=target,
            y_pred=art["pred_test"],
            indices=test_idx,
            model_name=model_name,
            split=np.array(["test"] * len(test_idx)),
        )

    fn_frames.append(_fn_test_only(df_rule, logreg_base_art, "logistic_regression_baseline"))
    fn_frames.append(_fn_test_only(df_bin6, logreg_bin6_art, "logistic_regression_bin6"))
    fn_frames.append(_fn_test_only(df_bin12, logreg_bin12_art, "logistic_regression_bin12"))

    fn_frames.append(_fn_test_only(df_rule, rf_base_art, "random_forest_baseline"))
    fn_frames.append(_fn_test_only(df_bin6, rf_bin6_art, "random_forest_bin6"))
    fn_frames.append(_fn_test_only(df_bin12, rf_bin12_art, "random_forest_bin12"))

    fn_all = pd.concat(fn_frames, ignore_index=True)
    fn_all.to_csv(out_dir / paths.false_negatives_name, index=False)

    # -------------------------------------------------------------------------
    # Reporting
    # -------------------------------------------------------------------------
    report_path = out_dir / paths.model_report_name
    with report_path.open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL REPORT (Lab 1)\n")
        f.write("=" * 80 + "\n\n")

        f.write("Dataset\n")
        f.write("-------\n")
        f.write(f"Rule cohort rows: {len(df_rule)}\n")
        f.write(f"Outcome prevalence (PosIntFinal==1): {int((df_rule[target] == 1).sum())}\n")
        f.write(f"Train/Test split: {len(train_idx)}/{len(test_idx)} (test_size={test_size}, seed={seed})\n\n")

        _write_dict_block(
            f,
            "Model 1: PECARN clinical decision rule (CT recommended if not low-risk) [evaluated on ALL]",
            pec_metrics,
        )

        _write_dict_block(
            f,
            "Model 2A: Logistic regression BASELINE (threshold chosen for target sensitivity on TRAIN) [evaluated on TEST]",
            logreg_base_metrics,
        )
        _write_dict_block(
            f,
            "Model 2B: Logistic regression BIN6 (Age binned at 6 months; same split) [evaluated on TEST]",
            logreg_bin6_metrics,
        )
        _write_dict_block(
            f,
            "Model 2C: Logistic regression BIN12 (Age binned at 12 months; same split) [evaluated on TEST]",
            logreg_bin12_metrics,
        )

        _write_dict_block(
            f,
            "Model 3A: Random forest BASELINE (threshold 0.5) [evaluated on TEST]",
            rf_base_metrics,
        )
        _write_dict_block(
            f,
            "Model 3B: Random forest BIN6 (Age binned at 6 months; threshold 0.5) [evaluated on TEST]",
            rf_bin6_metrics,
        )
        _write_dict_block(
            f,
            "Model 3C: Random forest BIN12 (Age binned at 12 months; threshold 0.5) [evaluated on TEST]",
            rf_bin12_metrics,
        )

        f.write("Interpretability artifacts\n")
        f.write("--------------------------\n")
        f.write(f"- {paths.coef_logreg_baseline_name}\n")
        f.write(f"- {paths.coef_logreg_bin6_name}\n")
        f.write(f"- {paths.coef_logreg_bin12_name}\n")
        f.write(f"- {paths.imp_rf_baseline_name}\n")
        f.write(f"- {paths.imp_rf_bin6_name}\n")
        f.write(f"- {paths.imp_rf_bin12_name}\n\n")

        f.write("Prediction artifacts\n")
        f.write("--------------------\n")
        for p in [
            paths.pred_pecarn_name,
            paths.pred_logreg_baseline_name,
            paths.pred_logreg_bin6_name,
            paths.pred_logreg_bin12_name,
            paths.pred_rf_baseline_name,
            paths.pred_rf_bin6_name,
            paths.pred_rf_bin12_name,
            paths.false_negatives_name,
            paths.model_report_name,
        ]:
            f.write(f"- {p}\n")

    impact_path = out_dir / paths.impact_summary_name
    with impact_path.open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("IMPACT SUMMARY (Lab 1)\n")
        f.write("=" * 80 + "\n\n")

        f.write("False negatives (ciTBI cases predicted low risk / CT not recommended)\n")
        f.write("-------------------------------------------------------------------\n")

        def _fn_count(model: str) -> int:
            return int((fn_all["model"] == model).sum())

        f.write(f"PECARN CDR (all rows):                {_fn_count('pecarn_cdr')}\n")
        f.write(f"Logistic regression baseline (test):  {_fn_count('logistic_regression_baseline')}\n")
        f.write(f"Logistic regression bin6 (test):      {_fn_count('logistic_regression_bin6')}\n")
        f.write(f"Logistic regression bin12 (test):     {_fn_count('logistic_regression_bin12')}\n")
        f.write(f"Random forest baseline (test):        {_fn_count('random_forest_baseline')}\n")
        f.write(f"Random forest bin6 (test):            {_fn_count('random_forest_bin6')}\n")
        f.write(f"Random forest bin12 (test):           {_fn_count('random_forest_bin12')}\n\n")

        f.write(
            "Framing note: y_pred=0 corresponds to 'low risk / CT not recommended'.\n"
            "False negatives are therefore the most clinically consequential error mode.\n"
        )

    print(f"Wrote model outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()