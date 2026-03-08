#!/usr/bin/env python3
"""
findings_plots_pub.py

Three publication-style finding figures using the cleaned rule cohort.

Inputs
------
- lab1/output/cleaned_rule_cohort.csv

Outputs
-------
- lab1/output/figures/finding1_citbi_by_age_months.(pdf|png)
- lab1/output/figures/finding2_ct_use_vs_yield_by_age.(pdf|png)
- lab1/output/figures/finding3_logreg_odds_ratios.(pdf|png)

Run
---
python lab1/code/findings_plots_pub.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ----------------------------
# Global style (publication-ish)
# ----------------------------
def set_pub_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
        }
    )


def ensure_outdir(lab1_dir: Path) -> Path:
    outdir = lab1_dir / "output" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def load_rule_cohort(lab1_dir: Path) -> pd.DataFrame:
    p = lab1_dir / "output" / "cleaned_rule_cohort.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}. Run clean.py first.")
    return pd.read_csv(p)


# ----------------------------
# Stats helpers
# ----------------------------
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt((phat * (1 - phat) / n) + (z**2 / (4 * n**2)))
    return (max(0.0, center - half), min(1.0, center + half))


def bin_rates_with_ci(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    bins: np.ndarray,
    y_is_one: int = 1,
) -> pd.DataFrame:
    """
    Bin xcol, compute y rate and Wilson CI in each bin.

    Returns columns:
    - x_mid, n, k, rate, lo, hi
    """
    d = df[[xcol, ycol]].dropna().copy()
    d["bin"] = pd.cut(d[xcol], bins=bins, right=False, include_lowest=True)

    rows = []
    for b, sub in d.groupby("bin", observed=True):
        n = int(len(sub))
        k = int((sub[ycol] == y_is_one).sum())
        rate = k / n if n else np.nan
        lo, hi = wilson_ci(k, n)
        # midpoint of bin for plotting
        left = float(b.left)
        right = float(b.right)
        x_mid = (left + right) / 2.0
        rows.append({"x_mid": x_mid, "n": n, "k": k, "rate": rate, "lo": lo, "hi": hi})

    out = pd.DataFrame(rows).sort_values("x_mid").reset_index(drop=True)
    return out


def save_both(fig: plt.Figure, outdir: Path, stem: str) -> None:
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Finding 1:
# ciTBI rate vs age in months (binned + CI ribbon)
# ----------------------------
def finding1_citbi_by_age_months(df: pd.DataFrame, outdir: Path) -> None:
    needed = {"AgeInMonth", "PosIntFinal"}
    if not needed.issubset(df.columns):
        return

    d = df.copy()
    d = d[d["PosIntFinal"].notna()].copy()

    # 6-month bins up to 216 months (your data max ~215)
    bins = np.arange(0, 216 + 6, 6)
    res = bin_rates_with_ci(d, "AgeInMonth", "PosIntFinal", bins=bins, y_is_one=1)

    # Rate per 1,000 is readable
    y = res["rate"] * 1000
    ylo = res["lo"] * 1000
    yhi = res["hi"] * 1000

    fig = plt.figure(figsize=(8.6, 4.9))
    ax = plt.gca()

    ax.plot(res["x_mid"], y, marker="o", linewidth=2)
    ax.fill_between(res["x_mid"], ylo, yhi, alpha=0.20)

    ax.axvline(24, linestyle="--", linewidth=1.5)
    ax.text(24 + 2, ax.get_ylim()[1] * 0.92, "PECARN split (24 mo)", fontsize=10)

    ax.set_title("Finding 1: ciTBI rate varies with age (binned; Wilson 95% CI)")
    ax.set_xlabel("Age (months)")
    ax.set_ylabel("ciTBI rate (per 1,000)")
    ax.set_xlim(0, 216)

    # annotate a few bins with n
    for i in [0, 3, 6, 12, 18, len(res) - 1]:
        if 0 <= i < len(res):
            ax.annotate(f"n={int(res.loc[i,'n'])}", (res.loc[i, "x_mid"], y.loc[i]),
                        textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

    fig.tight_layout()
    save_both(fig, outdir, "finding1_citbi_by_age_months")


# ----------------------------
# Finding 2:
# CT utilization and CT yield vs age (2-panel; binned + CI ribbon)
# ----------------------------
def finding2_ct_use_vs_yield_by_age(df: pd.DataFrame, outdir: Path) -> None:
    needed = {"AgeInMonth", "CTDone", "PosCT"}
    if not needed.issubset(df.columns):
        return

    d = df.copy()

    # Same bins as Finding 1
    bins = np.arange(0, 216 + 6, 6)

    # Panel A: CT utilization = P(CTDone==1)
    util = bin_rates_with_ci(d, "AgeInMonth", "CTDone", bins=bins, y_is_one=1)

    # Panel B: CT yield among scanned = P(PosCT==1 | CTDone==1)
    scanned = d[d["CTDone"] == 1].copy()
    # PosCT might still have NA among scanned; drop for yield computation
    scanned = scanned[scanned["PosCT"].notna()].copy()
    yld = bin_rates_with_ci(scanned, "AgeInMonth", "PosCT", bins=bins, y_is_one=1)

    fig, axes = plt.subplots(2, 1, figsize=(8.6, 7.2), sharex=True)

    # Utilization
    ax = axes[0]
    ax.plot(util["x_mid"], util["rate"], marker="o", linewidth=2)
    ax.fill_between(util["x_mid"], util["lo"], util["hi"], alpha=0.20)
    ax.axvline(24, linestyle="--", linewidth=1.5)
    ax.set_title("Finding 2A: CT utilization by age (binned; Wilson 95% CI)")
    ax.set_ylabel("P(CT performed)")
    ax.set_ylim(0, 1)

    # Yield
    ax = axes[1]
    ax.plot(yld["x_mid"], yld["rate"], marker="o", linewidth=2)
    ax.fill_between(yld["x_mid"], yld["lo"], yld["hi"], alpha=0.20)
    ax.axvline(24, linestyle="--", linewidth=1.5)
    ax.set_title("Finding 2B: CT positivity among scanned by age (binned; Wilson 95% CI)")
    ax.set_ylabel("P(PosCT=1 | CT performed)")
    ax.set_xlabel("Age (months)")
    ax.set_ylim(0, max(0.20, float(np.nanmax(yld["hi"])) * 1.10))  # keep readable scale

    fig.tight_layout()
    save_both(fig, outdir, "finding2_ct_use_vs_yield_by_age")


# ----------------------------
# Finding 3:
# Logistic regression forest plot of odds ratios (interpretable model-based graphic)
# ----------------------------
def finding3_logreg_odds_ratios(df: pd.DataFrame, outdir: Path) -> None:
    """
    Fit a simple logistic regression on an interpretable subset (not a "best model"),
    then plot odds ratios with 95% CI as a forest plot.

    This is publication-style and complements Findings 1–2 by quantifying associations.
    """
    # Choose a small, clinically interpretable feature set.
    # All are in cleaned_rule_cohort.csv under your current cleaning.
    candidate = [
        "AgeInMonth",
        "GCSTotal",
        "AMS",
        "High_impact_InjSev",
        "Vomit",
        "LOCSeparate",
        "SFxBas",
        "Hema",
        "HA_verb",
    ]
    needed = set(candidate + ["PosIntFinal"])
    if not needed.issubset(df.columns):
        # If some are missing due to earlier edits, degrade gracefully by using what exists.
        candidate = [c for c in candidate if c in df.columns]
        if not set(candidate + ["PosIntFinal"]).issubset(df.columns) or len(candidate) < 3:
            return

    d = df[candidate + ["PosIntFinal"]].copy()
    d = d[d["PosIntFinal"].notna()].copy()
    y = (d["PosIntFinal"] == 1).astype(int)

    # Numeric casting: treat these as numeric codes (we're explicit).
    X = d[candidate].apply(pd.to_numeric, errors="coerce")

    # For a clean forest plot, do complete-case on selected variables.
    keep = X.notna().all(axis=1)
    X = X.loc[keep].copy()
    y = y.loc[keep].copy()

    # Mild regularization for stability
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=214, stratify=y
    )

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )
    pipe.fit(X_train, y_train)

    clf = pipe.named_steps["clf"]
    # Approximate standard errors via observed Fisher information (common in practice).
    # This is adequate for a report figure; document it in caption if needed.
    # Compute on TRAIN design matrix after scaling:
    Xs = pipe.named_steps["scaler"].transform(X_train)
    Xs = np.asarray(Xs)
    p = clf.predict_proba(X_train)[:, 1]
    W = p * (1 - p)
    # Hessian approx: X^T W X
    XtW = Xs.T * W
    H = XtW @ Xs
    try:
        cov = np.linalg.inv(H)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        # fallback: no CI
        se = np.full_like(clf.coef_[0], np.nan, dtype=float)

    beta = clf.coef_[0]
    # 95% CI for beta
    lo = beta - 1.96 * se
    hi = beta + 1.96 * se

    # Convert to odds ratios
    OR = np.exp(beta)
    OR_lo = np.exp(lo)
    OR_hi = np.exp(hi)

    res = pd.DataFrame(
        {"feature": candidate, "OR": OR, "OR_lo": OR_lo, "OR_hi": OR_hi}
    )

    # Sort by |log(OR)| for readability
    res["abs_log_or"] = np.abs(np.log(res["OR"]))
    res = res.sort_values("abs_log_or", ascending=True).reset_index(drop=True)

    fig = plt.figure(figsize=(8.6, 5.4))
    ax = plt.gca()

    y_pos = np.arange(len(res))
    ax.hlines(y_pos, res["OR_lo"], res["OR_hi"], linewidth=2)
    ax.plot(res["OR"], y_pos, marker="o", linestyle="none")
    ax.axvline(1.0, linestyle="--", linewidth=1.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(res["feature"])
    ax.set_xscale("log")
    ax.set_xlabel("Odds ratio (log scale); 95% CI")
    ax.set_title("Finding 3: Logistic regression associations with ciTBI (forest plot)")

    # Annotate OR values
    for i, r in res.iterrows():
        ax.text(float(r["OR_hi"]) * 1.03, i, f"{r['OR']:.2f}", va="center", fontsize=9)

    fig.tight_layout()
    save_both(fig, outdir, "finding3_logreg_odds_ratios")


def main() -> None:
    set_pub_style()
    lab1_dir = Path(__file__).resolve().parents[1]
    outdir = ensure_outdir(lab1_dir)
    df = load_rule_cohort(lab1_dir)

    finding1_citbi_by_age_months(df, outdir)
    finding2_ct_use_vs_yield_by_age(df, outdir)
    finding3_logreg_odds_ratios(df, outdir)

    print(f"Wrote publication-style findings to: {outdir.resolve()}")


if __name__ == "__main__":
    main()