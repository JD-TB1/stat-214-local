#!/usr/bin/env python3
"""
eda_plots.py

Generate exploratory (non-findings) plots to give the reader an overall feel
for the PECARN TBI dataset and to motivate cleaning decisions.

Inputs
------
- lab1/output/cleaned_full.csv
- lab1/output/cleaned_rule_cohort.csv

Outputs
-------
Writes .png figures to lab1/output/figures/

Run
---
python lab1/code/eda_plots.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_outdir(lab1_dir: Path) -> Path:
    outdir = lab1_dir / "output" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt((phat * (1 - phat) / n) + (z**2 / (4 * n**2)))
    return (max(0.0, center - half), min(1.0, center + half))


def load_data(lab1_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_path = lab1_dir / "output" / "cleaned_full.csv"
    rule_path = lab1_dir / "output" / "cleaned_rule_cohort.csv"

    if not full_path.exists() or not rule_path.exists():
        raise FileNotFoundError(
            "Missing cleaned files. Expected:\n"
            f"- {full_path}\n"
            f"- {rule_path}\n"
            "Run clean.py first."
        )

    full_df = pd.read_csv(full_path)
    rule_df = pd.read_csv(rule_path)
    return full_df, rule_df


def plot_age_distribution(full_df: pd.DataFrame, outdir: Path) -> None:
    """
    Reason: age drives the PECARN rule split (<2 vs >=2). Show the age support and mass.
    """
    if "AgeInMonth" not in full_df.columns:
        return

    x = full_df["AgeInMonth"].dropna().to_numpy()
    plt.figure(figsize=(8.5, 5.2))
    plt.hist(x, bins=50)
    plt.axvline(24, linestyle="--")
    plt.title("Age distribution (months) with PECARN split at 24 months")
    plt.xlabel("Age (months)")
    plt.ylabel("Number of patients")
    plt.tight_layout()
    plt.savefig(outdir / "eda_age_distribution.png", dpi=300)
    plt.close()


def plot_outcome_prevalence(rule_df: pd.DataFrame, outdir: Path) -> None:
    """
    Reason: outcome is rare; prevalence affects PPV/NPV and model calibration.
    """
    if "PosIntFinal" not in rule_df.columns:
        return

    y = rule_df["PosIntFinal"].dropna()
    n = int(len(y))
    k = int((y == 1).sum())

    plt.figure(figsize=(6.5, 4.5))
    plt.bar(["No ciTBI", "ciTBI"], [n - k, k])
    plt.title(f"Outcome prevalence in rule cohort (n={n}, ciTBI={k}, {100*k/n:.2f}%)")
    plt.ylabel("Number of patients")
    plt.tight_layout()
    plt.savefig(outdir / "eda_outcome_prevalence.png", dpi=300)
    plt.close()


def plot_ct_usage_by_age(rule_df: pd.DataFrame, outdir: Path) -> None:
    """
    Reason: CT utilization is central to the domain tradeoff; show how CT use varies by age group.
    """
    needed = {"AgeTwoPlus", "CTDone"}
    if not needed.issubset(rule_df.columns):
        return

    df = rule_df.copy()
    # Dataset convention: AgeTwoPlus 1=<2, 2=>=2
    df["age_group"] = np.where(df["AgeTwoPlus"] == 1, "<2 years", ">=2 years")

    tab = (
        df.groupby("age_group")["CTDone"]
        .agg(n="size", ct_rate=lambda s: np.mean(s == 1))
        .reset_index()
    )

    plt.figure(figsize=(6.5, 4.5))
    plt.bar(tab["age_group"], tab["ct_rate"])
    plt.ylim(0, 1)
    plt.title("CT utilization rate by age group (rule cohort)")
    plt.ylabel("Proportion with CTDone==1")
    for i, r in tab.iterrows():
        plt.text(i, r["ct_rate"] + 0.02, f"n={int(r['n'])}", ha="center")
    plt.tight_layout()
    plt.savefig(outdir / "eda_ct_rate_by_age.png", dpi=300)
    plt.close()


def plot_key_missingness(rule_df: pd.DataFrame, outdir: Path) -> None:
    """
    Reason: motivate cleaning decisions by showing which clinically relevant variables are structurally missing.
    Uses the cleaned dataset (after your targeted 92->NA conversions).
    """
    key_cols = [
        "LOCSeparate",
        "LocLen",
        "HA_verb",
        "HASeverity",
        "Vomit",
        "Hema",
        "HemaLoc",
        "SFxPalp",
        "SFxBas",
        "GCSEye",
        "GCSVerbal",
        "GCSMotor",
        "GCSTotal",
        "AMS",
        "High_impact_InjSev",
    ]
    present = [c for c in key_cols if c in rule_df.columns]
    if not present:
        return

    miss = pd.DataFrame(
        {
            "column": present,
            "missing_rate": [float(rule_df[c].isna().mean()) for c in present],
        }
    ).sort_values("missing_rate", ascending=False)

    plt.figure(figsize=(9.5, 5.2))
    plt.bar(miss["column"], miss["missing_rate"])
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.title("Missingness rates for key clinical variables (rule cohort)")
    plt.ylabel("Proportion missing (NA)")
    plt.tight_layout()
    plt.savefig(outdir / "eda_key_missingness.png", dpi=300)
    plt.close()


def plot_gcs_distribution(rule_df: pd.DataFrame, outdir: Path) -> None:
    """
    Reason: GCS is clinically central; show concentration near 15 and the rare lower scores.
    """
    if "GCSTotal" not in rule_df.columns:
        return
    x = rule_df["GCSTotal"].dropna().to_numpy()

    plt.figure(figsize=(7.5, 4.8))
    bins = np.arange(2.5, 15.6, 1.0)
    plt.hist(x, bins=bins, edgecolor="black")
    plt.title("Distribution of Glasgow Coma Scale Total (rule cohort)")
    plt.xlabel("GCS Total")
    plt.ylabel("Number of patients")
    plt.tight_layout()
    plt.savefig(outdir / "eda_gcs_total_distribution.png", dpi=300)
    plt.close()


def main() -> None:
    # Resolve lab1 directory relative to this file: lab1/code/eda_plots.py -> lab1/
    lab1_dir = Path(__file__).resolve().parents[1]
    outdir = _ensure_outdir(lab1_dir)

    full_df, rule_df = load_data(lab1_dir)

    plot_age_distribution(full_df, outdir)
    plot_outcome_prevalence(rule_df, outdir)
    plot_ct_usage_by_age(rule_df, outdir)
    plot_key_missingness(rule_df, outdir)
    plot_gcs_distribution(rule_df, outdir)

    print(f"Wrote EDA figures to: {outdir.resolve()}")


if __name__ == "__main__":
    main()