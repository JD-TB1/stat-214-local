#!/usr/bin/env python3
"""
debug_vars.py

Purpose
-------
Diagnose variable semantics/mapping issues for PECARN reproduction:
- Check distributions of key predictors by age group and outcome
- Check whether "Hema" is too broad (needs non-frontal hematoma)
- Check whether HemaLoc exists and what values it takes
- Check InjSev/InjuryMech code distributions (for severe mechanism mapping)
- Export summary tables to lab1/output/debug_*.csv and debug_report.txt

Owner: Selina Yu
Last modified: 2026-02-20
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

LAB1_DIR = Path("/Users/yy/Desktop/UCB/stat214/stat-214/lab1")
OUTPUT_DIR = LAB1_DIR / "output"
CLEANED_PATH = OUTPUT_DIR / "cleaned_data.csv"

DEBUG_DIR = OUTPUT_DIR  # write into output/
REPORT_PATH = DEBUG_DIR / "debug_report.txt"


KEY_VARS = [
    "PosIntFinal",
    "AgeTwoPlus",
    "AgeYearsDerived",
    "GCSTotal",
    "AMS",
    "SFxPalp",
    "SFxBas",
    "Hema",
    "HemaLoc",
    "HemaSize",
    "LOCSeparate",
    "LocLen",
    "Vomit",
    "VomitNbr",
    "HA_verb",
    "HASeverity",
    "ActNorm",
    "InjSev",
    "InjuryMech",
    "CTDone",
    "PosCT",
]


def age_group(df: pd.DataFrame) -> pd.Series:
    if "AgeTwoPlus" in df.columns:
        s = pd.to_numeric(df["AgeTwoPlus"], errors="coerce")
        return pd.Series(np.where(s == 1, ">=2", "<2"), index=df.index)
    if "AgeYearsDerived" in df.columns:
        s = pd.to_numeric(df["AgeYearsDerived"], errors="coerce")
        return pd.Series(np.where(s >= 2, ">=2", "<2"), index=df.index)
    return pd.Series(["unknown"] * len(df), index=df.index)


def vc_table(s: pd.Series, name: str, top: int = 25) -> pd.DataFrame:
    vc = s.value_counts(dropna=False).head(top)
    out = vc.reset_index()
    out.columns = [name, "count"]
    return out


def by_age_counts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = age_group(df)
    s = df[col] if col in df.columns else pd.Series([np.nan] * len(df), index=df.index)
    tmp = pd.DataFrame({"age_group": g, col: s})
    tab = (
        tmp.groupby("age_group")[col]
        .value_counts(dropna=False)
        .rename("count")
        .reset_index()
        .sort_values(["age_group", "count"], ascending=[True, False])
    )
    return tab


def by_age_outcome_rate(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    For binary-ish predictors, compute ciTBI rate when predictor==1 vs predictor==0.
    """
    if "PosIntFinal" not in df.columns or col not in df.columns:
        return pd.DataFrame()

    y = pd.to_numeric(df["PosIntFinal"], errors="coerce")
    x = pd.to_numeric(df[col], errors="coerce")
    g = age_group(df)

    tmp = pd.DataFrame({"age_group": g, "x": x, "y": y}).dropna(subset=["y"])
    tmp = tmp[tmp["x"].isin([0, 1])]

    out = (
        tmp.groupby(["age_group", "x"])["y"]
        .agg(n="size", ciTBI_rate="mean")
        .reset_index()
        .rename(columns={"x": col})
    )
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CLEANED_PATH)

    df["age_group"] = age_group(df)

    # 1) Basic sanity: are key vars present?
    present = {c: (c in df.columns) for c in KEY_VARS}
    missing_cols = [c for c, ok in present.items() if not ok]

    # 2) Value counts for Hema/HemaLoc/InjSev/InjuryMech (overall + by age)
    tables = {}

    for col in ["Hema", "HemaLoc", "HemaSize", "InjSev", "InjuryMech", "SFxPalp", "SFxBas", "AMS", "Vomit", "HA_verb", "ActNorm", "LOCSeparate"]:
        if col in df.columns:
            tables[f"vc_{col}"] = vc_table(df[col], col, top=50)
            tables[f"by_age_{col}"] = by_age_counts(df, col)

    # 3) ciTBI rate conditional on predictor (for binary predictors)
    for col in ["Hema", "SFxPalp", "SFxBas", "AMS", "Vomit", "HA_verb", "ActNorm", "LOCSeparate"]:
        if col in df.columns:
            rate_tab = by_age_outcome_rate(df, col)
            if len(rate_tab):
                tables[f"rate_{col}"] = rate_tab

    # 4) “Hema too broad?”: among <2, what fraction has Hema==1?
    hema_lt2_rate = float("nan")
    if "Hema" in df.columns:
        lt2 = df["age_group"] == "<2"
        x = pd.to_numeric(df.loc[lt2, "Hema"], errors="coerce")
        hema_lt2_rate = float((x == 1).mean())

    # 5) If HemaLoc exists, check how often it is NA when Hema==1 (should NOT be all NA)
    hema_loc_na_given_hema1 = float("nan")
    if "Hema" in df.columns and "HemaLoc" in df.columns:
        hema1 = pd.to_numeric(df["Hema"], errors="coerce") == 1
        hema_loc_na_given_hema1 = float(df.loc[hema1, "HemaLoc"].isna().mean())

    # 6) Severe mechanism sanity: InjSev distribution by age group
    injsev_summary = pd.DataFrame()
    if "InjSev" in df.columns:
        s = pd.to_numeric(df["InjSev"], errors="coerce")
        injsev_summary = (
            df.assign(InjSev_num=s)
            .groupby("age_group")["InjSev_num"]
            .agg(n="size", na_rate=lambda x: float(x.isna().mean()), unique=lambda x: int(x.nunique(dropna=True)))
            .reset_index()
        )

    # Write CSV outputs
    for name, tab in tables.items():
        tab.to_csv(DEBUG_DIR / f"debug_{name}.csv", index=False)

    if len(injsev_summary):
        injsev_summary.to_csv(DEBUG_DIR / "debug_injsev_summary.csv", index=False)

    # Write a single text report
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write("DEBUG REPORT: variable mapping diagnostics\n")
        f.write("=" * 80 + "\n\n")

        f.write("MISSING COLUMNS (among KEY_VARS)\n")
        f.write("-" * 80 + "\n")
        if missing_cols:
            for c in missing_cols:
                f.write(f"- {c}\n")
        else:
            f.write("(none)\n")
        f.write("\n")

        f.write("HEMA SANITY CHECK\n")
        f.write("-" * 80 + "\n")
        f.write(f"Fraction of <2 with Hema==1: {hema_lt2_rate:.6f}\n")
        f.write("If this is very high (e.g., ~0.3-0.4), Hema is likely too broad vs PECARN's non-frontal hematoma.\n\n")

        f.write("HEMA LOC COMPLETENESS\n")
        f.write("-" * 80 + "\n")
        f.write(f"P(HemaLoc is NA | Hema==1): {hema_loc_na_given_hema1:.6f}\n")
        f.write("If this is near 1.0, HemaLoc isn't populated and you cannot distinguish frontal vs non-frontal from this dataset.\n\n")

        f.write("INJSEV SUMMARY (for severe mechanism mapping)\n")
        f.write("-" * 80 + "\n")
        if len(injsev_summary):
            f.write(injsev_summary.to_string(index=False))
            f.write("\n")
        else:
            f.write("InjSev not present.\n")
        f.write("\n")

        f.write("NEXT STEP\n")
        f.write("-" * 80 + "\n")
        f.write("Use debug_vc_* and debug_by_age_* tables to map:\n")
        f.write("1) severe mechanism: find the variable/codes that match PECARN definition.\n")
        f.write("2) non-frontal scalp hematoma for <2: use HemaLoc if available; otherwise, revise rule to match available data.\n")

    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Wrote: {REPORT_PATH}")
    print("Wrote: debug_*.csv files to output/")


if __name__ == "__main__":
    main()