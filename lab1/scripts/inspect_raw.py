#!/usr/bin/env python3
"""
inspect_raw.py

Purpose
-------
Run a thorough *pre-cleaning* inspection of the PECARN TBI raw CSV.
This script is intentionally "read-only": it does NOT mutate data.
It produces a structured set of outputs you can paste into your report and
use to justify cleaning decisions.

Inputs
------
- Raw CSV: lab1/data/TBI PUD 10-08-2013.csv (path set below)

Outputs (written to OUT_DIR)
----------------------------
- 00_overview.txt
- 01_columns.csv
- 02_na_summary.csv
- 03_sentinel_summary.csv
- 04_value_counts_topk/ (one CSV per column, top-k frequencies)
- 05_cross_checks.txt
- 06_dependency_violations.csv
- 07_duplicates.txt
- 08_ct_outcome_checks.txt
- 09_age_checks.txt

Owner: <YOUR NAME>
Last modified: 2026-02-20
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# -----------------------------
# Config
# -----------------------------
DATA_PATH = Path(
    "/Users/jayding/Desktop/DUKE/YY/歪歪长官的任务/Stat 214/stat-214-gsi/lab1/data/TBI PUD 10-08-2013.csv"
)

OUT_DIR = Path("/Users/jayding/Desktop/DUKE/YY/歪歪长官的任务/Stat 214/stat-214-local/output/inspect_raw")
TOPK = 15  # top categories shown per column
SENTINELS = [90, 91, 92]  # common codes in documentation: Other, Preverbal/Nonverbal, Not applicable
ENCODING = "utf-8"


@dataclass(frozen=True)
class DepRule:
    """
    Dependency rule: if trigger_col takes any value in trigger_vals,
    then dependent_col should be in allowed_when_triggered.
    """
    trigger_col: str
    trigger_vals: set[int]
    dependent_col: str
    allowed_when_triggered: set[int]


def _ensure_outdir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "04_value_counts_topk").mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding=ENCODING)


def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype="float64")
    return df[col]


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def _describe_numeric(s: pd.Series) -> pd.Series:
    return s.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])


def _value_counts_topk(s: pd.Series, topk: int) -> pd.DataFrame:
    vc = s.value_counts(dropna=False).head(topk)
    out = vc.rename("count").to_frame()
    out["prop"] = out["count"] / len(s)
    out.index.name = "value"
    return out.reset_index()


def _na_summary(df: pd.DataFrame) -> pd.DataFrame:
    na = df.isna().sum()
    out = na.rename("na_count").to_frame()
    out["na_prop"] = out["na_count"] / len(df)
    return out.sort_values(["na_count", "na_prop"], ascending=False).reset_index(names="column")


def _sentinel_summary(df: pd.DataFrame, sentinels: list[int]) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        if not _is_numeric(s):
            continue
        for code in sentinels:
            cnt = int((s == code).sum())
            if cnt > 0:
                rows.append(
                    {"column": col, "sentinel": code, "count": cnt, "prop": cnt / len(df)}
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["count", "column"], ascending=[False, True]).reset_index(drop=True)


def _columns_inventory(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "n_unique_including_na": int(s.nunique(dropna=False)),
                "n_unique_excluding_na": int(s.nunique(dropna=True)),
                "na_count": int(s.isna().sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("column").reset_index(drop=True)


def _duplicate_checks(df: pd.DataFrame) -> str:
    lines = []
    lines.append("DUPLICATE ROW CHECKS\n")
    dup_rows = int(df.duplicated().sum())
    lines.append(f"- Duplicated full rows: {dup_rows}\n")

    if "PatNum" in df.columns:
        dup_pat = int(df["PatNum"].duplicated().sum())
        lines.append(f"- Duplicated PatNum: {dup_pat}\n")
    else:
        lines.append("- PatNum not found.\n")

    return "".join(lines)


def _gcs_consistency(df: pd.DataFrame) -> str:
    needed = {"GCSEye", "GCSVerbal", "GCSMotor", "GCSTotal", "GCSGroup"}
    if not needed.issubset(df.columns):
        return "GCS CONSISTENCY: required columns not found.\n"

    s_eye = df["GCSEye"]
    s_ver = df["GCSVerbal"]
    s_mot = df["GCSMotor"]
    s_tot = df["GCSTotal"]

    gcs_sum = s_eye + s_ver + s_mot
    mismatch = (gcs_sum != s_tot) & (~gcs_sum.isna()) & (~s_tot.isna())
    mismatch_n = int(mismatch.sum())

    # Check GCSGroup encoding: docs say 1 = 3-13, 2 = 14-15
    grp = df["GCSGroup"]
    group_mismatch = pd.Series(False, index=df.index)
    # only check when total is present and within plausible range
    tot_ok = s_tot.between(3, 15, inclusive="both")
    group_mismatch |= (tot_ok & (s_tot <= 13) & (grp != 1))
    group_mismatch |= (tot_ok & (s_tot >= 14) & (grp != 2))
    group_mismatch_n = int(group_mismatch.sum())

    out = []
    out.append("GCS CONSISTENCY CHECKS\n")
    out.append(f"- Eye+Verbal+Motor != Total (where both non-missing): {mismatch_n}\n")
    out.append(f"- GCSGroup inconsistent with Total (where Total in [3,15]): {group_mismatch_n}\n")
    out.append("\nNumeric summaries:\n")
    out.append(str(_describe_numeric(s_tot)) + "\n")
    return "".join(out)


def _age_consistency(df: pd.DataFrame) -> str:
    cols = {"AgeInMonth", "AgeinYears", "AgeTwoPlus"}
    if not cols.issubset(df.columns):
        return "AGE CONSISTENCY: required columns not found.\n"

    m = df["AgeInMonth"]
    y = df["AgeinYears"]
    grp = df["AgeTwoPlus"]  # docs: 1 <2, 2 >=2

    out = []
    out.append("AGE CONSISTENCY CHECKS\n")
    out.append("\nAgeInMonth summary:\n")
    out.append(str(_describe_numeric(m)) + "\n")
    out.append("\nAgeinYears summary:\n")
    out.append(str(_describe_numeric(y)) + "\n")

    # Month/Year coherence (allow some rounding noise)
    diff = y - (m / 12.0)
    out.append("\nAgeinYears - AgeInMonth/12 summary:\n")
    out.append(str(_describe_numeric(diff)) + "\n")

    # AgeTwoPlus logical check
    # if AgeinYears >= 2 => AgeTwoPlus should be 2; if <2 => should be 1
    valid_year = y.notna()
    mismatch = pd.Series(False, index=df.index)
    mismatch |= valid_year & (y < 2) & (grp != 1)
    mismatch |= valid_year & (y >= 2) & (grp != 2)
    out.append(f"\n- AgeTwoPlus inconsistent with AgeinYears (where AgeinYears present): {int(mismatch.sum())}\n")
    return "".join(out)


def _ct_outcome_checks(df: pd.DataFrame) -> str:
    out = []
    out.append("CT / OUTCOME CHECKS\n")

    if "PosIntFinal" in df.columns:
        out.append("\nciTBI prevalence (PosIntFinal):\n")
        out.append(str(df["PosIntFinal"].value_counts(dropna=False)) + "\n")
    else:
        out.append("\nPosIntFinal not found.\n")

    if {"CTDone", "PosCT"}.issubset(df.columns):
        out.append("\nCTDone vs PosCT crosstab:\n")
        out.append(str(pd.crosstab(df["CTDone"], df["PosCT"], dropna=False)) + "\n")
    else:
        out.append("\nCTDone/PosCT not found.\n")

    # Basic logical check: if CTDone==0 then PosCT should be NA/92 (not applicable) in this coding
    # We cannot assume exact code; we just quantify how often PosCT is non-missing & non-92 when CT not done.
    if {"CTDone", "PosCT"}.issubset(df.columns):
        ctd = df["CTDone"]
        pos = df["PosCT"]
        suspicious = (ctd == 0) & pos.notna() & (pos != 92)
        out.append(f"\n- Suspicious: CTDone==0 but PosCT present and !=92: {int(suspicious.sum())}\n")

    return "".join(out)


def _dependency_rules() -> list[DepRule]:
    """
    Encode a few high-value conditional-logic checks from documentation.
    These are *diagnostics*, not cleaning decisions.
    """
    rules: list[DepRule] = []

    # LOC: if LOCSeparate==0 (no) then LocLen should be 92 (not applicable) per docs.
    rules.append(DepRule("LOCSeparate", {0}, "LocLen", {92}))

    # Seizure: if Seiz==0 then SeizOccur and SeizLen should be 92.
    rules.append(DepRule("Seiz", {0}, "SeizOccur", {92}))
    rules.append(DepRule("Seiz", {0}, "SeizLen", {92}))

    # Vomiting: if Vomit==0 then VomitNbr/VomitStart/VomitLast should be 92.
    rules.append(DepRule("Vomit", {0}, "VomitNbr", {92}))
    rules.append(DepRule("Vomit", {0}, "VomitStart", {92}))
    rules.append(DepRule("Vomit", {0}, "VomitLast", {92}))

    # Headache: if HA_verb==0 then HASeverity/HAStart should be 92 (or 91 for preverbal in some cases).
    # Docs for HASeverity: 92 not applicable when HA_verb is no OR preverbal/nonverbal OR missing.
    # Here we only enforce "no" -> 92; we do not enforce for preverbal cases.
    rules.append(DepRule("HA_verb", {0}, "HASeverity", {92}))
    rules.append(DepRule("HA_verb", {0}, "HAStart", {92}))

    # Basilar skull fracture composite: if SFxBas==0 then SFxBas* components should be 92.
    for sub in ["SFxBasHem", "SFxBasOto", "SFxBasPer", "SFxBasRet", "SFxBasRhi"]:
        rules.append(DepRule("SFxBas", {0}, sub, {92}))

    # Hematoma: if Hema==0 then HemaLoc/HemaSize should be 92.
    rules.append(DepRule("Hema", {0}, "HemaLoc", {92}))
    rules.append(DepRule("Hema", {0}, "HemaSize", {92}))

    return rules


def _check_dependencies(df: pd.DataFrame, rules: list[DepRule]) -> pd.DataFrame:
    rows = []
    for r in rules:
        if r.trigger_col not in df.columns or r.dependent_col not in df.columns:
            continue
        trig = df[r.trigger_col]
        dep = df[r.dependent_col]

        triggered = trig.isin(list(r.trigger_vals))
        violated = triggered & (~dep.isin(list(r.allowed_when_triggered))) & dep.notna()

        rows.append(
            {
                "trigger_col": r.trigger_col,
                "trigger_vals": sorted(list(r.trigger_vals)),
                "dependent_col": r.dependent_col,
                "allowed_dep_vals": sorted(list(r.allowed_when_triggered)),
                "n_triggered": int(triggered.sum()),
                "n_violated": int(violated.sum()),
                "violation_rate_given_trigger": (float(violated.sum()) / float(triggered.sum()))
                if float(triggered.sum()) > 0
                else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["n_violated", "violation_rate_given_trigger"], ascending=False).reset_index(drop=True)


def _numeric_range_flags(df: pd.DataFrame) -> str:
    """
    Light-touch range checks for core clinical scores.
    This is meant to surface obvious outliers, not enforce medical truth.
    """
    out = []
    out.append("BASIC NUMERIC RANGE FLAGS\n")

    def _count_outside(col: str, lo: float, hi: float) -> None:
        if col not in df.columns:
            return
        s = df[col]
        if not _is_numeric(s):
            return
        bad = s.notna() & ~s.between(lo, hi, inclusive="both")
        out.append(f"- {col} outside [{lo},{hi}] (non-missing): {int(bad.sum())}\n")

    # GCS components: typical ranges
    _count_outside("GCSEye", 1, 4)
    _count_outside("GCSVerbal", 1, 5)
    _count_outside("GCSMotor", 1, 6)
    _count_outside("GCSTotal", 3, 15)

    # Age: plausible bounds
    _count_outside("AgeInMonth", 0, 18 * 12 + 6)  # give a small buffer
    _count_outside("AgeinYears", 0, 18)

    return "".join(out)


def main() -> None:
    _ensure_outdir()

    df = pd.read_csv(DATA_PATH)

    # 00 Overview
    overview_lines = []
    overview_lines.append(f"DATA_PATH: {DATA_PATH}\n")
    overview_lines.append(f"SHAPE: {df.shape}\n")
    overview_lines.append(f"N_COLUMNS: {df.shape[1]}\n")
    overview_lines.append(f"N_ROWS: {df.shape[0]}\n")
    overview_lines.append("\nDTYPE COUNTS:\n")
    overview_lines.append(str(df.dtypes.value_counts()) + "\n")
    _write_text(OUT_DIR / "00_overview.txt", "".join(overview_lines))

    # 01 Columns inventory
    cols_df = _columns_inventory(df)
    cols_df.to_csv(OUT_DIR / "01_columns.csv", index=False)

    # 02 NA summary (true NA only)
    na_df = _na_summary(df)
    na_df.to_csv(OUT_DIR / "02_na_summary.csv", index=False)

    # 03 Sentinel summary (90/91/92) for numeric columns
    sent_df = _sentinel_summary(df, SENTINELS)
    sent_df.to_csv(OUT_DIR / "03_sentinel_summary.csv", index=False)

    # 04 Value counts topk for every column
    vc_dir = OUT_DIR / "04_value_counts_topk"
    for col in df.columns:
        s = df[col]
        # always output; helpful for spotting weird codes
        vc = _value_counts_topk(s, TOPK)
        vc.to_csv(vc_dir / f"{col}.csv", index=False)

    # 05 Cross checks (text)
    cross_lines = []
    cross_lines.append(_numeric_range_flags(df))
    cross_lines.append("\n")
    cross_lines.append(_age_consistency(df))
    cross_lines.append("\n")
    cross_lines.append(_gcs_consistency(df))
    cross_lines.append("\n")
    _write_text(OUT_DIR / "05_cross_checks.txt", "".join(cross_lines))

    # 06 Dependency violations
    dep = _check_dependencies(df, _dependency_rules())
    dep.to_csv(OUT_DIR / "06_dependency_violations.csv", index=False)

    # 07 Duplicate checks
    _write_text(OUT_DIR / "07_duplicates.txt", _duplicate_checks(df))

    # 08 CT/outcome checks
    _write_text(OUT_DIR / "08_ct_outcome_checks.txt", _ct_outcome_checks(df))

    # 09 Age checks duplicated as separate file (for report convenience)
    _write_text(OUT_DIR / "09_age_checks.txt", _age_consistency(df))

    print(f"Wrote inspection outputs to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()