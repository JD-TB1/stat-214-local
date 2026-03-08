#!/usr/bin/env python3
"""
clean.py  (PECARN-aligned)

Purpose
-------
Clean the PECARN TBI public-use dataset in a way that preserves the clinical
measurement logic used in the Kuppermann et al. (Lancet 2009) PECARN rule.

Key design choices (paper / PUD aligned)
----------------------------------------
1) DO NOT globally replace 90/91/92 with NA.
   - 92 = "not applicable" is STRUCTURAL and should be handled conditionally.
   - 91 = "preverbal/nonverbal" is NOT missing; it indicates "unassessable".
   - 90 = "other" is a valid category (e.g., InjuryMech).

2) Convert 92 -> NA ONLY for detail fields where 92 encodes "not applicable"
   (e.g., LocLen when LOCSeparate==0, seizure duration when Seiz==0, etc.).

3) Do NOT force {0,1} semantics onto multi-coded fields like:
   LOCSeparate, SFxPalp, HA_verb, PosCT.
   Instead, create derived predictors that implement the PECARN rule
   conservatively (treat "suspected/unclear" as positive, and treat unknown as
   not satisfying low-risk).

4) Create PECARN-ready derived variables (feature engineering) for <2 and >=2.

Outputs (if write_outputs=True)
-------------------------------
- lab1/output/cleaned_full.csv
- lab1/output/cleaned_rule_cohort.csv
- lab1/output/clean_report.txt
- lab1/output/missingness_key_vars.csv

Owner: Selina Yu
Last modified: 2026-02-20
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
DEFAULT_SENTINELS: tuple[int, ...] = (90, 91, 92)

# Core variables used by PECARN rule or cohort definition (plus outcome)
KEY_RULE_VARS = [
    "AgeInMonth",
    "AgeInYears",
    "AgeTwoPlus",
    "GCSGroup",
    "GCSEye",
    "GCSVerbal",
    "GCSMotor",
    "GCSTotal",
    "AMS",
    "LOCSeparate",
    "LocLen",
    "Vomit",
    "HA_verb",
    "HASeverity",
    "ActNorm",
    "SFxPalp",
    "SFxBas",
    "Hema",
    "HemaLoc",
    "High_impact_InjSev",
    "CTDone",
    "PosCT",
    "PosIntFinal",
]

# Columns where 92 specifically means "not applicable" per conditional design.
# We convert 92 -> NA in these detail fields for analysis convenience.
DETAIL_92_TO_NA = [
    # LOC / seizure / vomiting details
    "LocLen",
    "SeizOccur",
    "SeizLen",
    "VomitNbr",
    "VomitStart",
    "VomitLast",
    # Headache details
    "HASeverity",
    "HAStart",
    # Hematoma details
    "HemaLoc",
    "HemaSize",
    # Basilar skull fracture subcomponents
    "SFxBasHem",
    "SFxBasOto",
    "SFxBasPer",
    "SFxBasRet",
    "SFxBasRhi",
    # Neuro deficit subcomponents
    "NeuroDMotor",
    "NeuroDSensory",
    "NeuroDCranial",
    "NeuroDReflex",
    "NeuroDOth",
    # Other significant injuries subcomponents
    "OSIExtremity",
    "OSICut",
    "OSICspine",
    "OSIFlank",
    "OSIAbdomen",
    "OSIPelvis",
    "OSIOth",
    # CT sedation reason subfields
    "CTSedAgitate",
    "CTSedAge",
    "CTSedRqst",
    "CTSedOth",
]

# Coded categoricals we want to treat as category (do NOT destroy 90/91/92 here)
CODED_CATEGORICAL_VARS = [
    "InjuryMech",
    "High_impact_InjSev",
    "HemaLoc",
    "HemaSize",
    "HASeverity",
    "LocLen",
]


@dataclass
class CleanConfig:
    # Cohort
    drop_missing_outcome: bool = True  # drop rows with missing PosIntFinal
    create_rule_cohort_flag: bool = True  # rule cohort is GCS 14-15 + non-missing outcome

    # Sentinel handling
    # Only convert 92->NA for DETAIL_92_TO_NA columns; never global replace.
    detail_na_code: int = 92

    # GCS handling:
    # - "flag": add mismatch flags only
    # - "overwrite_total": set GCSTotal = sum where complete & mismatch
    # - "none": skip
    gcs_mode: str = "flag"

    # Age handling:
    fill_age_months_from_years: bool = False  # your dataset has 0 missing AgeInMonth
    ensure_age_two_plus: bool = True  # only if AgeTwoPlus missing

    # Typing / categoricals
    set_coded_categoricals: bool = True
    coerce_binary_int: bool = True

    # Outputs
    write_outputs: bool = False
    output_dir: str = "output"
    cleaned_full_filename: str = "cleaned_full.csv"
    cleaned_rule_filename: str = "cleaned_rule_cohort.csv"
    report_filename: str = "clean_report.txt"
    missingness_filename: str = "missingness_key_vars.csv"


@dataclass
class CleanReport:
    n_rows_raw: int
    n_cols_raw: int
    n_rows_clean_full: int
    n_cols_clean_full: int
    n_rows_rule_cohort: int
    n_cols_rule_cohort: int

    rename_applied: dict[str, str]

    rows_dropped_missing_outcome: int
    positfinal_counts_raw: dict[str, int] | None
    positfinal_counts_clean: dict[str, int] | None

    age_months_imputed_count: int
    gcs_mode: str
    gcs_mismatch_count_complete_cases: int | None

    ct_contradiction_count: int | None
    posct_set_na_count: int

    n_detail_92_to_na_converted: int
    remaining_92_counts_key_vars: dict[str, int]


# -----------------------------------------------------------------------------
# Helpers: column standardization and parsing
# -----------------------------------------------------------------------------
def _standardize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    rename_map = {
        "AgeinYears": "AgeInYears",
        "AgeinMonth": "AgeInMonth",
        "AgeinMonths": "AgeInMonth",
        "ageinyears": "AgeInYears",
        "ageinmonth": "AgeInMonth",
        "High_impact_InjSev": "High_impact_InjSev",  # keep exact
    }

    existing = set(df.columns)
    applied: dict[str, str] = {}
    for k, v in rename_map.items():
        if k in existing and v in existing and k != v:
            continue
        if k in existing and v not in existing:
            applied[k] = v

    if applied:
        df = df.rename(columns=applied)

    return df, applied


def _normalize_blank_strings_to_na(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Replace empty or whitespace-only strings with NA (helps if CSV has blanks)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].replace(r"^\s*$", np.nan, regex=True)
    return df


def _coerce_numeric_like_objects(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


# -----------------------------------------------------------------------------
# Sentinel handling: targeted 92 -> NA for detail fields
# -----------------------------------------------------------------------------
def _convert_detail_92_to_na(df: pd.DataFrame, cols: Iterable[str], na_code: int = 92) -> tuple[pd.DataFrame, int]:
    df = df.copy()
    converted = 0
    for c in cols:
        if c not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        before = int((df[c] == na_code).sum())
        if before > 0:
            df.loc[df[c] == na_code, c] = np.nan
            converted += before
    return df, converted


def _count_remaining_92(df: pd.DataFrame, cols: Iterable[str], na_code: int = 92) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            out[c] = int((df[c] == na_code).sum())
    return out


# -----------------------------------------------------------------------------
# Age handling (preserve dataset coding: AgeTwoPlus = 1 (<2), 2 (>=2))
# -----------------------------------------------------------------------------
def _handle_age(df: pd.DataFrame, cfg: CleanConfig) -> tuple[pd.DataFrame, int]:
    df = df.copy()
    imputed = 0

    has_months = "AgeInMonth" in df.columns
    has_years = "AgeInYears" in df.columns

    if cfg.fill_age_months_from_years and has_months and has_years:
        miss_months = df["AgeInMonth"].isna() & df["AgeInYears"].notna()
        imputed = int(miss_months.sum())
        if imputed > 0:
            df.loc[miss_months, "AgeInMonth"] = df.loc[miss_months, "AgeInYears"] * 12.0

    if has_months:
        df["AgeYearsDerived"] = df["AgeInMonth"] / 12.0
    elif has_years:
        df["AgeYearsDerived"] = df["AgeInYears"].astype(float)
    else:
        df["AgeYearsDerived"] = np.nan

    if cfg.ensure_age_two_plus and "AgeTwoPlus" not in df.columns:
        # Preserve dataset convention: 1 for <2, 2 for >=2
        df["AgeTwoPlus"] = np.where(df["AgeYearsDerived"] >= 2.0, 2, 1).astype("Int64")

    return df, imputed


# -----------------------------------------------------------------------------
# GCS checks
# -----------------------------------------------------------------------------
def _gcs_checks(df: pd.DataFrame, cfg: CleanConfig) -> tuple[pd.DataFrame, int | None]:
    df = df.copy()
    needed = {"GCSEye", "GCSVerbal", "GCSMotor", "GCSTotal"}
    if cfg.gcs_mode == "none" or not needed.issubset(df.columns):
        df["flag_gcs_complete"] = pd.Series([pd.NA] * len(df), dtype="boolean")
        df["flag_gcs_mismatch"] = pd.Series([pd.NA] * len(df), dtype="boolean")
        return df, None

    complete = df[list(needed)].notna().all(axis=1)
    gcs_sum = df["GCSEye"] + df["GCSVerbal"] + df["GCSMotor"]
    mismatch = complete & (gcs_sum != df["GCSTotal"])

    df["flag_gcs_complete"] = complete.astype("boolean")
    df["flag_gcs_mismatch"] = mismatch.astype("boolean")

    mismatch_count = int(mismatch.sum())
    if cfg.gcs_mode == "overwrite_total":
        df.loc[mismatch, "GCSTotal"] = gcs_sum.loc[mismatch]

    return df, mismatch_count


# -----------------------------------------------------------------------------
# CT logic: keep PosCT meaningful only if CTDone==1
# -----------------------------------------------------------------------------
def _ct_logic(df: pd.DataFrame) -> tuple[pd.DataFrame, int | None, int]:
    df = df.copy()
    posct_set_na = 0

    if {"CTDone", "PosCT"}.issubset(df.columns):
        contradiction = (df["CTDone"] == 0) & (df["PosCT"] == 1)
        df["flag_ct_contradiction"] = contradiction.astype("boolean")
        contradiction_count = int(contradiction.sum())

        affected = (df["CTDone"] == 0) & df["PosCT"].notna()
        posct_set_na = int(affected.sum())

        # When CT not done, PosCT is not observed -> set to NA
        df.loc[df["CTDone"] == 0, "PosCT"] = np.nan
        return df, contradiction_count, posct_set_na

    df["flag_ct_contradiction"] = pd.Series([pd.NA] * len(df), dtype="boolean")
    return df, None, 0


# -----------------------------------------------------------------------------
# Derived predictors for PECARN rule (conservative handling)
# -----------------------------------------------------------------------------
def _derive_pecarn_predictors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Age group flags (dataset convention: AgeTwoPlus 1=<2, 2=>=2)
    df["age_lt_2"] = (df["AgeTwoPlus"] == 1) if "AgeTwoPlus" in df.columns else pd.NA
    df["age_ge_2"] = (df["AgeTwoPlus"] == 2) if "AgeTwoPlus" in df.columns else pd.NA

    # AMS (Altered Mental Status): use PUD AMS indicator
    if "AMS" in df.columns:
        ams_arr = np.where(df["AMS"] == 1, 1, np.where(df["AMS"] == 0, 0, np.nan))
        df["ams"] = pd.Series(ams_arr, index=df.index, dtype="Float64")
    else:
        df["ams"] = pd.Series([np.nan] * len(df), index=df.index, dtype="Float64")

    # Palpable skull fracture / unclear exam combined (PUD note)
    if "SFxPalp" in df.columns:
        sfx_arr = np.where(df["SFxPalp"].isin([1, 2]), 1, np.where(df["SFxPalp"] == 0, 0, np.nan))
        df["palpable_or_unclear_skull_fx"] = pd.Series(sfx_arr, index=df.index, dtype="Float64")
    else:
        df["palpable_or_unclear_skull_fx"] = pd.Series([np.nan] * len(df), index=df.index, dtype="Float64")

    # Basilar skull fracture signs (>=2 rule)
    if "SFxBas" in df.columns:
        bas_arr = np.where(df["SFxBas"] == 1, 1, np.where(df["SFxBas"] == 0, 0, np.nan))
        df["basilar_skull_fx_signs"] = pd.Series(bas_arr, index=df.index, dtype="Float64")
    else:
        df["basilar_skull_fx_signs"] = pd.Series([np.nan] * len(df), index=df.index, dtype="Float64")

    # Scalp hematoma occipital/parietal/temporal (HemaLoc: 2=occ, 3=par/tem)
    if {"Hema", "HemaLoc"}.issubset(df.columns):
        scalp_opt = np.where(df["Hema"] == 0, 0, np.nan)
        scalp_opt = np.where((df["Hema"] == 1) & df["HemaLoc"].isin([2, 3]), 1, scalp_opt)
        scalp_opt = np.where((df["Hema"] == 1) & df["HemaLoc"].isin([1]), 0, scalp_opt)
        df["scalp_hematoma_opt"] = pd.Series(scalp_opt, index=df.index, dtype="Float64")
    else:
        df["scalp_hematoma_opt"] = pd.Series([np.nan] * len(df), index=df.index, dtype="Float64")

    # LOC any (treat suspected/unclear as present)
    if "LOCSeparate" in df.columns:
        loc_any_arr = np.where(df["LOCSeparate"].isin([1, 2]), 1, np.where(df["LOCSeparate"] == 0, 0, np.nan))
        df["loc_any"] = pd.Series(loc_any_arr, index=df.index, dtype="Float64")
    else:
        df["loc_any"] = pd.Series([np.nan] * len(df), index=df.index, dtype="Float64")

    # LOC >= 5 seconds for <2 rule
    if {"LOCSeparate", "LocLen"}.issubset(df.columns):
        loc_ge_5 = np.where(df["LOCSeparate"] == 0, 0, np.nan)
        loc_ge_5 = np.where(df["LOCSeparate"].isin([1, 2]) & df["LocLen"].isin([2, 3, 4]), 1, loc_ge_5)
        loc_ge_5 = np.where(df["LOCSeparate"].isin([1, 2]) & (df["LocLen"] == 1), 0, loc_ge_5)
        df["loc_ge_5s"] = pd.Series(loc_ge_5, index=df.index, dtype="Float64")
    else:
        df["loc_ge_5s"] = pd.Series([np.nan] * len(df), index=df.index, dtype="Float64")

    # Severe mechanism: High_impact_InjSev==3
    if "High_impact_InjSev" in df.columns:
        mech_arr = np.where(df["High_impact_InjSev"] == 3, 1, np.where(df["High_impact_InjSev"].isin([1, 2]), 0, np.nan))
        df["severe_mechanism"] = pd.Series(mech_arr, index=df.index, dtype="Float64")
    else:
        df["severe_mechanism"] = pd.Series([np.nan] * len(df), index=df.index, dtype="Float64")

    # Not acting normally per parent (<2 rule): ActNorm==0
    if "ActNorm" in df.columns:
        act_arr = np.where(df["ActNorm"] == 0, 1, np.where(df["ActNorm"] == 1, 0, np.nan))
        df["not_acting_normally"] = pd.Series(act_arr, index=df.index, dtype="Float64")
    else:
        df["not_acting_normally"] = pd.Series([np.nan] * len(df), index=df.index, dtype="Float64")

    # Vomiting (>=2 rule)
    if "Vomit" in df.columns:
        vom_arr = np.where(df["Vomit"] == 1, 1, np.where(df["Vomit"] == 0, 0, np.nan))
        df["vomiting"] = pd.Series(vom_arr, index=df.index, dtype="Float64")
    else:
        df["vomiting"] = pd.Series([np.nan] * len(df), index=df.index, dtype="Float64")

    # Headache assessability and severe headache (>=2 rule)
    if "HA_verb" in df.columns:
        assess_arr = np.where(df["HA_verb"].isin([0, 1]), 1, np.where(df["HA_verb"] == 91, 0, np.nan))
        df["headache_assessable"] = pd.Series(assess_arr, index=df.index, dtype="Float64")
    else:
        df["headache_assessable"] = pd.Series([np.nan] * len(df), index=df.index, dtype="Float64")

    if {"HA_verb", "HASeverity"}.issubset(df.columns):
        severe = np.where(df["HA_verb"] == 0, 0, np.nan)
        severe = np.where((df["HA_verb"] == 1) & (df["HASeverity"] == 3), 1, severe)
        severe = np.where((df["HA_verb"] == 1) & df["HASeverity"].isin([1, 2]), 0, severe)
        df["severe_headache"] = pd.Series(severe, index=df.index, dtype="Float64")
    else:
        df["severe_headache"] = pd.Series([np.nan] * len(df), index=df.index, dtype="Float64")

    # Optional seizure feature
    if "Seiz" in df.columns:
        seiz_arr = np.where(df["Seiz"] == 1, 1, np.where(df["Seiz"] == 0, 0, np.nan))
        df["seizure"] = pd.Series(seiz_arr, index=df.index, dtype="Float64")
    else:
        df["seizure"] = pd.Series([np.nan] * len(df), index=df.index, dtype="Float64")

    return df


# -----------------------------------------------------------------------------
# Typing: coerce strictly binary columns to Int64
# -----------------------------------------------------------------------------
def _coerce_binary_yesno(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            u = df[col].dropna().unique()
            if len(u) == 0:
                continue
            vals = set(pd.Series(u).astype(float).tolist())
            if vals.issubset({0.0, 1.0}):
                df[col] = df[col].astype("Int64")
    return df


def _set_coded_categoricals(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64").astype("category")
    return df


# -----------------------------------------------------------------------------
# Reporting helpers
# -----------------------------------------------------------------------------
def _missingness_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.DataFrame(columns=["column", "dtype", "na_count", "na_pct", "n_unique"])

    out = []
    n = len(df)
    for c in present:
        s = df[c]
        out.append(
            {
                "column": c,
                "dtype": str(s.dtype),
                "na_count": int(s.isna().sum()),
                "na_pct": float(s.isna().mean()) if n > 0 else np.nan,
                "n_unique": int(s.nunique(dropna=False)),
            }
        )
    return pd.DataFrame(out).sort_values(["na_pct", "na_count"], ascending=False)


def _value_counts_dict(s: pd.Series) -> dict[str, int]:
    vc = s.value_counts(dropna=False)
    return {str(k): int(v) for k, v in vc.items()}


# -----------------------------------------------------------------------------
# Main API
# -----------------------------------------------------------------------------
def clean_data(raw_df: pd.DataFrame, cfg: CleanConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame, CleanReport]:
    if cfg is None:
        cfg = CleanConfig()

    n_rows_raw, n_cols_raw = raw_df.shape

    df, rename_applied = _standardize_columns(raw_df)
    df = _normalize_blank_strings_to_na(df)
    df = _coerce_numeric_like_objects(df)

    # Convert 92 -> NA only in detail fields
    df, n_converted = _convert_detail_92_to_na(df, DETAIL_92_TO_NA, na_code=cfg.detail_na_code)

    # CT logic (after detail NA conversion; before dropping outcome)
    df, ct_contradiction_count, posct_set_na_count = _ct_logic(df)

    # Age + GCS checks
    df, age_imputed_count = _handle_age(df, cfg)
    df, gcs_mismatch_count = _gcs_checks(df, cfg)

    # Outcome counts before dropping
    positfinal_counts_raw = _value_counts_dict(df["PosIntFinal"]) if "PosIntFinal" in df.columns else None
    rows_dropped_missing_outcome = 0

    if "PosIntFinal" in df.columns and cfg.drop_missing_outcome:
        before = len(df)
        df = df.loc[df["PosIntFinal"].notna()].copy()
        rows_dropped_missing_outcome = before - len(df)

    positfinal_counts_clean = _value_counts_dict(df["PosIntFinal"]) if "PosIntFinal" in df.columns else None

    # Derived PECARN predictors and low-risk flags
    df = _derive_pecarn_predictors(df)

    # Rule cohort flag and dataset split
    if cfg.create_rule_cohort_flag and {"GCSGroup", "PosIntFinal"}.issubset(df.columns):
        df["rule_cohort"] = ((df["GCSGroup"] == 2) & df["PosIntFinal"].notna()).astype("boolean")
    else:
        df["rule_cohort"] = pd.Series([pd.NA] * len(df), dtype="boolean")

    rule_df = df.loc[df["rule_cohort"] == True].copy()  # noqa: E712

    # Typing adjustments
    if cfg.set_coded_categoricals:
        df = _set_coded_categoricals(df, CODED_CATEGORICAL_VARS)
        rule_df = _set_coded_categoricals(rule_df, CODED_CATEGORICAL_VARS)

    if cfg.coerce_binary_int:
        df = _coerce_binary_yesno(df)
        rule_df = _coerce_binary_yesno(rule_df)

    audit_cols = [c for c in KEY_RULE_VARS if c not in {"AgeInMonth", "AgeInYears"}]
    remaining_92_counts_key_vars = _count_remaining_92(df, audit_cols, na_code=cfg.detail_na_code)

    report = CleanReport(
        n_rows_raw=int(n_rows_raw),
        n_cols_raw=int(n_cols_raw),
        n_rows_clean_full=int(df.shape[0]),
        n_cols_clean_full=int(df.shape[1]),
        n_rows_rule_cohort=int(rule_df.shape[0]),
        n_cols_rule_cohort=int(rule_df.shape[1]),
        rename_applied=dict(rename_applied),
        rows_dropped_missing_outcome=int(rows_dropped_missing_outcome),
        positfinal_counts_raw=positfinal_counts_raw,
        positfinal_counts_clean=positfinal_counts_clean,
        age_months_imputed_count=int(age_imputed_count),
        gcs_mode=str(cfg.gcs_mode),
        gcs_mismatch_count_complete_cases=gcs_mismatch_count,
        ct_contradiction_count=ct_contradiction_count,
        posct_set_na_count=int(posct_set_na_count),
        n_detail_92_to_na_converted=int(n_converted),
        remaining_92_counts_key_vars=remaining_92_counts_key_vars,
    )

    return df, rule_df, report


# -----------------------------------------------------------------------------
# Output writer (CLI)
# -----------------------------------------------------------------------------
def write_outputs(
    lab1_dir: Path,
    full_df: pd.DataFrame,
    rule_df: pd.DataFrame,
    report: CleanReport,
    cfg: CleanConfig,
    key_vars: list[str] = KEY_RULE_VARS,
) -> None:
    out_dir = lab1_dir / cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    full_df.to_csv(out_dir / cfg.cleaned_full_filename, index=False)
    rule_df.to_csv(out_dir / cfg.cleaned_rule_filename, index=False)

    miss = _missingness_table(full_df, key_vars)
    miss.to_csv(out_dir / cfg.missingness_filename, index=False)

    with (out_dir / cfg.report_filename).open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("CLEANING REPORT (PECARN-aligned)\n")
        f.write("=" * 80 + "\n\n")

        f.write("CONFIG\n")
        f.write("-" * 80 + "\n")
        for k, v in asdict(cfg).items():
            f.write(f"{k}: {v}\n")
        f.write("\n")

        f.write("COLUMN STANDARDIZATION\n")
        f.write("-" * 80 + "\n")
        if report.rename_applied:
            for k, v in report.rename_applied.items():
                f.write(f"renamed: {k} -> {v}\n")
        else:
            f.write("No renames applied.\n")
        f.write("\n")

        f.write("BASIC SHAPES\n")
        f.write("-" * 80 + "\n")
        f.write(f"Raw shape:           ({report.n_rows_raw}, {report.n_cols_raw})\n")
        f.write(f"Clean full shape:    ({report.n_rows_clean_full}, {report.n_cols_clean_full})\n")
        f.write(f"Rule cohort shape:   ({report.n_rows_rule_cohort}, {report.n_cols_rule_cohort})\n")
        f.write(f"Rows dropped (missing PosIntFinal): {report.rows_dropped_missing_outcome}\n\n")

        f.write("TARGETED 92->NA CONVERSION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total '92' converted to NA in detail fields: {report.n_detail_92_to_na_converted}\n\n")

        f.write("CT LOGIC\n")
        f.write("-" * 80 + "\n")
        f.write(f"CT contradictions (CTDone==0 & PosCT==1): {report.ct_contradiction_count}\n")
        f.write(f"Rows where PosCT set to NA due to CTDone==0: {report.posct_set_na_count}\n\n")

        f.write("GCS CONSISTENCY\n")
        f.write("-" * 80 + "\n")
        f.write(f"GCS mode: {report.gcs_mode}\n")
        f.write(f"GCS mismatches among complete-cases: {report.gcs_mismatch_count_complete_cases}\n\n")

        f.write("OUTCOME DISTRIBUTION (PosIntFinal)\n")
        f.write("-" * 80 + "\n")
        f.write("Raw (before dropping missing outcome):\n")
        if report.positfinal_counts_raw is None:
            f.write("  PosIntFinal not present.\n")
        else:
            for k, v in report.positfinal_counts_raw.items():
                f.write(f"  {k}: {v}\n")
        f.write("\nClean (after dropping missing outcome):\n")
        if report.positfinal_counts_clean is None:
            f.write("  PosIntFinal not present.\n")
        else:
            for k, v in report.positfinal_counts_clean.items():
                f.write(f"  {k}: {v}\n")
        f.write("\n")

        f.write("REMAINING 92 COUNTS IN KEY VARIABLES (AUDIT)\n")
        f.write("-" * 80 + "\n")
        for c, cnt in report.remaining_92_counts_key_vars.items():
            if cnt > 0:
                f.write(f"{c}: {cnt}\n")
        f.write("\n")

        f.write("KEY VARIABLE MISSINGNESS (top 30 by NA%)\n")
        f.write("-" * 80 + "\n")
        miss_top = miss.sort_values("na_pct", ascending=False).head(30)
        f.write(miss_top.to_string(index=False))
        f.write("\n")


# -----------------------------------------------------------------------------
# CLI usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Adjust these two paths to your environment
    DATA_PATH = "/Users/jayding/Desktop/DUKE/YY/歪歪长官的任务/Stat 214/stat-214-gsi/lab1/data/TBI PUD 10-08-2013.csv"
    LAB1_DIR = Path("/Users/jayding/Desktop/DUKE/YY/歪歪长官的任务/Stat 214/stat-214-local/lab1")

    raw = pd.read_csv(DATA_PATH)

    cfg = CleanConfig(
        drop_missing_outcome=True,
        create_rule_cohort_flag=True,
        gcs_mode="flag",
        fill_age_months_from_years=False,  # your inspection suggests 0 missing AgeInMonth
        ensure_age_two_plus=True,
        set_coded_categoricals=True,
        coerce_binary_int=True,
        write_outputs=True,
    )

    full_df, rule_df, report = clean_data(raw, cfg)

    print("=" * 80)
    print("CLEANED SHAPES")
    print("=" * 80)
    print("Full:", (report.n_rows_clean_full, report.n_cols_clean_full))
    print("Rule cohort:", (report.n_rows_rule_cohort, report.n_cols_rule_cohort))
    print("Dropped missing PosIntFinal:", report.rows_dropped_missing_outcome)
    print("92->NA converted (detail fields):", report.n_detail_92_to_na_converted)
    print("PosCT set NA due to CTDone==0:", report.posct_set_na_count)

    # Quick sanity checks for rule flags

    if cfg.write_outputs:
        write_outputs(LAB1_DIR, full_df, rule_df, report, cfg)
        print(f"\nWrote outputs to: {(LAB1_DIR / cfg.output_dir).resolve()}")