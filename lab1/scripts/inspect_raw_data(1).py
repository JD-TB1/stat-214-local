#!/usr/bin/env python3

"""
inspect_raw_data.py

Purpose
-------
Basic console inspection of the raw PECARN TBI (PUD) dataset for Lab 1.

Report Owner
------------
Selina Yu

Last Modified
-------------
2026-02-20
"""

import pandas as pd
import numpy as np

# Absolute path based on your repo root
DATA_PATH = "/Users/yy/Desktop/UCB/stat214/stat-214/lab1/data/TBI PUD 10-08-2013.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 80)
print("BASIC SHAPE")
print("=" * 80)
print(df.shape)

print("\n\nCOLUMN NAMES")
print("=" * 80)
print(df.columns.tolist())

print("\n\nMISSING VALUE SUMMARY (NA only)")
print("=" * 80)
print(df.isna().sum().sort_values(ascending=False).head(20))

print("\n\nUNIQUE VALUE COUNTS (Top 30 columns)")
print("=" * 80)
for col in df.columns[:30]:
    print(f"\n--- {col} ---")
    print(df[col].value_counts(dropna=False).head(10))

print("\n\nCHECK FOR SENTINEL CODES (90, 91, 92)")
print("=" * 80)
for col in df.columns:
    if df[col].dtype != object:
        sentinel_counts = df[col].isin([90, 91, 92]).sum()
        if sentinel_counts > 0:
            print(f"{col}: {sentinel_counts} sentinel-coded entries")

print("\n\nAGE CONSISTENCY CHECK")
print("=" * 80)
if "AgeInYears" in df.columns and "AgeInMonth" in df.columns:
    age_diff = df["AgeInYears"] - (df["AgeInMonth"] / 12)
    print("Age difference summary:")
    print(age_diff.describe())

print("\n\nGCS CONSISTENCY CHECK")
print("=" * 80)
if {"GCSEye", "GCSVerbal", "GCSMotor", "GCSTotal"}.issubset(df.columns):
    gcs_sum = df["GCSEye"] + df["GCSVerbal"] + df["GCSMotor"]
    mismatch = (gcs_sum != df["GCSTotal"]).sum()
    print("Number of GCS mismatches:", mismatch)

print("\n\nciTBI PREVALENCE")
print("=" * 80)
if "PosIntFinal" in df.columns:
    print(df["PosIntFinal"].value_counts(dropna=False))

print("\n\nCTDone vs PosCT cross-tab")
print("=" * 80)
if {"CTDone", "PosCT"}.issubset(df.columns):
    print(pd.crosstab(df["CTDone"], df["PosCT"], dropna=False))