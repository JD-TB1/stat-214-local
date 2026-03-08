import sys
from pathlib import Path
import importlib.util
import pandas as pd

# --- locate lab1 root ---
ROOT = Path(__file__).resolve().parents[1]
print(f"Using ROOT = {ROOT}")

# --- load clean.py explicitly (avoid stdlib 'code' name collision) ---
clean_path = ROOT / "code" / "clean.py"
spec = importlib.util.spec_from_file_location("lab1_clean", clean_path)
lab1_clean = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(lab1_clean)

clean_data = lab1_clean.clean_data

# --- load data ---
DATA_PATH = r"/Users/jayding/Desktop/DUKE/YY/歪歪长官的任务/Stat 214/stat-214-gsi/lab1/data/TBI PUD 10-08-2013.csv"
df_raw = pd.read_csv(DATA_PATH, encoding="latin1")

# --- run ---
df_clean, report = clean_data(df_raw, mode="pud", return_report=True)
df_rule, report_rule = clean_data(df_raw, mode="kuppermann_rule_sample", return_report=True)

print(report)
print(report_rule)

# sanity checks
if "very_low_risk" in df_rule.columns:
    print(df_rule["very_low_risk"].value_counts(dropna=False))

if {"PosIntFinal", "very_low_risk"}.issubset(df_rule.columns):
    print(pd.crosstab(df_rule["very_low_risk"], df_rule["PosIntFinal"], dropna=False))