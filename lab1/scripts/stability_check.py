#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LAB1_DIR = Path(__file__).resolve().parents[1]
INFILE = LAB1_DIR / "output" / "cleaned_rule_cohort.csv"
FIG_DIR = LAB1_DIR / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def wilson_ci(k, n, z=1.96):
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2/(2*n)) / denom
    half = (z/denom) * np.sqrt((phat*(1-phat)/n) + (z**2/(4*n**2)))
    return center - half, center + half


def compute_binned(df, bins):
    df = df.dropna(subset=["AgeInMonth", "PosIntFinal"]).copy()
    df["bin"] = pd.cut(df["AgeInMonth"], bins=bins, right=False)

    rows = []
    for b, sub in df.groupby("bin"):
        n = len(sub)
        k = (sub["PosIntFinal"] == 1).sum()
        if n > 0:
            lo, hi = wilson_ci(k, n)
            mid = (b.left + b.right) / 2
            rows.append((mid, k/n, lo, hi))
    return pd.DataFrame(rows, columns=["mid", "rate", "lo", "hi"])


df = pd.read_csv(INFILE)

# Original: 6-month bins
bins6 = np.arange(0, 216+6, 6)
res6 = compute_binned(df, bins6)

# Perturbed: 12-month bins
bins12 = np.arange(0, 216+12, 12)
res12 = compute_binned(df, bins12)

plt.figure(figsize=(8,5))

plt.plot(res6["mid"], res6["rate"]*1000, label="6-month bins", marker="o")
plt.plot(res12["mid"], res12["rate"]*1000, label="12-month bins", marker="s")

plt.axvline(24, linestyle="--")
plt.xlabel("Age (months)")
plt.ylabel("ciTBI rate (per 1,000)")
plt.title("Stability Check: Effect of Bin Width on ciTBI vs Age")
plt.legend()

plt.tight_layout()
plt.savefig(FIG_DIR / "stability_age_bins.png", dpi=300)
plt.close()

print("Stability figure written to figs/stability_age_bins.png")