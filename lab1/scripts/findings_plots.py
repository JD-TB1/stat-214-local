#!/usr/bin/env python3
"""
findings_plots.py

Creates THREE publication-quality figures for the "Three Findings" section.

Finding 1:
  ciTBI rate by age group (<2 vs >=2) with Wilson 95% CI.

Finding 2:
  CT utilization vs CT yield (PosCT among those scanned), stratified by age group.

Finding 3:
  ciTBI rate by severe mechanism (High_impact_InjSev) and AMS (altered mental status):
  a 2x2 heatmap-like tile plot with rates and sample sizes.

Inputs
------
- lab1/output/cleaned_rule_cohort.csv

Outputs
-------
Writes .png figures to lab1/output/figures/

Run
---
python lab1/code/findings_plots.py
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
    if n == 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt((phat * (1 - phat) / n) + (z**2 / (4 * n**2)))
    return (max(0.0, center - half), min(1.0, center + half))


def load_rule_cohort(lab1_dir: Path) -> pd.DataFrame:
    rule_path = lab1_dir / "output" / "cleaned_rule_cohort.csv"
    if not rule_path.exists():
        raise FileNotFoundError(f"Missing: {rule_path}. Run clean.py first.")
    return pd.read_csv(rule_path)


def finding1_citbi_by_age(df: pd.DataFrame, outdir: Path) -> None:
    needed = {"AgeTwoPlus", "PosIntFinal"}
    if not needed.issubset(df.columns):
        return

    d = df.copy()
    d = d[d["PosIntFinal"].notna()]
    d["age_group"] = np.where(d["AgeTwoPlus"] == 1, "<2 years", ">=2 years")

    rows = []
    for g, sub in d.groupby("age_group"):
        n = int(len(sub))
        k = int((sub["PosIntFinal"] == 1).sum())
        lo, hi = _wilson_ci(k, n)
        rows.append({"age_group": g, "n": n, "k": k, "rate": k / n, "lo": lo, "hi": hi})
    res = pd.DataFrame(rows).sort_values("age_group")

    # Publication choices: rate per 1000 is more readable given low prevalence.
    rate_per_1000 = res["rate"] * 1000
    err_low = (res["rate"] - res["lo"]) * 1000
    err_high = (res["hi"] - res["rate"]) * 1000

    plt.figure(figsize=(7.2, 4.8))
    x = np.arange(len(res))
    plt.bar(x, rate_per_1000)
    plt.errorbar(x, rate_per_1000, yerr=[err_low, err_high], fmt="none", capsize=5)

    plt.xticks(x, res["age_group"])
    plt.ylabel("ciTBI rate (per 1,000)")
    plt.title("Finding 1: ciTBI rate differs by age group (Wilson 95% CI)")
    for i, r in res.iterrows():
        plt.text(
            int(np.where(res.index == i)[0][0]),
            float(rate_per_1000.loc[i]) + 0.15,
            f"{int(r['k'])}/{int(r['n'])}",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(outdir / "finding1_citbi_by_age.png", dpi=300)
    plt.close()


def finding2_ct_utilization_and_yield(df: pd.DataFrame, outdir: Path) -> None:
    needed = {"AgeTwoPlus", "CTDone", "PosCT"}
    if not needed.issubset(df.columns):
        return

    d = df.copy()
    d["age_group"] = np.where(d["AgeTwoPlus"] == 1, "<2 years", ">=2 years")

    # CT utilization: P(CTDone==1)
    util = d.groupby("age_group")["CTDone"].agg(
        n="size", ct_rate=lambda s: np.mean(s == 1)
    )

    # CT yield: P(PosCT==1 | CTDone==1)
    scanned = d[d["CTDone"] == 1].copy()
    # PosCT is NA for CTDone==0 by design; among scanned, PosCT should be 0/1 (plus maybe NA if missing)
    yield_tab = scanned.groupby("age_group")["PosCT"].agg(
        n_scanned="size",
        posct_rate=lambda s: np.mean(s == 1),
    )

    out = util.join(yield_tab, how="left").reset_index()

    plt.figure(figsize=(8.6, 5.0))
    x = np.arange(len(out))
    width = 0.35

    plt.bar(x - width / 2, out["ct_rate"], width=width, label="CT utilization: P(CTDone=1)")
    plt.bar(x + width / 2, out["posct_rate"], width=width, label="CT yield: P(PosCT=1 | CTDone=1)")

    plt.ylim(0, 1)
    plt.xticks(x, out["age_group"])
    plt.ylabel("Proportion")
    plt.title("Finding 2: CT use is common, but CT positivity is low (by age group)")
    plt.legend(loc="upper right")

    for i, r in out.iterrows():
        plt.text(i - width / 2, float(r["ct_rate"]) + 0.02, f"n={int(r['n'])}", ha="center", fontsize=9)
        if pd.notna(r["n_scanned"]):
            plt.text(
                i + width / 2,
                float(r["posct_rate"]) + 0.02,
                f"scanned={int(r['n_scanned'])}",
                ha="center",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(outdir / "finding2_ct_use_and_yield.png", dpi=300)
    plt.close()


def finding3_citbi_by_mechanism_and_ams(df: pd.DataFrame, outdir: Path) -> None:
    needed = {"High_impact_InjSev", "AMS", "PosIntFinal"}
    if not needed.issubset(df.columns):
        return

    d = df.copy()
    d = d[d["PosIntFinal"].notna()].copy()

    # Severe mechanism indicator: High_impact_InjSev==3 per your earlier coding summary.
    d["severe_mech"] = np.where(d["High_impact_InjSev"] == 3, 1, np.where(d["High_impact_InjSev"].isin([1, 2]), 0, np.nan))
    d["ams_bin"] = np.where(d["AMS"] == 1, 1, np.where(d["AMS"] == 0, 0, np.nan))

    # Conservative: drop unknowns for this descriptive finding plot (explicitly documented in caption).
    d = d.dropna(subset=["severe_mech", "ams_bin"])

    # Compute rates for 2x2 cells
    cells = []
    for sm in [0, 1]:
        for ams in [0, 1]:
            sub = d[(d["severe_mech"] == sm) & (d["ams_bin"] == ams)]
            n = int(len(sub))
            k = int((sub["PosIntFinal"] == 1).sum())
            rate = (k / n) if n else np.nan
            cells.append({"severe_mech": sm, "ams": ams, "n": n, "k": k, "rate": rate})

    tab = pd.DataFrame(cells)

    # Tile plot (heatmap-like) without external dependencies
    plt.figure(figsize=(7.2, 4.8))
    ax = plt.gca()

    # Coordinates: x=AMS (0/1), y=SevereMech (0/1)
    for _, r in tab.iterrows():
        x = int(r["ams"])
        y = int(r["severe_mech"])
        # color intensity based on rate
        color = plt.cm.Blues(0.15 + 0.85 * (0 if np.isnan(r["rate"]) else min(1.0, r["rate"] * 50)))  # scale for rare outcome
        rect = plt.Rectangle((x, y), 1, 1, facecolor=color, edgecolor="black")
        ax.add_patch(rect)

        label = "NA" if np.isnan(r["rate"]) else f"{r['rate']*1000:.2f}/1000\n{int(r['k'])}/{int(r['n'])}"
        ax.text(x + 0.5, y + 0.5, label, ha="center", va="center", fontsize=10)

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["AMS=0", "AMS=1"])
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["SevereMech=0", "SevereMech=1"])
    plt.title("Finding 3: ciTBI rate (per 1,000) rises with severe mechanism and AMS")
    plt.tight_layout()
    plt.savefig(outdir / "finding3_citbi_by_mech_and_ams.png", dpi=300)
    plt.close()


def main() -> None:
    lab1_dir = Path(__file__).resolve().parents[1]
    outdir = _ensure_outdir(lab1_dir)
    df = load_rule_cohort(lab1_dir)

    finding1_citbi_by_age(df, outdir)
    finding2_ct_utilization_and_yield(df, outdir)
    finding3_citbi_by_mechanism_and_ams(df, outdir)

    print(f"Wrote findings figures to: {outdir.resolve()}")


if __name__ == "__main__":
    main()