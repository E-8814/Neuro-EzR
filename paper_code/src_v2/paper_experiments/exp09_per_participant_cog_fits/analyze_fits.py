"""
Analyze per-participant cog fits: compute Pearson correlations between
each fitted cognitive parameter and reader-level summaries
(mean reading time, etc.).

Reads:    results/per_participant_cog_fits.csv
Writes:   results/cog_correlations.csv
"""

import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = Path(_HERE) / "results"
FITS_CSV = RESULTS_DIR / "per_participant_cog_fits.csv"
CORR_CSV = RESULTS_DIR / "cog_correlations.csv"


COG_PARAMS = [
    "alpha1_reichle", "alpha2_reichle", "delta",
    "epsilon", "M1", "M2_eq_I",
    "lambda_refix", "refix_pivot", "skip_temperature",
]


def main():
    if not FITS_CSV.exists():
        print(f"Run fit_per_participant.py first; missing: {FITS_CSV}")
        return

    df = pd.read_csv(FITS_CSV)

    # External variables to correlate against
    external_vars = ["mean_RT", "n_train_words", "fit_loss"]

    rows = []
    for param in COG_PARAMS:
        if param not in df.columns:
            continue
        for ext in external_vars:
            if ext not in df.columns:
                continue
            x = df[param].dropna()
            y = df[ext].dropna()
            common = x.index.intersection(y.index)
            if len(common) < 3:
                continue
            xv = df.loc[common, param].values
            yv = df.loc[common, ext].values
            r, p = stats.pearsonr(xv, yv)
            rows.append({
                "param": param,
                "correlated_with": ext,
                "pearson_r": float(r),
                "p_value": float(p),
                "n_readers": len(common),
            })

    # Also: cross-parameter correlations (intra-cog)
    for i, p1 in enumerate(COG_PARAMS):
        for p2 in COG_PARAMS[i + 1:]:
            if p1 not in df.columns or p2 not in df.columns:
                continue
            r, p = stats.pearsonr(df[p1].values, df[p2].values)
            rows.append({
                "param": p1,
                "correlated_with": p2,
                "pearson_r": float(r),
                "p_value": float(p),
                "n_readers": len(df),
            })

    with open(CORR_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["param", "correlated_with", "pearson_r", "p_value", "n_readers"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote {len(rows)} correlations to {CORR_CSV}")

    # Print significant correlations (|r| > 0.5 OR p < 0.05)
    print("\n=== Notable correlations (|r| > 0.4 OR p < 0.1) ===")
    for r in rows:
        if abs(r["pearson_r"]) > 0.4 or r["p_value"] < 0.1:
            print(f"  {r['param']:<22s} ↔ {r['correlated_with']:<18s}: "
                  f"r = {r['pearson_r']:+.3f}, p = {r['p_value']:.4f}")


if __name__ == "__main__":
    main()
