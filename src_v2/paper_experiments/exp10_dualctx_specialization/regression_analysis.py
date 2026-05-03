"""
Analysis 1 — what word features does each ctx head respond to?

For each corpus separately, runs two OLS regressions:
    ctx_FFD  ~ surprisal + log_freq_norm + word_length + position_in_sentence
    ctx_skip ~ surprisal + log_freq_norm + word_length + position_in_sentence

Reports standardized β coefficients, t-statistics, p-values, and R² per
regression. Side-by-side comparison shows which features each head
actually relies on.

Reads:    results/per_word_dualctx.csv
Writes:   results/regression_betas.csv
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = Path(_HERE) / "results"
PER_WORD_CSV = RESULTS_DIR / "per_word_dualctx.csv"
BETAS_CSV = RESULTS_DIR / "regression_betas.csv"


PREDICTORS = ["surprisal", "log_freq_norm", "word_length", "position_in_sentence"]
TARGETS = ["ctx_FFD", "ctx_skip"]


def standardize(arr):
    arr = np.asarray(arr, dtype=float)
    sd = arr.std()
    if sd == 0:
        return arr - arr.mean()
    return (arr - arr.mean()) / sd


def ols_with_t_p(y, X, predictor_names):
    """
    Plain OLS with intercept. Returns:
        list of dicts: per-predictor (β, t, p, std_β)
        plus R² and adjusted R²
    """
    from scipy import stats

    y = np.asarray(y, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    X1 = np.hstack([np.ones((n, 1)), X])
    coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
    yhat = X1 @ coef
    residuals = y - yhat
    ss_res = (residuals ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else r2

    # Standard errors of coefficients
    df = n - k - 1
    sigma2 = ss_res / df if df > 0 else 0.0
    XtX_inv = np.linalg.pinv(X1.T @ X1)
    se = np.sqrt(np.diag(XtX_inv) * sigma2)
    t_stats = np.where(se > 0, coef / se, 0.0)
    p_vals = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stats), df=df))

    rows = []
    for j, name in enumerate(["intercept"] + predictor_names):
        rows.append({
            "predictor": name,
            "beta_unstandardized": float(coef[j]),
            "se": float(se[j]),
            "t": float(t_stats[j]),
            "p": float(p_vals[j]),
        })
    return rows, r2, adj_r2


def main():
    if not PER_WORD_CSV.exists():
        print(f"Run extract_per_word_features.py first; missing: {PER_WORD_CSV}")
        return

    df = pd.read_csv(PER_WORD_CSV)
    print(f"Loaded {len(df):,} rows from {PER_WORD_CSV}")

    out_rows = []

    for corpus in sorted(df["corpus"].unique()):
        sub = df[df["corpus"] == corpus].copy()
        # Standardize predictors so β is comparable across them
        X_std = np.column_stack([
            standardize(sub[p].values) for p in PREDICTORS
        ])

        for target in TARGETS:
            y = sub[target].values
            # Standardize y too — gives standardized β.
            y_std = standardize(y)
            betas, r2, adj_r2 = ols_with_t_p(y_std, X_std, PREDICTORS)

            # Skip the intercept row in the output (always 0 with standardized vars)
            for b in betas:
                if b["predictor"] == "intercept":
                    continue
                out_rows.append({
                    "corpus": corpus,
                    "target": target,
                    "predictor": b["predictor"],
                    "standardized_beta": b["beta_unstandardized"],
                    "se": b["se"],
                    "t": b["t"],
                    "p": b["p"],
                    "r2": r2,
                    "adj_r2": adj_r2,
                    "n_words": len(sub),
                })

    with open(BETAS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)
    print(f"\nWrote {len(out_rows)} rows to {BETAS_CSV}")

    # Pretty-print side-by-side comparison per corpus
    print("\n" + "=" * 80)
    print("STANDARDIZED β COEFFICIENTS (each head ~ word features)")
    print("=" * 80)
    for corpus in sorted(df["corpus"].unique()):
        sub_betas = [r for r in out_rows if r["corpus"] == corpus]
        print(f"\n>> {corpus}")
        print(f"  {'predictor':<25s} {'ctx_FFD β':>12s}  {'ctx_skip β':>12s}  "
              f"{'Δ (FFD - skip)':>15s}")
        for predictor in PREDICTORS:
            ffd_beta = next(r["standardized_beta"] for r in sub_betas
                            if r["predictor"] == predictor and r["target"] == "ctx_FFD")
            skip_beta = next(r["standardized_beta"] for r in sub_betas
                             if r["predictor"] == predictor and r["target"] == "ctx_skip")
            delta = ffd_beta - skip_beta
            print(f"  {predictor:<25s} {ffd_beta:>+12.3f}  {skip_beta:>+12.3f}  {delta:>+15.3f}")
        # R² for each head
        ffd_r2 = next(r["adj_r2"] for r in sub_betas if r["target"] == "ctx_FFD")
        skip_r2 = next(r["adj_r2"] for r in sub_betas if r["target"] == "ctx_skip")
        print(f"  {'(adj R²)':<25s} {ffd_r2:>12.3f}  {skip_r2:>12.3f}")


if __name__ == "__main__":
    main()
