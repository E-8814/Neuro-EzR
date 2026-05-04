"""
Analysis 2 — cross-prediction matrix.

Tests specialization directly: does ctx_FFD predict reading time better
than skip rate? Does ctx_skip predict skip rate better than reading time?

For each corpus, computes:
    Pearson r between each model intermediate output and each human target.
    Plus partial correlations controlling for word features (log_freq,
    word_length, surprisal).

Reads:    results/per_word_dualctx.csv
Writes:   results/cross_prediction_matrix.csv
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from paper_experiments.utils.eval_metrics import corr, partial_corr

RESULTS_DIR = Path(_HERE) / "results"
PER_WORD_CSV = RESULTS_DIR / "per_word_dualctx.csv"
CROSS_CSV = RESULTS_DIR / "cross_prediction_matrix.csv"


# What model intermediates to score
PREDICTORS = [
    "ctx_FFD", "ctx_skip",
    "base_L1_FFD", "base_L1_skip",
    "L1", "pred_TRT", "pred_FFD", "pred_skip",
]

# What human targets to score against
TARGETS = ["h_TRT", "h_FFD", "h_Gaze", "h_skip"]

# Controls for partial correlation
CONTROLS = ["log_freq_norm", "word_length", "surprisal"]


def main():
    if not PER_WORD_CSV.exists():
        print(f"Run extract_per_word_features.py first; missing: {PER_WORD_CSV}")
        return

    df = pd.read_csv(PER_WORD_CSV)
    print(f"Loaded {len(df):,} rows from {PER_WORD_CSV}")

    out_rows = []

    for corpus in sorted(df["corpus"].unique()):
        sub = df[df["corpus"] == corpus]

        for predictor in PREDICTORS:
            if predictor not in sub.columns:
                continue
            x = sub[predictor].values
            for target in TARGETS:
                if target not in sub.columns:
                    continue
                y = sub[target].values

                # Pearson r
                r = corr(x, y)

                # Partial r controlling for word features
                # (only if predictor is not itself one of the controls)
                if predictor not in CONTROLS:
                    pr = partial_corr(
                        x, y,
                        controls=[sub[c].values for c in CONTROLS],
                    )
                else:
                    pr = float("nan")

                out_rows.append({
                    "corpus": corpus,
                    "predictor": predictor,
                    "target": target,
                    "pearson_r": r,
                    "partial_r": pr,
                    "n_words": len(sub),
                })

    with open(CROSS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)
    print(f"\nWrote {len(out_rows)} rows to {CROSS_CSV}")

    # Pretty-print specialization-key sub-matrix
    print("\n" + "=" * 80)
    print("SPECIALIZATION CHECK: Pearson r(predictor, target)")
    print("=" * 80)
    for corpus in sorted(df["corpus"].unique()):
        print(f"\n>> {corpus}")
        print(f"  {'predictor':<14s} | {'h_TRT':>8s} {'h_FFD':>8s} {'h_Gaze':>8s} {'h_skip':>8s}")
        print(f"  {'-' * 14} + {'-' * 36}")
        for predictor in ["ctx_FFD", "ctx_skip", "L1", "pred_TRT", "pred_skip"]:
            row = ""
            for target in TARGETS:
                r_val = next(
                    (r["pearson_r"] for r in out_rows
                     if r["corpus"] == corpus
                     and r["predictor"] == predictor
                     and r["target"] == target),
                    None,
                )
                if r_val is None:
                    row += f" {'--':>8s}"
                else:
                    row += f" {r_val:>+8.3f}"
            print(f"  {predictor:<14s} |{row}")

    # Highlight the specialization signal
    print("\n" + "=" * 80)
    print("SPECIALIZATION DIFFERENCE: r(ctx_FFD, target) − r(ctx_skip, target)")
    print("(Positive = ctx_FFD specializes; Negative = ctx_skip specializes)")
    print("=" * 80)
    for corpus in sorted(df["corpus"].unique()):
        print(f"\n>> {corpus}")
        for target in TARGETS:
            r_ffd = next(
                (r["pearson_r"] for r in out_rows
                 if r["corpus"] == corpus
                 and r["predictor"] == "ctx_FFD"
                 and r["target"] == target),
                0.0,
            )
            r_skip = next(
                (r["pearson_r"] for r in out_rows
                 if r["corpus"] == corpus
                 and r["predictor"] == "ctx_skip"
                 and r["target"] == target),
                0.0,
            )
            diff = r_ffd - r_skip
            note = ""
            if target in ("h_TRT", "h_FFD", "h_Gaze") and diff > 0.02:
                note = " ← FFD-head specializes ✓"
            elif target == "h_skip" and diff < -0.02:
                note = " ← skip-head specializes ✓"
            print(f"  {target:<10s}: r(ctx_FFD)={r_ffd:+.3f} − r(ctx_skip)={r_skip:+.3f} = {diff:+.3f}{note}")


if __name__ == "__main__":
    main()
