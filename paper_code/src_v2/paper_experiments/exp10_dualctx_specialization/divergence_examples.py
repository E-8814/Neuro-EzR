"""
Analysis 3 — qualitative inspection of divergent words.

Finds words where ctx_FFD and ctx_skip output very different values:
  - top-N where (ctx_FFD − ctx_skip) is most positive
        (FFD says "slow this word"; skip says "this word is easy")
  - top-N where (ctx_skip − ctx_FFD) is most positive
        (skip says "slow parafoveal preview"; FFD says "easy when fixated")

Shows the words alongside their features (frequency, length, surprisal)
and human metrics (h_TRT, h_skip) for inspection.

Reads:    results/per_word_dualctx.csv
Writes:   results/divergence_examples.csv
"""

import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = Path(_HERE) / "results"
PER_WORD_CSV = RESULTS_DIR / "per_word_dualctx.csv"
EXAMPLES_CSV = RESULTS_DIR / "divergence_examples.csv"


TOP_N = 30


COLUMNS_TO_SHOW = [
    "corpus", "sentence_idx", "word_position", "word",
    "ctx_FFD", "ctx_skip", "ctx_FFD_minus_skip",
    "log_freq", "word_length", "surprisal",
    "h_TRT", "h_FFD", "h_skip",
    "pred_TRT", "pred_FFD", "pred_skip",
]


def main():
    if not PER_WORD_CSV.exists():
        print(f"Run extract_per_word_features.py first; missing: {PER_WORD_CSV}")
        return

    df = pd.read_csv(PER_WORD_CSV)
    df["ctx_FFD_minus_skip"] = df["ctx_FFD"] - df["ctx_skip"]

    rows = []
    for corpus in sorted(df["corpus"].unique()):
        sub = df[df["corpus"] == corpus].copy()

        # Top-N where ctx_FFD > ctx_skip (FFD-leaning)
        top_ffd = sub.nlargest(TOP_N, "ctx_FFD_minus_skip")
        for _, r in top_ffd.iterrows():
            row = {col: r[col] for col in COLUMNS_TO_SHOW if col in r}
            row["bucket"] = "ctx_FFD>>ctx_skip (FFD says slower)"
            rows.append(row)

        # Top-N where ctx_skip > ctx_FFD (skip-leaning)
        top_skip = sub.nsmallest(TOP_N, "ctx_FFD_minus_skip")
        for _, r in top_skip.iterrows():
            row = {col: r[col] for col in COLUMNS_TO_SHOW if col in r}
            row["bucket"] = "ctx_skip>>ctx_FFD (skip says slower)"
            rows.append(row)

    # Write CSV
    fieldnames = ["bucket"] + COLUMNS_TO_SHOW
    with open(EXAMPLES_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote {len(rows)} rows to {EXAMPLES_CSV}")

    # Pretty-print summary
    for corpus in sorted(df["corpus"].unique()):
        print(f"\n{'=' * 90}")
        print(f"  {corpus.upper()}")
        print(f"{'=' * 90}")

        for direction in ["FFD>>skip", "skip>>FFD"]:
            print(f"\n  TOP {TOP_N} where ctx_{direction}:")
            print(f"  {'word':<20s} {'ctx_FFD':>8s} {'ctx_skip':>9s} {'Δ':>7s} "
                  f"{'len':>4s} {'logf':>6s} {'surp':>6s} {'h_TRT':>6s} {'h_skip':>7s}")
            print(f"  {'-' * 90}")

            sub = df[df["corpus"] == corpus]
            if direction == "FFD>>skip":
                examples = sub.nlargest(TOP_N, "ctx_FFD_minus_skip")
            else:
                examples = sub.nsmallest(TOP_N, "ctx_FFD_minus_skip")

            for _, r in examples.iterrows():
                print(
                    f"  {str(r['word'])[:20]:<20s} "
                    f"{r['ctx_FFD']:>+8.2f} {r['ctx_skip']:>+9.2f} "
                    f"{r['ctx_FFD_minus_skip']:>+7.2f} "
                    f"{int(r['word_length']):>4d} {r['log_freq']:>6.2f} "
                    f"{r['surprisal']:>6.2f} {r['h_TRT']:>6.0f} "
                    f"{r['h_skip']:>7.3f}"
                )


if __name__ == "__main__":
    main()
