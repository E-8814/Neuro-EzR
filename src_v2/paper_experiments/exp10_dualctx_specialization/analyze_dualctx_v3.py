"""
v3 dual-ctx analyses, run on results/per_word_dualctx_v3.csv:

1. Cross-prediction (H2 head specialization): partial correlation of each
   ctx head's output with each human metric, controlling for word length,
   log-frequency, and TinyLlama surprisal. Under the v3 'next' alignment
   ctx_skip[w] feeds the race that decides word w's own skip, so this
   analysis is mechanistically aligned (unlike v2, where ctx_skip[w]
   influenced word w-1's prediction).

2. Function/content divergence: the two heads disagree along the
   open/closed-class boundary. Reported as (a) mean ctx values by class,
   (b) standardized OLS of (ctx_FFD - ctx_skip) on is_function_word +
   frequency + length + surprisal + position, (c) top-N divergent words
   per side (deduplicated by word type, sentence-initial words flagged).

CPU-only; needs pandas + numpy.

Usage:
    python analyze_dualctx_v3.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = Path(_HERE) / "results"
PER_WORD_CSV = RESULTS_DIR / "per_word_dualctx_v3.csv"
OUT_CROSS = RESULTS_DIR / "cross_prediction_v3.csv"
OUT_REG = RESULTS_DIR / "divergence_regression_v3.csv"
OUT_EXAMPLES = RESULTS_DIR / "divergence_examples_v3.csv"

FUNC_WORDS = set("""a an the this that these those and or but nor so yet for of
to in on at by with from up out off over under again further then once here
there all any both each few more most other some such no not only own same
than too very can will just should now i me my we our you your he him his she
her it its they them their what which who whom is are was were be been being
have has had do does did would could may might must shall as if because while
although though since until unless about into through during before after
above below between against among""".split())


def is_function(word: str) -> bool:
    return str(word).lower().strip(".,;:!?'\"()[]") in FUNC_WORDS


def zscore(x):
    x = np.asarray(x, dtype=float)
    s = x.std()
    return (x - x.mean()) / (s if s > 0 else 1.0)


def residualize(y, controls):
    """OLS-residualize y on controls (adds intercept)."""
    X = np.column_stack([np.ones(len(y))] + controls)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def partial_r(a, b, controls):
    ra = residualize(np.asarray(a, float), controls)
    rb = residualize(np.asarray(b, float), controls)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    if not PER_WORD_CSV.exists():
        print(f"Missing {PER_WORD_CSV}; run extract_per_word_features_v3.py first.")
        sys.exit(2)
    df = pd.read_csv(PER_WORD_CSV)
    df["is_func"] = df["word"].map(is_function).astype(int)
    df["delta_heads"] = df["ctx_FFD"] - df["ctx_skip"]

    cross_rows, reg_rows = [], []

    for corpus in ["geco_test", "provo"]:
        g = df[df.corpus == corpus].copy()
        print(f"\n{'='*72}\n{corpus}  (n={len(g)})\n{'='*72}")

        controls = [np.asarray(g["word_length"], float),
                    np.asarray(g["log_freq"], float),
                    np.asarray(g["surprisal"], float)]

        # ---- 1. Cross-prediction partial correlations ---- #
        print("\nPartial r (controls: length, log_freq, surprisal):")
        print(f"{'':14}{'h_TRT':>8}{'h_FFD':>8}{'h_Gaze':>8}{'h_skip':>8}")
        for head in ["ctx_FFD", "ctx_skip"]:
            vals = {}
            for metric in ["h_TRT", "h_FFD", "h_Gaze", "h_skip"]:
                vals[metric] = partial_r(g[head], g[metric], controls)
                cross_rows.append({"corpus": corpus, "head": head,
                                   "metric": metric, "partial_r": vals[metric]})
            print(f"{head:14}" + "".join(f"{vals[m]:>8.3f}"
                                         for m in ["h_TRT", "h_FFD", "h_Gaze", "h_skip"]))

        # ---- 2. Function/content divergence ---- #
        print("\nMean head outputs by word class:")
        cls = g.groupby("is_func")[["ctx_FFD", "ctx_skip", "delta_heads"]].mean()
        for f_flag, label in [(0, "content"), (1, "function")]:
            if f_flag in cls.index:
                r = cls.loc[f_flag]
                print(f"  {label:9}: ctx_FFD={r.ctx_FFD:+7.2f}  "
                      f"ctx_skip={r.ctx_skip:+7.2f}  Δ={r.delta_heads:+7.2f}")

        y = zscore(g["delta_heads"])
        X = np.column_stack([
            np.ones(len(g)),
            g["is_func"].values,
            zscore(g["log_freq"]),
            zscore(g["word_length"]),
            zscore(g["surprisal"]),
            zscore(g["position_in_sentence"]),
        ])
        names = ["intercept", "is_function_word", "z_log_freq",
                 "z_length", "z_surprisal", "z_position"]
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"\nStd. OLS  zΔ(ctx_FFD−ctx_skip) ~ features   R²={r2:.3f}")
        for n_, b in zip(names, beta):
            print(f"  {n_:18} β = {b:+.3f}")
            reg_rows.append({"corpus": corpus, "term": n_, "beta": float(b),
                             "r2": float(r2)})

    pd.DataFrame(cross_rows).to_csv(OUT_CROSS, index=False)
    pd.DataFrame(reg_rows).to_csv(OUT_REG, index=False)

    # ---- 3. Divergence examples (deduped by type, GECO test) ---- #
    g = df[df.corpus == "geco_test"].copy()
    g["wl"] = g["word"].str.lower().str.strip(".,;:!?'\"()[]")
    typ = (g.groupby("wl")
             .agg(delta=("delta_heads", "mean"), n=("delta_heads", "size"),
                  is_func=("is_func", "max"), length=("word_length", "first"),
                  h_skip=("h_skip", "mean"))
             .reset_index())
    typ = typ[typ.n >= 3]  # types seen at least 3x
    top = typ.nlargest(15, "delta")
    bot = typ.nsmallest(15, "delta")
    ex = pd.concat([top.assign(side="FFD>>skip"), bot.assign(side="skip>>FFD")])
    ex.to_csv(OUT_EXAMPLES, index=False)
    print(f"\nTop divergent word TYPES (≥3 tokens, GECO test):")
    print("  FFD>>skip:", ", ".join(top.wl.head(10)))
    print("  skip>>FFD:", ", ".join(bot.wl.head(10)))
    print(f"\nWrote {OUT_CROSS}\nWrote {OUT_REG}\nWrote {OUT_EXAMPLES}")


if __name__ == "__main__":
    main()
