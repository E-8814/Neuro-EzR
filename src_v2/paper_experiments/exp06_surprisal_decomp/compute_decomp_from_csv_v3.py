"""
v3 variance partition (H3), computed from exp10's v3 per-word CSV
(../exp10_dualctx_specialization/results/per_word_dualctx_v3.csv) —
no GPU needed.

Reproduces the paper's exp06 quantities on the v3 model:
    r(L1, surprisal)
    ΔR²(L1 | surprisal + controls)   — L1 beyond surprisal
    ΔR²(surprisal | L1 + controls)   — surprisal beyond L1
with controls = log_freq, word_length and target = h_TRT,
for GECO test and Provo.

Outputs: results/surprisal_decomp_v3.csv (+ printed table).

Usage:
    python compute_decomp_from_csv_v3.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
PER_WORD_CSV = (Path(_HERE) / ".." / "exp10_dualctx_specialization" /
                "results" / "per_word_dualctx_v3.csv").resolve()
OUT_CSV = Path(_HERE) / "results" / "surprisal_decomp_v3.csv"


def ols_r2(y, cols):
    X = np.column_stack([np.ones(len(y))] + cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot


def main():
    if not PER_WORD_CSV.exists():
        print(f"Missing {PER_WORD_CSV}; run exp10's extract_per_word_features_v3.py first.")
        sys.exit(2)
    df = pd.read_csv(PER_WORD_CSV)

    rows = []
    for corpus in ["geco_test", "provo"]:
        g = df[df.corpus == corpus]
        y = np.asarray(g["h_TRT"], float)
        L1 = np.asarray(g["L1"], float)
        surp = np.asarray(g["surprisal"], float)
        ctrl = [np.asarray(g["log_freq"], float),
                np.asarray(g["word_length"], float)]

        r_l1_surp = float(np.corrcoef(L1, surp)[0, 1])
        r2_ctrl = ols_r2(y, ctrl)
        r2_ctrl_surp = ols_r2(y, ctrl + [surp])
        r2_ctrl_l1 = ols_r2(y, ctrl + [L1])
        r2_full = ols_r2(y, ctrl + [surp, L1])

        d_l1_beyond_surp = r2_full - r2_ctrl_surp
        d_surp_beyond_l1 = r2_full - r2_ctrl_l1

        print(f"\n=== {corpus} (n={len(g)}) ===")
        print(f"  r(L1, surprisal)              = {r_l1_surp:+.3f}")
        print(f"  R² controls only              = {r2_ctrl:.4f}")
        print(f"  ΔR² L1 beyond surprisal+ctrl  = {d_l1_beyond_surp:.4f}")
        print(f"  ΔR² surprisal beyond L1+ctrl  = {d_surp_beyond_l1:.4f}")
        ratio = (d_l1_beyond_surp / d_surp_beyond_l1
                 if d_surp_beyond_l1 > 1e-9 else float("inf"))
        print(f"  asymmetry (L1 : surprisal)    = {ratio:.1f} : 1")

        rows.append({
            "corpus": corpus, "n_words": len(g),
            "r_L1_surprisal": r_l1_surp,
            "r2_controls": r2_ctrl,
            "delta_r2_L1_beyond_surprisal": d_l1_beyond_surp,
            "delta_r2_surprisal_beyond_L1": d_surp_beyond_l1,
            "asymmetry_ratio": ratio,
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
