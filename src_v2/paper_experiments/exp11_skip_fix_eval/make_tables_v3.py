"""
Generate the paper's Tables 1 (Pearson r) and 2 (MAE) as LaTeX from the
exp11 raw JSONs — no hand-transcription.

Conventions (mirroring the published tables):
  - rows: linear, GPT-2, LightGBM, BERT, OSU RoBERTa | EZR (fitted),
    Diff-EZR (no LM), Surp-ablation | Ours
  - seed means where >1 seed; bold = best per column among TRAINED
    non-reference models (reference rows: EZR fitted, Diff-EZR no LM,
    surp ablation are excluded from bolding, as before)
  - skip columns are computed on words 1..L-1 for every model EXCEPT the
    classical EZR row, which is carried over from the v2 all-words run
    and marked with a dagger until re-scored (PAPER_CHANGES.md item T2).

Outputs:
    results/table1_r_v3.tex
    results/table2_mae_v3.tex

Usage:
    python -u make_tables_v3.py
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
RAW = Path(_HERE) / "results" / "raw"
CLASSICAL_JSON = (Path(_HERE) / ".." / "exp01_main_comparison" /
                  "complete_metrics" / "results" / "raw" /
                  "ez_reader_classical_seed1.json").resolve()
OUT1 = Path(_HERE) / "results" / "table1_r_v3.tex"
OUT2 = Path(_HERE) / "results" / "table2_mae_v3.tex"

CLASSICAL_V3 = RAW / "ez_reader_classical_v3_seed1.json"
HAVE_CLASSICAL_V3 = CLASSICAL_V3.exists()

# (display name, json model key, is_reference_row)
ROWS = [
    ("linear",            "linear_regression",        False),
    ("GPT-2",             "gpt2_surprisal",           False),
    ("LightGBM",          "lightgbm",                 False),
    ("BERT",              "bert_regression",          False),
    ("OSU RoBERTa",       "ohio_state_roberta",       False),
    ("EZR (fitted)" + ("" if HAVE_CLASSICAL_V3 else "$^\\dagger$"),
                          "ez_reader_classical",      True),
    ("Diff-EZR (no LM)",  "v4c_v3_dualctx_next_no_ai", True),
    ("Surp-only ablation", "v4c_v3_surp_next",        True),
    ("\\textbf{Ours}",    "v4c_v3_dualctx_next",      False),
]
CORPORA = ["geco_test", "provo"]


def load_blocks():
    acc = defaultdict(lambda: defaultdict(list))  # model -> corpus -> [block]
    for path in sorted(RAW.rglob("*_seed*.json")):
        d = json.loads(path.read_text())
        for corpus, b in d["datasets"].items():
            flat = dict(b)
            if "skip_cmp" in b:  # nested -> flatten cmp variant
                flat.update(b["skip_cmp"])
            acc[d["model"]][corpus].append(flat)
    # classical fallback: v2 all-words run (dagger) only if no v3 re-run yet
    if "ez_reader_classical" not in acc and CLASSICAL_JSON.exists():
        d = json.loads(CLASSICAL_JSON.read_text())
        for corpus, b in d["datasets"].items():
            acc["ez_reader_classical"][corpus].append(dict(b))
    return acc


def mean_of(blocks, key):
    vals = [b[key] for b in blocks if key in b and b[key] is not None]
    return float(np.mean(vals)) if vals else None


def fmt_r(v):
    return "—" if v is None else f"{v:.2f}".replace("0.", ".")


def fmt_ms(v):
    return "—" if v is None else f"{v:.1f}"


def fmt_skip_mae(v):
    return "—" if v is None else f"{v:.2f}".replace("0.", ".")


def build_table(acc, metrics, fmts, higher_better, caption, label, out_path):
    # collect cell values
    cells = {}   # (model_key, corpus, metric) -> value
    for _, key, _ in ROWS:
        for corpus in CORPORA:
            blocks = acc.get(key, {}).get(corpus, [])
            for m in metrics:
                cells[(key, corpus, m)] = mean_of(blocks, m) if blocks else None

    # best per column among non-reference rows
    best = {}
    for corpus in CORPORA:
        for m in metrics:
            vals = [(cells[(key, corpus, m)], key)
                    for _, key, ref in ROWS
                    if not ref and cells[(key, corpus, m)] is not None]
            if not vals:
                continue
            best[(corpus, m)] = (max if higher_better else min)(
                vals, key=lambda t: t[0])[1]

    lines = [
        "\\begin{table}[t]", "\\centering",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{l *{4}{r} @{\\hskip 0.6em} *{4}{r}}",
        "\\toprule",
        "& \\multicolumn{4}{c}{\\textbf{GECO}} & "
        "\\multicolumn{4}{c}{\\textbf{Provo}} \\\\",
        "\\cmidrule(lr){2-5} \\cmidrule(lr){6-9}",
        "Model & TRT & FFD & Gaze & skip & TRT & FFD & Gaze & skip \\\\",
        "\\midrule",
    ]
    prev_ref = False
    for name, key, ref in ROWS:
        if ref and not prev_ref:
            lines.append("\\addlinespace")
        prev_ref = ref
        row = [name]
        for corpus in CORPORA:
            for m, fmt in zip(metrics, fmts):
                v = cells[(key, corpus, m)]
                cell = fmt(v)
                if best.get((corpus, m)) == key and v is not None:
                    cell = f"\\textbf{{{cell}}}"
                row.append(cell)
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


def main():
    acc = load_blocks()
    present = {k: sorted(v.keys()) for k, v in acc.items()}
    print("Models found:", json.dumps(present, indent=1))

    build_table(
        acc,
        metrics=["r_trt", "r_ffd", "r_gaze", "r_skip"],
        fmts=[fmt_r] * 4,
        higher_better=True,
        caption=(
            "Pearson $r$ (higher is better) on GECO test and Provo. Means "
            "over 5 seeds where applicable. Skip is evaluated on non-initial "
            "words (words $2..L$) for all models"
            + ("" if HAVE_CLASSICAL_V3 else
               "; $^\\dagger$EZR (fitted) skip carried over from the "
               "all-words evaluation pending re-run")
            + ". Bold = best per column among trained models (reference rows "
            "excluded)."),
        label="tab:main-r",
        out_path=OUT1,
    )
    build_table(
        acc,
        metrics=["mae_trt", "mae_ffd", "mae_gaze", "mae_skip"],
        fmts=[fmt_ms, fmt_ms, fmt_ms, fmt_skip_mae],
        higher_better=False,
        caption=(
            "MAE (lower is better; milliseconds for TRT/FFD/Gaze, $[0,1]$ "
            "units for skip) on GECO test and Provo. Setup as in "
            "Table~\\ref{tab:main-r}."),
        label="tab:main-mae",
        out_path=OUT2,
    )


if __name__ == "__main__":
    main()
