"""
Generate final LaTeX tables for the paper.

Reads the per-experiment CSVs and writes:
    results/tables/table1_main_comparison.tex
    results/tables/table2_lesion.tex
    results/tables/table3_surprisal_decomp.tex
    results/tables/table4_ctx_vs_surp.tex
    results/tables/table5_per_participant_eval.tex
    results/tables/table6_per_participant_cog_params.tex
    results/tables/table7_recovered_params.tex
    results/tables/noise_ceiling.tex

Each table also gets a sibling .csv with the same data for inspection.

Tables use booktabs format. Pandas's `to_latex()` is used as the
backend with hand-tuned column formats.
"""

import os
import sys
from pathlib import Path

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from paper_experiments import config


PAPER_ROOT = Path(_HERE).parent
TABLES_DIR = config.PAPER_FINAL_TABLES
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def write_latex(df: pd.DataFrame, name: str, caption: str = "", label: str = ""):
    """Write df.to_latex with booktabs and a sibling CSV."""
    csv_path = TABLES_DIR / f"{name}.csv"
    tex_path = TABLES_DIR / f"{name}.tex"
    df.to_csv(csv_path, index=False)
    tex = df.to_latex(
        index=False, escape=True, na_rep="--",
        column_format="l" + "c" * (len(df.columns) - 1),
        float_format="%.3f",
    )
    if caption or label:
        tex = (
            "\\begin{table}[t]\n"
            "\\centering\n"
            f"{tex}"
            f"\\caption{{{caption}}}\n" if caption else ""
            f"\\label{{{label}}}\n" if label else ""
            "\\end{table}\n"
        )
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"  Wrote {csv_path}, {tex_path}")


def make_table1_main_comparison():
    src = PAPER_ROOT / "exp01_main_comparison" / "results" / "comparison_results.csv"
    if not src.exists():
        print(f"  [skip] {src} not found")
        return
    df = pd.read_csv(src)
    df["value_str"] = df.apply(
        lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1,
    )
    pivot = df.pivot_table(
        index=["model", "dataset"],
        columns="metric",
        values="value_str",
        aggfunc="first",
    ).reset_index()
    write_latex(
        pivot, "table1_main_comparison",
        caption="Main comparison: paper model vs NLP baselines on GECO test and Provo. "
        "Each cell is mean ± std across 5 seeds.",
        label="tab:main",
    )


def make_table2_lesion():
    src = PAPER_ROOT / "exp03_lesion_study" / "results" / "lesion_results.csv"
    if not src.exists():
        print(f"  [skip] {src} not found")
        return
    df = pd.read_csv(src)
    df = df[df["dataset"] == "geco_test"]
    df = df[df["metric"].isin(["r_trt", "r_ffd", "r_gaze", "r_skip"])]
    pivot = df.pivot_table(
        index="lesion", columns="metric", values="value",
    ).reset_index()
    write_latex(pivot, "table2_lesion",
                caption="Lesion study (GECO test).",
                label="tab:lesion")


def make_table3_surprisal_decomp():
    src = PAPER_ROOT / "exp06_surprisal_decomp" / "results" / "surprisal_decomp_results.csv"
    if not src.exists():
        print(f"  [skip] {src} not found")
        return
    df = pd.read_csv(src)
    write_latex(df, "table3_surprisal_decomp",
                caption="Surprisal decomposition.",
                label="tab:surprisal")


def make_table4_ctx_vs_surp():
    src = PAPER_ROOT / "exp07_ctx_vs_surprisal" / "results" / "ctx_vs_surp_summary.csv"
    if not src.exists():
        print(f"  [skip] {src} not found")
        return
    df = pd.read_csv(src)
    write_latex(df, "table4_ctx_vs_surp",
                caption="ctx_head vs TinyLlama-surprisal head-to-head.",
                label="tab:ctx-vs-surp")


def make_table5_per_participant_eval():
    src = PAPER_ROOT / "exp08_per_participant_eval" / "results" / "per_participant_eval.csv"
    if not src.exists():
        print(f"  [skip] {src} not found")
        return
    df = pd.read_csv(src)
    write_latex(df, "table5_per_participant_eval",
                caption="Per-participant evaluation (14 GECO readers).",
                label="tab:per-participant")


def make_table6_per_participant_cog():
    src = PAPER_ROOT / "exp09_per_participant_cog_fits" / "results" / "per_participant_cog_fits.csv"
    if not src.exists():
        print(f"  [skip] {src} not found")
        return
    df = pd.read_csv(src)
    write_latex(df, "table6_per_participant_cog_params",
                caption="Per-participant fitted cognitive parameters.",
                label="tab:per-participant-cog")


def make_table7_recovered_params():
    src = PAPER_ROOT / "exp02_randinit_recovery" / "results" / "recovery_summary.csv"
    if not src.exists():
        print(f"  [skip] {src} not found")
        return
    df = pd.read_csv(src)
    write_latex(df, "table7_recovered_params",
                caption="Random-init parameter recovery summary.",
                label="tab:recovery")


def make_noise_ceiling_table():
    src = PAPER_ROOT / "exp04_noise_ceiling" / "results" / "noise_ceiling_results.csv"
    if not src.exists():
        print(f"  [skip] {src} not found")
        return
    df = pd.read_csv(src)
    write_latex(df, "noise_ceiling",
                caption="Split-half reliability (noise ceiling) of GECO.",
                label="tab:ceiling")


def main():
    print("Generating LaTeX tables...")
    make_table1_main_comparison()
    make_table2_lesion()
    make_table3_surprisal_decomp()
    make_table4_ctx_vs_surp()
    make_table5_per_participant_eval()
    make_table6_per_participant_cog()
    make_table7_recovered_params()
    make_noise_ceiling_table()
    print(f"\nAll tables in: {TABLES_DIR}")


if __name__ == "__main__":
    main()
