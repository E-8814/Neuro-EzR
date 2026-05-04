#!/bin/bash
#
# Full paper-experiment pipeline.
# Each phase is idempotent: skips work if outputs already exist.
#
# Usage:
#   bash pipeline.sh             # run all phases
#   bash pipeline.sh phase B     # run only phase B
#   bash pipeline.sh phase C     # ... etc
#
# Designed to be re-runnable: re-launching after a crash continues
# from where the last successful step finished.

set -euo pipefail

cd "$(dirname "$0")"
PIPELINE_ROOT="$(pwd)"
REPO_ROOT="$(cd ../.. && pwd)"

PHASES_TO_RUN="${2:-all}"
echo ">> Pipeline root: $PIPELINE_ROOT"
echo ">> Repo root:     $REPO_ROOT"
echo ">> Running phase: $PHASES_TO_RUN"
echo ""

run_phase_B() {
    echo "================================================================="
    echo "PHASE B: training (can be parallelized across GPUs)"
    echo "================================================================="

    # Preflight 1: baselines look up `archive/baselines/../data/` for SUBTLEXus,
    # GECO, and Provo CSVs, but the data lives at `<repo>/data/`. Symlink it
    # if not already there.
    if [ ! -e "$REPO_ROOT/archive/data" ]; then
        echo ">> [preflight] linking $REPO_ROOT/archive/data -> ../data"
        ln -s ../data "$REPO_ROOT/archive/data"
    fi

    # Preflight 2: train_baselines_seeds.sh skips a seed if the checkpoint dir
    # is non-empty. Failed runs leave dirs that block retries — clean those up.
    echo ">> [preflight] removing empty baseline checkpoint dirs"
    find "$REPO_ROOT"/archive/baselines/checkpoints_*/seed* -maxdepth 0 \
        -type d -empty -print -delete 2>/dev/null || true

    echo ">> [exp01a] training NLP baselines (Table 1)"
    bash "$PIPELINE_ROOT/exp01_main_comparison/train_baselines_seeds_v3.sh"

    echo ">> [exp01b] training paper model (5 seeds)"
    bash "$PIPELINE_ROOT/exp01_main_comparison/train_paper_model_seeds.sh"

    echo ">> [exp07] precomputing TinyLlama surprisals (one-shot)"
    cd "$PIPELINE_ROOT/exp07_ctx_vs_surprisal"
    python precompute_surprisal.py
    cd "$PIPELINE_ROOT"

    echo ">> [exp07] training v4c_v2_surp ablation (5 seeds)"
    bash "$PIPELINE_ROOT/exp07_ctx_vs_surprisal/train_surp_seeds.sh"
    echo ""
}

run_phase_C() {
    echo "================================================================="
    echo "PHASE C: model evaluations (require trained checkpoints)"
    echo "================================================================="

    echo ">> [exp03] lesion study"
    cd "$PIPELINE_ROOT/exp03_lesion_study"
    python run_lesions.py
    cd "$PIPELINE_ROOT"

    echo ">> [exp06] surprisal decomposition"
    cd "$PIPELINE_ROOT/exp06_surprisal_decomp"
    python compute_surprisal_decomp.py
    cd "$PIPELINE_ROOT"

    echo ">> [exp09] per-group cog parameter fits"
    cd "$PIPELINE_ROOT/exp09_per_participant_cog_fits"
    python fit_per_participant.py
    python fit_per_group.py
    python analyze_fits.py
    cd "$PIPELINE_ROOT"

    echo ">> [exp10] dualctx specialization analyses"
    cd "$PIPELINE_ROOT/exp10_dualctx_specialization"
    python extract_per_word_features.py
    python regression_analysis.py
    python cross_prediction_analysis.py
    python divergence_examples.py
    python plot_scatter.py
    cd "$PIPELINE_ROOT"
    echo ""
}

run_phase_D() {
    echo "================================================================="
    echo "PHASE D: aggregation per experiment (long-form CSVs)"
    echo "================================================================="

    cd "$PIPELINE_ROOT/exp01_main_comparison" && python aggregate.py && cd "$PIPELINE_ROOT"
    cd "$PIPELINE_ROOT/exp07_ctx_vs_surprisal"  && python aggregate.py && cd "$PIPELINE_ROOT"
    echo ""
}

run_phase_E() {
    echo "================================================================="
    echo "PHASE E: final paper artifacts (tables + figures)"
    echo "================================================================="

    # Preflight: delete unparseable (header-less) CSVs so the skip path triggers.
    echo ">> [preflight] removing unparseable (header-less) CSVs in exp*/results/"
    find "$PIPELINE_ROOT"/exp*/results -maxdepth 1 -name "*.csv" -size -5c -print -delete 2>/dev/null || true

    # Render per-experiment PDFs. Each plot script self-skips if its CSV is missing.
    echo ">> [render] per-experiment plot scripts"
    for plot in \
        "$PIPELINE_ROOT/exp01_main_comparison/plot_comparison.py" \
        "$PIPELINE_ROOT/exp03_lesion_study/plot_lesion.py" \
        "$PIPELINE_ROOT/exp07_ctx_vs_surprisal/plot_ctx_vs_surp.py" \
        "$PIPELINE_ROOT/exp09_per_participant_cog_fits/plot_per_participant_cog.py" \
    ; do
        if [ -f "$plot" ]; then
            echo "   > $(basename "$(dirname "$plot")")/$(basename "$plot")"
            ( cd "$(dirname "$plot")" && python "$(basename "$plot")" ) || \
                echo "     [warn] plot failed; continuing"
        fi
    done

    cd "$PIPELINE_ROOT/analysis"
    PYTHONPATH="$REPO_ROOT/src_v2${PYTHONPATH:+:$PYTHONPATH}" python make_paper_tables.py
    PYTHONPATH="$REPO_ROOT/src_v2${PYTHONPATH:+:$PYTHONPATH}" python make_paper_figures.py
    cd "$PIPELINE_ROOT"
    echo ""
}

case "$PHASES_TO_RUN" in
    all)
        run_phase_B
        run_phase_C
        run_phase_D
        run_phase_E
        ;;
    B) run_phase_B ;;
    C) run_phase_C ;;
    D) run_phase_D ;;
    E) run_phase_E ;;
    *)
        echo "Unknown phase: $PHASES_TO_RUN"
        echo "Usage: bash pipeline.sh [phase B|C|D|E]"
        exit 1
        ;;
esac

echo "================================================================="
echo "Pipeline complete. See:"
echo "  $PIPELINE_ROOT/exp*/results/   for per-experiment raw outputs"
echo "  $PIPELINE_ROOT/results/        for paper-ready tables and figures"
echo "================================================================="
