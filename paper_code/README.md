# Differentiable E-Z Reader — code release

Code accompanying *Differentiable E-Z Reader: A Trainable Cognitive Cascade for Word-Level Eye-Tracking Prediction* (Kahraman, Dalbagno, Cipriani, Meda; Tilburg University).

The compiled paper is `neuro_ezr_paper.pdf`; LaTeX source is in `paper/`.

## What this is

An end-to-end trainable instantiation of an E-Z Reader-style cognitive cascade. A TinyLlama-1.1B backbone feeds two specialized contextual heads (`ctx-head_FFD`, `ctx-head_skip`); these drive a differentiable cascade with named cog scalars (α₁, α₂, δ, ε, M₁, M₂, λ_refix, refix_pivot, τ) that jointly predict FFD, gaze duration, TRT, and skip probability per word.

## Repository layout

```
paper_code/
├── neuro_ezr_paper.pdf              compiled paper
├── paper/                           LaTeX source + plots
│   ├── paper.tex
│   └── plots/
├── DATA.md                          how to obtain GECO + Provo
├── CHECKPOINTS.md                   how to retrain or obtain trained weights
├── requirements.txt                 Python deps
├── src_v2/
│   ├── lm_model/                    cascade model definitions
│   │   ├── model_llama_hybrid_v4c_v2_dualctx.py    paper model
│   │   └── model_llama_hybrid_v4c_v2_surp.py       exp07 (ctx-head replaced by α₃·surprisal)
│   ├── lm_train/                    training scripts (one per model variant)
│   │   ├── train_hybrid_v4c_v2_dualctx_geco.py
│   │   └── train_hybrid_v4c_v2_surp_geco.py
│   └── paper_experiments/           full experimental pipeline (6 experiments)
│       ├── README.md                 per-experiment overview
│       ├── DEPENDENCIES.md           execution-order graph
│       ├── pipeline.sh               end-to-end driver (4 phases: B, C, D, E)
│       ├── config.py                 single source of truth for paths/seeds/hparams
│       ├── utils/                    shared loaders (data, model, metrics, alignment)
│       ├── analysis/                 final-table and final-figure builders
│       ├── exp01_main_comparison/    Table 1 (baselines vs cascade, GECO + Provo)
│       ├── exp03_lesion_study/       lesions → §Lesion (double dissociation)
│       ├── exp06_surprisal_decomp/   §Surprisal decomposition (ΔR², partial r)
│       ├── exp07_ctx_vs_surprisal/   ctx-head vs α₃·TinyLlama-surprisal ablation
│       ├── exp09_per_participant_cog_fits/  Figure 2 (fast vs slow group)
│       └── exp10_dualctx_specialization/ cross-prediction (Table xpred)
└── archive/
    ├── original_ezreader/           classical E-Z Reader simulator + data loaders
    │   ├── data_loader.py            GECO/Provo CSV → SentenceData / AggregatedSentence
    │   ├── geco_loader.py            GECO-specific loading + train/val/test split
    │   ├── ez_reader_engine.py       hand-tuned simulator (reference, not the trainable model)
    │   ├── ez_wrapper.py             simulator wrapper
    │   └── utilities.py
    └── baselines/                   five baselines used in Table 1
        ├── linear_regression.py     log-freq + predictability + length
        ├── lightgbm_baseline.py     engineered features + GPT-2 surprisal
        ├── gpt2_surprisal.py        surprisal-only linear regression
        ├── bert_regression.py
        └── run_ohio_state_on_geco.py CMCL 2021 RoBERTa winner (Oh & Fazeli)
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download GECO + Provo into data/  (see DATA.md)
mkdir data
# ... place files as described in DATA.md ...

# 3. Run a single experiment, or the full pipeline
cd src_v2/paper_experiments
bash pipeline.sh             # end-to-end (multi-day on a single GPU)
# or
bash pipeline.sh phase B     # all training (~20 GPU-hours)
bash pipeline.sh phase C     # post-training analyses
bash pipeline.sh phase D     # aggregation
bash pipeline.sh phase E     # final tables + figures
```

Each experiment is also individually runnable from its own directory; see `src_v2/paper_experiments/README.md` and `DEPENDENCIES.md`.

## Reproducing paper artifacts

| Paper artifact | Produced by |
|---|---|
| Table 1 (main comparison, GECO + Provo) | `exp01_main_comparison/aggregate.py` → `exp01_main_comparison/results/per_seed_metrics_v2.csv` |
| Figure 1 (architecture) | TikZ in `paper/paper.tex` (no script) |
| Figure 2 / Table 6 (per-group cog scalars) | `exp09_per_participant_cog_fits/{fit_per_group.py,analyze_fits.py,plot_per_participant_cog.py}` |
| Lesion figure (`plot_lesion_pretty.pdf`) | `exp03_lesion_study/{run_lesions.py,plot_lesion.py}` |
| Group-comparison figure (`plot_group_comparison.pdf`) | `exp09_per_participant_cog_fits/plot_per_participant_cog.py` |
| Surprisal-decomposition stats (§ctxsurp) | `exp06_surprisal_decomp/compute_surprisal_decomp.py` |
| ctx-head vs surprisal ablation (Table ctxsurp) | `exp07_ctx_vs_surprisal/{aggregate.py,plot_ctx_vs_surp.py}` |
| Cross-prediction (Table xpred) | `exp10_dualctx_specialization/{cross_prediction_analysis.py,regression_analysis.py}` |

Most experiments require trained checkpoints (see `CHECKPOINTS.md`).

## Seeds and reproducibility

- Default seeds: `{1, 2, 3, 42, 100}` (`config.SEEDS`).
- Paper model recipe: `v4c_v2_dualctx` (`config.PAPER_MODEL_RECIPE`).
- All scripts are idempotent — re-running skips completed work.

## Citation

```bibtex
@inproceedings{kahraman2026diffezreader,
  title  = {Differentiable {E}-{Z} Reader: A Trainable Cognitive Cascade for Word-Level Eye-Tracking Prediction},
  author = {Kahraman, Efe and Dalbagno, Arturo and Cipriani, Lorenzo and Meda},
  year   = {2026},
  note   = {Department of Cognitive Science and Artificial Intelligence, Tilburg University}
}
```

## License

Code is released for academic / research use. GECO (Cop et al., 2017) and Provo (Luke & Christianson, 2018) are governed by their original licenses — see `DATA.md`.
