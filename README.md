# Differentiable E-Z Reader

A trainable cognitive cascade for word-level eye-tracking prediction.

> **Differentiable E-Z Reader: A Trainable Cognitive Cascade for Word-Level Eye-Tracking Prediction.**
> Department of Intelligent Systems, Tilburg University.

This repo accompanies the paper. It contains the model, the baselines, the full
experimental pipeline, and the scripts that produce every number reported in
the paper.

## What this is

E-Z Reader (Reichle, Rayner & Pollatsek, 2003) decomposes word processing into
named stages — familiarity check L₁, lexical access L₂, motor stages M₁/M₂ —
and predicts eye-movement measures from their latencies. The stages are
interpretable, but their parameters are hand-tuned to summary statistics.
Modern neural predictors of reading time are accurate but expose no stages.

This project closes the gap: we keep E-Z Reader's stage decomposition as a
fixed, differentiable functional form, and we **learn its cognitive parameters
by gradient descent** through a transformer language model. The model jointly
predicts word-level FFD, gaze duration, TRT, and skip rate from text and
word-frequency norms, with a TinyLlama-1.1B (Zhang et al., 2024) backbone
feeding two specialized contextual heads (one for the reading-time path, one
for the skip path) into a differentiable cascade with named cognitive scalars.

## Headline findings

| | claim | evidence in repo |
|---|---|---|
| **H1** | Competitive with strong neural baselines on FFD, Gaze, TRT (Pearson r within 0.03–0.06 of best). Lags on skip — a structural cost of the cascade sharing M₁ and L₁ between paths. | `src_v2/paper_experiments/exp01_main_comparison/` and the augmented metric suite in `exp01_main_comparison/complete_metrics/` |
| **H2** | The two contextual heads carry distinct content (partial-r cross-prediction is on-diagonal); refitting cog scalars on fast vs slow GECO readers shifts δ, λ_refix, ε significantly more than under a permutation null (p = 0.008 against 124 random splits). Motor and skip-decision parameters stay inside the null. | `exp09_per_participant_cog_fits/` (fast/slow refit) + `exp09_per_participant_cog_fits/permutation_null/` (significance) + `exp10_dualctx_specialization/` (cross-prediction) |
| **H3** | Contextual heads carry information beyond raw LM surprisal. L₁ adds ΔR² = 0.027 over surprisal-and-controls while surprisal adds only 0.003 over L₁ (≈ 9× asymmetry). Replacing the ctx head with α₃·surprisal degrades every reading-time r and skip r drops by 0.25 on Provo. | `exp06_surprisal_decomp/` (variance partition) + `exp07_ctx_vs_surprisal/` (architectural ablation) |

## Architecture in one paragraph

For each word *w*, TinyLlama-1.1B produces a hidden state that's projected to
256-d. Two MLPs (`ctx_head_FFD` and `ctx_head_skip` in the code; `ctx-head_RT`
and `ctx-head_skip` in the paper) produce a scalar context correction per
path. The reading-time path computes the familiarity-check time
`L₁ = (α₁ + α₂ · f + c) · ε^((ℓ−1)/2)`, lexical access `L₂ = δ · L₁`, and the
cascade summing latencies into FFD, Gaze, and TRT. The skip path runs a soft
race `P(skip) = σ((M₁ − L₁_next_para)/τ + r)` against motor preparation. All
nine cog scalars (α₁, α₂, δ, ε, M₁, M₂=I, λ_refix, refix_pivot, τ) are learned
by backprop through the cascade alongside the neural pieces.

## Repository layout

```
.
├── archive/
│   ├── baselines/                       Five baseline models for Table 1
│   │   ├── linear_regression.py           log-freq + predictability + length
│   │   ├── gpt2_surprisal.py              + per-word GPT-2 surprisal
│   │   ├── lightgbm_baseline.py           engineered features (CMCL 2021 winner-style)
│   │   ├── bert_regression.py             BERT with per-metric heads
│   │   └── run_ohio_state_on_geco.py      CMCL 2021 Ohio State system
│   └── original_ezreader/               Classical E-Z Reader sim + data loaders (GECO, Provo, SUBTLEX)
│
├── src_v2/
│   ├── lm_model/                        Model definitions (multiple variants)
│   │   ├── model_llama_hybrid_v4c_v2.py          single ctx head (v4c_v2 baseline)
│   │   ├── model_llama_hybrid_v4c_v2_dualctx.py  paper model: TWO ctx heads
│   │   └── model_llama_hybrid_v4c_v2_surp.py     H3 ablation: ctx → α₃·surprisal
│   │
│   ├── lm_train/                        Training scripts (one per model variant)
│   │   ├── train_hybrid_v4c_v2_geco.py
│   │   ├── train_hybrid_v4c_v2_dualctx_geco.py    paper-model training
│   │   └── train_hybrid_v4c_v2_surp_geco.py
│   │
│   └── paper_experiments/               Reproducible experimental pipeline
│       ├── config.py                      single source of truth for paths/seeds/hparams
│       ├── pipeline.sh                    end-to-end driver (5 phases)
│       ├── utils/                         shared loaders (data, model, metrics, alignment)
│       ├── analysis/                      final-table and final-figure builders
│       │
│       ├── exp01_main_comparison/         Table 1 (baselines vs paper model, GECO + Provo)
│       │   └── complete_metrics/          Augmented Table 1: adds mae_skip + classical EZ + Diff-EZR (no LM)
│       │       ├── 04_eval_ez_classical.py         Reichle 2003 simulator on GECO + Provo
│       │       ├── 05_fit_ez_classical_params.py   Nelder-Mead refit of classical EZR scalars
│       │       ├── 06_eval_v4c_v2_no_ai.py         Diff-EZR (no LM): cascade + trained scalars, AI heads zeroed
│       │       └── 07_eval_v4c_v2_classical_params.py   cascade with classical-fitted params, AI heads zeroed
│       │
│       ├── exp03_lesion_study/            §H2 lesions
│       ├── exp06_surprisal_decomp/        §H3 variance partition (ΔR², partial r)
│       ├── exp07_ctx_vs_surprisal/        §H3 ablation: ctx head → α₃·surprisal
│       │
│       ├── exp09_per_participant_cog_fits/  §H2 fast-vs-slow reader refit (Fig. 1 top)
│       │   └── permutation_null/          §H2 permutation test (Fig. 1 bottom, p = 0.008)
│       │
│       └── exp10_dualctx_specialization/  §H2 cross-prediction analysis
│
├── scripts/                             Utility scripts
├── requirements.txt                     Python dependencies
└── run.sh                               Top-level training driver (SLURM-ready)
```

## Data

GECO and Provo are publicly available eye-tracking corpora used under their
original licenses. They are **not redistributed in this repo**. Place the
following files under `data/` (sibling of `src_v2/`):

- `Geco_MonolingualReadingData.csv` — Cop et al. (2017), [https://expsy.ugent.be/downloads/geco/](https://expsy.ugent.be/downloads/geco/)
- `Geco_EnglishMaterial.csv` — same source
- `geco_predictability.pkl` — precomputed per-word cloze probabilities (see paper)
- `Provo_Corpus-Eyetracking_Data.csv` — Luke & Christianson (2018), [https://osf.io/sjefs/](https://osf.io/sjefs/)
- `Provo_Corpus-Predictability_Norms.csv` — same source
- `SUBTLEXus.txt` — SUBTLEX-US frequency norms, Brysbaert & New (2009)

Splits and aggregation conventions are documented in
`archive/original_ezreader/geco_loader.py` and applied automatically by the
loaders.

## Reproducing the paper

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full pipeline end to end:

```bash
cd src_v2/paper_experiments
bash pipeline.sh
```

Individual experiments are runnable from their own directories; each writes
results to a local `results/` folder. Targeted reproductions:

| paper section | script(s) |
|---|---|
| Table 1, Table 2 | `exp01_main_comparison/complete_metrics/run_slurm.sh` (orchestrator: paper model + BERT/Ohio + flat baselines + classical EZR + aggregator) |
| Diff-EZR (no LM) row | `exp01_main_comparison/complete_metrics/06_eval_v4c_v2_no_ai.py` |
| EZR (fitted) row | `exp01_main_comparison/complete_metrics/04_eval_ez_classical.py` (after fit in `05_fit_ez_classical_params.py`) |
| §H2 fast/slow refit (Fig. 1 top) | `exp09_per_participant_cog_fits/fit_per_group.py` |
| §H2 permutation null (Fig. 1 bottom, p = 0.008) | `exp09_per_participant_cog_fits/permutation_null/run_slurm.sh` |
| §H2 cross-prediction matrix | `exp10_dualctx_specialization/cross_prediction_analysis.py` |
| §H3 variance partition | `exp06_surprisal_decomp/compute_surprisal_decomp.py` |
| §H3 architectural ablation | `exp07_ctx_vs_surprisal/aggregate.py` (after `train_surp_seeds.sh`) |

The paper model is trained for 5 epochs across 5 seeds {1, 2, 3, 42, 100};
all reported numbers are seed means with std < 0.03. Single-GPU total
compute: roughly 25 GPU-hours for Phase B (all training) plus ~5 GPU-hours
for Phase C (evaluations).

## Notes for readers of the code

- The paper's **`ctx-head_RT`** and **`ctx-head_skip`** correspond to
  `ctx_head_FFD` and `ctx_head_skip` in
  `model_llama_hybrid_v4c_v2_dualctx.py` (the FFD name is a historical
  artefact; it drives the full reading-time path).
- The paper's **`EZR (fitted)`** baseline = classical Reichle 2003 simulator
  refit to GECO train via Nelder-Mead (see `05_fit_ez_classical_params.py`).
- The paper's **`Diff-EZR (no LM)`** row = the differentiable cascade with
  the LLaMA-derived contextual heads zeroed but the cog scalars kept at
  their trained values; see `06_eval_v4c_v2_no_ai.py`.
- The `complete_metrics/` subdirectory exists because Table 2 (MAE) requires
  metrics not present in the original `exp01` JSONs (specifically `mae_skip`
  on every model, `mae_gaze` on the flat baselines). It re-aggregates all
  rows with the full metric set.

