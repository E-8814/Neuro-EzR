# exp10 — Dual-ctx specialization analysis

## What this produces

Diagnostics that show what each ctx head (`ctx_head_FFD`, `ctx_head_skip`)
in the dualctx model has learned, and how their behaviors differ.

The goal is a paper figure/table showing that the architectural split
corresponds to a meaningful cognitive distinction — not just doubled
capacity.

## Files

| File | Purpose |
|---|---|
| `extract_per_word_features.py` | Walks GECO test + Provo. For each word: dumps `(ctx_FFD, ctx_skip, L1, log_freq, word_length, surprisal, h_TRT, h_FFD, h_skip)` to a CSV. Source-of-truth for downstream analyses. |
| `regression_analysis.py`       | Analysis 1 — two linear regressions: `ctx_FFD ~ word_features` and `ctx_skip ~ word_features`. Compares β coefficients to expose what each head responds to. |
| `cross_prediction_analysis.py` | Analysis 2 — correlation matrix between each ctx head and human metrics (h_TRT, h_FFD, h_skip). Tests whether ctx_FFD specializes for fixation duration and ctx_skip for skip rate. |
| `divergence_examples.py`       | Analysis 3 — finds top-N words where `ctx_FFD − ctx_skip` is most positive or most negative; produces a table for qualitative inspection. |
| `plot_scatter.py`              | Analysis 5 — scatter ctx_FFD vs ctx_skip per word, colored by log_freq, sized by word length, with the y=x diagonal. Visualizes how much they decorrelate. |

## Outputs (in `results/`)

- `per_word_dualctx.csv` — long form, one row per word, all features
- `regression_betas.csv` — β coefficients with t-stats and p-values
- `cross_prediction_matrix.csv` — pairwise correlations
- `divergence_examples.csv` — qualitative top-N table
- `plot_scatter.pdf`

## Run individually

```bash
# 1. Extract per-word features (slowest step, ~5 min)
python extract_per_word_features.py

# Steps 2-5 read the per-word CSV; each runs in <1 minute
python regression_analysis.py
python cross_prediction_analysis.py
python divergence_examples.py
python plot_scatter.py
```

## Prerequisites

- Trained dualctx model checkpoint: `checkpoints/hybrid_v4c_v2_dualctx/geco_TinyLlama_TinyLlama-1.1B-Chat-v1.0_seed42/best_model.pt`
- TinyLlama causal LM available locally (for surprisal).

## Expected findings

If the dualctx specialization is real:

1. **Regression β profiles differ**: ctx_FFD weights surprisal/log_freq more strongly (drives reading time prediction); ctx_skip weights word frequency and word_length more strongly (modulates skip-affordability).
2. **Cross-prediction asymmetry**: r(ctx_FFD, h_TRT) > r(ctx_skip, h_TRT); r(ctx_skip, h_skip) > r(ctx_FFD, h_skip).
3. **Divergence cases interpretable**: words where heads disagree are linguistically meaningful (e.g., function words vs content words).
4. **Scatter shows off-diagonal spread**: not collapsed to identity.

If these all hold, the paper has clean Section 6 evidence that dual-ctx is specialization, not redundancy.
