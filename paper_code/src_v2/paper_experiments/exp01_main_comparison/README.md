# exp01 — Main comparison vs NLP baselines (Table 1)

## What this produces

Paper Table 1: each row is a model (paper model + 6 NLP baselines + 2
formula baselines), each column is a metric × dataset (GECO test, Provo).
Mean ± std across 5 seeds for the trained models.

## Files

| File | Purpose |
|---|---|
| `train_paper_model_seeds.sh` | trains paper model with 5 seeds (idempotent) |
| `train_baselines_seeds.sh`   | trains all NLP baselines with 5 seeds each |
| `eval_all_models.py`         | evaluates each (model × seed) on GECO + Provo, writes per-run JSON |
| `aggregate.py`               | reads all per-run JSONs → `comparison_results.csv` (long form) |
| `plot_comparison.py`         | reads CSV → bar chart with error bars (PDF) |

## Outputs (in `results/`)

- `per_seed_metrics.csv` — long-form CSV: `(model, seed, dataset, metric, value)`
- `comparison_results.csv` — aggregated: `(model, dataset, metric, mean, std, n_seeds)`
- `plot_comparison.pdf` — bar chart of model × metric × dataset

## Run individually

```bash
# 1. train all required checkpoints (skip if exists)
bash train_paper_model_seeds.sh
bash train_baselines_seeds.sh

# 2. evaluate
python eval_all_models.py

# 3. aggregate
python aggregate.py

# 4. plot
python plot_comparison.py
```

## Prerequisites

- Paper model recipe decided (`PAPER_MODEL_RECIPE` in `config.py`).
- All baseline training scripts in `archive/baselines/` runnable with
  `--seed` argument (verify before bulk submission).
- GECO + Provo data files present (`config.GECO_*`, `config.PROVO_FILE`).

## Compute

- ~5 hrs per paper-model seed × 5 = 25 hrs sequential, 5 hrs parallel.
- Each NLP baseline ~1-2 hrs × 5 seeds × 6 baselines = 30-60 hrs sequential.
- Total: ~50 hrs sequential, ~10 hrs with 5 parallel GPUs.
