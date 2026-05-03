# Paper Experiments — Pipeline

Self-contained pipeline that produces every table and figure for the
paper. Each experiment is in its own directory and is **individually
runnable**. All experiments share `utils/` and `config.py`.

## Quick start

```bash
# Run the entire pipeline end-to-end (multi-day on single GPU):
bash pipeline.sh

# Or run a single experiment:
cd exp04_noise_ceiling/
python compute_noise_ceiling.py
```

Each experiment writes to its own `results/` directory in CSV format
(long form — one row per observation). Plot scripts read these CSVs
to produce figures.

## Experiments

| # | Name | Paper output | Compute | Prereqs |
|---|---|---|---|---|
| 1 | `exp01_main_comparison/` | Table 1 — model comparison vs NLP baselines | ~10 hrs GPU (5 seeds × 8 models) | paper-model recipe decided |
| 2 | `exp02_randinit_recovery/` | Figure 2 — parameter recovery from random init | ~5 hrs GPU (5 seeds × ±50% jitter) | paper-model recipe decided |
| 3 | `exp03_lesion_study/` | Table 2 — per-component lesion study | ~1 hr | trained paper model (1 seed) |
| 4 | `exp04_noise_ceiling/` | Reported in §4 — split-half reliability | ~30 min | none (data only) |
| 5 | `exp05_ceiling_curve_provo/` | Figure (TBD) — ceiling curve on Provo | ~1 hr | trained paper model |
| 6 | `exp06_surprisal_decomp/` | Table 3 — r(L1, surprisal), partial r, ΔR² | ~1 hr | trained paper model |
| 7 | `exp07_ctx_vs_surprisal/` | Table 4 — ctx_head vs TinyLlama-surprisal | ~5 hrs GPU (5 seeds) | paper-model recipe decided |
| 8 | `exp08_per_participant_eval/` | Table 5 — 14-reader evaluation | ~1 hr | trained paper model (1 seed) |
| 9 | `exp09_per_participant_cog_fits/` | Figure 5 + Table 6 — per-reader cog params | ~2 hrs | trained paper model (1 seed) |

See `DEPENDENCIES.md` for the full execution order.

## Output structure

```
paper_experiments/
├── exp{NN}_*/results/         # raw experiment outputs (long-form CSVs + per-seed JSONs)
└── results/                   # aggregated paper-ready artifacts
    ├── tables/                # final .tex (booktabs) + .csv
    └── figures/               # final .pdf (paper-formatted)
```

Run `python analysis/make_paper_tables.py` and
`python analysis/make_paper_figures.py` once the experiment results
exist to generate the final paper artifacts.

## Reproducibility

- All seeds set deterministically via `utils/seed_utils.py`.
- Default seeds: `{1, 2, 3, 42, 100}` (config.py).
- Checkpoint paths follow the convention from
  `train_hybrid_v4c_v2_*_geco.py`: `checkpoints/<model_dir>/geco_<model_short>_seed<seed>/best_model.pt`.
- All scripts skip steps if outputs already exist (idempotent).
- Configuration in `config.py` is the single source of truth.

## Decisions baked in

- Paper model recipe: `v4c_v2` (configurable in `config.py`)
- Random-init jitter: ±50%
- Seeds: {1, 2, 3, 42, 100}
- Surprisal source for #7: TinyLlama (same backbone as paper model)
- Per-participant fit LR: 3e-5 (cog_lr / 10)
- Statistical test for #7: paired t-test + bootstrap CI
