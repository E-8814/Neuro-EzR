# Paper Experiments — Pipeline

Self-contained pipeline that produces every table and figure for the
paper. Each experiment is in its own directory and is **individually
runnable**. All experiments share `utils/` and `config.py`.

## Quick start

```bash
# Run the entire pipeline end-to-end (multi-day on single GPU):
bash pipeline.sh

# Or run a single experiment:
cd exp03_lesion_study/
python run_lesions.py
```

Each experiment writes to its own `results/` directory in CSV format
(long form — one row per observation). Plot scripts read these CSVs
to produce figures.

## Experiments

| # | Name | Paper output | Compute | Prereqs |
|---|---|---|---|---|
| 1 | `exp01_main_comparison/` | Table 1 — paper model vs 5 NLP baselines on GECO + Provo | ~10 hrs GPU (5 seeds × 6 models) | — |
| 3 | `exp03_lesion_study/` | Lesion figure — per-component lesions, double dissociation | ~1 hr | trained paper model (1 seed) |
| 6 | `exp06_surprisal_decomp/` | r(L1, surprisal), partial r, ΔR² | ~1 hr | trained paper model |
| 7 | `exp07_ctx_vs_surprisal/` | ctx-head vs α₃·TinyLlama-surprisal ablation | ~5 hrs GPU (5 seeds) | — |
| 9 | `exp09_per_participant_cog_fits/` | Figure 2 — per-group cog parameter fits | ~2 hrs | trained paper model (1 seed) |
| 10 | `exp10_dualctx_specialization/` | Cross-prediction matrix (FFD-head vs skip-head) | ~30 min | trained paper model |

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

- Paper model recipe: `v4c_v2_dualctx`
- Seeds: {1, 2, 3, 42, 100}
- Surprisal source for exp07: TinyLlama (same backbone as paper model)
- Per-group fit LR: 3e-5 (cog_lr / 10)
