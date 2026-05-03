# Pipeline Dependency Graph

```
                  [v4c_v2_wide_prior result]
                         ↓
                [Decide PAPER_MODEL_RECIPE in config.py]
                         ↓
      ┌──────────────────┼──────────────────┐
      ↓                  ↓                  ↓
  exp01a               exp01b           exp02
  Train baselines      Train paper      Train randinit
  (5 seeds × 6)        model (5 seeds)  (5 seeds, ±50% jitter)
      │                  │                  │
      │                  ↓                  ↓
      │              [paper_model_seed42    [randinit checkpoints]
      │               checkpoint ready]            │
      │                  │                         │
      │                  ├→ exp03 lesion           │
      │                  ├→ exp06 surprisal        │
      │                  ├→ exp08 per_part eval    │
      │                  ├→ exp09 per_part fits    │
      │                  └→ exp05 ceiling curve    │
      │                                            │
      ↓                                            ↓
  exp01c aggregate ←──────────────────────────  exp02 aggregate
      │                                            │
      └─────────────────┬──────────────────────────┘
                        │
              exp07 ctx_vs_surprisal
              (depends on paper-model checkpoints from exp01b)


  [independent, run any time]
  exp04 noise_ceiling   (no model required)
```

## Phase ordering

**Phase A** (data-only, no model needed):
- exp04 noise_ceiling

**Phase B** (training, can be parallelized across GPUs):
- exp01a baselines (5 seeds × 6 models = 30 runs)
- exp01b paper model (5 seeds)
- exp02 randinit (5 seeds)
- exp07 surp variant (5 seeds, after exp01b checkpoints exist)

**Phase C** (model evaluations, after Phase B):
- exp03 lesion (uses seed=42 paper model)
- exp05 ceiling curve provo (paper model)
- exp06 surprisal decomposition (paper model)
- exp08 per-participant eval (seed=42 paper model)
- exp09 per-participant cog fits (seed=42 paper model)

**Phase D** (aggregation):
- Each experiment's `aggregate.py` reads its own raw outputs into a long-form CSV.

**Phase E** (final paper artifacts):
- `analysis/make_paper_tables.py` — all CSVs → LaTeX tables.
- `analysis/make_paper_figures.py` — all CSVs → PDFs.

## Idempotency

Every script checks for existing outputs (checkpoint files or CSVs) before
running. Re-running `pipeline.sh` after a partial completion will skip
already-finished steps. To force re-run, delete the relevant output files.

## Compute estimates (single RTX-class GPU, sequential)

| Phase | Time |
|---|---|
| A | 30 min |
| B | ~25 hrs (paper:5h, randinit:5h, surp:5h, baselines:10h) |
| C | ~5 hrs |
| D | <1 hr |
| E | <30 min |

With 4 parallel GPUs, Phase B drops to ~6 hrs.
