# Pipeline Dependency Graph

```
      ┌──────────────────┬──────────────────┐
      ↓                  ↓                  ↓
  exp01a               exp01b           exp07
  Train baselines      Train paper      Precompute surprisal +
  (5 seeds × 5)        model (5 seeds)  train surp ablation (5 seeds)
      │                  │                  │
      │                  ├→ exp03 lesion           
      │                  ├→ exp06 surprisal decomp 
      │                  ├→ exp09 per-group fits   
      │                  └→ exp10 dualctx specialization
      ↓                                            ↓
  exp01 aggregate ←──────────────────────────  exp07 aggregate
                        │
                        ↓
              analysis/make_paper_tables.py
              analysis/make_paper_figures.py
```

## Phase ordering

**Phase B** (training, can be parallelized across GPUs):
- exp01a baselines (5 baselines: 3 single-run + 2 × 5 seeds)
- exp01b paper model (5 seeds)
- exp07 surp variant (5 seeds, after exp07 surprisal precompute)

**Phase C** (model evaluations, after Phase B):
- exp03 lesion (uses seed=42 paper model)
- exp06 surprisal decomposition (paper model)
- exp09 per-group cog fits (seed=42 paper model)
- exp10 dualctx specialization analyses

**Phase D** (aggregation):
- `exp01_main_comparison/aggregate.py`
- `exp07_ctx_vs_surprisal/aggregate.py`

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
| B | ~20 hrs (paper:5h, surp:5h, baselines:10h) |
| C | ~5 hrs |
| D | <1 hr |
| E | <30 min |

With 4 parallel GPUs, Phase B drops to ~5 hrs.
