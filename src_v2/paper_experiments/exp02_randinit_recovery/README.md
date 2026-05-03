# exp02 — Random-init parameter recovery (Figure 2)

## What this produces

The strongest empirical figure in the paper. For each of 7 cognitive
parameters with a published Reichle 2003 value, show:
- 5 randomly-sampled init values (from ±50% of Reichle)
- 5 final converged values
- Reichle 2003 published value as horizontal line

If the converged values cluster near Reichle despite different starts,
that's recovery. The paper claim is "Optimization on real data drives
arbitrary cognitive scalars toward Reichle's published values."

## Files

| File | Purpose |
|---|---|
| `train_randinit_seeds.sh` | trains 5 seeds with ±50% jitter (idempotent) |
| `aggregate.py` | reads checkpoints → `recovery_results.csv` |
| `plot_recovery.py` | reads CSV → `plot_recovery.pdf` |

## Outputs (in `results/`)

- `recovery_results.csv` — long form: `(seed, param, init_value, converged_value, reichle_target, abs_drift_init, abs_drift_reichle)`
- `recovery_summary.csv` — per-param: `(param, mean_init, std_init, mean_converged, std_converged, tightening_ratio, mean_abs_distance_to_reichle)`
- `plot_recovery.pdf` — strip plot per parameter

## Run individually

```bash
bash train_randinit_seeds.sh
python aggregate.py
python plot_recovery.py
```

## Prerequisites

- Paper-model recipe decided (the randinit model uses v4c_v2 architecture).
- `src_v2/lm_model/model_llama_hybrid_v4c_v2_randinit.py` (already exists).
- `src_v2/lm_train/train_hybrid_v4c_v2_randinit_geco.py` (already exists).

## What "recovery" looks like in the output

For each parameter, compute `tightening_ratio = std(converged) / std(init)`.
- `< 0.30` → strong recovery (parameters tightened from random toward common value).
- `0.30–0.70` → partial recovery.
- `> 0.70` → no recovery (parameters stayed near init).

The `mean_abs_distance_to_reichle` column tells you whether the common
value the parameters converge to actually matches Reichle. Both metrics
are needed for the recovery claim.

## Compute

5 seeds × 1 hr ≈ 5 hrs sequential. 1 hr with 5 parallel GPUs.
