# exp04 — Noise ceiling

## What this produces

Split-half reliability of GECO eye-tracking data. The reported number
is what your model is correlating against — the ceiling is where any
model could reach.

Reported in §4 of the paper as: "On aggregated GECO, our model achieves
r_TRT=0.43 against a split-half reliability of X (%Y of available signal)."

## Files

| File | Purpose |
|---|---|
| `compute_noise_ceiling.py` | wrapper that runs `src_v2/break_the_ceiling/noise_ceiling.py` and emits structured CSV output |

## Outputs

- `noise_ceiling_results.csv` — `(corpus, metric, half_corr, full_corr_estimate, n_splits)`

## Run individually

```bash
python compute_noise_ceiling.py
```

## Prerequisites

- GECO data only (no model required).

## Compute

~30 minutes (200 random splits × 4 metrics).
