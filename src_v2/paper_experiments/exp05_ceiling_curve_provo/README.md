# exp05 — Ceiling curve on Provo

## What this produces

Curve showing model performance on Provo as a function of the fraction
of held-out data used. Compared against the noise ceiling (asymptote).
Tells reviewers: "We're feature-limited, not data-limited."

## Files

| File | Purpose |
|---|---|
| `compute_ceiling_curve.py` | runs paper model on Provo at various data fractions; emits CSV |
| `plot_ceiling_curve.py` | reads CSV → PDF |

## Outputs

- `ceiling_curve_results.csv` — `(data_fraction, metric, model_r, ceiling_r, gap)`
- `plot_ceiling_curve.pdf`

## Run individually

```bash
python compute_ceiling_curve.py
python plot_ceiling_curve.py
```

## Prerequisites

- Trained paper model (1 seed minimum).
- Provo data.
- Existing `src_v2/break_the_ceiling/ceiling_curve_provo.py` provides the heavy
  lifting — this script wraps + reformats.

## Compute

~1 hour.
