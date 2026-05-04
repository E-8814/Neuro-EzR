# exp09 — Per-participant cog parameter fits (Figure 5 + Table 6)

## What this produces

For each of the 14 GECO readers, fit the cognitive scalar parameters
(α1, α2, ε, M1, M2 = I, δ) to that reader's data, with the neural
backbone (LLaMA, projection, ctx_head, skip_residual_head) frozen.

This tests whether the cognitive parameters are *functional indicators*
of individual differences, or just decorative scalars.

If fitted ε correlates with reader speed, fitted M1 with motor speed,
etc., the parameters carry per-reader meaning.

## Files

| File | Purpose |
|---|---|
| `fit_per_participant.py` | for each reader, frozen-backbone fine-tune of cog scalars |
| `analyze_fits.py` | reads fits + per-reader summaries; computes correlations |
| `plot_per_participant_cog.py` | parallel-coordinates + correlation plot |

## Outputs

- `per_participant_cog_fits.csv` — `(participant_id, alpha1_reichle, alpha2_reichle, epsilon, M1, M2_eq_I, delta, lambda_refix, refix_pivot, skip_temperature, n_train_words, fit_loss)`
- `cog_correlations.csv` — `(param, correlated_with, pearson_r, p_value, n_readers)`
- `plot_per_participant_cog.pdf` — parameter distributions + correlations

## Run individually

```bash
python fit_per_participant.py --seed 42      # produces per_participant_cog_fits.csv
python analyze_fits.py                        # produces cog_correlations.csv
python plot_per_participant_cog.py            # produces PDF
```

## Hyperparameters (in `fit_per_participant.py`)

- LR for cog scalars: 3e-5 (= cog_lr / 10).
- Epochs: 3.
- Batch size: 8.
- Backbone frozen (LLaMA + projection + ctx_head + skip_residual_head).

## Prerequisites

- Trained paper model checkpoint (seed 42 default).

## Compute

~10 min per reader × 14 readers ≈ 2 hrs.
