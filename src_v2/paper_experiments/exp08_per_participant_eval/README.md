# exp08 — Per-participant evaluation (Table 5)

## What this produces

Take the trained paper model (1 seed, default 42). Evaluate it against
each of GECO's individual readers' eye-tracking data. Reports
per-reader r/MAE/bias.

Robustness check: does the aggregated-trained model generalize to
individual reader patterns?

## Files

| File | Purpose |
|---|---|
| `eval_per_participant.py` | runs evaluation per reader; emits CSV |

## Outputs

- `per_participant_eval.csv` — `(participant_id, n_sentences, n_words, r_TRT, r_FFD, r_Gaze, r_skip, MAE_TRT, MAE_FFD, MAE_Gaze, bias_TRT, bias_FFD, mean_RT)`

## Run individually

```bash
python eval_per_participant.py --seed 42
```

## Prerequisites

- Trained paper model checkpoint (seed 42 default).

## Compute

~1 hour.
