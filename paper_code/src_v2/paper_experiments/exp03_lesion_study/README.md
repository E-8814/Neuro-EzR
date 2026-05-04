# exp03 — Lesion study (Table 2)

## What this produces

Per-component ablation. Loads the seed=42 paper model. For each lesion:
1. Apply (zero a head, replace L1 with mean, etc.)
2. Re-evaluate on GECO test + Provo
3. Report Δr in each metric vs the full (un-lesioned) model

## Lesions tested

Existing lesions (from prior eval_lesion.py):
- `const_l1`           — replace per-word L1 with mean L1
- `const_l2`           — replace per-word L2 with mean L2
- `const_skip`         — set every skip prediction to mean skip rate
- `swap_l1l2`          — swap L1 and L2 values
- `shuffle_l1`         — randomly permute L1 across words
- `shuffle_l2`         — randomly permute L2 across words
- `zero_ecc`           — set ε = 1.0 (eliminates eccentricity scaling)
- `no_l2_to_ffd`       — exclude L2 path from gaze/TRT computation

v4c_v2-specific lesions (new):
- `zero_ctx_head`      — set ctx_head output to 0 (tests LLaMA contribution)
- `zero_skip_residual` — set skip residual to 0 (tests neural skip contribution)
- `no_first_word_mask` — disable first-word skip flooring

## Files

| File | Purpose |
|---|---|
| `run_lesions.py` | applies all lesions, evaluates, writes CSV |
| `plot_lesion.py` | reads CSV → bar chart of Δr |

## Outputs (in `results/`)

- `lesion_results.csv` — long-form: `(lesion_type, dataset, metric, value, delta_vs_full)`
- `plot_lesion.pdf`

## Run individually

```bash
python run_lesions.py --seed 42
python plot_lesion.py
```

## Prerequisites

- Trained paper model checkpoint at seed=42 (`config.paper_model_ckpt_path(42)`).

## Compute

~1 hr (each lesion = a forward pass over GECO test + Provo).
