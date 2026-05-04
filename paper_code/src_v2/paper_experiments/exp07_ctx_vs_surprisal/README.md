# exp07 — ctx_head vs TinyLlama-surprisal head-to-head (Table 4)

## What this produces

The killer comparison: same Reichle cascade, two different ways to fill
the predictability slot:

- **(a) ctx_head**: paper model — `base_L1 += ctx_head(LLaMA_hidden)`
- **(b) surp**: ablation — `base_L1 += α3 · TinyLlama_surprisal`

Both use the same TinyLlama backbone. Surprisal is precomputed from the
same model that produces hidden states. The difference: rich hidden
state vs scalar surprisal.

## Files

| File | Purpose |
|---|---|
| `precompute_surprisal.py` | precompute TinyLlama per-word surprisals → cache .pt files |
| `train_surp_seeds.sh`     | trains v4c_v2_surp with 5 seeds (idempotent) |
| `aggregate.py`            | reads paper-model + surp checkpoints → CSVs |
| `plot_ctx_vs_surp.py`     | bar chart with error bars + per-seed dots |

## Outputs

- `ctx_vs_surp_results.csv` — long form: `(variant, seed, dataset, metric, value)`
- `ctx_vs_surp_summary.csv` — paired t-test, bootstrap CI, mean ± std
- `plot_ctx_vs_surp.pdf`

## Run individually

```bash
# 1. Precompute surprisals (once; ~1 hr)
python precompute_surprisal.py

# 2. Train surp variant (5 seeds × ~1 hr)
bash train_surp_seeds.sh

# 3. Aggregate (reads exp01 paper-model + this exp's surp results)
python aggregate.py

# 4. Plot
python plot_ctx_vs_surp.py
```

## Prerequisites

- Trained paper model (5 seeds) from exp01.
- TinyLlama causal LM accessible.
- `src_v2/lm_model/model_llama_hybrid_v4c_v2_surp.py` (already exists).
- `src_v2/lm_train/train_hybrid_v4c_v2_surp_geco.py` (already exists).
- Surprisal cache files under `data/cache/`.

## Statistical test

Paired t-test on r_TRT across the 5 paired seeds (ctx vs surp at the
same seed). Plus bootstrap CI on the difference for transparency.

## Compute

- Precompute: ~1 hr
- Training: 5 hrs sequential (5 seeds), ~1 hr parallel.
