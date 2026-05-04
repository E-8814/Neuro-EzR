# Checkpoints

Trained model weights are not redistributed in this repository (the full set is ~16 GB across baselines and cascade variants). Reproduce them by running the training scripts; all results in the paper come from the recipes documented here.

## Where checkpoints are expected

Paths are resolved by helpers in `src_v2/paper_experiments/config.py`:

```python
paper_model_ckpt_path(seed)  # paper cascade (dualctx)
surp_ckpt_path(seed)         # exp07 (ctx-head replaced by α₃·surprisal)
baseline_ckpt_path(name, seed)  # baselines from archive/baselines/
```

Default layout (relative to repository root):

```
checkpoints/
├── hybrid_v4c_v2_dualctx/geco_TinyLlama_TinyLlama-1.1B-Chat-v1.0_seed{1,2,3,42,100}/best_model.pt
└── hybrid_v4c_v2_surp/geco_..._seed{1,2,3,42,100}/best_model.pt

archive/baselines/
├── checkpoints_linear_regression/seed{1,2,3,42,100}/...
├── checkpoints_lightgbm/seed{1,2,3,42,100}/...
├── checkpoints_gpt2_surprisal/seed{1,2,3,42,100}/...
├── checkpoints_bert_regression/seed{1,2,3,42,100}/...
└── checkpoints_ohio_state_roberta/seed{1,2,3,42,100}/...
```

## Retraining

### Paper cascade (dualctx) — exp01b

```bash
cd src_v2/paper_experiments/exp01_main_comparison
bash train_paper_model_seeds.sh   # 5 seeds × ~1h on a single RTX-class GPU
```

This calls `src_v2/lm_train/train_hybrid_v4c_v2_dualctx_geco.py` with seeds from `config.SEEDS`. Hyperparameters (LM lr 2e-5, head lr 5e-4, cog lr 3e-4, 5 epochs) are fixed in `config.py`.

### ctx-head vs surprisal ablation — exp07

```bash
cd src_v2/paper_experiments/exp07_ctx_vs_surprisal
python precompute_surprisal.py    # one-shot; writes per-token TinyLlama surprisals
bash train_surp_seeds.sh
```

### Baselines — exp01a

```bash
cd src_v2/paper_experiments/exp01_main_comparison
bash train_baselines_seeds_v3.sh   # 3 single-run + 2 × 5-seed = ~10h GPU
```

The five baselines documented in Table 1: `linear_regression`, `lightgbm`, `gpt2_surprisal`, `bert_regression`, `ohio_state_roberta`. The Ohio-State and BERT baselines run 5 seeds; the three flat regressors are deterministic and run once.

## GPU requirements

- Cascade variants (dualctx, surp): TinyLlama-1.1B with 75% of layers frozen; runs on a single 24 GB GPU. ~1 hour per seed.
- BERT and Ohio-State RoBERTa: similar memory profile; ~30 min per seed.
- LightGBM, linear, GPT-2 surprisal: CPU-only or any GPU, < 30 min total.

A single seed of the full pipeline (Phase B) is feasible in ~5 GPU-hours; the 5-seed paper run is ~20 GPU-hours sequential or ~5 hours across 4 GPUs.

## CPU/disk

- Tokenizing GECO + Provo and computing aggregations is cached under `data/cache/` after the first run.
- Each `best_model.pt` is ~0.5–4 GB depending on the variant (TinyLlama backbone dominates).

## Backbone download

The TinyLlama backbone (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`) is auto-downloaded by `transformers` on first use. Set `HF_HOME` if you want to control the cache location.
