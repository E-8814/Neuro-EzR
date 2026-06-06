# Complete-metrics augmentation of Table 1

Adds the metrics that were missing from the original Table 1 evaluation:

- **`mae_skip`** for **all** models (in raw `[0, 1]` fraction-of-readers units)
- **`mae_gaze`** for the three flat baselines (linear, lightgbm, gpt2_surprisal) — was already present for BERT/Ohio
- **A new row** for the **classical E-Z Reader** simulator (Reichle, Rayner, Pollatsek 2003) so the differentiable cascade is benchmarked against its hand-tuned ancestor

All scripts in this directory are **new** — they do not modify any existing
files. They produce JSONs under `complete_metrics/results/raw/{,baselines/}`
which the local `aggregate.py` collates into augmented CSVs.

## Files

```
complete_metrics/
├── README.md                          this file
├── metrics.py                          metrics_summary_complete()
├── 01_eval_paper_model.py              re-eval paper model (5 seeds, loads checkpoints)
├── 02_eval_bert_ohio.py                re-eval BERT + Ohio (5 seeds, loads checkpoints)
├── 03_train_eval_flat_baselines.py     train + eval linear/lightgbm/gpt2 (no saved ckpts)
├── 04_eval_ez_classical.py             N=200 Monte Carlo on GECO test + Provo
├── ez_classical/
│   ├── __init__.py
│   ├── wrapper_with_gaze.py            extends ez_wrapper.py with gaze-duration extraction
│   └── provo_predictability.py         (unused; kept for reference — Provo CSV
│                                        already includes OrthographicMatch as predictability)
├── aggregate.py                         long-form + summary CSVs
└── run_slurm.sh                         end-to-end orchestrator
```

## Outputs

```
complete_metrics/results/
├── raw/
│   ├── v4c_v2_dualctx_seed{1,2,3,42,100}.json
│   ├── ez_reader_classical_seed1.json
│   └── baselines/
│       ├── bert_regression_seed{1,2,3,42,100}.json
│       ├── ohio_state_roberta_seed{1,2,3,42,100}.json
│       ├── linear_regression_seed1.json
│       ├── lightgbm_seed1.json
│       └── gpt2_surprisal_seed1.json
├── per_seed_metrics_complete.csv         long form: (model, seed, dataset, metric, value)
└── comparison_results_complete.csv       (model, dataset, metric, mean, std, n_seeds)
```

Every JSON's `datasets["geco_test"]` and `datasets["provo"]` block contains:

```
r_trt, r_ffd, r_gaze, r_skip
mae_trt, mae_ffd, mae_gaze, mae_skip
bias_trt, bias_ffd, bias_gaze, bias_skip
mean_pred_skip, mean_human_skip
n_words
```

## Running

### Single shot

```bash
sbatch src_v2/paper_experiments/exp01_main_comparison/complete_metrics/run_slurm.sh
```

Or interactively in a tmux on a GPU node:

```bash
bash src_v2/paper_experiments/exp01_main_comparison/complete_metrics/run_slurm.sh
```

Total time on a single GPU node with ~16 CPUs: ~1–1.5 hours.

### One step at a time

```bash
cd /home/u384661/Neuro_EZR
python -u src_v2/paper_experiments/exp01_main_comparison/complete_metrics/01_eval_paper_model.py
python -u src_v2/paper_experiments/exp01_main_comparison/complete_metrics/02_eval_bert_ohio.py
python -u src_v2/paper_experiments/exp01_main_comparison/complete_metrics/03_train_eval_flat_baselines.py
python -u src_v2/paper_experiments/exp01_main_comparison/complete_metrics/04_eval_ez_classical.py
python -u src_v2/paper_experiments/exp01_main_comparison/complete_metrics/aggregate.py
```

Each step is independently idempotent — finished JSONs are skipped. Pass
`--force` to recompute.

### Smoke test for the classical E-Z Reader

```bash
python -u .../04_eval_ez_classical.py --num_runs 50 --limit 100 --workers 8
```

Runs 50 MC × 100 sentences × 2 corpora ≈ 5 minutes; sanity-check the
output before kicking off the full N=200 run.

## Conventions

- **`mae_skip` is in raw `[0, 1]` fraction-of-readers units**, NOT scaled to
  percentage points. CMCL 2021 uses `× 100` scaling for FIXPROP MAE; we
  prefer the unscaled form for direct interpretability ("predicted skip
  rate is off by 0.13 from observed" rather than "13 percentage points").
- The classical E-Z Reader uses the **unmodified** simulator from
  `archive/original_ezreader/ez_reader_engine.py`. The wrapper extension
  in `ez_classical/wrapper_with_gaze.py` adds gaze-duration extraction
  (first-pass fixation sum) without modifying the original wrapper.
- N=200 Monte Carlo runs per sentence for the classical model. Reichle 2003
  used N=1,000 per condition on the 48-sentence Schilling corpus; for our
  ~8,000-sentence corpus, N=200 keeps per-word noise under ~3.5% relative
  while staying inside a sensible compute budget.
