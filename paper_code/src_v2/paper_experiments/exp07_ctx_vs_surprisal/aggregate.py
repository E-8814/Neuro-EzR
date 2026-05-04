"""
Aggregate ctx vs surp results.

Loads:
    - paper-model (ctx_head) per-seed eval results from exp01's results/raw/
    - surp checkpoints for each seed from exp07
Evaluates surp checkpoints on GECO test + Provo, then computes:
    - per-seed paired metrics for both variants
    - paired t-test on r_TRT
    - bootstrap CI on the difference

Usage:
    python aggregate.py
"""

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from paper_experiments import config
from paper_experiments.utils.load_data import (
    load_geco_aggregated, load_provo_aggregated, load_subtlex,
    word_frequency,
)
from paper_experiments.utils.load_model import load_paper_model, load_surp_model
from paper_experiments.utils.eval_metrics import (
    paired_t_test, bootstrap_ci_difference, corr,
)


RESULTS_DIR = Path(_HERE) / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LONG_CSV = RESULTS_DIR / "ctx_vs_surp_results.csv"
SUMMARY_CSV = RESULTS_DIR / "ctx_vs_surp_summary.csv"


def _eval_surp_one_seed(seed, geco_test, provo, subtlex, device):
    """Evaluate v4c_v2_surp at the given seed on both datasets."""
    model, _ = load_surp_model(seed=seed, device=device)

    # We need surps for evaluation. For test set, use cached test surprisals.
    cache_dir = config.DATA_DIR / "cache"
    surp_test = torch.load(
        str(cache_dir / "tinyllama_surprisal_geco_test.pt"),
        weights_only=False,
    )
    surp_provo = torch.load(
        str(cache_dir / "tinyllama_surprisal_provo.pt"),
        weights_only=False,
    )

    def eval_on(agg, surp_cache):
        pt, ht = [], []
        pf, hf = [], []
        pg, hg = [], []
        ps, hs = [], []
        with torch.no_grad():
            for s in agg:
                key = (s.text_id, getattr(s, "sentence_number", 0))
                surp_arr = surp_cache.get(
                    key, np.zeros(len(s.tokens), dtype=np.float32),
                )
                surps = torch.tensor(surp_arr, dtype=torch.float32).unsqueeze(0).to(device)
                freqs = torch.tensor(
                    [float(word_frequency(t, subtlex)) for t in s.tokens],
                    dtype=torch.float32,
                ).unsqueeze(0).to(device)
                wlens = torch.tensor(
                    [len(t) for t in s.tokens], dtype=torch.float32,
                ).unsqueeze(0).to(device)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    p = model([s.tokens], freqs, wlens, surps)
                seq_len = len(s.tokens)
                pt.extend(p['conditional_trt'][0, :seq_len].cpu().tolist())
                ht.extend(s.mean_trt)
                pf.extend(p['first_fixation'][0, :seq_len].cpu().tolist())
                hf.extend(s.mean_ffd)
                pg.extend(p['gaze_duration'][0, :seq_len].cpu().tolist())
                hg.extend(s.mean_gaze)
                ps.extend(p['skip_prob'][0, :seq_len].cpu().tolist())
                hs.extend(s.skip_rate)
        return {
            "r_trt": corr(pt, ht), "r_ffd": corr(pf, hf),
            "r_gaze": corr(pg, hg), "r_skip": corr(ps, hs),
        }

    out = {
        "geco_test": eval_on(geco_test, surp_test),
        "provo": eval_on(provo, surp_provo),
    }
    del model
    torch.cuda.empty_cache()
    return out


def _load_paper_model_seed_results():
    """Read per-seed JSONs from exp01 main_comparison."""
    exp01_raw = Path(_HERE).parent / "exp01_main_comparison" / "results" / "raw"
    out = {}
    for path in sorted(exp01_raw.glob(f"{config.PAPER_MODEL_RECIPE}_seed*.json")):
        with open(path) as f:
            payload = json.load(f)
        out[payload["seed"]] = payload["datasets"]
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading data...")
    subtlex = load_subtlex()
    geco_test = load_geco_aggregated("test")
    provo = load_provo_aggregated()

    # Paper-model (ctx) results
    print("Loading paper-model per-seed results from exp01/results/raw/...")
    ctx_results = _load_paper_model_seed_results()
    if not ctx_results:
        print("WARNING: no paper-model per-seed JSONs found. "
              "Run exp01/eval_all_models.py first.")

    # Surp results: evaluate each seed's checkpoint
    surp_results = {}
    for seed in config.SEEDS:
        ckpt = config.surp_ckpt_path(seed)
        if not ckpt.exists():
            print(f"  surp seed={seed}: missing checkpoint {ckpt}")
            continue
        print(f"  Evaluating surp seed={seed}...")
        surp_results[seed] = _eval_surp_one_seed(
            seed, geco_test, provo, subtlex, device,
        )

    # Long-form CSV: per (variant, seed, dataset, metric)
    rows = []
    for variant_name, results in [("ctx_head", ctx_results), ("surp", surp_results)]:
        for seed, datasets in results.items():
            for ds_name, metrics in datasets.items():
                for metric in ["r_trt", "r_ffd", "r_gaze", "r_skip"]:
                    if metric in metrics:
                        rows.append({
                            "variant": variant_name,
                            "seed": seed,
                            "dataset": ds_name,
                            "metric": metric,
                            "value": metrics[metric],
                        })

    with open(LONG_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["variant", "seed", "dataset", "metric", "value"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nWrote {len(rows)} rows to {LONG_CSV}")

    # Paired tests: per (dataset, metric), compare ctx vs surp seeds
    summary_rows = []
    common_seeds = sorted(
        set(ctx_results.keys()) & set(surp_results.keys())
    )
    for ds_name in ["geco_test", "provo"]:
        for metric in ["r_trt", "r_ffd", "r_gaze", "r_skip"]:
            ctx_vals = [
                ctx_results[s][ds_name][metric]
                for s in common_seeds
                if metric in ctx_results[s][ds_name]
            ]
            surp_vals = [
                surp_results[s][ds_name][metric]
                for s in common_seeds
                if metric in surp_results[s][ds_name]
            ]
            if len(ctx_vals) != len(surp_vals) or len(ctx_vals) < 2:
                continue
            ctx_mean, ctx_std = float(np.mean(ctx_vals)), float(np.std(ctx_vals, ddof=1))
            surp_mean, surp_std = float(np.mean(surp_vals)), float(np.std(surp_vals, ddof=1))
            try:
                t_stat, p_val = paired_t_test(ctx_vals, surp_vals)
            except Exception:
                t_stat, p_val = float("nan"), float("nan")
            diff_mean, lo, hi = bootstrap_ci_difference(ctx_vals, surp_vals)
            summary_rows.append({
                "dataset": ds_name,
                "metric": metric,
                "ctx_mean": ctx_mean, "ctx_std": ctx_std,
                "surp_mean": surp_mean, "surp_std": surp_std,
                "diff_mean": diff_mean,
                "diff_ci_low": lo, "diff_ci_high": hi,
                "paired_t": t_stat, "p_value": p_val,
                "n_seeds_paired": len(ctx_vals),
            })

    with open(SUMMARY_CSV, "w", newline="") as f:
        if summary_rows:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            for r in summary_rows:
                writer.writerow(r)
    print(f"Wrote {len(summary_rows)} rows to {SUMMARY_CSV}")

    print("\n=== ctx vs surp summary (geco_test, r_TRT) ===")
    for r in summary_rows:
        if r["dataset"] == "geco_test" and r["metric"] == "r_trt":
            print(f"  ctx:   {r['ctx_mean']:.3f} ± {r['ctx_std']:.3f}")
            print(f"  surp:  {r['surp_mean']:.3f} ± {r['surp_std']:.3f}")
            print(f"  diff:  {r['diff_mean']:+.3f}  (CI: [{r['diff_ci_low']:+.3f}, {r['diff_ci_high']:+.3f}])")
            print(f"  paired t = {r['paired_t']:.3f}, p = {r['p_value']:.4f}")


if __name__ == "__main__":
    main()
