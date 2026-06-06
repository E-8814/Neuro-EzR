"""
Evaluate the CLASSICAL E-Z Reader simulator (Reichle 2003) on
GECO test + Provo, with N=200 Monte Carlo runs per sentence.

Output:
    complete_metrics/results/raw/ez_reader_classical_seed1.json

The classical model has no learnable parameters — only stochasticity in
the simulator. We use a single 'seed' label (=1) and `random.seed(1)`
for reproducibility. Per-word predictions are means over the 200 MC
runs:
    pred_FFD   = mean of first_fixation_duration across runs
    pred_Gaze  = mean of gaze_duration (first-pass sum) across runs
    pred_TRT   = mean of total_reading_time across runs
    pred_skip  = fraction of runs where the word was skipped (in [0,1])

Usage:
    python -u .../04_eval_ez_classical.py
    python -u .../04_eval_ez_classical.py --num_runs 100
    python -u .../04_eval_ez_classical.py --workers 16
    python -u .../04_eval_ez_classical.py --limit 100   # smoke test on first 100 sentences
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
SRC_V2 = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
REPO_ROOT = os.path.dirname(SRC_V2)
ARCHIVE_BASELINES = os.path.join(REPO_ROOT, "archive", "baselines")
ORIG_EZ = os.path.join(REPO_ROOT, "archive", "original_ezreader")

for p in (SRC_V2, ARCHIVE_BASELINES, ORIG_EZ, _HERE,
          os.path.join(_HERE, "ez_classical")):
    if p not in sys.path:
        sys.path.insert(0, p)

from paper_experiments.utils.load_data import (  # noqa: E402
    load_geco_aggregated, load_provo_aggregated, load_subtlex, word_frequency,
)

from ez_classical.wrapper_with_gaze import run_classical_averaged  # noqa: E402

from metrics import metrics_summary_complete  # local


OUT_DIR = Path(_HERE) / "results" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
#  Per-sentence worker (top-level for picklability)
# --------------------------------------------------------------------------- #


def _simulate_one_sentence(args):
    """
    Args (passed as a single tuple for Pool.imap):
        (sent_idx, tokens, frequencies, predictabilities, num_runs, model_params)

    Returns (sent_idx, dict) where dict has averaged per-word predictions:
        total_reading_time, first_fixation_duration, gaze_duration, skip_rate
    """
    sent_idx, tokens, freqs, preds, num_runs, model_params = args
    out = run_classical_averaged(
        tokens=list(tokens),
        frequencies=list(freqs),
        predictabilities=list(preds),
        num_runs=num_runs,
        model_params=model_params,
    )
    return sent_idx, out


# --------------------------------------------------------------------------- #
#  Run the simulator over a list of AggregatedSentence
# --------------------------------------------------------------------------- #


def _run_corpus(agg_list, subtlex, num_runs, workers, label,
                pred_default: float = 0.05,
                model_params=None):
    """
    Returns four numpy arrays (pred_trt, pred_ffd, pred_gaze, pred_skip)
    aligned with the four arrays of human targets (h_trt, h_ffd, h_gaze, h_skip).
    Per-word predictions are means over `num_runs` MC simulations.

    `model_params`, if provided, overrides classical-engine parameters
    (alpha1, alpha2, eccentricity, delta, lambda, etc.) for every
    simulation.
    """
    # Build per-sentence (tokens, frequencies, predictabilities) tuples.
    tasks = []
    human_per_sent = []  # list of dicts per-sentence with h_trt, h_ffd, h_gaze, h_skip
    for sent_idx, agg in enumerate(agg_list):
        tokens = list(agg.tokens)
        freqs = [float(word_frequency(t, subtlex)) for t in tokens]
        preds = []
        for i in range(len(tokens)):
            try:
                p = float(agg.predictabilities[i])
            except (TypeError, IndexError, ValueError):
                p = pred_default
            if p != p:  # NaN guard
                p = pred_default
            preds.append(max(0.0, min(1.0, p)))
        tasks.append((sent_idx, tokens, freqs, preds, num_runs, model_params))
        human_per_sent.append({
            "h_trt":  list(agg.mean_trt),
            "h_ffd":  list(agg.mean_ffd),
            "h_gaze": list(agg.mean_gaze),
            "h_skip": list(agg.skip_rate),
        })

    print(f"  {label}: {len(tasks)} sentences × {num_runs} MC runs each")

    pred_per_sent = [None] * len(tasks)
    t0 = time.time()
    if workers > 1:
        with Pool(processes=workers) as pool:
            for i, (sent_idx, out) in enumerate(
                    pool.imap_unordered(_simulate_one_sentence, tasks, chunksize=4),
                    start=1):
                pred_per_sent[sent_idx] = out
                if i % 200 == 0:
                    elapsed = time.time() - t0
                    print(f"    [{i:>5d}/{len(tasks)}]  ({elapsed:.0f}s)")
    else:
        for i, t in enumerate(tasks, start=1):
            sent_idx, out = _simulate_one_sentence(t)
            pred_per_sent[sent_idx] = out
            if i % 100 == 0:
                elapsed = time.time() - t0
                print(f"    [{i:>5d}/{len(tasks)}]  ({elapsed:.0f}s)")
    elapsed = time.time() - t0
    print(f"  {label}: simulation done in {elapsed:.0f}s")

    # Flatten per-word predictions and human targets
    pred_trt, pred_ffd, pred_gaze, pred_skip = [], [], [], []
    h_trt, h_ffd, h_gaze, h_skip = [], [], [], []
    n_failed = 0
    for sent_idx, out in enumerate(pred_per_sent):
        h = human_per_sent[sent_idx]
        n = len(h["h_trt"])
        if out is None or not out.get("success"):
            # Fallback: zero predictions; will count toward MAE/r unfavorably.
            pred_trt.extend([0.0] * n)
            pred_ffd.extend([0.0] * n)
            pred_gaze.extend([0.0] * n)
            pred_skip.extend([1.0] * n)
            n_failed += 1
        else:
            pred_trt.extend(out["total_reading_time"])
            pred_ffd.extend(out["first_fixation_duration"])
            pred_gaze.extend(out["gaze_duration"])
            pred_skip.extend(out["skip_rate"])
        h_trt.extend(h["h_trt"])
        h_ffd.extend(h["h_ffd"])
        h_gaze.extend(h["h_gaze"])
        h_skip.extend(h["h_skip"])

    if n_failed:
        print(f"  {label}: {n_failed} sentences had simulation failures (filled with zeros).")

    return (
        np.array(pred_trt),  np.array(pred_ffd),
        np.array(pred_gaze), np.array(pred_skip),
        np.array(h_trt),     np.array(h_ffd),
        np.array(h_gaze),    np.array(h_skip),
    )


# --------------------------------------------------------------------------- #
#  Driver
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", type=int, default=200,
                        help="Monte Carlo runs per sentence (default 200).")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 1),
                        help=f"Parallel workers (default {max(1, cpu_count()-1)}).")
    parser.add_argument("--limit", type=int, default=0,
                        help="If >0, only run the first N sentences per corpus (smoke test).")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if JSON exists.")
    parser.add_argument("--no_fitted_params", action="store_true",
                        help="Use Reichle 2003 default parameters even if "
                             "ez_classical/fitted_params.json is present.")
    args = parser.parse_args()

    out_path = OUT_DIR / "ez_reader_classical_seed1.json"
    if out_path.exists() and not args.force:
        print(f">> {out_path.name} exists, skipping (use --force to redo).")
        return

    # ---- Load fitted parameters if available ---- #
    fitted_path = Path(_HERE) / "ez_classical" / "fitted_params.json"
    model_params = None
    params_source = "Reichle 2003 defaults"
    if fitted_path.exists() and not args.no_fitted_params:
        try:
            payload = json.loads(fitted_path.read_text())
            model_params = payload.get("fitted_params", None)
            if model_params:
                params_source = (
                    f"FITTED params from {fitted_path.name} "
                    f"(loss {payload.get('loss_default', float('nan')):.4f} -> "
                    f"{payload.get('loss_fitted', float('nan')):.4f}; "
                    f"{payload.get('n_evals', '?')} evals on "
                    f"{payload.get('n_sentences', '?')} train sentences)"
                )
        except Exception as exc:
            print(f"  [warn] could not parse {fitted_path}: {exc!r}; "
                  f"falling back to defaults.")
            model_params = None
    print(f"\nClassical-engine parameters: {params_source}")
    if model_params:
        for k, v in sorted(model_params.items()):
            print(f"  {k:<22s} = {v:.4f}")

    random.seed(1)
    np.random.seed(1)

    print("\nLoading SUBTLEX...")
    subtlex = load_subtlex()

    print("Loading GECO test...")
    geco_test = load_geco_aggregated("test")
    if args.limit:
        geco_test = geco_test[:args.limit]
    print(f"  {len(geco_test)} sentences")

    print("Loading Provo...")
    provo = load_provo_aggregated()
    if args.limit:
        provo = provo[:args.limit]
    print(f"  {len(provo)} sentences")

    t_total = time.time()
    print("\n========== GECO test ==========")
    p_trt_g, p_ffd_g, p_gaze_g, p_skip_g, \
        h_trt_g, h_ffd_g, h_gaze_g, h_skip_g = _run_corpus(
            geco_test, subtlex, args.num_runs, args.workers, "GECO test",
            model_params=model_params,
        )
    geco_summary = metrics_summary_complete(
        p_trt_g, p_ffd_g, p_gaze_g, p_skip_g,
        h_trt_g, h_ffd_g, h_gaze_g, h_skip_g,
    )

    print("\n========== Provo ==========")
    p_trt_p, p_ffd_p, p_gaze_p, p_skip_p, \
        h_trt_p, h_ffd_p, h_gaze_p, h_skip_p = _run_corpus(
            provo, subtlex, args.num_runs, args.workers, "Provo",
            model_params=model_params,
        )
    provo_summary = metrics_summary_complete(
        p_trt_p, p_ffd_p, p_gaze_p, p_skip_p,
        h_trt_p, h_ffd_p, h_gaze_p, h_skip_p,
    )

    payload = {
        "model": "ez_reader_classical",
        "seed": 1,
        "num_mc_runs": args.num_runs,
        "is_classical": True,
        "params_source": params_source,
        "model_params": model_params,
        "datasets": {
            "geco_test": geco_summary,
            "provo": provo_summary,
        },
    }
    out_path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nWrote {out_path}  (total {time.time() - t_total:.0f}s)")

    # Print a compact summary
    print("\n========== Summary ==========")
    for label, s in (("GECO test", geco_summary), ("Provo", provo_summary)):
        print(f"\n{label}:")
        for m in ("trt", "ffd", "gaze", "skip"):
            unit = "" if m == "skip" else " ms"
            print(f"  {m.upper():<5s}  r={s[f'r_{m}']:+.3f}  "
                  f"MAE={s[f'mae_{m}']:.3f}{unit}  "
                  f"bias={s[f'bias_{m}']:+.3f}")


if __name__ == "__main__":
    main()
