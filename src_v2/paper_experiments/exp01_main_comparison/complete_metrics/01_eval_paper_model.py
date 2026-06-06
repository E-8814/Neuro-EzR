"""
Re-evaluate the paper model (v4c_v2_dualctx) across all 5 seeds and
write JSONs with the COMPLETE metric set (incl. mae_skip, bias_skip,
mae_gaze) into:

    complete_metrics/results/raw/v4c_v2_dualctx_seed<N>.json

Loads existing trained checkpoints — does NOT retrain.

Usage:
    python -u .../01_eval_paper_model.py
    python -u .../01_eval_paper_model.py --seeds 1 42 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))   # so paper_experiments imports work
sys.path.insert(0, _HERE)                                    # for metrics.py

from paper_experiments import config
from paper_experiments.utils.load_data import (
    load_geco_aggregated, load_provo_aggregated, load_subtlex,
)
from paper_experiments.utils.load_model import load_paper_model
from paper_experiments.utils.eval_metrics import eval_predictions_on_aggregated

from metrics import metrics_summary_complete


OUT_DIR = Path(_HERE) / "results" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _eval_one_seed(seed: int, device: torch.device, subtlex):
    print(f"\n>> seed={seed}: loading checkpoint")
    model, _ = load_paper_model(seed=seed, device=device)

    print(f"   GECO test: predicting...")
    geco_test = load_geco_aggregated("test")
    arr_geco, _ = eval_predictions_on_aggregated(model, geco_test, device, subtlex)

    print(f"   Provo: predicting...")
    provo = load_provo_aggregated()
    arr_provo, _ = eval_predictions_on_aggregated(model, provo, device, subtlex)

    out = {
        "model": "v4c_v2_dualctx",
        "seed": seed,
        "datasets": {
            "geco_test": metrics_summary_complete(
                arr_geco["pred_trt"], arr_geco["pred_ffd"],
                arr_geco["pred_gaze"], arr_geco["pred_skip"],
                arr_geco["human_trt"], arr_geco["human_ffd"],
                arr_geco["human_gaze"], arr_geco["human_skip"],
            ),
            "provo": metrics_summary_complete(
                arr_provo["pred_trt"], arr_provo["pred_ffd"],
                arr_provo["pred_gaze"], arr_provo["pred_skip"],
                arr_provo["human_trt"], arr_provo["human_ffd"],
                arr_provo["human_gaze"], arr_provo["human_skip"],
            ),
        },
    }

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=config.SEEDS,
                        help=f"Seeds to evaluate (default {config.SEEDS}).")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if JSON already exists.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading SUBTLEX...")
    subtlex = load_subtlex()

    for seed in args.seeds:
        out_path = OUT_DIR / f"v4c_v2_dualctx_seed{seed}.json"
        if out_path.exists() and not args.force:
            print(f">> seed={seed}: {out_path.name} exists, skipping (use --force to redo)")
            continue
        t0 = time.time()
        out = _eval_one_seed(seed, device, subtlex)
        out_path.write_text(json.dumps(out, indent=2))
        print(f">> seed={seed}: wrote {out_path}  ({time.time() - t0:.1f}s)")

    print("\nDone.")


if __name__ == "__main__":
    main()
