"""
Re-evaluate BERT regression and Ohio State RoBERTa baselines using
existing trained checkpoints, with the COMPLETE metric set.

For each model × seed × dataset, writes:

    complete_metrics/results/raw/baselines/<name>_seed<N>.json

Re-uses the existing data loaders and inference helpers from
eval_baselines.py (imported, not modified). The only difference is
that we route results through metrics_summary_complete() so mae_skip,
bias_skip, and the n_words diagnostic are present.

Usage:
    python -u .../02_eval_bert_ohio.py
    python -u .../02_eval_bert_ohio.py --bert_only
    python -u .../02_eval_bert_ohio.py --ohio_only
    python -u .../02_eval_bert_ohio.py --seeds 1 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
EXP01_DIR = os.path.dirname(_HERE)
SRC_V2 = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
REPO_ROOT = os.path.dirname(SRC_V2)
ARCHIVE_BASELINES = os.path.join(REPO_ROOT, "archive", "baselines")
ORIG_EZ = os.path.join(REPO_ROOT, "archive", "original_ezreader")

# Ensure all required modules are importable.
for p in (EXP01_DIR, SRC_V2, ARCHIVE_BASELINES, ORIG_EZ):
    if p not in sys.path:
        sys.path.insert(0, p)
sys.path.insert(0, _HERE)  # for local metrics.py

from paper_experiments import config  # noqa: E402  (just to confirm path resolution)

# Import the reusable helpers from the existing eval_baselines.py — without modifying it.
from eval_baselines import (  # noqa: E402
    load_eval_data, collect_bert_preds, get_human,
    eval_ohio_seed,  # we'll wrap this differently below; see notes
    _ohio_predict_all_metrics, _ohio_human_targets,
    OHIO_METRICS,
)

from bert_regression import BertDirectRegression  # noqa: E402

from metrics import metrics_summary_complete  # local


SEEDS = config.SEEDS
OUT_DIR = Path(_HERE) / "results" / "raw" / "baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
#  BERT
# --------------------------------------------------------------------------- #


def _eval_bert_seed_complete(seed: int, device: torch.device) -> dict:
    """Mirror eval_baselines.eval_bert_seed but with metrics_summary_complete."""
    geco_test, provo_agg = load_eval_data()
    ckpt_path = os.path.join(
        ARCHIVE_BASELINES, "checkpoints_bert_regression",
        f"seed{seed}", "best_model.pt",
    )
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    print(f"  loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = BertDirectRegression(
        bert_model_name=ckpt.get("bert_model_name", "bert-base-uncased"),
        freeze_bert_layers=ckpt.get("freeze_bert_layers", 8),
        hidden_dim=ckpt.get("hidden_dim", 256),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("  predicting GECO test...")
    p_trt, p_ffd, p_gaze, p_skip = collect_bert_preds(model, geco_test, device)
    h_trt, h_ffd, h_gaze, h_skip = get_human(geco_test)
    geco = metrics_summary_complete(p_trt, p_ffd, p_gaze, p_skip,
                                    h_trt, h_ffd, h_gaze, h_skip)

    print("  predicting Provo...")
    p_trt, p_ffd, p_gaze, p_skip = collect_bert_preds(model, provo_agg, device)
    h_trt, h_ffd, h_gaze, h_skip = get_human(provo_agg)
    provo = metrics_summary_complete(p_trt, p_ffd, p_gaze, p_skip,
                                     h_trt, h_ffd, h_gaze, h_skip)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {"geco_test": geco, "provo": provo}


# --------------------------------------------------------------------------- #
#  Ohio State (4 metric-specific RoBERTa heads)
# --------------------------------------------------------------------------- #


def _eval_ohio_seed_complete(seed: int, device: torch.device) -> dict:
    """Mirror eval_baselines.eval_ohio_seed but with metrics_summary_complete."""
    from transformers import RobertaTokenizer
    from run_ohio_state_on_geco import convert_to_ohio_format

    geco_test, provo_agg = load_eval_data()
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    print("  converting data to ohio format...")
    geco_data = convert_to_ohio_format(geco_test, tokenizer)
    provo_data = convert_to_ohio_format(provo_agg, tokenizer)

    out = {}
    for label, data in (("geco_test", geco_data), ("provo", provo_data)):
        print(f"  predicting {label}...")
        per_metric = _ohio_predict_all_metrics(seed, data, tokenizer, device)
        h_trt, h_ffd, h_gaze, h_skip = _ohio_human_targets(data)

        def safe(arr, h):
            return arr if arr is not None else np.zeros_like(h)

        p_trt  = safe(per_metric["trt"],  h_trt)
        p_ffd  = safe(per_metric["ffd"],  h_ffd)
        p_gaze = safe(per_metric["gaze"], h_gaze)
        p_skip = safe(per_metric["skip"], h_skip)

        out[label] = metrics_summary_complete(p_trt, p_ffd, p_gaze, p_skip,
                                              h_trt, h_ffd, h_gaze, h_skip)

    return out


# --------------------------------------------------------------------------- #
#  Driver
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_only", action="store_true")
    parser.add_argument("--ohio_only", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.bert_only and args.ohio_only:
        parser.error("--bert_only and --ohio_only are exclusive")

    do_bert = not args.ohio_only
    do_ohio = not args.bert_only

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if do_bert:
        print("\n========== BERT regression ==========")
        for seed in args.seeds:
            out_path = OUT_DIR / f"bert_regression_seed{seed}.json"
            if out_path.exists() and not args.force:
                print(f"  seed={seed}: {out_path.name} exists, skipping (use --force).")
                continue
            t0 = time.time()
            try:
                summaries = _eval_bert_seed_complete(seed, device)
            except FileNotFoundError as exc:
                print(f"  seed={seed}: {exc}")
                continue
            payload = {
                "model": "bert_regression",
                "seed": seed,
                "datasets": summaries,
            }
            out_path.write_text(json.dumps(payload, indent=2))
            print(f"  seed={seed}: wrote {out_path.name}  ({time.time()-t0:.1f}s)")

    if do_ohio:
        print("\n========== Ohio State RoBERTa ==========")
        for seed in args.seeds:
            out_path = OUT_DIR / f"ohio_state_roberta_seed{seed}.json"
            if out_path.exists() and not args.force:
                print(f"  seed={seed}: {out_path.name} exists, skipping (use --force).")
                continue
            t0 = time.time()
            try:
                summaries = _eval_ohio_seed_complete(seed, device)
            except Exception as exc:
                print(f"  seed={seed}: failed -> {exc!r}")
                continue
            payload = {
                "model": "ohio_state_roberta",
                "seed": seed,
                "datasets": summaries,
            }
            out_path.write_text(json.dumps(payload, indent=2))
            print(f"  seed={seed}: wrote {out_path.name}  ({time.time()-t0:.1f}s)")

    print("\nDone.")


if __name__ == "__main__":
    main()
