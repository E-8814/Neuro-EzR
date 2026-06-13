"""
Evaluate the v4c_v3_dualctx (skip_align=next) checkpoints on GECO test
and Provo, with the skip metric computed on the comparable population
(words 1..L-1, next-aligned). Also evaluates the no-LM variant of each
checkpoint (Diff-EZR no LM: both ctx heads and the skip residual head
zeroed, trained cognitive scalars kept), mirroring exp01's
06_eval_v4c_v2_no_ai.py for the new model family.

Outputs per seed:
    results/raw/v4c_v3_dualctx_next_seed<N>.json
    results/raw/v4c_v3_dualctx_next_no_ai_seed<N>.json
    results/perword/v4c_v3_seed<N>_<corpus>.csv          (per-word dump)

Usage:
    python -u eval_v4c_v3_seeds.py
    python -u eval_v4c_v3_seeds.py --seeds 42 --skip_no_ai
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
SRC_V2 = os.path.abspath(os.path.join(_HERE, "..", ".."))
REPO_ROOT = os.path.dirname(SRC_V2)
LM_MODEL = os.path.join(SRC_V2, "lm_model")
ORIG_EZ = os.path.join(REPO_ROOT, "archive", "original_ezreader")

for p in (SRC_V2, LM_MODEL, ORIG_EZ, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from paper_experiments.utils.load_data import (  # noqa: E402
    load_geco_aggregated, load_provo_aggregated, load_subtlex,
)
from paper_experiments.utils.eval_metrics import (  # noqa: E402
    eval_predictions_on_aggregated,
)
from model_llama_hybrid_v4c_v3_dualctx import NeuralEZReaderHybrid  # noqa: E402

from skip_metrics import next_aligned_pairs, skip_summary  # noqa: E402


SEEDS = [1, 2, 3, 42, 100]
CKPT_TMPL = os.path.join(
    REPO_ROOT, "checkpoints", "hybrid_v4c_v3_dualctx_next",
    "geco_TinyLlama_TinyLlama-1.1B-Chat-v1.0_seed{seed}", "best_model.pt",
)

RAW_DIR = Path(_HERE) / "results" / "raw"
PW_DIR = Path(_HERE) / "results" / "perword"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PW_DIR.mkdir(parents=True, exist_ok=True)


def load_v3(seed: int, device: torch.device) -> NeuralEZReaderHybrid:
    path = CKPT_TMPL.format(seed=seed)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing v4c_v3 checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = NeuralEZReaderHybrid(
        model_name=ckpt["model_name"],
        freeze_layers=ckpt["freeze_layers"],
        hidden_dim=ckpt.get("hidden_dim", 256),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def zero_ai_heads(model: NeuralEZReaderHybrid) -> None:
    """Diff-EZR (no LM): zero the FINAL linear of each neural head so its
    output is exactly 0 for every word; cognitive scalars stay trained."""
    with torch.no_grad():
        for head in (model.ctx_head_FFD, model.ctx_head_skip,
                     model.skip_residual_head):
            last = head[-1]
            last.weight.zero_()
            last.bias.zero_()


def time_summary(arrays) -> dict:
    """Time metrics on ALL words (same definition as the existing tables)."""
    def corr(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        if len(a) > 2 and a.std() > 0 and b.std() > 0:
            return float(np.corrcoef(a, b)[0, 1])
        return 0.0
    out = {}
    for m, pk, hk in (("trt", "pred_trt", "human_trt"),
                      ("ffd", "pred_ffd", "human_ffd"),
                      ("gaze", "pred_gaze", "human_gaze")):
        p, h = np.asarray(arrays[pk], float), np.asarray(arrays[hk], float)
        out[f"r_{m}"] = corr(p, h)
        out[f"mae_{m}"] = float(np.mean(np.abs(p - h)))
        out[f"bias_{m}"] = float(np.mean(p) - np.mean(h))
    out["n_words_all"] = int(len(arrays["pred_trt"]))
    return out


def dump_perword(arrays, out_path: Path) -> None:
    cols = ["sentence_idx", "word_position", "word",
            "pred_trt", "pred_ffd", "pred_gaze", "pred_skip",
            "human_trt", "human_ffd", "human_gaze", "human_skip",
            "L1", "L2", "race_logit", "residual_logit"]
    n = len(arrays["pred_trt"])
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(n):
            w.writerow([arrays[c][i] for c in cols])


def eval_variant(model, label: str, seed: int, device, subtlex,
                 save_perword: bool) -> dict:
    out = {"model": label, "seed": seed, "skip_align": "next",
           "skip_population": "words 1..L-1", "datasets": {}}
    for corpus, data in (("geco_test", load_geco_aggregated("test")),
                         ("provo", load_provo_aggregated())):
        print(f"   {corpus}: predicting...")
        arrays, _ = eval_predictions_on_aggregated(model, data, device, subtlex)
        block = time_summary(arrays)
        sp, st = next_aligned_pairs(
            arrays["pred_skip"], arrays["human_skip"], arrays["word_position"],
        )
        block.update(skip_summary(sp, st))
        out["datasets"][corpus] = block
        if save_perword:
            dump_perword(arrays, PW_DIR / f"{label}_seed{seed}_{corpus}.csv")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--skip_no_ai", action="store_true",
                        help="skip the no-LM (heads-zeroed) variant")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    subtlex = load_subtlex()

    for seed in args.seeds:
        for no_ai in ([False] if args.skip_no_ai else [False, True]):
            label = "v4c_v3_dualctx_next" + ("_no_ai" if no_ai else "")
            out_path = RAW_DIR / f"{label}_seed{seed}.json"
            if out_path.exists() and not args.force:
                print(f">> {label} seed={seed}: exists, skipping")
                continue
            t0 = time.time()
            print(f"\n>> {label} seed={seed}: loading checkpoint")
            model = load_v3(seed, device)
            if no_ai:
                zero_ai_heads(model)
            payload = eval_variant(model, label, seed, device, subtlex,
                                   save_perword=not no_ai)
            out_path.write_text(json.dumps(payload, indent=2, default=float))
            print(f">> wrote {out_path.name} ({time.time()-t0:.1f}s)")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
