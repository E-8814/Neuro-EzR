"""
Evaluate all model checkpoints (paper model × seeds + each baseline × seeds)
on GECO test + Provo. Writes one JSON per (model, seed, dataset) combination
into results/raw/.

Per-baseline evaluation logic in this script handles only the paper-model
family. Pre-existing baseline evaluation is delegated to
`src_v2/evaluation/eval_all_models_v2.py` for the heavy lifting where
applicable; for the paper-model checkpoints we evaluate directly using
`utils.eval_metrics`.

Usage:
    python eval_all_models.py
    python eval_all_models.py --models v4c_v2 ohio_state_roberta
    python eval_all_models.py --seeds 42
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_HERE, ".."))

from paper_experiments import config
from paper_experiments.utils.load_data import (
    load_geco_aggregated,
    load_provo_aggregated,
    load_subtlex,
)
from paper_experiments.utils.load_model import load_paper_model
from paper_experiments.utils.eval_metrics import eval_predictions_on_aggregated


RESULTS_DIR = Path(_HERE) / "results"
RAW_DIR = RESULTS_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def eval_paper_model(seed: int, datasets: dict, subtlex, device):
    """Evaluate the paper model checkpoint at the given seed."""
    model, ckpt = load_paper_model(seed=seed, device=device)
    model_name = config.PAPER_MODEL_RECIPE

    out = {}
    for ds_name, ds_data in datasets.items():
        _, summary = eval_predictions_on_aggregated(
            model, ds_data, device, subtlex
        )
        out[ds_name] = summary

    # Free memory
    del model
    torch.cuda.empty_cache()
    return model_name, out, ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="Subset of models to eval. Default: paper model only "
             "(baselines handled by their own eval scripts).",
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=None,
                        help="Subset of seeds (default: all from config.SEEDS)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    seeds = args.seeds or config.SEEDS
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    print("Loading data...")
    subtlex = load_subtlex()
    geco_test = load_geco_aggregated("test")
    provo = load_provo_aggregated()
    print(f"  GECO test: {len(geco_test)} sentences")
    print(f"  Provo:     {len(provo)} sentences")

    datasets = {"geco_test": geco_test, "provo": provo}

    # === Paper model ===
    if args.models is None or config.PAPER_MODEL_RECIPE in args.models:
        for seed in seeds:
            out_path = RAW_DIR / f"{config.PAPER_MODEL_RECIPE}_seed{seed}.json"
            if out_path.exists():
                print(f"  [skip] {out_path.name} exists.")
                continue

            try:
                model_name, results, ckpt = eval_paper_model(
                    seed, datasets, subtlex, device
                )
            except FileNotFoundError as e:
                print(f"  [missing checkpoint] seed={seed}: {e}")
                continue

            payload = {
                "model": model_name,
                "seed": seed,
                "epoch": ckpt.get("epoch"),
                "val_step": ckpt.get("val_step"),
                "datasets": results,
                "cog_params": ckpt.get("cog_params"),
            }
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2, default=str)
            print(f"  [done] {model_name} seed={seed} → {out_path.name}")

    print(
        "\nNB: NLP baselines (BERT, RoBERTa, GPT-2, LightGBM, LinReg) are "
        "evaluated separately. The existing src_v2/evaluation/eval_all_models_v2.py "
        "produces compatible-format results which `aggregate.py` reads."
    )


if __name__ == "__main__":
    main()
