"""
v3 version of extract_per_word_features.py: extracts per-word features
from the trained v4c_v3_dualctx (skip_align=next) model for GECO test +
Provo. Output: results/per_word_dualctx_v3.csv.

Columns are the same as the v2 extraction, plus:

    pred_skip_word    the model's skip prediction FOR THIS WORD under the
                      race-faithful alignment: the race computed at row
                      i-1 targets word i, so pred_skip_word[i] =
                      skip_prob[i-1] (empty for sentence-initial words,
                      which the v3 model does not predict).

The raw `pred_skip` column keeps the row quantity (the race computed AT
this word, i.e. P(skip of the NEXT word)) for mechanistic analyses.

Under the v3 'next' alignment, ctx_skip[w] feeds the race that decides
word w's own skip, so ctx_skip is per-word aligned with its own word —
unlike v2, where it influenced the previous word's prediction.

Usage:
    python extract_per_word_features_v3.py --seed 42
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "lm_model"))

from paper_experiments import config
from paper_experiments.utils.load_data import (
    load_geco_aggregated, load_provo_aggregated, load_subtlex,
    word_frequency,
)
from paper_experiments.utils.word_align import per_word_surprisal_from_subword


RESULTS_DIR = Path(_HERE) / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PER_WORD_CSV = RESULTS_DIR / "per_word_dualctx_v3.csv"


def _v3_ckpt_path(seed=42):
    return (
        config.CHECKPOINTS_DIR
        / "hybrid_v4c_v3_dualctx_next"
        / f"geco_{config.BACKBONE_MODEL_SHORT}_seed{seed}"
        / "best_model.pt"
    )


def load_v3_model(seed: int, device):
    from model_llama_hybrid_v4c_v3_dualctx import NeuralEZReaderHybrid

    ckpt_path = _v3_ckpt_path(seed)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"v3 checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = NeuralEZReaderHybrid(
        model_name=ckpt.get("model_name", config.BACKBONE_MODEL),
        freeze_layers=ckpt.get("freeze_layers", config.FREEZE_LAYERS),
        hidden_dim=ckpt.get("hidden_dim", 256),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded v3 checkpoint (skip_align={ckpt.get('skip_align')}): "
          f"epoch {ckpt.get('epoch')}, val_step {ckpt.get('val_step')}")
    return model, ckpt


def _model_forward_one_sentence(model, sentence, device, subtlex):
    tokens = sentence.tokens
    freqs = torch.tensor(
        [float(word_frequency(t, subtlex)) for t in tokens],
        dtype=torch.float32,
    ).unsqueeze(0).to(device)
    wlens = torch.tensor(
        [len(t) for t in tokens], dtype=torch.float32,
    ).unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            return model([tokens], freqs, wlens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--corpora", nargs="+",
                        default=["geco_test", "provo"],
                        choices=["geco_test", "provo"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\nLoading v4c_v3_dualctx_next model (seed={args.seed})...")
    model, _ = load_v3_model(args.seed, device)

    print(f"\nLoading TinyLlama causal LM for surprisal computation...")
    tokenizer = AutoTokenizer.from_pretrained(config.BACKBONE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    causal_lm = AutoModelForCausalLM.from_pretrained(
        config.BACKBONE_MODEL, torch_dtype=torch.float32,
    ).to(device)
    causal_lm.eval()

    subtlex = load_subtlex()

    rows = []
    for corpus_name in args.corpora:
        if corpus_name == "geco_test":
            sentences = load_geco_aggregated("test")
        else:
            sentences = load_provo_aggregated()
        print(f"\n>> {corpus_name}: {len(sentences)} sentences")

        for sent_idx, s in enumerate(sentences):
            if sent_idx % 50 == 0:
                print(f"  [{corpus_name}] {sent_idx}/{len(sentences)}",
                      flush=True)
            n = len(s.tokens)
            if n == 0:
                continue

            p = _model_forward_one_sentence(model, s, device, subtlex)
            surps = per_word_surprisal_from_subword(
                s.tokens, tokenizer, causal_lm, device,
            )

            skip_row = p["skip_prob"][0].float().cpu().numpy()

            for i, tok in enumerate(s.tokens):
                freq = word_frequency(tok, subtlex)
                log_freq = math.log(max(freq, 1.0))
                log_freq_norm = (log_freq - 10.0) / 5.0
                pos = i / max(1, n - 1)
                # race-faithful prediction for THIS word lives at row i-1
                pred_skip_word = float(skip_row[i - 1]) if i > 0 else ""
                rows.append({
                    "corpus": corpus_name,
                    "sentence_idx": sent_idx,
                    "word_position": i,
                    "word": tok,
                    "ctx_FFD": float(p["ctx_FFD"][0, i].item()),
                    "ctx_skip": float(p["ctx_skip"][0, i].item()),
                    "base_L1_FFD": float(p["base_L1_FFD"][0, i].item()),
                    "base_L1_skip": float(p["base_L1_skip"][0, i].item()),
                    "L1": float(p["L1"][0, i].item()),
                    "L2": float(p["L2"][0, i].item()),
                    "pred_TRT": float(p["conditional_trt"][0, i].item()),
                    "pred_FFD": float(p["first_fixation"][0, i].item()),
                    "pred_Gaze": float(p["gaze_duration"][0, i].item()),
                    "pred_skip": float(skip_row[i]),
                    "pred_skip_word": pred_skip_word,
                    "log_freq": log_freq,
                    "log_freq_norm": log_freq_norm,
                    "word_length": len(tok),
                    "surprisal": float(surps[i]),
                    "position_in_sentence": pos,
                    "h_TRT": float(s.mean_trt[i]),
                    "h_FFD": float(s.mean_ffd[i]),
                    "h_Gaze": float(s.mean_gaze[i]),
                    "h_skip": float(s.skip_rate[i]),
                })

    if not rows:
        print("No words extracted. Aborting.")
        return

    fieldnames = list(rows[0].keys())
    with open(PER_WORD_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nWrote {len(rows):,} rows to {PER_WORD_CSV}")

    arr_FFD = np.array([r["ctx_FFD"] for r in rows])
    arr_skip = np.array([r["ctx_skip"] for r in rows])
    print(f"\nQuick stats:")
    print(f"  ctx_FFD:  mean={arr_FFD.mean():+.2f}ms std={arr_FFD.std():.2f}")
    print(f"  ctx_skip: mean={arr_skip.mean():+.2f}ms std={arr_skip.std():.2f}")
    print(f"  r(ctx_FFD, ctx_skip) = {np.corrcoef(arr_FFD, arr_skip)[0, 1]:+.3f}")


if __name__ == "__main__":
    main()
