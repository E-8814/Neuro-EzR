"""
Extract per-word features from the trained dualctx model for all words
in GECO test + Provo. The output CSV is the source of truth for all
downstream analyses in exp10.

Per word, records:
    corpus            "geco_test" or "provo"
    sentence_idx      sentence index within corpus
    word_position     position within sentence (0-based)
    word              the word string
    ctx_FFD           v4c_v2_dualctx ctx_head_FFD output (ms scale)
    ctx_skip          v4c_v2_dualctx ctx_head_skip output (ms scale)
    base_L1_FFD       formula + ctx_FFD (after softplus floor)
    base_L1_skip      formula + ctx_skip (after softplus floor)
    L1                final L1 used for FFD (after eccentricity)
    L2                δ * base_L1_FFD
    pred_TRT, pred_FFD, pred_Gaze, pred_skip   model outputs
    log_freq          log SUBTLEX frequency
    log_freq_norm     centered/scaled log_freq used by the model
    word_length       chars
    surprisal         TinyLlama per-word surprisal (nats)
    position_in_sentence  word_position / (n_words - 1) — relative position
    h_TRT, h_FFD, h_Gaze, h_skip              human (aggregated) targets

Usage:
    python extract_per_word_features.py
    python extract_per_word_features.py --seed 42 --corpora geco_test provo
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
PER_WORD_CSV = RESULTS_DIR / "per_word_dualctx.csv"


def _dualctx_ckpt_path(seed=42):
    return (
        config.CHECKPOINTS_DIR
        / "hybrid_v4c_v2_dualctx"
        / f"geco_{config.BACKBONE_MODEL_SHORT}_seed{seed}"
        / "best_model.pt"
    )


def load_dualctx_model(seed: int, device):
    """Load the trained dualctx model from its best_model.pt checkpoint."""
    from model_llama_hybrid_v4c_v2_dualctx import NeuralEZReaderHybrid

    ckpt_path = _dualctx_ckpt_path(seed)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"DualCtx checkpoint not found: {ckpt_path}\n"
            f"Train it first via train_hybrid_v4c_v2_dualctx_geco.py."
        )

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = NeuralEZReaderHybrid(
        model_name=ckpt.get("model_name", config.BACKBONE_MODEL),
        freeze_layers=ckpt.get("freeze_layers", config.FREEZE_LAYERS),
        hidden_dim=ckpt.get("hidden_dim", 256),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded dualctx checkpoint: epoch {ckpt.get('epoch')}, "
          f"val_step {ckpt.get('val_step')}")
    print(f"  val_metrics: r_TRT={ckpt['val_metrics']['r_trt']:.3f}, "
          f"r_FFD={ckpt['val_metrics']['r_ffd']:.3f}, "
          f"r_Gaze={ckpt['val_metrics']['r_gaze']:.3f}, "
          f"r_skip={ckpt['val_metrics']['r_skip']:.3f}")
    return model, ckpt


def _model_forward_one_sentence(model, sentence, device, subtlex):
    """Run dualctx model on a single aggregated sentence, return prediction dict."""
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

    print(f"\nLoading dualctx model (seed={args.seed})...")
    model, _ = load_dualctx_model(args.seed, device)

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
        elif corpus_name == "provo":
            sentences = load_provo_aggregated()
        else:
            continue
        print(f"\n>> {corpus_name}: {len(sentences)} sentences")

        for sent_idx, s in enumerate(sentences):
            if sent_idx % 50 == 0:
                print(f"  [{corpus_name}] {sent_idx}/{len(sentences)}",
                      flush=True)
            n = len(s.tokens)
            if n == 0:
                continue

            # Dualctx model outputs
            p = _model_forward_one_sentence(model, s, device, subtlex)

            # Per-word TinyLlama surprisal (one forward pass per sentence)
            surps = per_word_surprisal_from_subword(
                s.tokens, tokenizer, causal_lm, device,
            )

            for i, tok in enumerate(s.tokens):
                freq = word_frequency(tok, subtlex)
                log_freq = math.log(max(freq, 1.0))
                log_freq_norm = (log_freq - 10.0) / 5.0
                pos = i / max(1, n - 1)
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
                    "pred_skip": float(p["skip_prob"][0, i].item()),
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

    # Quick sanity print
    arr_FFD = np.array([r["ctx_FFD"] for r in rows])
    arr_skip = np.array([r["ctx_skip"] for r in rows])
    print(f"\nQuick stats:")
    print(f"  ctx_FFD:    mean={arr_FFD.mean():+.2f}ms  std={arr_FFD.std():.2f}  |·|={np.abs(arr_FFD).mean():.2f}")
    print(f"  ctx_skip:   mean={arr_skip.mean():+.2f}ms  std={arr_skip.std():.2f}  |·|={np.abs(arr_skip).mean():.2f}")
    print(f"  r(ctx_FFD, ctx_skip) = {np.corrcoef(arr_FFD, arr_skip)[0, 1]:+.3f}")


if __name__ == "__main__":
    main()
