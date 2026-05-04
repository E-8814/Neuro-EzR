"""
Surprisal decomposition (exp06).

Computes:
    r(L1, surprisal)
    partial r(L1, h_TRT | surprisal + controls)
    ΔR² L1 beyond surprisal

Where:
    surprisal = TinyLlama per-word surprisal (same backbone as paper model)
    L1 = paper model's first-stage processing time per word
    controls = log_freq, word_length

Outputs CSVs ready for the paper Table 3.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from paper_experiments import config
from paper_experiments.utils.load_model import load_paper_model
from paper_experiments.utils.load_data import (
    load_geco_aggregated, load_provo_aggregated, load_subtlex,
    word_frequency,
)
from paper_experiments.utils.word_align import per_word_surprisal_from_subword
from paper_experiments.utils.eval_metrics import corr, partial_corr, delta_r2


RESULTS_DIR = Path(_HERE) / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DECOMP_CSV = RESULTS_DIR / "surprisal_decomp_results.csv"
PER_WORD_CSV = RESULTS_DIR / "per_word_features.csv"


def _model_l1_for_sentence(model, sentence, device, subtlex):
    """Run paper model on a single sentence and return per-word L1, h_TRT, h_FFD."""
    word_lists = [sentence.tokens]
    freqs = torch.tensor(
        [float(word_frequency(t, subtlex)) for t in sentence.tokens],
        dtype=torch.float32,
    ).unsqueeze(0).to(device)
    wlens = torch.tensor(
        [len(t) for t in sentence.tokens], dtype=torch.float32
    ).unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(word_lists, freqs, wlens)
    seq_len = len(sentence.tokens)
    L1 = out['L1'][0, :seq_len].cpu().numpy()
    return L1


def collect_features(model, causal_lm, causal_tokenizer, agg_data, device,
                     subtlex, corpus_name):
    """Walk every aggregated sentence, collect per-word features.

    Returns a list of dict rows."""
    rows = []
    for sent_idx, s in enumerate(agg_data):
        # Per-word L1 from paper model
        L1 = _model_l1_for_sentence(model, s, device, subtlex)
        # Per-word surprisal from causal LM (same TinyLlama)
        surp = per_word_surprisal_from_subword(
            s.tokens, causal_tokenizer, causal_lm, device,
        )
        for i, tok in enumerate(s.tokens):
            freq = word_frequency(tok, subtlex)
            log_freq = np.log(max(freq, 1.0))
            wlen = len(tok)
            rows.append({
                "corpus": corpus_name,
                "sentence_idx": sent_idx,
                "word_position": i,
                "word": tok,
                "L1": float(L1[i]),
                "surprisal": float(surp[i]),
                "log_freq": float(log_freq),
                "word_length": int(wlen),
                "h_TRT": float(s.mean_trt[i]),
                "h_FFD": float(s.mean_ffd[i]),
                "h_Gaze": float(s.mean_gaze[i]),
                "h_Skip": float(s.skip_rate[i]),
            })
    return rows


def compute_statistics(rows, corpus_name):
    """Compute the three statistics on the rows from one corpus."""
    df = {k: np.array([r[k] for r in rows]) for k in
          ["L1", "surprisal", "log_freq", "word_length",
           "h_TRT", "h_FFD"]}

    # Optionally exclude skipped words (skip_rate > 0.5)?
    # For simplicity here, include all words.

    stats = []

    # 1. r(L1, surprisal)
    r1 = corr(df["L1"], df["surprisal"])
    stats.append({
        "corpus": corpus_name,
        "statistic": "r(L1, surprisal)",
        "value": r1,
        "n": len(df["L1"]),
        "interpretation": "L1 correlation with TinyLlama surprisal",
    })

    # 2. partial r(L1, h_TRT | surprisal + log_freq + word_length)
    pr = partial_corr(
        df["L1"], df["h_TRT"],
        controls=[df["surprisal"], df["log_freq"], df["word_length"]],
    )
    stats.append({
        "corpus": corpus_name,
        "statistic": "partial_r(L1, h_TRT | surprisal+controls)",
        "value": pr,
        "n": len(df["L1"]),
        "interpretation": "L1's unique correlation with TRT after controls",
    })

    # 3. ΔR² L1 beyond (surprisal + log_freq + word_length)
    X_baseline = np.column_stack([df["surprisal"], df["log_freq"], df["word_length"]])
    dr2 = delta_r2(df["h_TRT"], X_baseline, df["L1"])
    stats.append({
        "corpus": corpus_name,
        "statistic": "deltaR2_L1_beyond_surprisal_controls",
        "value": dr2,
        "n": len(df["L1"]),
        "interpretation": "Variance L1 adds beyond surprisal + log_freq + word_length",
    })

    # 4. Reverse: deltaR² surprisal beyond (L1 + controls)
    X_baseline2 = np.column_stack([df["L1"], df["log_freq"], df["word_length"]])
    dr2_surp = delta_r2(df["h_TRT"], X_baseline2, df["surprisal"])
    stats.append({
        "corpus": corpus_name,
        "statistic": "deltaR2_surprisal_beyond_L1_controls",
        "value": dr2_surp,
        "n": len(df["L1"]),
        "interpretation": "Variance surprisal adds beyond L1 + controls",
    })

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading paper model...")
    model, _ = load_paper_model(seed=args.seed, device=device)

    print(f"Loading causal LM ({config.BACKBONE_MODEL}) for surprisal...")
    causal_tokenizer = AutoTokenizer.from_pretrained(config.BACKBONE_MODEL)
    causal_lm = AutoModelForCausalLM.from_pretrained(
        config.BACKBONE_MODEL, torch_dtype=torch.float32,
    ).to(device)
    causal_lm.eval()
    if causal_tokenizer.pad_token is None:
        causal_tokenizer.pad_token = causal_tokenizer.eos_token

    subtlex = load_subtlex()

    all_per_word = []
    all_stats = []

    for corpus_name, ds in [
        ("geco_test", load_geco_aggregated("test")),
        ("provo", load_provo_aggregated()),
    ]:
        print(f"\n>> {corpus_name}: {len(ds)} sentences")
        rows = collect_features(
            model, causal_lm, causal_tokenizer, ds, device, subtlex, corpus_name,
        )
        all_per_word.extend(rows)
        stats = compute_statistics(rows, corpus_name)
        all_stats.extend(stats)
        for s in stats:
            print(f"   {s['statistic']:<55s} = {s['value']:+.4f}  (n={s['n']})")

    # Write per-word CSV
    with open(PER_WORD_CSV, "w", newline="") as f:
        if all_per_word:
            fieldnames = list(all_per_word[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_per_word:
                writer.writerow(r)
    print(f"\nWrote {len(all_per_word)} rows to {PER_WORD_CSV}")

    # Write summary CSV
    with open(DECOMP_CSV, "w", newline="") as f:
        if all_stats:
            fieldnames = list(all_stats[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_stats:
                writer.writerow(r)
    print(f"Wrote {len(all_stats)} stats to {DECOMP_CSV}")


if __name__ == "__main__":
    main()
