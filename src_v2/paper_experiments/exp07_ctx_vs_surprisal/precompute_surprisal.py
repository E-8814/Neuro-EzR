"""
Precompute TinyLlama per-word surprisals for GECO + Provo, save to .pt
files so the training loop is fast.

This script runs once. Subsequent training runs read the cached
surprisals and skip the LM forward pass entirely.

Output:
    data/cache/tinyllama_surprisal_geco_train.pt
    data/cache/tinyllama_surprisal_geco_val.pt
    data/cache/tinyllama_surprisal_geco_test.pt
    data/cache/tinyllama_surprisal_provo.pt

Each file is a dict:
    {(text_id, sentence_number): np.ndarray (per-word surprisal in nats)}
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "archive", "original_ezreader"))

from paper_experiments import config
from paper_experiments.utils.word_align import per_word_surprisal_from_subword
from data_loader import aggregate_by_sentence, load_provo
from geco_loader import load_geco, split_geco


CACHE_DIR = config.DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def precompute_for_sentences(sentences, causal_lm, tokenizer, device,
                             label="sentences"):
    """Compute per-word surprisal for every sentence; return dict keyed by
    (text_id, sentence_number) -> np.ndarray."""
    out = {}
    n = len(sentences)
    for i, s in enumerate(sentences):
        if i % 100 == 0:
            print(f"  [{label}] {i}/{n}")
        key = (s.text_id, getattr(s, "sentence_number", i))
        sp = per_word_surprisal_from_subword(
            s.tokens, tokenizer, causal_lm, device,
        )
        out[key] = sp.astype(np.float32)
    print(f"  [{label}] done ({n} sentences)")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=["all", "geco", "provo"], default="all")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if cache file exists.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading {config.BACKBONE_MODEL} (causal LM)...")
    tokenizer = AutoTokenizer.from_pretrained(config.BACKBONE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    causal_lm = AutoModelForCausalLM.from_pretrained(
        config.BACKBONE_MODEL, torch_dtype=torch.float32,
    ).to(device)
    causal_lm.eval()

    if args.corpus in ("all", "geco"):
        print("\n=== GECO ===")
        raw = load_geco(
            str(config.GECO_READING_FILE),
            str(config.GECO_MATERIAL_FILE),
            str(config.GECO_PRED_FILE),
        )
        train_raw, val_raw, test_raw = split_geco(raw)
        agg = aggregate_by_sentence(raw, min_participants=5)
        train_ids = set(s.text_id for s in train_raw)
        val_ids = set(s.text_id for s in val_raw)
        train_agg = [a for a in agg if a.text_id in train_ids]
        val_agg = [a for a in agg if a.text_id in val_ids]
        test_agg = [a for a in agg if a.text_id not in train_ids
                    and a.text_id not in val_ids]

        for split, sentences in [
            ("train", train_agg), ("val", val_agg), ("test", test_agg),
        ]:
            cache_path = CACHE_DIR / f"tinyllama_surprisal_geco_{split}.pt"
            if cache_path.exists() and not args.force:
                print(f"  [skip] {cache_path.name} exists")
                continue
            print(f"  Computing GECO {split} ({len(sentences)} sentences)...")
            surps = precompute_for_sentences(
                sentences, causal_lm, tokenizer, device,
                label=f"geco_{split}",
            )
            torch.save(surps, str(cache_path))
            print(f"  Wrote {cache_path}")

    if args.corpus in ("all", "provo"):
        print("\n=== Provo ===")
        raw_provo = load_provo(str(config.PROVO_FILE))
        provo_agg = aggregate_by_sentence(raw_provo, min_participants=5)
        cache_path = CACHE_DIR / "tinyllama_surprisal_provo.pt"
        if cache_path.exists() and not args.force:
            print(f"  [skip] {cache_path.name} exists")
        else:
            print(f"  Computing Provo ({len(provo_agg)} sentences)...")
            surps = precompute_for_sentences(
                provo_agg, causal_lm, tokenizer, device, label="provo",
            )
            torch.save(surps, str(cache_path))
            print(f"  Wrote {cache_path}")


if __name__ == "__main__":
    main()
