"""
Evaluate the v4c_v3_surp (H3 ablation, skip_align=next) checkpoints on
GECO test AND Provo, with the same protocol as eval_v4c_v3_seeds.py
(time metrics on all words; skip next-aligned on words 1..L-1).

The surp model's forward needs precomputed TinyLlama surprisals:
    data/cache/tinyllama_surprisal_geco_test.pt
    data/cache/tinyllama_surprisal_provo.pt
Keys: (text_id, sentence_number). The script ABORTS if fewer than 90%
of sentences hit the cache (a silent miss would zero the surprisal term
and corrupt the ablation).

Outputs per seed:
    results/raw/v4c_v3_surp_next_seed<N>.json

Usage:
    python -u eval_v4c_v3_surp_seeds.py
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
SRC_V2 = os.path.abspath(os.path.join(_HERE, "..", ".."))
REPO_ROOT = os.path.dirname(SRC_V2)
LM_MODEL = os.path.join(SRC_V2, "lm_model")
ORIG_EZ = os.path.join(REPO_ROOT, "archive", "original_ezreader")
CACHE_DIR = os.path.join(REPO_ROOT, "data", "cache")

for p in (SRC_V2, LM_MODEL, ORIG_EZ, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from paper_experiments.utils.load_data import (  # noqa: E402
    load_geco_aggregated, load_provo_aggregated, load_subtlex, word_frequency,
)
from model_llama_hybrid_v4c_v3_surp import NeuralEZReaderHybrid  # noqa: E402

from skip_metrics import next_aligned_pairs, skip_summary  # noqa: E402


SEEDS = [1, 2, 3, 42, 100]
CKPT_TMPL = os.path.join(
    REPO_ROOT, "checkpoints", "hybrid_v4c_v3_surp_next",
    "geco_TinyLlama_TinyLlama-1.1B-Chat-v1.0_seed{seed}", "best_model.pt",
)

RAW_DIR = Path(_HERE) / "results" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def load_surp_model(seed: int, device: torch.device):
    path = CKPT_TMPL.format(seed=seed)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing v4c_v3_surp checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = NeuralEZReaderHybrid(
        model_name=ckpt["model_name"],
        freeze_layers=ckpt["freeze_layers"],
        hidden_dim=ckpt.get("hidden_dim", 256),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def surp_lookup(sentence, surp_cache):
    key = (sentence.text_id, getattr(sentence, "sentence_number", 0))
    return surp_cache.get(key)


def eval_corpus(model, sentences, surp_cache, device, subtlex):
    """Per-sentence forward; returns flat arrays + word positions."""
    out = {k: [] for k in ("pred_trt", "pred_ffd", "pred_gaze", "pred_skip",
                           "h_trt", "h_ffd", "h_gaze", "h_skip", "pos")}
    hits = misses = 0
    with torch.no_grad():
        for s in sentences:
            n = len(s.tokens)
            if n == 0:
                continue
            sp = surp_lookup(s, surp_cache)
            if sp is None:
                misses += 1
                sp = np.zeros(n, dtype=np.float32)
            else:
                hits += 1
            freqs = torch.tensor(
                [float(word_frequency(t, subtlex)) for t in s.tokens],
                dtype=torch.float32).unsqueeze(0).to(device)
            wlens = torch.tensor([len(t) for t in s.tokens],
                                 dtype=torch.float32).unsqueeze(0).to(device)
            surps = torch.tensor(np.asarray(sp, dtype=np.float32)
                                 ).unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                p = model([s.tokens], freqs, wlens, surps)
            out["pred_trt"].extend(p["conditional_trt"][0, :n].cpu().tolist())
            out["pred_ffd"].extend(p["first_fixation"][0, :n].cpu().tolist())
            out["pred_gaze"].extend(p["gaze_duration"][0, :n].cpu().tolist())
            out["pred_skip"].extend(p["skip_prob"][0, :n].cpu().tolist())
            out["h_trt"].extend(s.mean_trt)
            out["h_ffd"].extend(s.mean_ffd)
            out["h_gaze"].extend(s.mean_gaze)
            out["h_skip"].extend(s.skip_rate)
            out["pos"].extend(range(n))

    total = hits + misses
    hit_rate = hits / max(1, total)
    print(f"   surprisal cache hit-rate: {hits}/{total} = {hit_rate:.1%}")
    if hit_rate < 0.9:
        raise RuntimeError(
            f"Surprisal cache hit-rate {hit_rate:.1%} < 90% — keys mismatch; "
            f"aborting to avoid a silently-zeroed surprisal term.")
    return {k: np.asarray(v) for k, v in out.items()}


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) > 2 and a.std() > 0 and b.std() > 0:
        return float(np.corrcoef(a, b)[0, 1])
    return 0.0


def block_for(arr) -> dict:
    out = {}
    for m in ("trt", "ffd", "gaze"):
        p, h = arr[f"pred_{m}"], arr[f"h_{m}"]
        out[f"r_{m}"] = corr(p, h)
        out[f"mae_{m}"] = float(np.mean(np.abs(p - h)))
        out[f"bias_{m}"] = float(np.mean(p) - np.mean(h))
    sp, st = next_aligned_pairs(arr["pred_skip"], arr["h_skip"], arr["pos"])
    out.update(skip_summary(sp, st))
    out["n_words_all"] = int(len(arr["pred_trt"]))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    subtlex = load_subtlex()

    surp_geco = torch.load(os.path.join(
        CACHE_DIR, "tinyllama_surprisal_geco_test.pt"), weights_only=False)
    surp_provo = torch.load(os.path.join(
        CACHE_DIR, "tinyllama_surprisal_provo.pt"), weights_only=False)
    print(f"Caches: geco_test {len(surp_geco)} | provo {len(surp_provo)} sentences")

    geco_test = load_geco_aggregated("test")
    provo = load_provo_aggregated()

    for seed in args.seeds:
        out_path = RAW_DIR / f"v4c_v3_surp_next_seed{seed}.json"
        if out_path.exists() and not args.force:
            print(f">> seed {seed}: exists, skipping")
            continue
        t0 = time.time()
        print(f"\n>> v4c_v3_surp_next seed={seed}: loading checkpoint")
        model = load_surp_model(seed, device)
        payload = {"model": "v4c_v3_surp_next", "seed": seed,
                   "skip_align": "next", "skip_population": "words 1..L-1",
                   "datasets": {}}
        for corpus, data, cache in (("geco_test", geco_test, surp_geco),
                                    ("provo", provo, surp_provo)):
            print(f"   {corpus}: predicting...")
            arr = eval_corpus(model, data, cache, device, subtlex)
            payload["datasets"][corpus] = block_for(arr)
        out_path.write_text(json.dumps(payload, indent=2, default=float))
        print(f">> wrote {out_path.name} ({time.time()-t0:.1f}s)")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
