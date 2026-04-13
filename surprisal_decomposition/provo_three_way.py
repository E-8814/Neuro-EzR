"""
Three-way surprisal decomposition on Provo.

For every word in the Provo test split, compute four quantities:
  1. L1               — the trained model's learned familiarity check time
  2. LLaMA surprisal  — -log P(word | context) from the SAME LLaMA base
                        the model was built on (circular, shared encoder)
  3. GPT-2 surprisal  — -log P(word | context) from a completely
                        independent GPT-2 (no shared weights)
  4. Human cloze surp — -log(cloze_prob) from Provo's OrthographicMatch
                        column (actual humans guessing words in context)

Then correlate L1 against each surprisal source on the same word set
and produce:
  - results .txt  with correlation table, partial correlations, quintiles
  - scatter plots for L1 vs each surprisal (3 subplots)
  - bar chart of the three correlation magnitudes
  - quintile plot: mean L1 by surprisal bin, one line per source

Multiple seeds are supported via --checkpoints (pass more than one
checkpoint path) — the script reports mean ± std across runs.

Usage:
  python3 -u surprisal_decomposition/provo_three_way.py \
      --checkpoints checkpoints/faithful_sh/geco_TinyLlama_TinyLlama-1.1B-Chat-v1.0/best_model.pt \
      --llama_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
      --gpt2_model gpt2 \
      --output_dir surprisal_decomposition/out
"""

import argparse
import math
import os
import sys
from typing import List

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src_v2"))
sys.path.insert(0, os.path.join(ROOT, "src_v2", "lm_model"))


# --------------------------------------------------------------------------- #
#  Data loading
# --------------------------------------------------------------------------- #

def load_provo_test_split(data_dir: str):
    """Load Provo eyetracking, aggregate by sentence, return test split only."""
    from data_loader import load_provo, aggregate_by_sentence

    et_path = os.path.join(data_dir, "Provo_Corpus-Eyetracking_Data.csv")
    print(f"Loading Provo from {et_path}")
    raw = load_provo(et_path)
    agg = aggregate_by_sentence(raw, min_participants=10)

    # Reproduce the same 70/15/15 split-by-text_id the training code uses
    text_ids = sorted({a.text_id for a in agg})
    rng = np.random.RandomState(42)
    shuffled = text_ids.copy()
    rng.shuffle(shuffled)
    n_train = int(0.70 * len(shuffled))
    n_val = int(0.15 * len(shuffled))
    test_ids = set(shuffled[n_train + n_val:])
    test = [a for a in agg if a.text_id in test_ids]
    print(f"  Aggregated sentences: {len(agg)} | Test: {len(test)}")
    return test


def flatten(eval_data):
    """Flatten AggregatedSentence list into per-word arrays + sentence structure."""
    word_lists, preds, wlens = [], [], []
    for a in eval_data:
        word_lists.append(a.tokens)
        preds.append(a.predictabilities)
        wlens.append([len(t) for t in a.tokens])
    return word_lists, preds, wlens


# --------------------------------------------------------------------------- #
#  Causal LM surprisal
# --------------------------------------------------------------------------- #

def compute_surprisal(word_lists, model_name, device, batch_size=8):
    """-log P(word | context) per word. Multi-subword: sum subword surprisals."""
    print(f"  Loading causal LM: {model_name}")
    lm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lm.eval()

    out = []
    with torch.no_grad():
        for i in range(0, len(word_lists), batch_size):
            batch = word_lists[i:i + batch_size]
            enc = tok(
                batch,
                is_split_into_words=True,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            logits = lm(input_ids=input_ids, attention_mask=attention_mask).logits
            log_probs = torch.log_softmax(logits, dim=-1)

            for b in range(len(batch)):
                word_ids = enc.word_ids(batch_index=b)
                n_words = len(batch[b])
                surps = [0.0] * n_words
                has_any = [False] * n_words
                for tok_idx in range(1, len(word_ids)):
                    w_id = word_ids[tok_idx]
                    if w_id is None or w_id >= n_words:
                        continue
                    token_id = input_ids[b, tok_idx].item()
                    token_logprob = log_probs[b, tok_idx - 1, token_id].item()
                    surps[w_id] += -token_logprob
                    has_any[w_id] = True
                # Mark untokenised positions as NaN so they are dropped downstream
                out.append([s if h else float("nan") for s, h in zip(surps, has_any)])

    del lm
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------- #
#  Extract L1 from a trained checkpoint
# --------------------------------------------------------------------------- #

def extract_l1(word_lists, preds, wlens, ckpt_path, device, batch_size=8):
    from model_llama_faithful_sh import NeuralEZReaderLLaMA

    print(f"  Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = NeuralEZReaderLLaMA(
        model_name=ckpt.get("model_name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        freeze_layers=ckpt.get("freeze_layers", 16),
        hidden_dim=ckpt.get("hidden_dim", 256),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    delta_val = float(torch.sigmoid(model._delta_raw).item())
    l1_scale_val = float(model.l1_scale.item())
    print(f"    delta={delta_val:.4f}  l1_scale={l1_scale_val:.2f}")

    l1_out = []
    with torch.no_grad():
        for i in range(0, len(word_lists), batch_size):
            batch_w = word_lists[i:i + batch_size]
            batch_p = preds[i:i + batch_size]
            batch_l = wlens[i:i + batch_size]

            pred_t = pad_sequence(
                [torch.tensor(p, dtype=torch.float32) for p in batch_p],
                batch_first=True,
            ).to(device)
            wlen_t = pad_sequence(
                [torch.tensor(w, dtype=torch.float32) for w in batch_l],
                batch_first=True,
            ).to(device)

            pred = model(batch_w, pred_t, wlen_t)
            for b in range(len(batch_w)):
                n = len(batch_w[b])
                l1_out.append(pred["L1"][b, :n].cpu().tolist())

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return l1_out, {"delta": delta_val, "l1_scale": l1_scale_val}


# --------------------------------------------------------------------------- #
#  Stats
# --------------------------------------------------------------------------- #

def pearson(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan"), 0
    r = float(np.corrcoef(x, y)[0, 1])
    # Two-sided p-value from Fisher transformation
    n = len(x)
    if abs(r) >= 1.0:
        p = 0.0
    else:
        t = r * math.sqrt((n - 2) / (1 - r * r))
        # Normal approximation (good for our sample sizes)
        from math import erf
        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / math.sqrt(2))))
    return r, float(p), int(n)


def partial_corr(x, y, z):
    """Partial correlation x~y controlling for z."""
    from numpy.linalg import lstsq
    x, y, z = map(lambda a: np.asarray(a, dtype=np.float64), (x, y, z))
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    if len(x) < 10:
        return float("nan")
    Z = np.column_stack([z, np.ones(len(z))])
    rx = x - Z @ lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ lstsq(Z, y, rcond=None)[0]
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def quintile_means(x, y, n_bins=5):
    """Mean y per quintile of x. Returns (bin_centers, means, counts)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    qs = np.percentile(x, np.linspace(0, 100, n_bins + 1))
    centers, means, counts = [], [], []
    for i in range(n_bins):
        lo, hi = qs[i], qs[i + 1]
        if i == n_bins - 1:
            m = (x >= lo) & (x <= hi)
        else:
            m = (x >= lo) & (x < hi)
        if m.sum() > 0:
            centers.append(0.5 * (lo + hi))
            means.append(float(np.mean(y[m])))
            counts.append(int(m.sum()))
    return np.array(centers), np.array(means), np.array(counts)


# --------------------------------------------------------------------------- #
#  Plotting
# --------------------------------------------------------------------------- #

def scatter_plots(l1, surps_dict, out_path):
    fig, axes = plt.subplots(1, len(surps_dict), figsize=(5 * len(surps_dict), 4.5), sharey=True)
    if len(surps_dict) == 1:
        axes = [axes]
    for ax, (name, s) in zip(axes, surps_dict.items()):
        x = np.asarray(s)
        y = np.asarray(l1)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        ax.scatter(x, y, s=6, alpha=0.25, edgecolors="none")
        # Linear fit line
        if len(x) > 2 and np.std(x) > 0:
            beta = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, beta[0] * xs + beta[1], color="red", lw=1.5)
        r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan")
        ax.set_title(f"L1 vs {name}\nr = {r:.3f}  (n={len(x)})")
        ax.set_xlabel(f"{name} (nats)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("L1 (ms)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def bar_plot(corrs_mean, corrs_std, out_path):
    names = list(corrs_mean.keys())
    means = [corrs_mean[n] for n in names]
    stds = [corrs_std.get(n, 0.0) for n in names]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=["#6a8eae", "#c47e4a", "#74a37c"])
    ax.set_ylabel("Pearson r (L1 vs surprisal source)")
    ax.set_title("L1 alignment across surprisal sources (Provo)")
    ax.axhline(0, color="black", lw=0.8)
    ax.grid(axis="y", alpha=0.3)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{m:.3f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def quintile_plot(l1, surps_dict, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colors = ["#6a8eae", "#c47e4a", "#74a37c"]
    for (name, s), c in zip(surps_dict.items(), colors):
        centers, means, counts = quintile_means(s, l1, n_bins=5)
        ax.plot(range(1, len(means) + 1), means, "-o", label=name, color=c, lw=2)
    ax.set_xlabel("Surprisal quintile (1 = easiest, 5 = hardest)")
    ax.set_ylabel("Mean L1 (ms)")
    ax.set_title("Mean L1 by surprisal quintile, Provo test")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(range(1, 6))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="One or more best_model.pt paths. Multiple = seed runs.")
    p.add_argument("--llama_model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--gpt2_model", type=str, default="gpt2")
    p.add_argument("--data_dir", type=str, default=os.path.join(ROOT, "data"))
    p.add_argument("--output_dir", type=str, default=os.path.join(HERE, "out_provo_threeway"))
    p.add_argument("--cloze_floor_n", type=int, default=40,
                   help="Additive smoothing N for zero-cloze words (surprisal floor = log(N+1))")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load Provo test split ---
    eval_data = load_provo_test_split(args.data_dir)
    word_lists, preds, wlens = flatten(eval_data)
    n_words_total = sum(len(w) for w in word_lists)
    print(f"  Total words in Provo test: {n_words_total:,}")

    # --- Surprisal sources (compute once, reuse for all seeds) ---
    print("\nComputing LLaMA surprisal")
    llama_surp = compute_surprisal(word_lists, args.llama_model, device)

    print("\nComputing GPT-2 surprisal")
    gpt2_surp = compute_surprisal(word_lists, args.gpt2_model, device)

    print("\nComputing human cloze surprisal from Provo predictabilities")
    # preds already hold OrthographicMatch (cloze probability). Apply additive
    # smoothing so zero-cloze words get a finite (large) surprisal value.
    eps = 1.0 / (args.cloze_floor_n + 1)
    human_surp = [[-math.log(max(c, eps)) for c in row] for row in preds]

    # --- Flatten to per-word arrays ---
    def flat(xs):
        return np.array([v for row in xs for v in row], dtype=np.float64)

    flat_llama = flat(llama_surp)
    flat_gpt2 = flat(gpt2_surp)
    flat_human = flat(human_surp)
    flat_pred = flat(preds)
    flat_wlen = flat(wlens)

    # --- Per-checkpoint L1 extraction ---
    per_seed = []
    all_l1 = []
    learned_params = []
    for ckpt in args.checkpoints:
        print(f"\n=== Checkpoint: {ckpt} ===")
        l1_lists, params = extract_l1(word_lists, preds, wlens, ckpt, device)
        l1_arr = flat(l1_lists)
        all_l1.append(l1_arr)
        learned_params.append(params)

        # Correlations for THIS seed
        r_llama, p_llama, n_l = pearson(l1_arr, flat_llama)
        r_gpt2, p_gpt2, n_g = pearson(l1_arr, flat_gpt2)
        r_human, p_human, n_h = pearson(l1_arr, flat_human)

        # Partial correlation: L1 vs human cloze controlling for GPT-2 surprisal.
        # Asks whether human cloze contributes signal beyond neural LM surprisal.
        r_human_over_gpt2 = partial_corr(l1_arr, flat_human, flat_gpt2)

        per_seed.append({
            "ckpt": ckpt,
            "r_llama": r_llama, "p_llama": p_llama, "n": n_l,
            "r_gpt2": r_gpt2, "p_gpt2": p_gpt2,
            "r_human": r_human, "p_human": p_human,
            "r_human_partial": r_human_over_gpt2,
            "params": params,
        })
        print(f"  r(L1, LLaMA surp)  = {r_llama:.4f}  (n={n_l}, p={p_llama:.2e})")
        print(f"  r(L1, GPT-2 surp)  = {r_gpt2:.4f}  (p={p_gpt2:.2e})")
        print(f"  r(L1, human cloze) = {r_human:.4f}  (p={p_human:.2e})")
        print(f"  partial r(L1, human | GPT-2) = {r_human_over_gpt2:.4f}")

    # --- Aggregate across seeds ---
    mean_l1 = np.nanmean(np.stack(all_l1), axis=0)

    r_llama_mean = float(np.mean([s["r_llama"] for s in per_seed]))
    r_gpt2_mean = float(np.mean([s["r_gpt2"] for s in per_seed]))
    r_human_mean = float(np.mean([s["r_human"] for s in per_seed]))
    r_llama_std = float(np.std([s["r_llama"] for s in per_seed]))
    r_gpt2_std = float(np.std([s["r_gpt2"] for s in per_seed]))
    r_human_std = float(np.std([s["r_human"] for s in per_seed]))

    # --- Cross-source sanity: how aligned are the three surprisals themselves ---
    r_llama_gpt2, _, _ = pearson(flat_llama, flat_gpt2)
    r_llama_human, _, _ = pearson(flat_llama, flat_human)
    r_gpt2_human, _, _ = pearson(flat_gpt2, flat_human)

    # --- Quintile table using averaged L1 ---
    quintile_data = {
        "LLaMA surprisal": quintile_means(flat_llama, mean_l1, 5),
        "GPT-2 surprisal": quintile_means(flat_gpt2, mean_l1, 5),
        "Human cloze surprisal": quintile_means(flat_human, mean_l1, 5),
    }

    # --- Write results file ---
    txt_path = os.path.join(args.output_dir, "results.txt")
    with open(txt_path, "w") as f:
        f.write("Three-way surprisal decomposition on Provo test split\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"n checkpoints (seeds): {len(args.checkpoints)}\n")
        for c in args.checkpoints:
            f.write(f"  - {c}\n")
        f.write(f"LLaMA base (for surprisal): {args.llama_model}\n")
        f.write(f"GPT-2 model: {args.gpt2_model}\n")
        f.write(f"n words (Provo test): {n_words_total}\n")
        f.write(f"n words with finite L1: {int(np.isfinite(mean_l1).sum())}\n\n")

        f.write("-- Learned cognitive parameters per seed --\n")
        for i, params in enumerate(learned_params):
            f.write(f"  seed {i}: delta={params['delta']:.4f}  l1_scale={params['l1_scale']:.2f}\n")
        f.write("\n")

        f.write("-- Main correlations (mean ± std across seeds) --\n")
        f.write(f"  r(L1, LLaMA surprisal)       = {r_llama_mean:+.4f} ± {r_llama_std:.4f}\n")
        f.write(f"  r(L1, GPT-2 surprisal)       = {r_gpt2_mean:+.4f} ± {r_gpt2_std:.4f}\n")
        f.write(f"  r(L1, human cloze surprisal) = {r_human_mean:+.4f} ± {r_human_std:.4f}\n\n")

        f.write("-- Per-seed correlations with p-values --\n")
        for i, s in enumerate(per_seed):
            f.write(f"  seed {i}:\n")
            f.write(f"    LLaMA  r={s['r_llama']:+.4f}  p={s['p_llama']:.2e}\n")
            f.write(f"    GPT-2  r={s['r_gpt2']:+.4f}  p={s['p_gpt2']:.2e}\n")
            f.write(f"    human  r={s['r_human']:+.4f}  p={s['p_human']:.2e}\n")
            f.write(f"    partial r(L1, human | GPT-2) = {s['r_human_partial']:+.4f}\n")
        f.write("\n")

        f.write("-- Cross-source agreement between surprisal measures --\n")
        f.write(f"  r(LLaMA, GPT-2)        = {r_llama_gpt2:+.4f}\n")
        f.write(f"  r(LLaMA, human cloze)  = {r_llama_human:+.4f}\n")
        f.write(f"  r(GPT-2, human cloze)  = {r_gpt2_human:+.4f}\n\n")

        f.write("-- Quintile means: averaged L1 by surprisal quintile --\n")
        for name, (centers, means, counts) in quintile_data.items():
            f.write(f"  {name}:\n")
            for q, (m, c) in enumerate(zip(means, counts), start=1):
                f.write(f"    Q{q}  mean_L1={m:7.2f} ms  n={c}\n")
        f.write("\n")
    print(f"\nWrote results: {txt_path}")

    # --- Plots ---
    surps_dict = {
        "LLaMA surprisal": flat_llama,
        "GPT-2 surprisal": flat_gpt2,
        "Human cloze surp.": flat_human,
    }
    scatter_plots(mean_l1, surps_dict, os.path.join(args.output_dir, "scatter_l1_vs_surprisal.png"))
    bar_plot(
        {"LLaMA": r_llama_mean, "GPT-2": r_gpt2_mean, "Human cloze": r_human_mean},
        {"LLaMA": r_llama_std, "GPT-2": r_gpt2_std, "Human cloze": r_human_std},
        os.path.join(args.output_dir, "bar_correlations.png"),
    )
    quintile_plot(mean_l1, surps_dict, os.path.join(args.output_dir, "quintile_mean_l1.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
