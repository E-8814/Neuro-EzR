"""
Surprisal Decomposition: Does L1 ≈ surprisal?

Extracts per-word surprisal from TinyLlama and correlates it with
the learned L1 values from trained EZ Reader models.

Surprisal = -log P(word | context), computed from the LM's own
next-token distribution. L1 is the model's learned "familiarity
check time" (ms).

If L1 ≈ surprisal:
  → EZ Reader's familiarity stage learns information-theoretic surprise
  → Connects two major theories: staged lexical processing + surprisal theory

If L1 ≠ surprisal:
  → EZ Reader learns something different under eye-tracking supervision
  → What is it? Word length? Morphological complexity? Position effects?

Also analyzes:
  - L1 vs word_length, word_frequency (log), predictability
  - L2 vs surprisal (does integration track surprise differently?)
  - Residual analysis: what does L1 capture beyond surprisal?
  - Partial correlations: unique variance explained by each feature

Usage:
  python3 -u surprisal_decomposition/analyze_surprisal_vs_l1.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
  python3 -u surprisal_decomposition/analyze_surprisal_vs_l1.py --corpus provo
"""

import os
import sys
import argparse
import torch
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src_v2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src_v2', 'lm_model'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'archive', 'original_ezreader'))

from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer


# --------------------------------------------------------------------------- #
#  Compute surprisal from TinyLlama
# --------------------------------------------------------------------------- #

def compute_surprisal(word_lists, model_name, device, batch_size=8):
    """
    Compute per-word surprisal using the causal LM.

    For each word, surprisal = -log P(word | previous context).
    Multi-subword words: sum of subword log-probs (joint probability).

    Returns: list of lists of floats (surprisal per word per sentence)
    """
    print(f"  Loading causal LM: {model_name}")
    lm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    lm.eval()

    all_surprisals = []

    print(f"  Computing surprisal for {len(word_lists)} sentences...")
    with torch.no_grad():
        for i in range(0, len(word_lists), batch_size):
            batch_words = word_lists[i:i + batch_size]

            # Tokenize with word alignment
            encodings = tokenizer(
                batch_words,
                is_split_into_words=True,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = lm(input_ids=input_ids, attention_mask=attention_mask)

            # logits: (batch, seq_len, vocab_size)
            log_probs = torch.log_softmax(outputs.logits, dim=-1)

            for b in range(len(batch_words)):
                word_ids = encodings.word_ids(batch_index=b)
                n_words = len(batch_words[b])
                word_surprisals = [0.0] * n_words

                for tok_idx in range(1, len(word_ids)):
                    w_id = word_ids[tok_idx]
                    if w_id is None or w_id >= n_words:
                        continue
                    # surprisal of this token given previous tokens
                    token_id = input_ids[b, tok_idx].item()
                    token_logprob = log_probs[b, tok_idx - 1, token_id].item()
                    word_surprisals[w_id] += -token_logprob  # sum subword surprisals

                all_surprisals.append(word_surprisals)

            if (i // batch_size + 1) % 50 == 0:
                print(f"    {i + len(batch_words)}/{len(word_lists)} sentences done")

    del lm
    torch.cuda.empty_cache() if device.type == "cuda" else None
    print(f"  Surprisal computed for {len(all_surprisals)} sentences")
    return all_surprisals


# --------------------------------------------------------------------------- #
#  Extract L1/L2 from trained model
# --------------------------------------------------------------------------- #

def extract_l1_l2(word_lists, predictabilities, word_lengths_list,
                  model, device, batch_size=8):
    """
    Run the trained EZ Reader model and extract L1, L2, skip_prob per word.
    """
    model.eval()
    all_l1, all_l2, all_skip = [], [], []

    with torch.no_grad():
        for i in range(0, len(word_lists), batch_size):
            batch_words = word_lists[i:i + batch_size]
            batch_pred = predictabilities[i:i + batch_size]
            batch_wlen = word_lengths_list[i:i + batch_size]

            pred_vals = pad_sequence(
                [torch.tensor(p, dtype=torch.float32) for p in batch_pred],
                batch_first=True,
            ).to(device)
            wlens = pad_sequence(
                [torch.tensor(w, dtype=torch.float32) for w in batch_wlen],
                batch_first=True,
            ).to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(batch_words, pred_vals, wlens)

            for b in range(len(batch_words)):
                seq_len = len(batch_words[b])
                all_l1.append(pred['L1'][b, :seq_len].cpu().tolist())
                all_l2.append(pred['L2'][b, :seq_len].cpu().tolist())
                all_skip.append(pred['skip_prob'][b, :seq_len].cpu().tolist())

    return all_l1, all_l2, all_skip


# --------------------------------------------------------------------------- #
#  Analysis
# --------------------------------------------------------------------------- #

def correlate(x, y):
    x, y = np.array(x), np.array(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0:
        return np.corrcoef(x, y)[0, 1]
    return 0.0


def partial_correlation(x, y, covariates):
    """Partial correlation between x and y, controlling for covariates."""
    from numpy.linalg import lstsq
    x, y = np.array(x), np.array(y)
    covariates = np.column_stack(covariates) if len(covariates) > 0 else np.zeros((len(x), 1))

    mask = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(covariates), axis=1)
    x, y, covariates = x[mask], y[mask], covariates[mask]

    if len(x) < 10:
        return 0.0

    # Residualize x and y on covariates
    C = np.column_stack([covariates, np.ones(len(x))])
    res_x = x - C @ lstsq(C, x, rcond=None)[0]
    res_y = y - C @ lstsq(C, y, rcond=None)[0]

    if np.std(res_x) > 0 and np.std(res_y) > 0:
        return np.corrcoef(res_x, res_y)[0, 1]
    return 0.0


def run_analysis(word_lists, surprisals, l1_values, l2_values, skip_values,
                 predictabilities, word_lengths_list, human_trt, human_ffd):
    """Run all correlation and partial correlation analyses."""

    # Flatten everything to word level
    flat_surp, flat_l1, flat_l2, flat_skip = [], [], [], []
    flat_pred, flat_wlen, flat_htrt, flat_hffd = [], [], [], []
    flat_words = []

    for s_idx in range(len(word_lists)):
        n = len(word_lists[s_idx])
        for w_idx in range(n):
            flat_surp.append(surprisals[s_idx][w_idx])
            flat_l1.append(l1_values[s_idx][w_idx])
            flat_l2.append(l2_values[s_idx][w_idx])
            flat_skip.append(skip_values[s_idx][w_idx])
            flat_pred.append(predictabilities[s_idx][w_idx])
            flat_wlen.append(word_lengths_list[s_idx][w_idx])
            flat_words.append(word_lists[s_idx][w_idx])
            if human_trt and human_ffd:
                flat_htrt.append(human_trt[s_idx][w_idx])
                flat_hffd.append(human_ffd[s_idx][w_idx])

    flat_surp = np.array(flat_surp)
    flat_l1 = np.array(flat_l1)
    flat_l2 = np.array(flat_l2)
    flat_skip = np.array(flat_skip)
    flat_pred = np.array(flat_pred)
    flat_wlen = np.array(flat_wlen)
    log_wlen = np.log(flat_wlen + 1)

    n_words = len(flat_surp)
    print(f"\n{'='*80}")
    print(f"SURPRISAL DECOMPOSITION ANALYSIS ({n_words:,} words)")
    print(f"{'='*80}")

    # --- 1. Basic correlations ---
    print(f"\n--- 1. Pairwise Correlations ---")
    print(f"  L1 vs surprisal:       r = {correlate(flat_l1, flat_surp):.4f}")
    print(f"  L1 vs word_length:     r = {correlate(flat_l1, flat_wlen):.4f}")
    print(f"  L1 vs log(word_length):r = {correlate(flat_l1, log_wlen):.4f}")
    print(f"  L1 vs predictability:  r = {correlate(flat_l1, flat_pred):.4f}")
    print(f"  L2 vs surprisal:       r = {correlate(flat_l2, flat_surp):.4f}")
    print(f"  L2 vs word_length:     r = {correlate(flat_l2, flat_wlen):.4f}")
    print(f"  skip vs surprisal:     r = {correlate(flat_skip, flat_surp):.4f}")
    print(f"  skip vs predictability: r = {correlate(flat_skip, flat_pred):.4f}")
    print(f"  skip vs word_length:   r = {correlate(flat_skip, flat_wlen):.4f}")
    print(f"  surprisal vs pred:     r = {correlate(flat_surp, flat_pred):.4f}")
    print(f"  surprisal vs word_len: r = {correlate(flat_surp, flat_wlen):.4f}")

    if flat_htrt:
        flat_htrt = np.array(flat_htrt)
        flat_hffd = np.array(flat_hffd)
        print(f"\n  surprisal vs human_TRT: r = {correlate(flat_surp, flat_htrt):.4f}")
        print(f"  surprisal vs human_FFD: r = {correlate(flat_surp, flat_hffd):.4f}")
        print(f"  L1 vs human_FFD:        r = {correlate(flat_l1, flat_hffd):.4f}")
        print(f"  L1 vs human_TRT:        r = {correlate(flat_l1, flat_htrt):.4f}")

    # --- 2. Partial correlations ---
    print(f"\n--- 2. Partial Correlations (unique variance) ---")

    r_l1_surp_ctrl_wlen = partial_correlation(flat_l1, flat_surp, [flat_wlen])
    r_l1_wlen_ctrl_surp = partial_correlation(flat_l1, flat_wlen, [flat_surp])
    r_l1_surp_ctrl_all = partial_correlation(flat_l1, flat_surp, [flat_wlen, flat_pred])
    r_l1_wlen_ctrl_all = partial_correlation(flat_l1, flat_wlen, [flat_surp, flat_pred])
    r_l1_pred_ctrl_all = partial_correlation(flat_l1, flat_pred, [flat_surp, flat_wlen])

    print(f"  L1 vs surprisal | word_length:            r = {r_l1_surp_ctrl_wlen:.4f}")
    print(f"  L1 vs word_length | surprisal:            r = {r_l1_wlen_ctrl_surp:.4f}")
    print(f"  L1 vs surprisal | word_length, pred:      r = {r_l1_surp_ctrl_all:.4f}")
    print(f"  L1 vs word_length | surprisal, pred:      r = {r_l1_wlen_ctrl_all:.4f}")
    print(f"  L1 vs predictability | surprisal, wlen:   r = {r_l1_pred_ctrl_all:.4f}")

    # --- 3. Regression: L1 = a*surprisal + b*word_length + c*pred + d ---
    print(f"\n--- 3. Linear Regression: L1 ~ surprisal + word_length + predictability ---")
    X = np.column_stack([flat_surp, flat_wlen, flat_pred, np.ones(n_words)])
    mask = np.all(np.isfinite(X), axis=1) & np.isfinite(flat_l1)
    X_clean = X[mask]
    y_clean = flat_l1[mask]

    from numpy.linalg import lstsq
    beta, residuals, _, _ = lstsq(X_clean, y_clean, rcond=None)
    y_pred = X_clean @ beta
    ss_res = np.sum((y_clean - y_pred) ** 2)
    ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    print(f"  β_surprisal    = {beta[0]:.4f} ms/nat")
    print(f"  β_word_length  = {beta[1]:.4f} ms/char")
    print(f"  β_predictability = {beta[2]:.4f} ms")
    print(f"  β_intercept    = {beta[3]:.4f} ms")
    print(f"  R²             = {r_squared:.4f}")
    print(f"  → {r_squared*100:.1f}% of L1 variance explained by surprisal + word_length + pred")

    # Surprisal only
    X_surp = np.column_stack([flat_surp[mask], np.ones(mask.sum())])
    beta_s, _, _, _ = lstsq(X_surp, y_clean, rcond=None)
    y_pred_s = X_surp @ beta_s
    r2_surp = 1 - np.sum((y_clean - y_pred_s)**2) / ss_tot
    print(f"\n  Surprisal only:    R² = {r2_surp:.4f} ({r2_surp*100:.1f}%)")

    # Word length only
    X_wlen = np.column_stack([flat_wlen[mask], np.ones(mask.sum())])
    beta_w, _, _, _ = lstsq(X_wlen, y_clean, rcond=None)
    y_pred_w = X_wlen @ beta_w
    r2_wlen = 1 - np.sum((y_clean - y_pred_w)**2) / ss_tot
    print(f"  Word length only:  R² = {r2_wlen:.4f} ({r2_wlen*100:.1f}%)")

    # --- 4. Descriptive stats ---
    print(f"\n--- 4. Descriptive Statistics ---")
    print(f"  L1:         mean={np.mean(flat_l1):.1f}  std={np.std(flat_l1):.1f}  "
          f"min={np.min(flat_l1):.1f}  max={np.max(flat_l1):.1f} ms")
    print(f"  L2:         mean={np.mean(flat_l2):.1f}  std={np.std(flat_l2):.1f}  "
          f"min={np.min(flat_l2):.1f}  max={np.max(flat_l2):.1f} ms")
    print(f"  Surprisal:  mean={np.mean(flat_surp):.2f}  std={np.std(flat_surp):.2f}  "
          f"min={np.min(flat_surp):.2f}  max={np.max(flat_surp):.2f} nats")
    print(f"  Skip prob:  mean={np.mean(flat_skip):.3f}  std={np.std(flat_skip):.3f}")
    print(f"  Word len:   mean={np.mean(flat_wlen):.1f}  std={np.std(flat_wlen):.1f}")
    print(f"  Pred:       mean={np.mean(flat_pred):.3f}  std={np.std(flat_pred):.3f}")

    # --- 5. Binned analysis: surprisal quintiles ---
    print(f"\n--- 5. L1 by Surprisal Quintile ---")
    valid = np.isfinite(flat_surp) & np.isfinite(flat_l1)
    surp_v, l1_v, wlen_v = flat_surp[valid], flat_l1[valid], flat_wlen[valid]
    quintiles = np.percentile(surp_v, [0, 20, 40, 60, 80, 100])
    print(f"  {'Quintile':<12s} {'Surp range':>14s} {'mean_L1':>8s} {'std_L1':>8s} "
          f"{'mean_wlen':>9s} {'n':>6s}")
    print(f"  {'-'*60}")
    for q in range(5):
        lo, hi = quintiles[q], quintiles[q + 1]
        mask_q = (surp_v >= lo) & (surp_v < hi + (1 if q == 4 else 0))
        if mask_q.sum() > 0:
            print(f"  Q{q+1} (low→high) {lo:5.2f}-{hi:5.2f}   {np.mean(l1_v[mask_q]):8.1f} "
                  f"{np.std(l1_v[mask_q]):8.1f} {np.mean(wlen_v[mask_q]):9.1f} "
                  f"{mask_q.sum():6d}")

    # --- 6. Sample words ---
    print(f"\n--- 6. Sample Words (sorted by surprisal) ---")
    indices = np.argsort(flat_surp)
    print(f"  {'word':<20s} {'surprisal':>9s} {'L1':>6s} {'L2':>6s} {'skip':>6s} "
          f"{'wlen':>5s} {'pred':>5s}")
    print(f"  {'-'*65}")

    # 10 lowest surprisal
    print(f"  LOWEST SURPRISAL:")
    for idx in indices[:10]:
        print(f"  {flat_words[idx]:<20s} {flat_surp[idx]:9.3f} {flat_l1[idx]:6.1f} "
              f"{flat_l2[idx]:6.1f} {flat_skip[idx]:6.3f} {flat_wlen[idx]:5.0f} "
              f"{flat_pred[idx]:5.3f}")

    # 10 highest surprisal
    print(f"  HIGHEST SURPRISAL:")
    for idx in indices[-10:]:
        print(f"  {flat_words[idx]:<20s} {flat_surp[idx]:9.3f} {flat_l1[idx]:6.1f} "
              f"{flat_l2[idx]:6.1f} {flat_skip[idx]:6.3f} {flat_wlen[idx]:5.0f} "
              f"{flat_pred[idx]:5.3f}")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--corpus", type=str, default="both", choices=["geco", "provo", "both"])
    parser.add_argument("--checkpoint_variant", type=str, default="faithful_sh",
                        help="Which model variant checkpoint to load")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    base_dir = os.path.join(os.path.dirname(__file__), '..')
    data_dir = os.path.join(base_dir, "data")

    corpora = []
    if args.corpus in ("geco", "both"):
        corpora.append("geco")
    if args.corpus in ("provo", "both"):
        corpora.append("provo")

    for corpus_name in corpora:
        print(f"\n{'='*80}")
        print(f"  CORPUS: {corpus_name.upper()}")
        print(f"{'='*80}")

        # --- Load data ---
        if corpus_name == "geco":
            from geco_loader import load_geco, split_geco
            from data_loader import aggregate_by_sentence

            reading_path = os.path.join(data_dir, "Geco_MonolingualReadingData.csv")
            material_path = os.path.join(data_dir, "Geco_EnglishMaterial.csv")
            pred_path = os.path.join(data_dir, "geco_predictability.pkl")

            print("Loading GECO...")
            raw = load_geco(reading_path, material_path, pred_path)
            train_raw, val_raw, test_raw = split_geco(raw)
            agg = aggregate_by_sentence(raw, min_participants=5)
            train_ids = set(sd.text_id for sd in train_raw)
            val_ids = set(sd.text_id for sd in val_raw)
            test_agg = [a for a in agg if a.text_id not in train_ids and a.text_id not in val_ids]
            eval_data = test_agg
            print(f"  Test set: {len(eval_data)} sentences")

            model_short = args.model.replace("/", "_")
            ckpt_path = os.path.join(base_dir, "checkpoints", args.checkpoint_variant,
                                     f"geco_{model_short}", "best_model.pt")
        else:
            from data_loader import load_provo, split_dataset, aggregate_by_sentence

            et_path = os.path.join(data_dir, "Provo_Corpus-Eyetracking_Data.csv")
            print("Loading Provo...")
            raw = load_provo(et_path)
            train_raw, val_raw, test_raw = split_dataset(raw)
            agg = aggregate_by_sentence(raw, min_participants=10)
            train_ids = set(sd.text_id for sd in train_raw)
            val_ids = set(sd.text_id for sd in val_raw)
            test_agg = [a for a in agg if a.text_id not in train_ids and a.text_id not in val_ids]
            eval_data = test_agg
            print(f"  Test set: {len(eval_data)} sentences")

            model_short = args.model.replace("/", "_")
            ckpt_path = os.path.join(base_dir, "checkpoints", args.checkpoint_variant,
                                     f"provo_{model_short}", "best_model.pt")

        # Extract word lists, predictabilities, word lengths, human data
        word_lists = [a.tokens for a in eval_data]
        predictabilities = [a.predictabilities for a in eval_data]
        word_lengths_list = [[len(t) for t in a.tokens] for a in eval_data]
        human_trt = [a.mean_trt for a in eval_data]
        human_ffd = [a.mean_ffd for a in eval_data]

        n_words = sum(len(w) for w in word_lists)
        print(f"  Words: {n_words:,}")

        # --- Compute surprisal ---
        surprisals = compute_surprisal(word_lists, args.model, device)

        # --- Load trained model and extract L1/L2 ---
        print(f"\n  Loading trained model from: {ckpt_path}")
        if not os.path.exists(ckpt_path):
            print(f"  WARNING: checkpoint not found at {ckpt_path}")
            print(f"  Skipping L1/L2 extraction for {corpus_name}")
            continue

        from model_llama_faithful_sh import NeuralEZReaderLLaMA

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        freeze_layers = ckpt.get('freeze_layers', 16)
        hidden_dim = ckpt.get('hidden_dim', 256)
        ckpt_model_name = ckpt.get('model_name', args.model)

        model = NeuralEZReaderLLaMA(
            model_name=ckpt_model_name,
            freeze_layers=freeze_layers,
            hidden_dim=hidden_dim,
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        print(f"  Model loaded (delta={ckpt.get('delta', '?')}, "
              f"l1_scale={ckpt.get('l1_scale', '?')})")

        l1_values, l2_values, skip_values = extract_l1_l2(
            word_lists, predictabilities, word_lengths_list, model, device
        )

        # --- Run analysis ---
        run_analysis(
            word_lists, surprisals, l1_values, l2_values, skip_values,
            predictabilities, word_lengths_list, human_trt, human_ffd,
        )

        # Cleanup
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None


if __name__ == "__main__":
    main()
