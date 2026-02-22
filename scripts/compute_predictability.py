"""
Compute GPT-2 Large surprisal → predictability for GECO words.

For each word in the GECO corpus, computes:
    surprisal = -log2(P(word | left context))
    predictability = exp(-surprisal / temperature)

Temperature is calibrated so that the resulting predictability values
have a similar distribution to Provo cloze norms.

Results are cached to data/geco_predictability.pkl so this only runs once.

Usage:
    python compute_predictability.py
"""

import os
import sys
import math
import pickle

import torch
import numpy as np
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


# --------------------------------------------------------------------------- #
#  GPT-2 surprisal computation
# --------------------------------------------------------------------------- #

def compute_surprisal_for_sentences(sentences, model, tokenizer, device, batch_size=8):
    """
    Compute per-word surprisal for a list of sentences.

    Args:
        sentences: list of list of str (each sentence is a list of words)
        model: GPT2LMHeadModel
        tokenizer: GPT2TokenizerFast
        device: torch device

    Returns:
        list of list of float — surprisal in bits per word
    """
    model.eval()
    all_surprisals = []

    for sent_idx, words in enumerate(sentences):
        text = " ".join(words)

        # Tokenize with offset mapping to align subwords → words
        encoding = tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=True,
            max_length=1024,
        )

        input_ids = encoding["input_ids"].to(device)
        offsets = encoding["offset_mapping"][0]  # (n_subwords, 2)

        # Get log probabilities
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0]  # (seq_len, vocab_size)

        log_probs = torch.log_softmax(logits, dim=-1)

        # For each subword token at position i, its surprisal is
        # -log2(P(token_i | tokens_<i))
        # The prediction for token i comes from logits[i-1]
        subword_surprisals = []
        for i in range(len(input_ids[0])):
            if i == 0:
                # First token has no context — use uniform prior estimate
                subword_surprisals.append(10.0)  # ~uniform over vocab
            else:
                token_id = input_ids[0, i].item()
                log_p = log_probs[i - 1, token_id].item()
                surprisal_bits = -log_p / math.log(2)
                subword_surprisals.append(surprisal_bits)

        # Map subwords back to words using character offsets
        # Build word → character span mapping
        word_char_spans = []
        char_pos = 0
        for w in words:
            start = text.index(w, char_pos)
            end = start + len(w)
            word_char_spans.append((start, end))
            char_pos = end

        # For each word, collect surprisals of its subword tokens
        # and sum them (joint probability = product → sum of log probs)
        word_surprisals = []
        subword_idx = 0

        for w_idx, (w_start, w_end) in enumerate(word_char_spans):
            word_surp = 0.0
            n_sub = 0

            while subword_idx < len(offsets):
                s_start, s_end = offsets[subword_idx].tolist()

                # Check if this subword overlaps with current word
                if s_start >= w_end:
                    break
                if s_end > w_start:
                    word_surp += subword_surprisals[subword_idx]
                    n_sub += 1
                    subword_idx += 1
                else:
                    subword_idx += 1

            if n_sub == 0:
                word_surp = 10.0  # fallback
            word_surprisals.append(word_surp)

        all_surprisals.append(word_surprisals)

        if (sent_idx + 1) % 200 == 0:
            print(f"  Processed {sent_idx + 1}/{len(sentences)} sentences...")

    return all_surprisals


def surprisal_to_predictability(surprisal, temperature=4.0):
    """
    Convert surprisal (bits) to predictability (0-1).

    predictability = exp(-surprisal / temperature)

    Temperature=4.0 is calibrated so that:
      - Very predictable words (surprisal ~0) → pred ~1.0
      - Average words (surprisal ~8) → pred ~0.13
      - Rare words (surprisal ~15) → pred ~0.02
    This roughly matches the distribution of Provo cloze norms.
    """
    return math.exp(-surprisal / temperature)


# --------------------------------------------------------------------------- #
#  Load GECO sentences
# --------------------------------------------------------------------------- #

def load_geco_sentences(material_path):
    """
    Load GECO English material and group into sentences.

    Returns:
        list of (sentence_id, list of words)
    """
    print(f"Loading GECO material from {material_path}...")
    df = pd.read_excel(material_path)

    sentences = []
    current_sent_id = None
    current_words = []

    for _, row in df.iterrows():
        sent_id = str(row["SENTENCE_ID"])
        word = str(row["WORD"]).strip()

        if sent_id != current_sent_id:
            if current_sent_id is not None and current_words:
                sentences.append((current_sent_id, current_words))
            current_sent_id = sent_id
            current_words = [word]
        else:
            current_words.append(word)

    # Last sentence
    if current_sent_id is not None and current_words:
        sentences.append((current_sent_id, current_words))

    print(f"  {len(sentences)} sentences, {sum(len(w) for _, w in sentences)} words")
    return sentences


# --------------------------------------------------------------------------- #
#  Calibrate temperature against Provo cloze norms
# --------------------------------------------------------------------------- #

def calibrate_temperature(model, tokenizer, device):
    """
    Compute GPT-2 surprisal on Provo and find the temperature that best
    matches the distribution of Provo cloze norms (OrthographicMatch).
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ez_reader'))
    from data_loader import load_provo, aggregate_by_sentence  # noqa: E402

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    et_path = os.path.join(data_dir, "Provo_Corpus-Eyetracking_Data.csv")

    if not os.path.exists(et_path):
        print("  Provo not found, using default temperature=4.0")
        return 4.0

    print("Calibrating temperature against Provo cloze norms...")
    raw = load_provo(et_path)
    agg = aggregate_by_sentence(raw, min_participants=10)

    # Get sentences and their cloze predictabilities
    provo_sentences = [a.tokens for a in agg[:50]]  # Use first 50 for speed
    provo_preds = []
    for a in agg[:50]:
        provo_preds.extend(a.predictabilities)

    # Compute GPT-2 surprisal on same sentences
    surprisals_per_sent = compute_surprisal_for_sentences(
        provo_sentences, model, tokenizer, device
    )
    flat_surprisals = []
    for s in surprisals_per_sent:
        flat_surprisals.extend(s)

    # Ensure same length
    n = min(len(provo_preds), len(flat_surprisals))
    provo_preds = np.array(provo_preds[:n])
    flat_surprisals = np.array(flat_surprisals[:n])

    # Search for best temperature
    best_temp = 4.0
    best_corr = -1.0

    for temp in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0]:
        gpt2_preds = np.array([surprisal_to_predictability(s, temp) for s in flat_surprisals])
        if np.std(gpt2_preds) > 0 and np.std(provo_preds) > 0:
            r = np.corrcoef(gpt2_preds, provo_preds)[0, 1]
            if r > best_corr:
                best_corr = r
                best_temp = temp

    print(f"  Best temperature: {best_temp} (r={best_corr:.3f} with cloze norms)")
    return best_temp


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    material_path = os.path.join(data_dir, "Geco_EnglishMaterial.xlsx")
    output_path = os.path.join(data_dir, "geco_predictability.pkl")

    if os.path.exists(output_path):
        print(f"Cache already exists: {output_path}")
        with open(output_path, "rb") as f:
            data = pickle.load(f)
        print(f"  {len(data)} sentence entries")
        # Show sample
        for sent_id, preds in list(data.items())[:3]:
            words_preview = preds['words'][:5]
            preds_preview = preds['predictability'][:5]
            print(f"  {sent_id}: {words_preview} → {[f'{p:.3f}' for p in preds_preview]}")
        return

    # Load GPT-2 Large
    print("Loading GPT-2 Large...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-large")
    model.eval()
    print(f"  Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    # Calibrate temperature
    temperature = calibrate_temperature(model, tokenizer, device)

    # Load GECO sentences
    sentences = load_geco_sentences(material_path)

    # Compute surprisal
    print(f"\nComputing surprisal for {len(sentences)} GECO sentences...")
    word_lists = [words for _, words in sentences]
    sent_ids = [sid for sid, _ in sentences]

    surprisals = compute_surprisal_for_sentences(word_lists, model, tokenizer, device)

    # Convert to predictability and build output dict
    result = {}
    for sent_id, words, surps in zip(sent_ids, word_lists, surprisals):
        preds = [surprisal_to_predictability(s, temperature) for s in surps]
        result[sent_id] = {
            'words': words,
            'surprisal': surps,
            'predictability': preds,
        }

    # Save
    with open(output_path, "wb") as f:
        pickle.dump(result, f)
    print(f"\nSaved to {output_path}")
    print(f"  {len(result)} sentences")

    # Summary stats
    all_preds = []
    all_surps = []
    for v in result.values():
        all_preds.extend(v['predictability'])
        all_surps.extend(v['surprisal'])

    print(f"\n  Surprisal:       mean={np.mean(all_surps):.2f}  std={np.std(all_surps):.2f}  "
          f"min={np.min(all_surps):.2f}  max={np.max(all_surps):.2f}")
    print(f"  Predictability:  mean={np.mean(all_preds):.3f}  std={np.std(all_preds):.3f}  "
          f"min={np.min(all_preds):.3f}  max={np.max(all_preds):.3f}")


if __name__ == "__main__":
    main()
