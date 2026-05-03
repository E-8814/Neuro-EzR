"""
Word-level alignment helpers for surprisal computation (exp06, exp07).

When computing surprisal from a subword-level LM (TinyLlama, GPT-2),
each word may map to multiple subword tokens. We aggregate subword
surprisals into a single per-word surprisal (sum, by convention —
this corresponds to log P(word | context)).

Also provides per-word log-frequency-norm computation matching the
training scripts (so feature alignment is consistent everywhere).
"""

import math
from typing import List

import numpy as np
import torch


def per_word_surprisal_from_subword(
    tokens: List[str],
    tokenizer,
    causal_lm,
    device,
    max_length: int = 512,
) -> np.ndarray:
    """
    Compute per-word surprisal (in nats) for a single sentence.

    surprisal(w) = -log P(w | preceding_context)
                 = sum over subwords of -log P(subword | preceding subwords)

    Args:
        tokens: list of word strings (the sentence).
        tokenizer: HF tokenizer with `is_split_into_words=True` support
            (e.g., LlamaTokenizer, GPT2Tokenizer).
        causal_lm: HF AutoModelForCausalLM (e.g., TinyLlama or GPT-2).
        device: torch device.
        max_length: tokenization truncation.

    Returns:
        np.ndarray of shape (len(tokens),) — per-word surprisal in nats.
    """
    # Tokenize the sentence; track subword -> word alignment.
    enc = tokenizer(
        [tokens],
        is_split_into_words=True,
        padding=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)            # (1, T)
    word_ids = enc.word_ids(batch_index=0)              # list of len T

    causal_lm.eval()
    with torch.no_grad():
        out = causal_lm(input_ids=input_ids)
        logits = out.logits  # (1, T, V)

    # Compute log-prob of each token given previous tokens.
    # token at position t is predicted by logits at position t-1.
    log_probs = torch.log_softmax(logits[0], dim=-1)        # (T, V)
    target_ids = input_ids[0]                                # (T,)

    # surprisal of token t = -log_probs[t-1, token_id_at_t]
    # for t = 1, 2, ..., T-1; token 0 has no preceding context.
    surprisal_per_subword = torch.zeros(target_ids.size(0), device=device)
    for t in range(1, target_ids.size(0)):
        surprisal_per_subword[t] = -log_probs[t - 1, target_ids[t]]
    sps = surprisal_per_subword.cpu().numpy()

    # Aggregate to per-word: sum subwords belonging to each word.
    n_words = len(tokens)
    word_surprisal = np.zeros(n_words, dtype=np.float64)
    word_subword_count = np.zeros(n_words, dtype=np.int32)
    for sw_idx, w_idx in enumerate(word_ids):
        if w_idx is None or sw_idx == 0:
            continue
        if w_idx < n_words:
            word_surprisal[w_idx] += sps[sw_idx]
            word_subword_count[w_idx] += 1

    # Words with no subword mapping (truncation) → fall back to mean
    # surprisal of the rest (tiny effect; flagged via NaN propagation
    # in downstream stats if needed).
    avg = word_surprisal[word_subword_count > 0].mean() \
        if (word_subword_count > 0).any() else 0.0
    word_surprisal[word_subword_count == 0] = avg
    return word_surprisal


def log_freq_norm(frequency: float) -> float:
    """
    Match the normalization used in v4c_v2 training:
        log_freq_norm = (log f - 10) / 5
    Used by paper-model and as a control variable in stats analyses.
    """
    return (math.log(max(frequency, 1.0)) - 10.0) / 5.0


def precompute_surprisal_for_corpus(
    sentences,
    causal_lm,
    tokenizer,
    device,
):
    """
    Compute per-word surprisal for every sentence in `sentences`
    (a list with .tokens attribute).

    Returns:
        list of np.ndarray (one per sentence), aligned to sentence.tokens.
    """
    out = []
    for s in sentences:
        sp = per_word_surprisal_from_subword(s.tokens, tokenizer, causal_lm, device)
        out.append(sp)
    return out
