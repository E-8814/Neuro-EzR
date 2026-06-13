"""
Shared skip-metric helpers for exp11 (skip-fix fair comparison).

Comparable population: words 1..L-1 of each sentence ("cmp" = sentence-
initial words excluded). The cascade family does not model boundary
skips (the original E-Z Reader simulation starts with the eyes on
word 1), so all models are scored on the same non-initial word set.

Alignments:
  - Baselines and v4c_v2-family models predict each word's skip at the
    word's own row -> same-index selection (drop word_position == 0).
  - v4c_v3 'next' models compute the race for word i+1 at row i ->
    the prediction for word i sits at row i-1 of the same sentence.
"""

from __future__ import annotations

import numpy as np


def auc_binarized(scores, fractions, threshold: float = 0.5) -> float:
    """Rank-based AUC against the skip fraction binarized at threshold."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = (np.asarray(fractions, dtype=np.float64) > threshold).astype(np.int64)
    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = scores.argsort().argsort() + 1
    rank_sum = ranks[labels == 1].sum()
    return float((rank_sum - pos * (pos + 1) / 2) / (pos * neg))


def skip_summary(pred, human) -> dict:
    """r / AUC / Brier / MAE / bias / means for one (pred, target) pairing."""
    pred = np.asarray(pred, dtype=np.float64)
    human = np.asarray(human, dtype=np.float64)
    if len(pred) > 2 and pred.std() > 0 and human.std() > 0:
        r = float(np.corrcoef(pred, human)[0, 1])
    else:
        r = 0.0
    return {
        "r_skip": r,
        "skip_auc": auc_binarized(pred, human),
        "skip_brier": float(np.mean((pred - human) ** 2)),
        "mae_skip": float(np.mean(np.abs(pred - human))),
        "bias_skip": float(np.mean(pred) - np.mean(human)),
        "mean_pred_skip": float(np.mean(pred)),
        "mean_human_skip": float(np.mean(human)),
        "n_words": int(len(pred)),
    }


def same_index_pairs(pred_skip, human_skip, word_position):
    """Comparable-population pairs for same-index models (baselines)."""
    pred_skip = np.asarray(pred_skip)
    human_skip = np.asarray(human_skip)
    pos = np.asarray(word_position)
    sel = pos > 0
    return pred_skip[sel], human_skip[sel]


def next_aligned_pairs(pred_skip, human_skip, word_position):
    """
    Comparable-population pairs for next-aligned models (v4c_v3 'next').

    Rows are emitted sentence-by-sentence in word order, so for any row i
    with word_position > 0, row i-1 is the previous word of the SAME
    sentence, and the race computed there is the prediction for word i.
    """
    pred_skip = np.asarray(pred_skip)
    human_skip = np.asarray(human_skip)
    pos = np.asarray(word_position)
    sel = pos[1:] > 0
    return pred_skip[:-1][sel], human_skip[1:][sel]


def positions_from_agg(agg_data):
    """Flattened word_position array matching get_human()-style flattening."""
    return np.array([i for a in agg_data for i in range(len(a.tokens))])


def positions_from_sublists(sublists):
    """Flattened word_position array from a list of per-sentence lists."""
    return np.array([i for sub in sublists for i in range(len(sub))])
