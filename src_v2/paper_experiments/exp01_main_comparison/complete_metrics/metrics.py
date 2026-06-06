"""
Complete per-model metric computation for the augmented Table 1.

Same shape as the existing eval_baselines.py:metrics_summary, but adds
mae_skip in raw [0, 1] units (fraction-of-readers). All four metrics
(TRT, FFD, Gaze, skip) get full coverage:

    r_<m>      Pearson r between predicted and human-aggregated values
    mae_<m>    mean absolute error
                 - TRT/FFD/Gaze:  ms
                 - skip:          fraction in [0, 1] (NOT scaled to %)
    bias_<m>   mean(pred) - mean(human)

Plus the existing skip diagnostics (mean_pred_skip, mean_human_skip).

Usage:
    from metrics import metrics_summary_complete
    out = metrics_summary_complete(pred_trt, pred_ffd, pred_gaze, pred_skip,
                                   h_trt, h_ffd, h_gaze, h_skip)
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats


def _pearson(pred, human) -> float:
    pred = np.asarray(pred, dtype=float).ravel()
    human = np.asarray(human, dtype=float).ravel()
    if pred.size == 0 or pred.size != human.size:
        return 0.0
    if np.std(pred) == 0 or np.std(human) == 0:
        return 0.0
    return float(sp_stats.pearsonr(pred, human)[0])


def _mae(pred, human) -> float:
    pred = np.asarray(pred, dtype=float).ravel()
    human = np.asarray(human, dtype=float).ravel()
    if pred.size == 0 or pred.size != human.size:
        return 0.0
    return float(np.mean(np.abs(pred - human)))


def _bias(pred, human) -> float:
    pred = np.asarray(pred, dtype=float).ravel()
    human = np.asarray(human, dtype=float).ravel()
    if pred.size == 0:
        return 0.0
    return float(np.mean(pred) - np.mean(human))


def metrics_summary_complete(
    pred_trt, pred_ffd, pred_gaze, pred_skip,
    h_trt, h_ffd, h_gaze, h_skip,
) -> dict:
    """
    Return the complete per-(model, dataset) metrics block.

    All four metrics get r, mae, bias coverage. Skip's MAE and bias are
    in raw [0, 1] fraction units.
    """
    return {
        # Pearson r
        "r_trt":  _pearson(pred_trt,  h_trt),
        "r_ffd":  _pearson(pred_ffd,  h_ffd),
        "r_gaze": _pearson(pred_gaze, h_gaze),
        "r_skip": _pearson(pred_skip, h_skip),
        # MAE — TRT/FFD/Gaze in ms; skip in fraction (raw [0, 1])
        "mae_trt":  _mae(pred_trt,  h_trt),
        "mae_ffd":  _mae(pred_ffd,  h_ffd),
        "mae_gaze": _mae(pred_gaze, h_gaze),
        "mae_skip": _mae(pred_skip, h_skip),
        # Bias (signed)
        "bias_trt":  _bias(pred_trt,  h_trt),
        "bias_ffd":  _bias(pred_ffd,  h_ffd),
        "bias_gaze": _bias(pred_gaze, h_gaze),
        "bias_skip": _bias(pred_skip, h_skip),
        # Skip diagnostics
        "mean_pred_skip":  float(np.mean(pred_skip)),
        "mean_human_skip": float(np.mean(h_skip)),
        # Sample-size diagnostic
        "n_words": int(np.asarray(pred_trt).size),
    }
