"""
Evaluation-metric utilities for the paper-experiments pipeline.

Provides:
    - corr(a, b)                Pearson correlation, NaN-safe.
    - mae(a, b)                 Mean absolute error.
    - bias(pred, human)         pred.mean() - human.mean().
    - partial_corr(...)         Partial correlation, controlling for vars.
    - delta_r2(...)             ΔR² from hierarchical regression.
    - bootstrap_ci(...)         Bootstrap CI for any statistic.
    - paired_t_test(...)        Paired t-test wrapper.
    - eval_predictions_on_aggregated(model, agg_data, ...)
                                Single-pass evaluator that returns
                                per-word arrays AND summary metrics.
"""

from typing import Tuple

import numpy as np


# --------------------------------------------------------------------------- #
#  Basic statistics
# --------------------------------------------------------------------------- #


def corr(a, b) -> float:
    """Pearson correlation, returning 0.0 if either vector is constant."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return 0.0
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def mae(pred, human) -> float:
    pred = np.asarray(pred, dtype=float)
    human = np.asarray(human, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(human)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(pred[mask] - human[mask])))


def bias(pred, human) -> float:
    pred = np.asarray(pred, dtype=float)
    human = np.asarray(human, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(human)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(pred[mask]) - np.mean(human[mask]))


# --------------------------------------------------------------------------- #
#  Partial correlation and ΔR²
# --------------------------------------------------------------------------- #


def _residualize(y, X):
    """Linear-regress y on X (with intercept) and return residuals."""
    y = np.asarray(y, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]
    X = np.hstack([np.ones((n, 1)), X])
    # Least squares
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ coef


def partial_corr(x, y, controls) -> float:
    """
    Pearson partial correlation between x and y, controlling for `controls`.

    Args:
        x, y: 1-D arrays.
        controls: 2-D array (n_obs × n_controls) or list of 1-D arrays.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if isinstance(controls, list):
        controls = np.column_stack([np.asarray(c, dtype=float) for c in controls])
    else:
        controls = np.asarray(controls, dtype=float)
        if controls.ndim == 1:
            controls = controls.reshape(-1, 1)
    mask = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(controls), axis=1)
    x = x[mask]
    y = y[mask]
    controls = controls[mask]
    if len(x) < 3:
        return 0.0
    rx = _residualize(x, controls)
    ry = _residualize(y, controls)
    return corr(rx, ry)


def linear_r2(y, X) -> float:
    """R² of OLS regression y ~ X (with intercept)."""
    y = np.asarray(y, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]
    X1 = np.hstack([np.ones((n, 1)), X])
    coef, *_ = np.linalg.lstsq(X1, y, rcond=None)
    yhat = X1 @ coef
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def delta_r2(y, X_baseline, x_added) -> float:
    """
    Hierarchical regression: how much R² does `x_added` add to a model
    that already contains `X_baseline`?
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    X_baseline = np.asarray(X_baseline, dtype=float)
    if X_baseline.ndim == 1:
        X_baseline = X_baseline.reshape(-1, 1)
    x_added = np.asarray(x_added, dtype=float).reshape(-1, 1)
    r2_b = linear_r2(y, X_baseline)
    X_full = np.hstack([X_baseline, x_added])
    r2_f = linear_r2(y, X_full)
    return r2_f - r2_b


# --------------------------------------------------------------------------- #
#  Bootstrap CI
# --------------------------------------------------------------------------- #


def bootstrap_ci(values, n_boot: int = 1000, alpha: float = 0.05,
                 statistic=np.mean, seed: int = 0) -> Tuple[float, float, float]:
    """
    Compute (mean, ci_low, ci_high) of `statistic` over bootstrap resamples.

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(statistic(values))
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        samples.append(statistic(values[idx]))
    samples = np.array(samples)
    lo = float(np.percentile(samples, 100 * alpha / 2))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return point, lo, hi


def bootstrap_ci_difference(values_a, values_b, n_boot: int = 1000,
                            alpha: float = 0.05, seed: int = 0):
    """
    Bootstrap CI for the mean of (a - b), paired by index.

    Returns:
        (mean_diff, ci_low, ci_high)
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    assert len(a) == len(b), "Paired bootstrap requires same-length inputs"
    n = len(a)
    diffs = a - b
    point = float(np.mean(diffs))
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        samples.append(np.mean(diffs[idx]))
    samples = np.array(samples)
    lo = float(np.percentile(samples, 100 * alpha / 2))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return point, lo, hi


# --------------------------------------------------------------------------- #
#  Paired t-test
# --------------------------------------------------------------------------- #


def paired_t_test(values_a, values_b):
    """
    Paired-samples t-test wrapping scipy. Returns (t_statistic, p_value).
    """
    from scipy import stats  # imported lazily so this module is light
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    res = stats.ttest_rel(a, b)
    return float(res.statistic), float(res.pvalue)


# --------------------------------------------------------------------------- #
#  High-level evaluator
# --------------------------------------------------------------------------- #


def eval_predictions_on_aggregated(model, agg_data, device, subtlex,
                                   batch_size: int = 8):
    """
    Run model on aggregated GECO/Provo data, collect per-word predictions
    + targets, return everything needed for downstream analysis.

    Returns dict with arrays:
        pred_trt, pred_ffd, pred_gaze, pred_skip
        human_trt, human_ffd, human_gaze, human_skip
        L1, L2, base_L1, ctx, formula
        race_logit, residual_logit
        word, sentence_id (optional)
    Plus summary metrics: r_*, mae_*, bias_*, etc.
    """
    import torch
    from torch.nn.utils.rnn import pad_sequence

    from .load_data import word_frequency

    model.eval()
    out_lists = {
        'pred_trt': [], 'pred_ffd': [], 'pred_gaze': [], 'pred_skip': [],
        'human_trt': [], 'human_ffd': [], 'human_gaze': [], 'human_skip': [],
        'L1': [], 'L2': [], 'base_L1': [], 'ctx': [], 'formula': [],
        'race_logit': [], 'residual_logit': [],
        'word': [], 'sentence_idx': [], 'word_position': [],
    }

    with torch.no_grad():
        for sent_idx, s in enumerate(agg_data):
            word_lists = [s.tokens]
            freqs = torch.tensor(
                [float(word_frequency(t, subtlex)) for t in s.tokens],
                dtype=torch.float32,
            ).unsqueeze(0).to(device)
            wlens = torch.tensor(
                [len(t) for t in s.tokens], dtype=torch.float32
            ).unsqueeze(0).to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                p = model(word_lists, freqs, wlens)

            seq_len = len(s.tokens)
            for i in range(seq_len):
                out_lists['pred_trt'].append(p['conditional_trt'][0, i].item())
                out_lists['pred_ffd'].append(p['first_fixation'][0, i].item())
                out_lists['pred_gaze'].append(p['gaze_duration'][0, i].item())
                out_lists['pred_skip'].append(p['skip_prob'][0, i].item())
                out_lists['human_trt'].append(s.mean_trt[i])
                out_lists['human_ffd'].append(s.mean_ffd[i])
                out_lists['human_gaze'].append(s.mean_gaze[i])
                out_lists['human_skip'].append(s.skip_rate[i])
                out_lists['L1'].append(p['L1'][0, i].item())
                out_lists['L2'].append(p['L2'][0, i].item())
                out_lists['base_L1'].append(p['base_L1'][0, i].item())
                if 'ctx' in p:
                    out_lists['ctx'].append(p['ctx'][0, i].item())
                else:
                    out_lists['ctx'].append(0.0)
                if 'base_L1_formula' in p:
                    out_lists['formula'].append(p['base_L1_formula'][0, i].item())
                else:
                    out_lists['formula'].append(0.0)
                out_lists['race_logit'].append(p['race_logit'][0, i].item())
                out_lists['residual_logit'].append(p['residual_skip_logit'][0, i].item())
                out_lists['word'].append(s.tokens[i])
                out_lists['sentence_idx'].append(sent_idx)
                out_lists['word_position'].append(i)

    # Convert to arrays
    arrays = {k: np.array(v) if not isinstance(v[0], str) else v
              for k, v in out_lists.items()}

    summary = {
        'r_trt': corr(arrays['pred_trt'], arrays['human_trt']),
        'r_ffd': corr(arrays['pred_ffd'], arrays['human_ffd']),
        'r_gaze': corr(arrays['pred_gaze'], arrays['human_gaze']),
        'r_skip': corr(arrays['pred_skip'], arrays['human_skip']),
        'mae_trt': mae(arrays['pred_trt'], arrays['human_trt']),
        'mae_ffd': mae(arrays['pred_ffd'], arrays['human_ffd']),
        'mae_gaze': mae(arrays['pred_gaze'], arrays['human_gaze']),
        'bias_trt': bias(arrays['pred_trt'], arrays['human_trt']),
        'bias_ffd': bias(arrays['pred_ffd'], arrays['human_ffd']),
        'bias_gaze': bias(arrays['pred_gaze'], arrays['human_gaze']),
        'mean_pred_trt': float(np.mean(arrays['pred_trt'])),
        'mean_human_trt': float(np.mean(arrays['human_trt'])),
        'mean_pred_skip': float(np.mean(arrays['pred_skip'])),
        'mean_human_skip': float(np.mean(arrays['human_skip'])),
        'std_pred_skip': float(np.std(arrays['pred_skip'])),
        'n_words': len(arrays['pred_trt']),
    }
    return arrays, summary
