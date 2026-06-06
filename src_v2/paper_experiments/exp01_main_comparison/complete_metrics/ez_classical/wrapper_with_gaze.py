"""
Classical-E-Z-Reader wrapper that:
  - tracks fixations by fixation_point CHANGES (each saccade = new fixation),
    so refixations are properly counted as separate fixations,
  - computes proper FFD / Gaze / TRT distinctions,
  - optionally accepts a `model_params` override dict to swap the default
    Reichle 2003 parameter values.

The original archive/original_ezreader/ez_wrapper.py only tracked transitions
between words (via fixation_point's word-membership), which conflated
within-word refixations into a single "fixation" — this version fixes that.

Definitions (standard eye-tracking convention):
  - skipped[i]: word i was never fixated
  - FFD[i]:     duration of the FIRST fixation on word i
  - Gaze[i]:    sum of consecutive fixations on word i during first pass
                (first pass = from when the eye first lands on i until it
                 leaves to a different word)
  - TRT[i]:     sum of all fixations on i, including regressions
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional

import simpy

# Import the existing simulator without modifying it.
_ARCHIVE_EZ = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "..",
    "archive", "original_ezreader",
))
if _ARCHIVE_EZ not in sys.path:
    sys.path.insert(0, _ARCHIVE_EZ)

from ez_reader_engine import Simulation, Word  # noqa: E402


# --------------------------------------------------------------------------- #
#  Parameter override: Simulation.model_parameters is a CLASS attribute, so
#  swapping it in/out for a single run requires care to be re-entrant.
# --------------------------------------------------------------------------- #


@contextmanager
def _override_params(model_params: Optional[Dict[str, float]]):
    """Temporarily swap Simulation.model_parameters with overrides."""
    if not model_params:
        yield
        return
    saved = dict(Simulation.model_parameters)
    Simulation.model_parameters = dict(saved)  # fresh dict per call
    Simulation.model_parameters.update(model_params)
    try:
        yield
    finally:
        Simulation.model_parameters = saved


def _which_word(fp: float, position_map):
    """Return word index for a given fixation_point, or None if it's between words."""
    for (start, end), idx in position_map.items():
        if start <= fp <= end:
            return idx
    return None


def _compute_metrics_from_timeline(timeline, n_words):
    """
    Compute per-word FFD / Gaze / TRT / skip from a timeline of
    (start_time, end_time, word_idx) entries in time order.

    Each entry is one fixation. Fixations were recorded by detecting changes
    in fixation_point; consecutive fixations on the same word_idx represent
    refixations within first pass.
    """
    ffd  = [0.0] * n_words
    gaze = [0.0] * n_words
    trt  = [0.0] * n_words
    n_fix = [0] * n_words
    has_first_fix = [False] * n_words
    in_first_pass = [True] * n_words   # initially True for all words
    last_word = None

    for (s, e, w) in timeline:
        if w is None:
            continue
        dur = max(0.0, e - s)

        # When the eye moves from a different word to this one,
        # the previous word's first pass ends.
        if last_word is not None and last_word != w:
            in_first_pass[last_word] = False

        # TRT: every fixation contributes
        trt[w] += dur
        n_fix[w] += 1

        # FFD: only the first fixation
        if not has_first_fix[w]:
            ffd[w] = dur
            has_first_fix[w] = True

        # Gaze: only fixations during this word's first pass
        if in_first_pass[w]:
            gaze[w] += dur

        last_word = w

    skipped = [trt[i] == 0.0 for i in range(n_words)]
    return ffd, gaze, trt, skipped, n_fix


def _run_one_simulation(
    tokens: List[str],
    frequencies: List[float],
    predictabilities: List[float],
    integration_time: float = 25.0,
    integration_failure: float = 0.01,
    timeout_seconds: float = 5.0,
    model_params: Optional[Dict[str, float]] = None,
):
    """
    Run the classical Simulation once and return per-word metrics.

    Returns dict with:
        total_reading_time         list[float]
        first_fixation_duration    list[float]
        gaze_duration              list[float]
        was_skipped                list[bool]
        n_fixations                list[int]
        success                    bool
        error                      (only if success=False)
    """
    n_words = len(tokens)

    sentence = []
    for i in range(n_words):
        sentence.append(Word(
            token=tokens[i],
            frequency=max(1.0, float(frequencies[i])),
            predictability=float(predictabilities[i]),
            integration_time=integration_time,
            integration_failure=integration_failure,
        ))

    # Build position -> word index map (same scheme as the engine)
    position_map = {}
    pos = 1
    for i, w in enumerate(sentence):
        start = pos
        end = pos + 1 + len(w.token)
        position_map[(start, end)] = i
        pos = end

    fail = {
        "total_reading_time": [0.0] * n_words,
        "first_fixation_duration": [0.0] * n_words,
        "gaze_duration": [0.0] * n_words,
        "was_skipped": [True] * n_words,
        "n_fixations": [0] * n_words,
        "success": False,
    }

    with _override_params(model_params):
        try:
            sim = Simulation(sentence=sentence, realtime=False, trace=False)
        except Exception as exc:
            return {**fail, "error": f"init: {exc!r}"}

        # Track the global fixation timeline by detecting fixation_point CHANGES.
        timeline = []  # [(start_time, end_time, word_idx), ...]
        prev_fp = sim.fixation_point
        prev_word = _which_word(prev_fp, position_map)
        fix_start = 0.0

        max_steps = 10000
        step_count = 0
        sim_crashed = False

        while step_count < max_steps:
            try:
                sim.step()
                step_count += 1

                cur_fp = sim.fixation_point
                # Detect saccade: fixation_point changed from prev_fp
                if abs(cur_fp - prev_fp) > 1e-6:
                    # End the previous fixation
                    if prev_word is not None:
                        timeline.append((fix_start, sim.time, prev_word))
                    # Start a new fixation
                    prev_fp = cur_fp
                    prev_word = _which_word(prev_fp, position_map)
                    fix_start = sim.time

                if sim.time > timeout_seconds * 1000:
                    break

            except simpy.core.EmptySchedule:
                # Simulator finished; close out the last fixation.
                if prev_word is not None:
                    timeline.append((fix_start, sim.time, prev_word))
                break

            except Exception:
                # Latent bug in ez_reader_engine.py (_attend_again can hit
                # AttributeError on some refixation paths). Treat as a
                # failed MC run; the outer averager will skip this run.
                sim_crashed = True
                break

    if sim_crashed:
        return {**fail, "error": "simulator step exception"}

    ffd, gaze, trt, skipped, n_fix = _compute_metrics_from_timeline(timeline, n_words)
    return {
        "total_reading_time": trt,
        "first_fixation_duration": ffd,
        "gaze_duration": gaze,
        "was_skipped": skipped,
        "n_fixations": n_fix,
        "success": True,
    }


def run_classical_averaged(
    tokens: List[str],
    frequencies: List[float],
    predictabilities: List[float],
    num_runs: int = 200,
    integration_time: float = 25.0,
    integration_failure: float = 0.01,
    timeout_seconds: float = 5.0,
    model_params: Optional[Dict[str, float]] = None,
):
    """
    Run the classical Simulation `num_runs` times and average per-word
    predictions across runs.

    Returns dict with averaged per-word metrics:
        total_reading_time        list[float]   (ms)
        first_fixation_duration   list[float]   (ms)
        gaze_duration             list[float]   (ms)
        skip_rate                 list[float]   ([0, 1] fraction skipped)
        n_successful_runs         int
        success                   bool
    """
    n_words = len(tokens)
    accum_trt  = [0.0] * n_words
    accum_ffd  = [0.0] * n_words
    accum_gaze = [0.0] * n_words
    accum_skip = [0.0] * n_words
    n_success = 0

    for _ in range(num_runs):
        r = _run_one_simulation(
            tokens=tokens,
            frequencies=frequencies,
            predictabilities=predictabilities,
            integration_time=integration_time,
            integration_failure=integration_failure,
            timeout_seconds=timeout_seconds,
            model_params=model_params,
        )
        if not r["success"]:
            continue
        for i in range(n_words):
            accum_trt[i]  += r["total_reading_time"][i]
            accum_ffd[i]  += r["first_fixation_duration"][i]
            accum_gaze[i] += r["gaze_duration"][i]
            accum_skip[i] += 1.0 if r["was_skipped"][i] else 0.0
        n_success += 1

    if n_success == 0:
        return {
            "total_reading_time": [0.0] * n_words,
            "first_fixation_duration": [0.0] * n_words,
            "gaze_duration": [0.0] * n_words,
            "skip_rate": [1.0] * n_words,
            "n_successful_runs": 0,
            "success": False,
        }

    return {
        "total_reading_time":      [v / n_success for v in accum_trt],
        "first_fixation_duration": [v / n_success for v in accum_ffd],
        "gaze_duration":           [v / n_success for v in accum_gaze],
        "skip_rate":               [v / n_success for v in accum_skip],
        "n_successful_runs": n_success,
        "success": True,
    }


# --------------------------------------------------------------------------- #
#  Quick sanity test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    tokens = ["The", "incomprehensible", "consciousness", "fluctuated", "enormously"]
    freqs  = [1e7, 100, 500, 200, 800]
    preds  = [0.5, 0.01, 0.05, 0.05, 0.05]
    print("Default params, 200 MC...")
    out = run_classical_averaged(tokens, freqs, preds, num_runs=200)
    for i, t in enumerate(tokens):
        print(f"  {t:>20s}  TRT={out['total_reading_time'][i]:7.1f}  "
              f"Gaze={out['gaze_duration'][i]:7.1f}  "
              f"FFD={out['first_fixation_duration'][i]:7.1f}  "
              f"skip={out['skip_rate'][i]:.2f}")
    print(f"  successful runs: {out['n_successful_runs']}/200")
