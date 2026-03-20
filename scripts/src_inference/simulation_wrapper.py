"""
Run the Neural EZ Reader simulation and extract per-word reading metrics.

Provides:
  - run_simulation():          single run, returns TRT/FFD/Gaze/Skip per word
  - run_simulation_averaged(): average over K stochastic runs
  - run_original_simulation(): original formula-based EZR (for comparison)
  - run_original_simulation_averaged(): averaged original EZR
"""

from collections import defaultdict
import simpy

from simulation_engine import Simulation, NeuralEZReaderSim, Word


# --------------------------------------------------------------------------- #
#  Fixation tracker — extracts TRT, FFD, Gaze, Skip from simulation steps
# --------------------------------------------------------------------------- #

class FixationTracker:
    """
    Tracks fixation positions step-by-step and extracts reading measures.

    Measures:
      - TRT (total reading time): sum of ALL fixation durations on a word
      - FFD (first fixation duration): duration of the FIRST fixation on a word
      - Gaze duration: sum of fixation durations during first-pass
        (from first fixation until the reader leaves the word for the first time)
      - Skip: word was never fixated
    """

    def __init__(self, n_words, position_map):
        """
        Args:
            n_words: number of words in the sentence
            position_map: dict of (char_start, char_end) -> word_index
        """
        self.n_words = n_words
        self.position_map = position_map

        self.all_fixations = defaultdict(list)         # word_idx -> [(start, end)]
        self.first_pass_fixations = defaultdict(list)  # word_idx -> [(start, end)]
        self.gaze_complete = set()                     # words whose first-pass is done

        self.current_word_idx = None
        self.current_fixation_start = 0.0

    def _get_word_idx(self, fixation_point):
        """Map a character-level fixation point to a word index."""
        for (start, end), idx in self.position_map.items():
            if start <= fixation_point <= end:
                return idx
        return None

    def update(self, fixation_point, time_ms):
        """Call after each simulation step with current fixation point and time."""
        new_word_idx = self._get_word_idx(fixation_point)

        if new_word_idx != self.current_word_idx:
            # Leaving current word — record the fixation interval
            if self.current_word_idx is not None:
                interval = (self.current_fixation_start, time_ms)
                self.all_fixations[self.current_word_idx].append(interval)
                if self.current_word_idx not in self.gaze_complete:
                    self.first_pass_fixations[self.current_word_idx].append(interval)
                # First pass on this word is now over (reader left it)
                self.gaze_complete.add(self.current_word_idx)

            self.current_word_idx = new_word_idx
            self.current_fixation_start = time_ms

    def finalize(self, time_ms):
        """Call when simulation ends to flush the last fixation."""
        if self.current_word_idx is not None:
            interval = (self.current_fixation_start, time_ms)
            self.all_fixations[self.current_word_idx].append(interval)
            if self.current_word_idx not in self.gaze_complete:
                self.first_pass_fixations[self.current_word_idx].append(interval)

    def get_metrics(self):
        """
        Extract per-word reading measures.

        Returns:
            dict with lists of length n_words:
              'total_reading_time', 'first_fixation_duration',
              'gaze_duration', 'was_skipped', 'n_fixations'
        """
        total_reading_time = []
        first_fixation_duration = []
        gaze_duration = []
        was_skipped = []
        n_fixations = []

        for i in range(self.n_words):
            all_fix = self.all_fixations.get(i, [])
            fp_fix = self.first_pass_fixations.get(i, [])

            if len(all_fix) == 0:
                total_reading_time.append(0.0)
                first_fixation_duration.append(0.0)
                gaze_duration.append(0.0)
                was_skipped.append(True)
                n_fixations.append(0)
            else:
                trt = sum(end - start for start, end in all_fix)
                ffd = all_fix[0][1] - all_fix[0][0]
                gaze = sum(end - start for start, end in fp_fix)

                total_reading_time.append(max(0.0, trt))
                first_fixation_duration.append(max(0.0, ffd))
                gaze_duration.append(max(0.0, gaze))
                was_skipped.append(False)
                n_fixations.append(len(all_fix))

        return {
            'total_reading_time': total_reading_time,
            'first_fixation_duration': first_fixation_duration,
            'gaze_duration': gaze_duration,
            'was_skipped': was_skipped,
            'n_fixations': n_fixations,
        }


# --------------------------------------------------------------------------- #
#  Build helpers
# --------------------------------------------------------------------------- #

def _build_sentence_and_map(tokens, predictabilities=None,
                            integration_time=25, integration_failure=0.01,
                            default_freq=1e4):
    """Build Word tuples and position->word_index map."""
    n_words = len(tokens)
    if predictabilities is None:
        predictabilities = [0.0] * n_words

    sentence = []
    for i in range(n_words):
        sentence.append(Word(
            token=tokens[i],
            frequency=default_freq,
            predictability=predictabilities[i],
            integration_time=integration_time,
            integration_failure=integration_failure,
        ))

    position_map = {}
    pos = 1
    for i, w in enumerate(sentence):
        start = pos
        end = pos + 1 + len(w.token)
        position_map[(start, end)] = i
        pos = end

    return sentence, position_map


def _run_sim_steps(sim, position_map, n_words, timeout_seconds=5.0, max_steps=10000):
    """Step through simulation, track fixations, return metrics dict."""
    tracker = FixationTracker(n_words, position_map)
    step_count = 0

    while step_count < max_steps:
        try:
            sim.step()
            step_count += 1
            tracker.update(sim.fixation_point, sim.time)

            if sim.time > timeout_seconds * 1000:
                break
        except simpy.core.EmptySchedule:
            tracker.finalize(sim.time)
            break

    return tracker.get_metrics()


# --------------------------------------------------------------------------- #
#  Neural simulation (single run + averaged)
# --------------------------------------------------------------------------- #

def run_simulation(tokens, l1_times, l2_times,
                   skip_probs=None,
                   predictabilities=None,
                   integration_time=25,
                   integration_failure=0.01,
                   timeout_seconds=5.0):
    """
    Run Neural EZ Reader simulation once.

    Args:
        tokens:           list[str] - word tokens
        l1_times:         list[float] - L1 time per word (ms) from neural model
        l2_times:         list[float] - L2 time per word (ms) from neural model
        skip_probs:       list[float] or None - skip probability per word (0-1)
        predictabilities: list[float] or None - cloze predictability (fallback for skip)
        integration_time: float - integration time (ms)
        integration_failure: float - integration failure probability
        timeout_seconds:  float - max simulation time (seconds)

    Returns:
        dict with per-word metrics + 'success' flag
    """
    n_words = len(tokens)
    sentence, position_map = _build_sentence_and_map(
        tokens, predictabilities, integration_time, integration_failure,
    )

    try:
        sim = NeuralEZReaderSim(
            sentence=sentence,
            l1_times=l1_times,
            l2_times=l2_times,
            skip_probs=skip_probs,
            trace=False,
        )

        metrics = _run_sim_steps(sim, position_map, n_words, timeout_seconds)
        metrics['success'] = True
        return metrics

    except Exception as e:
        return {
            'total_reading_time': [0.0] * n_words,
            'first_fixation_duration': [0.0] * n_words,
            'gaze_duration': [0.0] * n_words,
            'was_skipped': [True] * n_words,
            'n_fixations': [0] * n_words,
            'success': False,
            'error': str(e),
        }


def run_simulation_averaged(tokens, l1_times, l2_times,
                            skip_probs=None,
                            predictabilities=None,
                            num_runs=50,
                            integration_time=25,
                            integration_failure=0.01,
                            timeout_seconds=5.0):
    """
    Run Neural EZ Reader simulation K times and average results.

    Returns:
        dict with averaged per-word metrics:
            'total_reading_time', 'first_fixation_duration',
            'gaze_duration', 'skip_rate', 'mean_n_fixations', 'success'
    """
    n_words = len(tokens)
    all_trt, all_ffd, all_gaze, all_skip, all_nfix = [], [], [], [], []
    n_success = 0

    for _ in range(num_runs):
        result = run_simulation(
            tokens, l1_times, l2_times,
            skip_probs=skip_probs,
            predictabilities=predictabilities,
            integration_time=integration_time,
            integration_failure=integration_failure,
            timeout_seconds=timeout_seconds,
        )
        if result['success']:
            all_trt.append(result['total_reading_time'])
            all_ffd.append(result['first_fixation_duration'])
            all_gaze.append(result['gaze_duration'])
            all_skip.append([1.0 if s else 0.0 for s in result['was_skipped']])
            all_nfix.append([float(n) for n in result['n_fixations']])
            n_success += 1

    if n_success == 0:
        return {
            'total_reading_time': [0.0] * n_words,
            'first_fixation_duration': [0.0] * n_words,
            'gaze_duration': [0.0] * n_words,
            'skip_rate': [1.0] * n_words,
            'mean_n_fixations': [0.0] * n_words,
            'success': False,
        }

    avg = lambda arrs, i: sum(run[i] for run in arrs) / n_success
    return {
        'total_reading_time':      [avg(all_trt, i) for i in range(n_words)],
        'first_fixation_duration': [avg(all_ffd, i) for i in range(n_words)],
        'gaze_duration':           [avg(all_gaze, i) for i in range(n_words)],
        'skip_rate':               [avg(all_skip, i) for i in range(n_words)],
        'mean_n_fixations':        [avg(all_nfix, i) for i in range(n_words)],
        'success': True,
        'n_successful_runs': n_success,
    }


# --------------------------------------------------------------------------- #
#  Original EZ Reader simulation (formula-based, for comparison)
# --------------------------------------------------------------------------- #

def run_original_simulation(tokens, frequencies, predictabilities,
                            integration_time=25, integration_failure=0.01,
                            timeout_seconds=5.0, model_params=None):
    """Run the original formula-based EZ Reader simulation once."""
    n_words = len(tokens)

    sentence = []
    for i in range(n_words):
        sentence.append(Word(
            token=tokens[i],
            frequency=max(1, frequencies[i]),
            predictability=predictabilities[i],
            integration_time=integration_time,
            integration_failure=integration_failure,
        ))

    position_map = {}
    pos = 1
    for i, w in enumerate(sentence):
        start = pos
        end = pos + 1 + len(w.token)
        position_map[(start, end)] = i
        pos = end

    try:
        sim = Simulation(sentence=sentence, trace=False)
        if model_params:
            sim.model_parameters.update(model_params)

        metrics = _run_sim_steps(sim, position_map, n_words, timeout_seconds)
        metrics['success'] = True
        return metrics

    except Exception as e:
        return {
            'total_reading_time': [0.0] * n_words,
            'first_fixation_duration': [0.0] * n_words,
            'gaze_duration': [0.0] * n_words,
            'was_skipped': [True] * n_words,
            'n_fixations': [0] * n_words,
            'success': False,
            'error': str(e),
        }


def run_original_simulation_averaged(tokens, frequencies, predictabilities,
                                      num_runs=50, integration_time=25,
                                      integration_failure=0.01,
                                      timeout_seconds=5.0, model_params=None):
    """Run original EZ Reader K times and average results."""
    n_words = len(tokens)
    all_trt, all_ffd, all_gaze, all_skip, all_nfix = [], [], [], [], []
    n_success = 0

    for _ in range(num_runs):
        result = run_original_simulation(
            tokens, frequencies, predictabilities,
            integration_time=integration_time,
            integration_failure=integration_failure,
            timeout_seconds=timeout_seconds,
            model_params=model_params,
        )
        if result['success']:
            all_trt.append(result['total_reading_time'])
            all_ffd.append(result['first_fixation_duration'])
            all_gaze.append(result['gaze_duration'])
            all_skip.append([1.0 if s else 0.0 for s in result['was_skipped']])
            all_nfix.append([float(n) for n in result['n_fixations']])
            n_success += 1

    if n_success == 0:
        return {
            'total_reading_time': [0.0] * n_words,
            'first_fixation_duration': [0.0] * n_words,
            'gaze_duration': [0.0] * n_words,
            'skip_rate': [1.0] * n_words,
            'mean_n_fixations': [0.0] * n_words,
            'success': False,
        }

    avg = lambda arrs, i: sum(run[i] for run in arrs) / n_success
    return {
        'total_reading_time':      [avg(all_trt, i) for i in range(n_words)],
        'first_fixation_duration': [avg(all_ffd, i) for i in range(n_words)],
        'gaze_duration':           [avg(all_gaze, i) for i in range(n_words)],
        'skip_rate':               [avg(all_skip, i) for i in range(n_words)],
        'mean_n_fixations':        [avg(all_nfix, i) for i in range(n_words)],
        'success': True,
        'n_successful_runs': n_success,
    }


# --------------------------------------------------------------------------- #
#  Quick test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    tokens = ["The", "quick", "brown", "fox", "jumps"]
    l1_times = [30.0, 80.0, 75.0, 90.0, 85.0]
    l2_times = [10.0, 25.0, 22.0, 30.0, 28.0]
    skip_probs = [0.7, 0.1, 0.1, 0.2, 0.15]

    print("=" * 60)
    print("  Neural EZ Reader Simulation (single run)")
    print("=" * 60)
    result = run_simulation(tokens, l1_times, l2_times, skip_probs=skip_probs)
    print(f"Success: {result['success']}")
    for i, tok in enumerate(tokens):
        print(f"  {tok:10s}  TRT={result['total_reading_time'][i]:7.1f}  "
              f"FFD={result['first_fixation_duration'][i]:7.1f}  "
              f"Gaze={result['gaze_duration'][i]:7.1f}  "
              f"Skip={result['was_skipped'][i]}")

    print(f"\n{'=' * 60}")
    print("  Neural EZ Reader Simulation (averaged, 50 runs)")
    print("=" * 60)
    result = run_simulation_averaged(tokens, l1_times, l2_times,
                                     skip_probs=skip_probs, num_runs=50)
    print(f"Success: {result['success']}  ({result.get('n_successful_runs', 0)} runs)")
    for i, tok in enumerate(tokens):
        print(f"  {tok:10s}  TRT={result['total_reading_time'][i]:7.1f}  "
              f"FFD={result['first_fixation_duration'][i]:7.1f}  "
              f"Gaze={result['gaze_duration'][i]:7.1f}  "
              f"Skip={result['skip_rate'][i]:.3f}")
