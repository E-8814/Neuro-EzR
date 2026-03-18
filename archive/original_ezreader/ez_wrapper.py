"""
EZ Reader Wrapper for RL Training.

Wraps the original EZ Reader simulation so it can be used as a
"black-box environment" in the RL loop.

Given L1/L2 times (from the LSTM policy), it:
    1. Builds Word tuples for the simulation
    2. Runs the full discrete-event simulation
    3. Extracts per-word reading time metrics
    4. Returns them as plain Python lists (no gradients)
"""

import sys
import os
import math
from collections import defaultdict

import simpy

# Ensure src is importable
sys.path.insert(0, os.path.dirname(__file__))

from ez_reader_engine import Simulation, Word, Action


# --------------------------------------------------------------------------- #
#  Neural EZ Reader Simulation
# --------------------------------------------------------------------------- #

class NeuralEZReaderSim(Simulation):
    """
    Modified EZ Reader that uses externally-provided L1/L2 times
    instead of the frequency/predictability formulas.

    The L1/L2 values come from the LSTM policy.
    Everything else (saccades, integration, regressions) stays original.
    """

    def __init__(self, sentence, l1_times, l2_times, **kwargs):
        """
        Args:
            sentence: list of Word namedtuples
            l1_times: list of floats, L1 time (ms) per word (from LSTM)
            l2_times: list of floats, L2 time (ms) per word (from LSTM)
            **kwargs: passed to Simulation.__init__
        """
        self._neural_l1 = l1_times
        self._neural_l2 = l2_times
        # Track per-word fixation times
        self._word_fixation_log = defaultdict(float)
        self._word_fixation_start = {}
        self._current_fixated_word = None
        super().__init__(sentence, **kwargs)

    def visual_processing(self, sentence):
        """
        Override visual_processing to use neural L1/L2 times.
        Everything else is identical to the original.
        """
        from numpy.random import uniform
        import utilities as ut

        first_letter = 1

        for i, elem in enumerate(sentence):
            self.attended_word = elem

            distance = first_letter - self.fixation_point

            # --- NEURAL L1 ---
            # Use the LSTM's predicted L1 instead of the formula
            # Still apply the skip logic based on predictability
            random_draw = uniform()

            if float(elem.predictability) > random_draw:
                time_familiarity_check = 0
            else:
                # Use neural L1, scaled by eccentricity (visual factor)
                word_len = len(elem.token)
                eccentricity = self.model_parameters['eccentricity']
                ecc_factor = pow(eccentricity, abs(distance + (word_len - 1) / 2.0))
                time_familiarity_check = max(0, self._neural_l1[i]) * ecc_factor

            yield self._timeout(time_familiarity_check)
            yield self._timeout(self._repeated_attention)
            self._repeated_attention = 0

            self._collect_action(Action('L1', " ".join(["Word:", str(elem.token)]), self.time))

            # Start programming movement to the next word
            try:
                next_elem = sentence[i + 1]
            except IndexError:
                pass
            else:
                new_fixation_point = first_letter + len(elem.token) + 0.5 + len(next_elem.token) / 2
                self._prepare_saccade(new_fixation_point, str(next_elem.token))

            # --- NEURAL L2 ---
            # Use the LSTM's predicted L2 instead of the formula
            time_lexical_access = max(0, self._neural_l2[i])

            yield self._timeout(time_lexical_access)
            yield self._timeout(self._repeated_attention)
            self._repeated_attention = 0

            self._collect_action(Action('L2', " ".join(["Word:", str(elem.token)]), self.time))

            ########################
            #  start integration   #
            ########################
            if i > 0:
                prev_pos = first_letter - 0.5 - len(sentence[i - 1].token) / 2
                prev_word = sentence[i - 1].token
            else:
                prev_pos = 0
                prev_elem = Word('None', 1e06, 1, 0, 0)

            random_draw = uniform()

            if float(self.model_parameters["probability_correct_regression"]) >= random_draw:
                self.env.process(self._integration(
                    last_letter=first_letter + len(elem.token),
                    new_fixation_point=first_letter - 0.5 + len(elem.token) / 2,
                    new_fixation_point2=new_fixation_point if 'new_fixation_point' in dir() else 0,
                    elem=elem,
                    elem_for_attention=elem,
                    next_elem=next_elem if 'next_elem' in dir() else elem,
                ))
            else:
                self.env.process(self._integration(
                    last_letter=first_letter - 2,
                    new_fixation_point=prev_pos,
                    new_fixation_point2=new_fixation_point if 'new_fixation_point' in dir() else 0,
                    elem=elem,
                    elem_for_attention=prev_elem if 'prev_elem' in dir() else elem,
                    next_elem=next_elem if 'next_elem' in dir() else elem,
                ))

            ########################
            #   end integration    #
            ########################

            time_attention_shift = self.model_parameters["time_attention_shift"]
            yield self._timeout(time_attention_shift)
            yield self._timeout(self._repeated_attention)
            self._repeated_attention = 0

            self._collect_action(Action('Attention shift', " ".join(["From word:", str(elem.token)]), self.time))

            first_letter += len(elem.token) + 1


# --------------------------------------------------------------------------- #
#  Wrapper: run simulation and extract reading times
# --------------------------------------------------------------------------- #

def run_simulation(tokens, l1_times, l2_times,
                   predictabilities=None,
                   integration_time=25,
                   integration_failure=0.01,
                   default_freq=1e4,
                   timeout_seconds=5.0):
    """
    Run the Neural EZ Reader simulation and extract per-word reading metrics.

    Args:
        tokens:           list[str] - word tokens
        l1_times:         list[float] - L1 time per word (from LSTM, in ms)
        l2_times:         list[float] - L2 time per word (from LSTM, in ms)
        predictabilities: list[float] or None - cloze predictability per word
        integration_time: float - integration time (ms), typically constant
        integration_failure: float - integration failure probability
        default_freq:     float - default word frequency
        timeout_seconds:  float - max sim time in seconds (safety limit)

    Returns:
        dict with per-word metrics:
            'total_reading_time': list[float] - total fixation time per word (ms)
            'first_fixation_duration': list[float] - first fixation duration (ms)
            'was_skipped': list[bool] - whether word was skipped
            'n_fixations': list[int] - number of fixations per word
            'success': bool - whether simulation completed without error
    """
    n_words = len(tokens)

    if predictabilities is None:
        predictabilities = [0.0] * n_words

    # Build Word tuples
    sentence = []
    for i in range(n_words):
        w = Word(
            token=tokens[i],
            frequency=default_freq,
            predictability=predictabilities[i],
            integration_time=integration_time,
            integration_failure=integration_failure,
        )
        sentence.append(w)

    # Build position -> word mapping (for tracking fixations)
    position_map = {}  # (start, end) -> word_index
    pos = 1
    for i, w in enumerate(sentence):
        start = pos
        end = pos + 1 + len(w.token)
        position_map[(start, end)] = i
        pos = end

    # Run simulation
    try:
        sim = NeuralEZReaderSim(
            sentence=sentence,
            l1_times=l1_times,
            l2_times=l2_times,
            realtime=False,
            trace=False,
        )

        # Track fixations per word
        fixations_per_word = defaultdict(list)  # word_idx -> [(start_time, end_time)]
        current_word_idx = None
        current_fixation_start = 0.0

        # Run step by step, tracking which word is fixated
        max_steps = 10000  # safety limit
        step_count = 0

        while step_count < max_steps:
            try:
                sim.step()
                step_count += 1

                # Determine which word is currently fixated
                new_word_idx = None
                for (start, end), idx in position_map.items():
                    if start <= sim.fixation_point <= end:
                        new_word_idx = idx
                        break

                # Track fixation transitions
                if new_word_idx != current_word_idx:
                    if current_word_idx is not None:
                        fixations_per_word[current_word_idx].append(
                            (current_fixation_start, sim.time)
                        )
                    current_word_idx = new_word_idx
                    current_fixation_start = sim.time

                # Safety: check total time
                if sim.time > timeout_seconds * 1000:
                    break

            except simpy.core.EmptySchedule:
                # Simulation complete
                if current_word_idx is not None:
                    fixations_per_word[current_word_idx].append(
                        (current_fixation_start, sim.time)
                    )
                break

        # Extract metrics
        total_reading_time = []
        first_fixation_duration = []
        was_skipped = []
        n_fixations = []

        for i in range(n_words):
            fixations = fixations_per_word.get(i, [])
            if len(fixations) == 0:
                total_reading_time.append(0.0)
                first_fixation_duration.append(0.0)
                was_skipped.append(True)
                n_fixations.append(0)
            else:
                total_time = sum(end - start for start, end in fixations)
                first_fix = fixations[0][1] - fixations[0][0]
                total_reading_time.append(max(0.0, total_time))
                first_fixation_duration.append(max(0.0, first_fix))
                was_skipped.append(False)
                n_fixations.append(len(fixations))

        return {
            'total_reading_time': total_reading_time,
            'first_fixation_duration': first_fixation_duration,
            'was_skipped': was_skipped,
            'n_fixations': n_fixations,
            'success': True,
        }

    except Exception as e:
        # If simulation crashes, return zeros (bad reward will discourage this)
        return {
            'total_reading_time': [0.0] * n_words,
            'first_fixation_duration': [0.0] * n_words,
            'was_skipped': [True] * n_words,
            'n_fixations': [0] * n_words,
            'success': False,
            'error': str(e),
        }


def run_simulation_averaged(tokens, l1_times, l2_times,
                            predictabilities=None,
                            num_runs=10,
                            integration_time=25,
                            integration_failure=0.01,
                            default_freq=1e4,
                            timeout_seconds=5.0):
    """
    Run the EZ Reader simulation K times with the same L1/L2 values
    and AVERAGE the reading time metrics. This dramatically reduces
    simulation stochasticity.

    Args:
        tokens:           list[str]
        l1_times:         list[float] - L1 per word (ms)
        l2_times:         list[float] - L2 per word (ms)
        predictabilities: list[float] or None
        num_runs:         int - number of simulation runs to average
        (other args same as run_simulation)

    Returns:
        dict with AVERAGED per-word metrics:
            'total_reading_time': list[float]
            'first_fixation_duration': list[float]
            'skip_rate': list[float] - fraction of runs where word was skipped
            'mean_n_fixations': list[float]
            'success': bool
    """
    n_words = len(tokens)
    all_trt = []
    all_ffd = []
    all_skip = []
    all_nfix = []
    n_success = 0

    for _ in range(num_runs):
        result = run_simulation(
            tokens=tokens,
            l1_times=l1_times,
            l2_times=l2_times,
            predictabilities=predictabilities,
            integration_time=integration_time,
            integration_failure=integration_failure,
            default_freq=default_freq,
            timeout_seconds=timeout_seconds,
        )
        if result['success']:
            all_trt.append(result['total_reading_time'])
            all_ffd.append(result['first_fixation_duration'])
            all_skip.append([1.0 if s else 0.0 for s in result['was_skipped']])
            all_nfix.append([float(n) for n in result['n_fixations']])
            n_success += 1

    if n_success == 0:
        return {
            'total_reading_time': [0.0] * n_words,
            'first_fixation_duration': [0.0] * n_words,
            'skip_rate': [1.0] * n_words,
            'mean_n_fixations': [0.0] * n_words,
            'success': False,
        }

    # Average across runs
    avg_trt = [sum(run[i] for run in all_trt) / n_success for i in range(n_words)]
    avg_ffd = [sum(run[i] for run in all_ffd) / n_success for i in range(n_words)]
    avg_skip = [sum(run[i] for run in all_skip) / n_success for i in range(n_words)]
    avg_nfix = [sum(run[i] for run in all_nfix) / n_success for i in range(n_words)]

    return {
        'total_reading_time': avg_trt,
        'first_fixation_duration': avg_ffd,
        'skip_rate': avg_skip,
        'mean_n_fixations': avg_nfix,
        'success': True,
        'n_successful_runs': n_success,
    }


# --------------------------------------------------------------------------- #
#  Run the REAL original EZ Reader (uses its own L1/L2 formulas internally)
# --------------------------------------------------------------------------- #

def run_original_simulation(tokens, frequencies, predictabilities,
                            integration_time=25, integration_failure=0.01,
                            timeout_seconds=5.0, model_params=None):
    """
    Run the ACTUAL original EZ Reader simulation.

    Unlike run_simulation() which takes pre-computed L1/L2 times,
    this uses the real Simulation class that computes L1/L2 internally
    using the formula with proper fixation distance.

    Args:
        tokens:           list[str] - word tokens
        frequencies:      list[float] - word frequency counts (from SUBTLEXus etc.)
        predictabilities: list[float] - cloze predictability per word (0-1)
        integration_time: float - integration time (ms)
        integration_failure: float - integration failure probability
        timeout_seconds:  float - max sim time in seconds
        model_params:     dict - optional custom EZ Reader parameters
                          (alpha1, alpha2, alpha3, delta, eccentricity, etc.)

    Returns:
        dict with per-word metrics (same format as run_simulation)
    """
    n_words = len(tokens)

    # Build Word tuples with REAL frequencies
    sentence = []
    for i in range(n_words):
        w = Word(
            token=tokens[i],
            frequency=max(1, frequencies[i]),
            predictability=predictabilities[i],
            integration_time=integration_time,
            integration_failure=integration_failure,
        )
        sentence.append(w)

    # Build position -> word mapping
    position_map = {}
    pos = 1
    for i, w in enumerate(sentence):
        start = pos
        end = pos + 1 + len(w.token)
        position_map[(start, end)] = i
        pos = end

    try:
        # Temporarily override model parameters if custom ones provided
        original_params = None
        if model_params:
            original_params = dict(Simulation.model_parameters)
            Simulation.model_parameters.update(model_params)

        # Use the REAL Simulation class (not NeuralEZReaderSim)
        sim = Simulation(
            sentence=sentence,
            realtime=False,
            trace=False,
        )

        fixations_per_word = defaultdict(list)
        current_word_idx = None
        current_fixation_start = 0.0
        max_steps = 10000
        step_count = 0

        while step_count < max_steps:
            try:
                sim.step()
                step_count += 1

                new_word_idx = None
                for (start, end), idx in position_map.items():
                    if start <= sim.fixation_point <= end:
                        new_word_idx = idx
                        break

                if new_word_idx != current_word_idx:
                    if current_word_idx is not None:
                        fixations_per_word[current_word_idx].append(
                            (current_fixation_start, sim.time)
                        )
                    current_word_idx = new_word_idx
                    current_fixation_start = sim.time

                if sim.time > timeout_seconds * 1000:
                    break

            except simpy.core.EmptySchedule:
                if current_word_idx is not None:
                    fixations_per_word[current_word_idx].append(
                        (current_fixation_start, sim.time)
                    )
                break

        total_reading_time = []
        first_fixation_duration = []
        was_skipped = []
        n_fixations = []

        for i in range(n_words):
            fixations = fixations_per_word.get(i, [])
            if len(fixations) == 0:
                total_reading_time.append(0.0)
                first_fixation_duration.append(0.0)
                was_skipped.append(True)
                n_fixations.append(0)
            else:
                total_time = sum(end - start for start, end in fixations)
                first_fix = fixations[0][1] - fixations[0][0]
                total_reading_time.append(max(0.0, total_time))
                first_fixation_duration.append(max(0.0, first_fix))
                was_skipped.append(False)
                n_fixations.append(len(fixations))

        return {
            'total_reading_time': total_reading_time,
            'first_fixation_duration': first_fixation_duration,
            'was_skipped': was_skipped,
            'n_fixations': n_fixations,
            'success': True,
        }

    except Exception as e:
        return {
            'total_reading_time': [0.0] * n_words,
            'first_fixation_duration': [0.0] * n_words,
            'was_skipped': [True] * n_words,
            'n_fixations': [0] * n_words,
            'success': False,
            'error': str(e),
        }

    finally:
        if original_params is not None:
            Simulation.model_parameters = original_params


def run_original_simulation_averaged(tokens, frequencies, predictabilities,
                                      num_runs=20, integration_time=25,
                                      integration_failure=0.01,
                                      timeout_seconds=5.0, model_params=None):
    """
    Run the real original EZ Reader K times and average the results.

    Args:
        tokens:           list[str]
        frequencies:      list[float] - real word frequencies
        predictabilities: list[float]
        num_runs:         int - number of runs to average
        model_params:     dict - optional custom EZ Reader parameters
        (other args same as run_original_simulation)

    Returns:
        dict with averaged per-word metrics
    """
    n_words = len(tokens)
    all_trt, all_ffd, all_skip, all_nfix = [], [], [], []
    n_success = 0

    for _ in range(num_runs):
        result = run_original_simulation(
            tokens=tokens,
            frequencies=frequencies,
            predictabilities=predictabilities,
            integration_time=integration_time,
            integration_failure=integration_failure,
            timeout_seconds=timeout_seconds,
            model_params=model_params,
        )
        if result['success']:
            all_trt.append(result['total_reading_time'])
            all_ffd.append(result['first_fixation_duration'])
            all_skip.append([1.0 if s else 0.0 for s in result['was_skipped']])
            all_nfix.append([float(n) for n in result['n_fixations']])
            n_success += 1

    if n_success == 0:
        return {
            'total_reading_time': [0.0] * n_words,
            'first_fixation_duration': [0.0] * n_words,
            'skip_rate': [1.0] * n_words,
            'mean_n_fixations': [0.0] * n_words,
            'success': False,
        }

    avg_trt = [sum(run[i] for run in all_trt) / n_success for i in range(n_words)]
    avg_ffd = [sum(run[i] for run in all_ffd) / n_success for i in range(n_words)]
    avg_skip = [sum(run[i] for run in all_skip) / n_success for i in range(n_words)]
    avg_nfix = [sum(run[i] for run in all_nfix) / n_success for i in range(n_words)]

    return {
        'total_reading_time': avg_trt,
        'first_fixation_duration': avg_ffd,
        'skip_rate': avg_skip,
        'mean_n_fixations': avg_nfix,
        'success': True,
        'n_successful_runs': n_success,
    }


# --------------------------------------------------------------------------- #
#  Quick test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    tokens = ["The", "quick", "brown", "fox"]
    l1_times = [30.0, 60.0, 55.0, 70.0]
    l2_times = [15.0, 25.0, 20.0, 30.0]
    preds = [0.9, 0.1, 0.1, 0.3]

    print("Running Neural EZ Reader simulation...")
    result = run_simulation(tokens, l1_times, l2_times, predictabilities=preds)

    print(f"\nSuccess: {result['success']}")
    for i, token in enumerate(tokens):
        print(f"  {token:10s}  TRT={result['total_reading_time'][i]:7.1f}ms  "
              f"FFD={result['first_fixation_duration'][i]:7.1f}ms  "
              f"skip={result['was_skipped'][i]}  "
              f"nfix={result['n_fixations'][i]}")
