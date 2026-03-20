"""
Discrete-event E-Z Reader simulation for inference.

Two simulation classes:
  - Simulation:           Original formula-based EZR (for comparison)
  - NeuralEZReaderSim:    Neural EZR that takes L1/L2/skip from a trained model

NeuralEZReaderSim is parameterized by the trained model's learned parameters
(via sim_params dict from model.get_sim_params()). This ensures the simulation
uses the same eccentricity, motor timing, and integration failure semantics
that the model was trained with.

Based on the original ez_reader/ez_reader_engine.py with these fixes:
  - sim_params dict overrides model_parameters (no hardcoded mismatch)
  - Linear eccentricity matching diff_ezreader (not exponential)
  - Per-word integration_failure from L2 (not flat 0.01)
  - Neural skip_probs used by default (not cloze predictability)
  - _attend_again word lookup fixed (sentence iteration, not id())
"""

import math
from collections import namedtuple, defaultdict

import numpy as np
from numpy.random import uniform, normal
import simpy


OPTIMAL_SACCADE_LENGTH = 7

Word = namedtuple(
    'Word', 'token frequency predictability integration_time integration_failure'
)
Action = namedtuple('Action', 'name details time')


# --------------------------------------------------------------------------- #
#  Original L1/L2 formulas (used by base Simulation only)
# --------------------------------------------------------------------------- #

def _time_familiarity_check(distance, wordlength, frequency, predictability,
                            eccentricity, alpha1, alpha2, alpha3):
    tL1 = alpha1 - alpha2 * math.log(max(1, frequency)) - alpha3 * predictability
    ecc_factor = pow(eccentricity, abs(distance + (wordlength - 1) / 2.0))
    return max(0, tL1 * ecc_factor)


def _time_lexical_access(frequency, predictability, delta, alpha1, alpha2, alpha3):
    tL2 = delta * (alpha1 - alpha2 * math.log(max(1, frequency)) - alpha3 * predictability)
    return max(0, tL2)


# --------------------------------------------------------------------------- #
#  Base Simulation (original E-Z Reader, formula-based)
# --------------------------------------------------------------------------- #

class Simulation(object):
    """Original E-Z Reader discrete-event simulation."""

    _default_parameters = {
        "alpha1": 104,
        "alpha2": 3.4,
        "alpha3": 39,
        "eccentricity": 1.15,
        "delta": 0.34,
        "predictability_repeated_attention": 0.9,
        "saccade_programming": 125,
        "saccade_finishing": 25,
        "time_attention_shift": 25,
        "omega1": 6,
        "omega2": 3,
        "eta1": 0.5,
        "eta2": 0.15,
        "lambda": 0.16,
        "probability_correct_regression": 0.6,
    }

    def __init__(self, sentence, realtime=False, initial_time=0,
                 initial_fixation=1, trace=False):
        self.model_parameters = dict(self._default_parameters)

        if realtime:
            self.env = simpy.RealtimeEnvironment(initial_time=initial_time)
        else:
            self.env = simpy.Environment(initial_time=initial_time)

        self.env.process(self.visual_processing(sentence))
        self.fixation_point = initial_fixation
        self.attended_word = None
        self.fixated_word = None
        self.last_action = None
        self.trace = trace
        self._canbeinterrupted = True
        self._plan_saccade = False
        self._saccade = None
        self._repeated_attention = 0
        self._fixation_launch_site = 0
        self._word_position_dict = {}

        position = 1
        for word in sentence:
            self._word_position_dict[
                (position, position + 1 + len(word.token))
            ] = word.token
            position += 1 + len(word.token)

        for pos_range in self._word_position_dict:
            if (initial_fixation >= pos_range[0]
                    and initial_fixation <= pos_range[1]):
                self.fixated_word = self._word_position_dict[pos_range]
                break

    @property
    def time(self):
        return 1000 * self.env.now

    def _timeout(self, time_in_ms):
        return self.env.timeout(time_in_ms / 1000)

    def _collect_action(self, action):
        self.last_action = action
        if self.trace:
            print(self.last_action)

    def _prepare_saccade(self, new_fixation_point, word,
                         canbeinterrupted=True):
        if self._canbeinterrupted:
            try:
                self._saccade.interrupt()
            except (AttributeError, RuntimeError):
                pass
            if ((float(self.fixation_point)
                 < float(new_fixation_point) - len(word) / 2)
                or (float(self.fixation_point)
                    > float(new_fixation_point) + len(word) / 2)):
                self._saccade = self.env.process(
                    self._saccadic_programming(
                        new_fixation_point=new_fixation_point,
                        word=word,
                        canbeinterrupted=canbeinterrupted,
                    )
                )
        else:
            if self.fixation_point != new_fixation_point:
                self._plan_saccade = (
                    new_fixation_point, word, canbeinterrupted
                )

    def _saccadic_programming(self, new_fixation_point, word,
                              regression=False, canbeinterrupted=True):
        self._collect_action(Action(
            'Started saccade',
            f'Planned: {self.fixation_point} -> {new_fixation_point} '
            f'Word: {word}',
            self.time,
        ))
        self._canbeinterrupted = canbeinterrupted
        tM1 = self.model_parameters['saccade_programming']

        try:
            yield self._timeout(tM1)
        except simpy.Interrupt:
            self._collect_action(Action(
                'Interrupted saccade',
                f'Planned: {self.fixation_point} -> {new_fixation_point} '
                f'Word: {word}',
                self.time,
            ))
            self._canbeinterrupted = True
        else:
            self._canbeinterrupted = False
            tM2 = self.model_parameters['saccade_finishing']
            yield self._timeout(tM2)

            intended_length = abs(
                self.fixation_point - new_fixation_point
            )
            launch_dt = max(1e-6, self.time - self._fixation_launch_site)
            systematic_error = (
                (OPTIMAL_SACCADE_LENGTH - intended_length)
                * ((self.model_parameters["omega1"] - math.log(launch_dt))
                   / self.model_parameters["omega2"])
            )
            self._fixation_launch_site = self.time
            self.fixation_point = normal(
                new_fixation_point + systematic_error,
                self.model_parameters["eta1"]
                + self.model_parameters["eta2"] * intended_length,
            )

            for pos_range in self._word_position_dict:
                if (self.fixation_point >= pos_range[0]
                        and self.fixation_point <= pos_range[1]):
                    self.fixated_word = self._word_position_dict[pos_range]
                    break

            if self._plan_saccade:
                self._saccade = self.env.process(
                    self._saccadic_programming(
                        new_fixation_point=self._plan_saccade[0],
                        word=self._plan_saccade[1],
                        canbeinterrupted=self._plan_saccade[2],
                    )
                )
                self._plan_saccade = False
            else:
                random_draw = uniform()
                lam = self.model_parameters["lambda"]
                if lam * abs(self.fixation_point - new_fixation_point) >= random_draw:
                    self._saccade = self.env.process(
                        self._saccadic_programming(
                            new_fixation_point=new_fixation_point,
                            word=word,
                            canbeinterrupted=canbeinterrupted,
                        )
                    )
                else:
                    self._saccade = None
                    self._canbeinterrupted = True

    def _integration(self, last_letter, new_fixation_point,
                     new_fixation_point2, elem, elem_for_attention, next_elem):
        yield self._timeout(float(elem.integration_time))
        random_draw = uniform()

        if float(elem.integration_failure) >= random_draw:
            self._collect_action(Action(
                'Failed integration', f'Word: {elem.token}', self.time,
            ))
            self._prepare_saccade(
                new_fixation_point, str(elem_for_attention.token),
                canbeinterrupted=False,
            )
            self.env.process(self._attend_again(
                last_letter, new_fixation_point2,
                elem=elem_for_attention, next_elem=next_elem,
            ))
        else:
            self._collect_action(Action(
                'Successful integration', f'Word: {elem.token}', self.time,
            ))

    def _attend_again(self, last_letter, new_fixation_point, elem, next_elem):
        old_attended_word = str(elem.token)
        if self.attended_word != old_attended_word:
            yield self._timeout(
                self.model_parameters["time_attention_shift"]
            )
            self.attended_word = elem

        if elem.token == "None":
            self._repeated_attention += 50
            yield self._timeout(50)
            self._prepare_saccade(
                new_fixation_point, str(next_elem.token)
            )
        else:
            distance = last_letter - self.fixation_point
            random_draw = uniform()

            if (self.model_parameters["predictability_repeated_attention"]
                    > random_draw):
                time_l1 = 0
            else:
                time_l1 = _time_familiarity_check(
                    distance, len(elem.token), elem.frequency,
                    elem.predictability,
                    self.model_parameters['eccentricity'],
                    self.model_parameters['alpha1'],
                    self.model_parameters['alpha2'],
                    self.model_parameters['alpha3'],
                )
            self._repeated_attention += time_l1
            yield self._timeout(time_l1)

            self._prepare_saccade(
                new_fixation_point, str(next_elem.token)
            )

            time_l2 = _time_lexical_access(
                elem.frequency, elem.predictability,
                self.model_parameters['delta'],
                self.model_parameters['alpha1'],
                self.model_parameters['alpha2'],
                self.model_parameters['alpha3'],
            )
            self._repeated_attention += time_l2
            yield self._timeout(time_l2)

            self._repeated_attention += float(elem.integration_time)
            yield self._timeout(float(elem.integration_time))

            self.attended_word = old_attended_word

    def visual_processing(self, sentence):
        first_letter = 1
        prev_elem = Word('None', 1e06, 1, 0, 0)
        new_fixation_point = 0
        next_elem = sentence[0] if sentence else prev_elem

        for i, elem in enumerate(sentence):
            self.attended_word = elem
            distance = first_letter - self.fixation_point

            random_draw = uniform()
            if float(elem.predictability) > random_draw:
                time_l1 = 0
            else:
                time_l1 = _time_familiarity_check(
                    distance, len(elem.token), elem.frequency,
                    float(elem.predictability),
                    self.model_parameters['eccentricity'],
                    self.model_parameters['alpha1'],
                    self.model_parameters['alpha2'],
                    self.model_parameters['alpha3'],
                )

            yield self._timeout(time_l1)
            yield self._timeout(self._repeated_attention)
            self._repeated_attention = 0
            self._collect_action(
                Action('L1', f'Word: {elem.token}', self.time)
            )

            try:
                next_elem = sentence[i + 1]
            except IndexError:
                pass
            else:
                new_fixation_point = (
                    first_letter + len(elem.token)
                    + 0.5 + len(next_elem.token) / 2
                )
                self._prepare_saccade(
                    new_fixation_point, str(next_elem.token)
                )

            time_l2 = _time_lexical_access(
                elem.frequency, elem.predictability,
                self.model_parameters['delta'],
                self.model_parameters['alpha1'],
                self.model_parameters['alpha2'],
                self.model_parameters['alpha3'],
            )
            yield self._timeout(time_l2)
            yield self._timeout(self._repeated_attention)
            self._repeated_attention = 0
            self._collect_action(
                Action('L2', f'Word: {elem.token}', self.time)
            )

            if i > 0:
                prev_pos = (
                    first_letter - 0.5 - len(sentence[i - 1].token) / 2
                )
                prev_elem = sentence[i - 1]
            else:
                prev_pos = 0
                prev_elem = Word('None', 1e06, 1, 0, 0)

            random_draw = uniform()
            if (float(self.model_parameters["probability_correct_regression"])
                    >= random_draw):
                self.env.process(self._integration(
                    last_letter=first_letter + len(elem.token),
                    new_fixation_point=(
                        first_letter - 0.5 + len(elem.token) / 2
                    ),
                    new_fixation_point2=new_fixation_point,
                    elem=elem,
                    elem_for_attention=elem,
                    next_elem=next_elem,
                ))
            else:
                self.env.process(self._integration(
                    last_letter=first_letter - 2,
                    new_fixation_point=prev_pos,
                    new_fixation_point2=new_fixation_point,
                    elem=elem,
                    elem_for_attention=prev_elem,
                    next_elem=next_elem,
                ))

            yield self._timeout(
                self.model_parameters["time_attention_shift"]
            )
            yield self._timeout(self._repeated_attention)
            self._repeated_attention = 0
            self._collect_action(Action(
                'Attention shift', f'From word: {elem.token}', self.time,
            ))
            first_letter += len(elem.token) + 1

    def step(self):
        self.env.step()

    def run(self, until):
        self.env.run(until=until)


# --------------------------------------------------------------------------- #
#  Neural E-Z Reader Simulation (parameterized by trained model)
# --------------------------------------------------------------------------- #

class NeuralEZReaderSim(Simulation):
    """
    E-Z Reader simulation using neural L1/L2/skip predictions.

    Key differences from base Simulation:
      - L1/L2 come from the trained neural model (pre-computed per word)
      - Skip uses neural skip probabilities (not cloze predictability)
      - Eccentricity uses the same linear formula as the diff EZR
      - Motor parameters come from the trained model (via sim_params)
      - Per-word integration_failure is set from L2 (not flat 0.01)
      - _attend_again word lookup uses sentence iteration (not id())
    """

    def __init__(self, sentence, l1_times, l2_times, skip_probs=None,
                 sim_params=None, **kwargs):
        """
        Args:
            sentence:    list of Word namedtuples
            l1_times:    list of float, L1 time (ms) per word
            l2_times:    list of float, L2 time (ms) per word
            skip_probs:  list of float (0-1), skip probability per word
            sim_params:  dict from model.get_sim_params(), overrides defaults
        """
        # Store neural data BEFORE super().__init__
        # (super starts visual_processing which needs these)
        self._neural_l1 = l1_times
        self._neural_l2 = l2_times
        self._skip_probs = skip_probs
        self._sentence = sentence

        # Extract eccentricity for linear formula
        self._eccentricity = (
            sim_params.get('eccentricity', 0.1) if sim_params else 0.1
        )

        super().__init__(sentence, **kwargs)

        # Override motor parameters with learned values
        if sim_params:
            for key in ['saccade_programming', 'saccade_finishing',
                        'time_attention_shift']:
                if key in sim_params:
                    self.model_parameters[key] = sim_params[key]

    def _get_word_index(self, elem):
        """Find word index by identity check, then token fallback."""
        for i, w in enumerate(self._sentence):
            if w is elem:
                return i
        # Fallback: match by token string (for re-created Word objects)
        for i, w in enumerate(self._sentence):
            if w.token == elem.token:
                return i
        return None

    def visual_processing(self, sentence):
        """Override: use neural L1/L2/skip with aligned eccentricity."""
        first_letter = 1
        prev_elem = Word('None', 1e06, 1, 0, 0)
        new_fixation_point = 0
        next_elem = sentence[0] if sentence else prev_elem

        for i, elem in enumerate(sentence):
            self.attended_word = elem

            # --- Skip gate: neural skip probability ---
            random_draw = uniform()
            if self._skip_probs is not None:
                skip = self._skip_probs[i] > random_draw
            else:
                skip = float(elem.predictability) > random_draw

            if skip:
                time_l1 = 0
            else:
                time_l1 = max(0, self._neural_l1[i])
                # Linear eccentricity (same formula as diff_ezreader.py):
                #   ecc_factor = 1.0 + eccentricity * max(0, word_length - 4)
                word_len = len(elem.token)
                ecc_factor = 1.0 + self._eccentricity * max(0, word_len - 4)
                time_l1 *= ecc_factor

            yield self._timeout(time_l1)
            yield self._timeout(self._repeated_attention)
            self._repeated_attention = 0
            self._collect_action(
                Action('L1', f'Word: {elem.token}', self.time)
            )

            # Program saccade to next word
            try:
                next_elem = sentence[i + 1]
            except IndexError:
                pass
            else:
                new_fixation_point = (
                    first_letter + len(elem.token)
                    + 0.5 + len(next_elem.token) / 2
                )
                self._prepare_saccade(
                    new_fixation_point, str(next_elem.token)
                )

            # --- Neural L2 ---
            time_l2 = max(0, self._neural_l2[i])
            yield self._timeout(time_l2)
            yield self._timeout(self._repeated_attention)
            self._repeated_attention = 0
            self._collect_action(
                Action('L2', f'Word: {elem.token}', self.time)
            )

            # --- Integration (uses per-word integration_failure from Word tuple) ---
            if i > 0:
                prev_pos = (
                    first_letter - 0.5 - len(sentence[i - 1].token) / 2
                )
                prev_elem = sentence[i - 1]
            else:
                prev_pos = 0
                prev_elem = Word('None', 1e06, 1, 0, 0)

            random_draw = uniform()
            if (float(self.model_parameters["probability_correct_regression"])
                    >= random_draw):
                self.env.process(self._integration(
                    last_letter=first_letter + len(elem.token),
                    new_fixation_point=(
                        first_letter - 0.5 + len(elem.token) / 2
                    ),
                    new_fixation_point2=new_fixation_point,
                    elem=elem,
                    elem_for_attention=elem,
                    next_elem=next_elem,
                ))
            else:
                self.env.process(self._integration(
                    last_letter=first_letter - 2,
                    new_fixation_point=prev_pos,
                    new_fixation_point2=new_fixation_point,
                    elem=elem,
                    elem_for_attention=prev_elem,
                    next_elem=next_elem,
                ))

            yield self._timeout(
                self.model_parameters["time_attention_shift"]
            )
            yield self._timeout(self._repeated_attention)
            self._repeated_attention = 0
            self._collect_action(Action(
                'Attention shift', f'From word: {elem.token}', self.time,
            ))
            first_letter += len(elem.token) + 1

    def _attend_again(self, last_letter, new_fixation_point, elem, next_elem):
        """Override: use neural L1/L2 with fixed word lookup."""
        old_attended_word = str(elem.token)
        if self.attended_word != old_attended_word:
            yield self._timeout(
                self.model_parameters["time_attention_shift"]
            )
            self.attended_word = elem

        if elem.token == "None":
            self._repeated_attention += 50
            yield self._timeout(50)
            self._prepare_saccade(
                new_fixation_point, str(next_elem.token)
            )
        else:
            random_draw = uniform()

            # 90% chance L1=0 on re-attending (word already familiar)
            if (self.model_parameters["predictability_repeated_attention"]
                    > random_draw):
                time_l1 = 0
            else:
                word_idx = self._get_word_index(elem)
                if word_idx is not None:
                    time_l1 = max(0, self._neural_l1[word_idx])
                    # Same linear eccentricity as visual_processing
                    word_len = len(elem.token)
                    ecc_factor = (
                        1.0 + self._eccentricity * max(0, word_len - 4)
                    )
                    time_l1 *= ecc_factor
                else:
                    time_l1 = 50  # fallback

            self._repeated_attention += time_l1
            yield self._timeout(time_l1)

            self._prepare_saccade(
                new_fixation_point, str(next_elem.token)
            )

            # Neural L2
            word_idx = self._get_word_index(elem)
            if word_idx is not None:
                time_l2 = max(0, self._neural_l2[word_idx])
            else:
                time_l2 = 20  # fallback

            self._repeated_attention += time_l2
            yield self._timeout(time_l2)

            self._repeated_attention += float(elem.integration_time)
            yield self._timeout(float(elem.integration_time))

            self.attended_word = old_attended_word


# --------------------------------------------------------------------------- #
#  Fixation Tracker
# --------------------------------------------------------------------------- #

class FixationTracker:
    """
    Tracks fixation positions step-by-step and extracts reading measures.

    TRT:  sum of ALL fixation durations on a word
    FFD:  duration of the FIRST fixation on a word
    Gaze: sum of fixation durations during first-pass
    Skip: word was never fixated
    """

    def __init__(self, n_words, position_map):
        self.n_words = n_words
        self.position_map = position_map
        self.all_fixations = defaultdict(list)
        self.first_pass_fixations = defaultdict(list)
        self.gaze_complete = set()
        self.current_word_idx = None
        self.current_fixation_start = 0.0

    def _get_word_idx(self, fixation_point):
        for (start, end), idx in self.position_map.items():
            if start <= fixation_point <= end:
                return idx
        return None

    def update(self, fixation_point, time_ms):
        new_word_idx = self._get_word_idx(fixation_point)
        if new_word_idx != self.current_word_idx:
            if self.current_word_idx is not None:
                interval = (self.current_fixation_start, time_ms)
                self.all_fixations[self.current_word_idx].append(interval)
                if self.current_word_idx not in self.gaze_complete:
                    self.first_pass_fixations[
                        self.current_word_idx
                    ].append(interval)
                self.gaze_complete.add(self.current_word_idx)
            self.current_word_idx = new_word_idx
            self.current_fixation_start = time_ms

    def finalize(self, time_ms):
        if self.current_word_idx is not None:
            interval = (self.current_fixation_start, time_ms)
            self.all_fixations[self.current_word_idx].append(interval)
            if self.current_word_idx not in self.gaze_complete:
                self.first_pass_fixations[
                    self.current_word_idx
                ].append(interval)

    def get_metrics(self):
        trt, ffd, gaze, skipped, nfix = [], [], [], [], []

        for i in range(self.n_words):
            all_fix = self.all_fixations.get(i, [])
            fp_fix = self.first_pass_fixations.get(i, [])

            if len(all_fix) == 0:
                trt.append(0.0)
                ffd.append(0.0)
                gaze.append(0.0)
                skipped.append(True)
                nfix.append(0)
            else:
                trt.append(max(0.0, sum(e - s for s, e in all_fix)))
                ffd.append(max(0.0, all_fix[0][1] - all_fix[0][0]))
                gaze.append(max(0.0, sum(e - s for s, e in fp_fix)))
                skipped.append(False)
                nfix.append(len(all_fix))

        return {
            'total_reading_time': trt,
            'first_fixation_duration': ffd,
            'gaze_duration': gaze,
            'was_skipped': skipped,
            'n_fixations': nfix,
        }


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _build_sentence(tokens, integration_failures=None, integration_time=25.0):
    """Build Word tuples for the simulation."""
    sentence = []
    for i, token in enumerate(tokens):
        int_fail = integration_failures[i] if integration_failures else 0.01
        sentence.append(Word(
            token=token,
            frequency=1e4,         # unused for neural sim
            predictability=0.0,    # unused when skip_probs provided
            integration_time=integration_time,
            integration_failure=int_fail,
        ))
    return sentence


def _build_position_map(sentence):
    """Build character-position -> word-index map for fixation tracking."""
    position_map = {}
    pos = 1
    for i, w in enumerate(sentence):
        start = pos
        end = pos + 1 + len(w.token)
        position_map[(start, end)] = i
        pos = end
    return position_map


def _run_sim_steps(sim, position_map, n_words,
                   timeout_seconds=5.0, max_steps=10000):
    """Step through simulation, track fixations, return metrics."""
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
#  Neural simulation: single run + averaged
# --------------------------------------------------------------------------- #

def compute_integration_failures(l2_times, sim_params):
    """Compute per-word integration failure probabilities from L2.

    Uses the same sigmoid formula as DifferentiableEZReader, ensuring
    alignment between training and inference.
    """
    if sim_params is None:
        return [0.01] * len(l2_times)

    threshold = sim_params.get('integration_threshold', 50.0)
    sharpness = sim_params.get('integration_sharpness', 0.1)

    failures = []
    for l2 in l2_times:
        x = sharpness * (l2 - threshold)
        prob = 1.0 / (1.0 + math.exp(-x))
        failures.append(prob)
    return failures


def run_simulation(tokens, l1_times, l2_times, skip_probs=None,
                   sim_params=None, integration_failures=None,
                   timeout_seconds=5.0):
    """Run Neural EZ Reader simulation once.

    Args:
        tokens:               list[str]
        l1_times:             list[float] - L1 per word (ms)
        l2_times:             list[float] - L2 per word (ms)
        skip_probs:           list[float] or None - skip probability per word
        sim_params:           dict from model.get_sim_params()
        integration_failures: list[float] or None - per-word integration failure.
                              If None, computed from L2 using sim_params.
        timeout_seconds:      float - max simulation time

    Returns:
        dict with per-word metrics + 'success' flag
    """
    n_words = len(tokens)

    # Compute integration failures from L2 if not provided
    if integration_failures is None:
        integration_failures = compute_integration_failures(
            l2_times, sim_params
        )

    sentence = _build_sentence(tokens, integration_failures)
    position_map = _build_position_map(sentence)

    try:
        sim = NeuralEZReaderSim(
            sentence=sentence,
            l1_times=l1_times,
            l2_times=l2_times,
            skip_probs=skip_probs,
            sim_params=sim_params,
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


def run_simulation_averaged(tokens, l1_times, l2_times, skip_probs=None,
                             sim_params=None, integration_failures=None,
                             num_runs=50, timeout_seconds=5.0):
    """Run Neural EZ Reader simulation K times and average results.

    Returns per-word metrics averaged over successful runs, with
    separate averages for fixated-only instances (for fair comparison
    to human data which excludes skipped words).
    """
    n_words = len(tokens)

    if integration_failures is None:
        integration_failures = compute_integration_failures(
            l2_times, sim_params
        )

    # Accumulators
    sum_trt = [0.0] * n_words
    sum_ffd = [0.0] * n_words
    sum_gaze = [0.0] * n_words
    sum_skip = [0.0] * n_words
    fixated_count = [0] * n_words
    sum_trt_fixated = [0.0] * n_words
    sum_ffd_fixated = [0.0] * n_words
    sum_gaze_fixated = [0.0] * n_words
    n_success = 0

    for _ in range(num_runs):
        result = run_simulation(
            tokens, l1_times, l2_times,
            skip_probs=skip_probs,
            sim_params=sim_params,
            integration_failures=integration_failures,
            timeout_seconds=timeout_seconds,
        )
        if not result['success']:
            continue

        n_success += 1
        for i in range(n_words):
            was_skip = result['was_skipped'][i]
            sum_skip[i] += 1.0 if was_skip else 0.0

            trt_val = result['total_reading_time'][i]
            ffd_val = result['first_fixation_duration'][i]
            gaze_val = result['gaze_duration'][i]

            sum_trt[i] += trt_val
            sum_ffd[i] += ffd_val
            sum_gaze[i] += gaze_val

            if not was_skip:
                fixated_count[i] += 1
                sum_trt_fixated[i] += trt_val
                sum_ffd_fixated[i] += ffd_val
                sum_gaze_fixated[i] += gaze_val

    if n_success == 0:
        return {
            'total_reading_time': [0.0] * n_words,
            'first_fixation_duration': [0.0] * n_words,
            'gaze_duration': [0.0] * n_words,
            'skip_rate': [1.0] * n_words,
            'trt_fixated': [0.0] * n_words,
            'ffd_fixated': [0.0] * n_words,
            'gaze_fixated': [0.0] * n_words,
            'success': False,
        }

    # Average over all runs (including zeros for skipped)
    avg_trt = [sum_trt[i] / n_success for i in range(n_words)]
    avg_ffd = [sum_ffd[i] / n_success for i in range(n_words)]
    avg_gaze = [sum_gaze[i] / n_success for i in range(n_words)]
    skip_rate = [sum_skip[i] / n_success for i in range(n_words)]

    # Average over fixated instances only (for comparison to human data)
    trt_fix = [
        sum_trt_fixated[i] / fixated_count[i] if fixated_count[i] > 0 else 0.0
        for i in range(n_words)
    ]
    ffd_fix = [
        sum_ffd_fixated[i] / fixated_count[i] if fixated_count[i] > 0 else 0.0
        for i in range(n_words)
    ]
    gaze_fix = [
        sum_gaze_fixated[i] / fixated_count[i] if fixated_count[i] > 0 else 0.0
        for i in range(n_words)
    ]

    return {
        'total_reading_time': avg_trt,
        'first_fixation_duration': avg_ffd,
        'gaze_duration': avg_gaze,
        'skip_rate': skip_rate,
        'trt_fixated': trt_fix,
        'ffd_fixated': ffd_fix,
        'gaze_fixated': gaze_fix,
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

    position_map = _build_position_map(sentence)

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
            all_skip.append(
                [1.0 if s else 0.0 for s in result['was_skipped']]
            )
            all_nfix.append([float(n) for n in result['n_fixations']])
            n_success += 1

    if n_success == 0:
        return {
            'total_reading_time': [0.0] * n_words,
            'first_fixation_duration': [0.0] * n_words,
            'gaze_duration': [0.0] * n_words,
            'skip_rate': [1.0] * n_words,
            'success': False,
        }

    avg = lambda arrs, i: sum(run[i] for run in arrs) / n_success
    return {
        'total_reading_time':      [avg(all_trt, i) for i in range(n_words)],
        'first_fixation_duration': [avg(all_ffd, i) for i in range(n_words)],
        'gaze_duration':           [avg(all_gaze, i) for i in range(n_words)],
        'skip_rate':               [avg(all_skip, i) for i in range(n_words)],
        'success': True,
        'n_successful_runs': n_success,
    }
