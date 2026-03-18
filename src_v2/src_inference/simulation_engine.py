"""
E-Z Reader discrete-event simulation, adapted for neural L1/L2/skip inputs.

Based on the original ez_reader/ez_reader_engine.py, with these changes:
  - model_parameters are instance-level (so each simulation can have its own)
  - NeuralEZReaderSim accepts skip_probs from the neural model
  - _attend_again uses neural L1/L2 instead of formula fallback
  - Fixed prev_elem / new_fixation_point variable scoping bugs
"""

from collections import namedtuple
import math
import numpy as np
from numpy.random import uniform, normal
import simpy

OPTIMAL_SACCADE_LENGTH = 7

Word = namedtuple('Word', 'token frequency predictability integration_time integration_failure')
Action = namedtuple('Action', 'name details time')


def _time_familiarity_check(distance, wordlength, frequency, predictability,
                            eccentricity, alpha1, alpha2, alpha3):
    """Original EZR L1 formula (only used for _attend_again fallback)."""
    tL1 = alpha1 - alpha2 * math.log(max(1, frequency)) - alpha3 * predictability
    ecc_factor = pow(eccentricity, abs(distance + (wordlength - 1) / 2.0))
    return max(0, tL1 * ecc_factor)


def _time_lexical_access(frequency, predictability, delta, alpha1, alpha2, alpha3):
    """Original EZR L2 formula (only used for _attend_again fallback)."""
    tL2 = delta * (alpha1 - alpha2 * math.log(max(1, frequency)) - alpha3 * predictability)
    return max(0, tL2)


class Simulation(object):
    """Base E-Z Reader simulation (unchanged logic, instance-level params)."""

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
        # Instance-level copy so each simulation can have its own parameters
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
            self._word_position_dict[(position, position + 1 + len(word.token))] = word.token
            position += 1 + len(word.token)

        for pos_range in self._word_position_dict:
            if initial_fixation >= pos_range[0] and initial_fixation <= pos_range[1]:
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

    def _prepare_saccade(self, new_fixation_point, word, canbeinterrupted=True):
        if self._canbeinterrupted:
            try:
                self._saccade.interrupt()
            except (AttributeError, RuntimeError):
                pass
            if (float(self.fixation_point) < float(new_fixation_point) - len(word) / 2) or \
               (float(self.fixation_point) > float(new_fixation_point) + len(word) / 2):
                self._saccade = self.env.process(
                    self._saccadic_programming(
                        new_fixation_point=new_fixation_point,
                        word=word,
                        canbeinterrupted=canbeinterrupted,
                    )
                )
        else:
            if self.fixation_point != new_fixation_point:
                self._plan_saccade = (new_fixation_point, word, canbeinterrupted)

    def _saccadic_programming(self, new_fixation_point, word,
                              regression=False, canbeinterrupted=True):
        self._collect_action(Action(
            'Started saccade',
            f'Planned saccade: {self.fixation_point} -> {new_fixation_point} Word: {word}',
            self.time,
        ))

        self._canbeinterrupted = canbeinterrupted
        tM1 = self.model_parameters['saccade_programming']

        try:
            yield self._timeout(tM1)
        except simpy.Interrupt:
            self._collect_action(Action(
                'Interrupted saccade',
                f'Planned saccade: {self.fixation_point} -> {new_fixation_point} Word: {word}',
                self.time,
            ))
            self._canbeinterrupted = True
        else:
            self._canbeinterrupted = False
            tM2 = self.model_parameters['saccade_finishing']
            yield self._timeout(tM2)

            intended_length = abs(self.fixation_point - new_fixation_point)
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
                if self.fixation_point >= pos_range[0] and self.fixation_point <= pos_range[1]:
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
                if self.model_parameters["lambda"] * abs(self.fixation_point - new_fixation_point) >= random_draw:
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

    def _integration(self, last_letter, new_fixation_point, new_fixation_point2,
                     elem, elem_for_attention, next_elem):
        yield self._timeout(float(elem.integration_time))
        random_draw = uniform()

        if float(elem.integration_failure) >= random_draw:
            self._collect_action(Action('Failed integration', f'Word: {elem.token}', self.time))
            self._prepare_saccade(new_fixation_point, str(elem_for_attention.token),
                                  canbeinterrupted=False)
            self.env.process(self._attend_again(
                last_letter, new_fixation_point2,
                elem=elem_for_attention, next_elem=next_elem,
            ))
        else:
            self._collect_action(Action('Successful integration', f'Word: {elem.token}', self.time))

    def _attend_again(self, last_letter, new_fixation_point, elem, next_elem):
        old_attended_word = str(elem.token)
        if self.attended_word != old_attended_word:
            yield self._timeout(self.model_parameters["time_attention_shift"])
            self.attended_word = elem

        if elem.token == "None":
            self._repeated_attention += 50
            yield self._timeout(50)
            self._prepare_saccade(new_fixation_point, str(next_elem.token))
        else:
            distance = last_letter - self.fixation_point
            random_draw = uniform()

            if self.model_parameters["predictability_repeated_attention"] > random_draw:
                time_l1 = 0
            else:
                time_l1 = _time_familiarity_check(
                    distance, len(elem.token), elem.frequency, elem.predictability,
                    self.model_parameters['eccentricity'],
                    self.model_parameters['alpha1'],
                    self.model_parameters['alpha2'],
                    self.model_parameters['alpha3'],
                )
            self._repeated_attention += time_l1
            yield self._timeout(time_l1)

            self._prepare_saccade(new_fixation_point, str(next_elem.token))

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

            self._collect_action(Action('L1', f'Word: {elem.token}', self.time))

            try:
                next_elem = sentence[i + 1]
            except IndexError:
                pass
            else:
                new_fixation_point = (first_letter + len(elem.token)
                                      + 0.5 + len(next_elem.token) / 2)
                self._prepare_saccade(new_fixation_point, str(next_elem.token))

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

            self._collect_action(Action('L2', f'Word: {elem.token}', self.time))

            # Integration
            if i > 0:
                prev_pos = first_letter - 0.5 - len(sentence[i - 1].token) / 2
                prev_elem = sentence[i - 1]
            else:
                prev_pos = 0
                prev_elem = Word('None', 1e06, 1, 0, 0)

            random_draw = uniform()
            if float(self.model_parameters["probability_correct_regression"]) >= random_draw:
                self.env.process(self._integration(
                    last_letter=first_letter + len(elem.token),
                    new_fixation_point=first_letter - 0.5 + len(elem.token) / 2,
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

            yield self._timeout(self.model_parameters["time_attention_shift"])
            yield self._timeout(self._repeated_attention)
            self._repeated_attention = 0

            self._collect_action(Action('Attention shift', f'From word: {elem.token}', self.time))
            first_letter += len(elem.token) + 1

    def step(self):
        self.env.step()

    def run(self, until):
        self.env.run(until=until)


# --------------------------------------------------------------------------- #
#  Neural EZ Reader Simulation
# --------------------------------------------------------------------------- #

class NeuralEZReaderSim(Simulation):
    """
    E-Z Reader simulation that uses neural network L1/L2/skip predictions
    instead of the frequency/predictability formulas.

    Key differences from base Simulation:
      - L1/L2 come from a trained neural network (pre-computed per word)
      - Skip gate uses neural skip probabilities instead of raw predictability
      - _attend_again uses neural L1/L2 (with 90% skip on re-attending)
      - Eccentricity scaling still applied to L1 (visual acuity effect)
    """

    def __init__(self, sentence, l1_times, l2_times, skip_probs=None,
                 apply_eccentricity=False, **kwargs):
        """
        Args:
            sentence:    list of Word namedtuples
            l1_times:    list of float, L1 time (ms) per word from neural model
            l2_times:    list of float, L2 time (ms) per word from neural model
            skip_probs:  list of float (0-1), skip probability per word from neural model.
                         If None, falls back to predictability-based skip (original behavior).
            apply_eccentricity: bool - whether to apply eccentricity scaling to neural L1.
                         Default False: the neural L1 already encodes word difficulty,
                         and the simulation's exponential eccentricity (1.15^distance)
                         causes cascading skips when attention races ahead of fixation.
        """
        self._neural_l1 = l1_times
        self._neural_l2 = l2_times
        self._skip_probs = skip_probs
        self._apply_eccentricity = apply_eccentricity
        super().__init__(sentence, **kwargs)
        # Map word objects to indices for _attend_again lookups
        self._word_index = {id(w): i for i, w in enumerate(sentence)}

    def visual_processing(self, sentence):
        first_letter = 1
        prev_elem = Word('None', 1e06, 1, 0, 0)
        new_fixation_point = 0
        next_elem = sentence[0] if sentence else prev_elem

        for i, elem in enumerate(sentence):
            self.attended_word = elem
            distance = first_letter - self.fixation_point

            # --- Skip gate ---
            random_draw = uniform()
            if self._skip_probs is not None:
                skip = self._skip_probs[i] > random_draw
            else:
                skip = float(elem.predictability) > random_draw

            if skip:
                time_l1 = 0
            else:
                time_l1 = max(0, self._neural_l1[i])
                if self._apply_eccentricity:
                    word_len = len(elem.token)
                    ecc = self.model_parameters['eccentricity']
                    ecc_factor = pow(ecc, abs(distance + (word_len - 1) / 2.0))
                    time_l1 *= ecc_factor

            yield self._timeout(time_l1)
            yield self._timeout(self._repeated_attention)
            self._repeated_attention = 0

            self._collect_action(Action('L1', f'Word: {elem.token}', self.time))

            # Program saccade to next word
            try:
                next_elem = sentence[i + 1]
            except IndexError:
                pass
            else:
                new_fixation_point = (first_letter + len(elem.token)
                                      + 0.5 + len(next_elem.token) / 2)
                self._prepare_saccade(new_fixation_point, str(next_elem.token))

            # --- Neural L2 ---
            time_l2 = max(0, self._neural_l2[i])

            yield self._timeout(time_l2)
            yield self._timeout(self._repeated_attention)
            self._repeated_attention = 0

            self._collect_action(Action('L2', f'Word: {elem.token}', self.time))

            # Integration
            if i > 0:
                prev_pos = first_letter - 0.5 - len(sentence[i - 1].token) / 2
                prev_elem = sentence[i - 1]
            else:
                prev_pos = 0
                prev_elem = Word('None', 1e06, 1, 0, 0)

            random_draw = uniform()
            if float(self.model_parameters["probability_correct_regression"]) >= random_draw:
                self.env.process(self._integration(
                    last_letter=first_letter + len(elem.token),
                    new_fixation_point=first_letter - 0.5 + len(elem.token) / 2,
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

            yield self._timeout(self.model_parameters["time_attention_shift"])
            yield self._timeout(self._repeated_attention)
            self._repeated_attention = 0

            self._collect_action(Action('Attention shift', f'From word: {elem.token}', self.time))
            first_letter += len(elem.token) + 1

    def _attend_again(self, last_letter, new_fixation_point, elem, next_elem):
        """Override: use neural L1/L2 when re-attending after integration failure."""
        old_attended_word = str(elem.token)
        if self.attended_word != old_attended_word:
            yield self._timeout(self.model_parameters["time_attention_shift"])
            self.attended_word = elem

        if elem.token == "None":
            self._repeated_attention += 50
            yield self._timeout(50)
            self._prepare_saccade(new_fixation_point, str(next_elem.token))
        else:
            distance = last_letter - self.fixation_point
            random_draw = uniform()

            # 90% chance L1 = 0 on re-attending (word is already familiar)
            if self.model_parameters["predictability_repeated_attention"] > random_draw:
                time_l1 = 0
            else:
                word_idx = self._word_index.get(id(elem))
                if word_idx is not None:
                    time_l1 = max(0, self._neural_l1[word_idx])
                    if self._apply_eccentricity:
                        word_len = len(elem.token)
                        ecc = self.model_parameters['eccentricity']
                        ecc_factor = pow(ecc, abs(distance + (word_len - 1) / 2.0))
                        time_l1 *= ecc_factor
                else:
                    time_l1 = 50  # fallback for unknown words

            self._repeated_attention += time_l1
            yield self._timeout(time_l1)

            self._prepare_saccade(new_fixation_point, str(next_elem.token))

            # Use neural L2
            word_idx = self._word_index.get(id(elem))
            if word_idx is not None:
                time_l2 = max(0, self._neural_l2[word_idx])
            else:
                time_l2 = 20  # fallback

            self._repeated_attention += time_l2
            yield self._timeout(time_l2)

            self._repeated_attention += float(elem.integration_time)
            yield self._timeout(float(elem.integration_time))

            self.attended_word = old_attended_word
