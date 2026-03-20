"""
Differentiable E-Z Reader -- aligned with the discrete simulation.

Every parameter is named to match the simulation's model_parameters dict,
so learned values transfer directly at inference time. No renaming, no
mapping, no calibration layer.

Key design decisions:
  - TRT = (1-skip) * (gaze + overhead + regression), matching the expected
    value of the simulation's stochastic skip gate (v2 style)
  - Eccentricity uses the same linear formula as NeuralEZReaderSim
  - Integration failure probability is derived from L2 using a sigmoid gate,
    and the same formula produces per-word integration_failure for the simulation
  - Motor timing parameters (saccade_programming, saccade_finishing,
    time_attention_shift) are named identically to the simulation's dict keys
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableEZReader(nn.Module):
    """
    Differentiable approximation of the E-Z Reader simulation.

    All parameters use the same names and semantics as the discrete simulation,
    so get_sim_params() can be passed directly to the simulation at inference.
    """

    def __init__(self):
        super().__init__()

        # --- Motor timing (simulation dict keys) ---
        self.saccade_programming = nn.Parameter(torch.tensor(125.0))   # M1 labile stage (ms)
        self.saccade_finishing = nn.Parameter(torch.tensor(25.0))      # M2 ballistic stage (ms)
        self.time_attention_shift = nn.Parameter(torch.tensor(25.0))   # attention shift (ms)

        # --- Eccentricity (linear, same formula in simulation) ---
        self.eccentricity = nn.Parameter(torch.tensor(0.1))

        # --- L2 contribution to FFD ---
        # Some of L2 processing "leaks" into the first fixation duration
        self.l2_contribution = nn.Parameter(torch.tensor(0.3))

        # --- Integration failure (maps to simulation's per-word integration_failure) ---
        # integration_failure_prob = sigmoid(sharpness * (L2 - threshold))
        self.integration_threshold = nn.Parameter(torch.tensor(50.0))  # ms
        self.integration_sharpness = nn.Parameter(torch.tensor(0.1))

    def get_sim_params(self):
        """Export learned parameters in the simulation's model_parameters format.

        Returns a plain dict of floats that can be passed directly to
        NeuralEZReaderSim's sim_params argument.
        """
        return {
            'saccade_programming': self.saccade_programming.item(),
            'saccade_finishing': self.saccade_finishing.item(),
            'time_attention_shift': self.time_attention_shift.item(),
            'eccentricity': self.eccentricity.item(),
            'integration_threshold': self.integration_threshold.item(),
            'integration_sharpness': self.integration_sharpness.item(),
        }

    def compute_integration_failure(self, L2):
        """Per-word integration failure probability from L2.

        Same formula is used to set per-word integration_failure in the
        simulation's Word tuples, ensuring alignment.
        """
        return torch.sigmoid(
            self.integration_sharpness * (L2 - self.integration_threshold)
        )

    def forward(self, L1, L2, skip_prob, word_lengths):
        """
        Args:
            L1:           (B, T) familiarity check time per word (ms)
            L2:           (B, T) lexical access time per word (ms)
            skip_prob:    (B, T) skip probability (0-1), from neural skip head
            word_lengths: (B, T) number of characters per word

        Returns:
            dict with:
                total_reading_time:      (B, T) expected TRT (ms)
                first_fixation:          (B, T) FFD (ms)
                gaze_duration:           (B, T) Gaze duration (ms)
                skip_prob:               (B, T) skip probability
                integration_failure_prob:(B, T) per-word integration failure prob
        """
        # --- 1. Eccentricity scaling on L1 ---
        # Same formula as NeuralEZReaderSim.visual_processing:
        #   ecc_factor = 1.0 + eccentricity * max(0, word_length - 4)
        ecc_scale = 1.0 + self.eccentricity * (word_lengths - 4.0).clamp(min=0)
        L1_scaled = L1 * ecc_scale

        # --- 2. First fixation duration ---
        # L1 determines first fixation, but some L2 processing leaks in
        first_fixation = L1_scaled + F.softplus(self.l2_contribution) * L2

        # --- 3. Gaze duration (first-pass reading) ---
        gaze_duration = L1_scaled + L2

        # --- 4. Integration failure -> regression penalty ---
        integration_failure_prob = self.compute_integration_failure(L2)
        # When integration fails, the reader re-fixates the previous word.
        # Cost approximated by previous word's gaze duration.
        prev_gaze = torch.zeros_like(gaze_duration)
        prev_gaze[:, 1:] = gaze_duration[:, :-1]
        regression_penalty = integration_failure_prob * prev_gaze

        # --- 5. Motor overhead per fixation ---
        overhead = (self.saccade_programming
                    + self.saccade_finishing
                    + self.time_attention_shift)

        # --- 6. Total reading time ---
        # TRT = (1-skip) * (gaze + overhead + regression)
        # This is the expected value of the simulation's stochastic skip gate:
        # with prob skip: TRT=0, with prob (1-skip): TRT = gaze + overhead + reg
        fixate_prob = 1.0 - skip_prob
        total_reading_time = fixate_prob * (gaze_duration + overhead + regression_penalty)

        return {
            'total_reading_time': total_reading_time,
            'first_fixation': first_fixation,
            'gaze_duration': gaze_duration,
            'skip_prob': skip_prob,
            'integration_failure_prob': integration_failure_prob,
        }
