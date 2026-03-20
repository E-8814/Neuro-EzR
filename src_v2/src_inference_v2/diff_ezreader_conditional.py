"""
Differentiable E-Z Reader -- conditional (sim-aligned) variant.

Key differences from diff_ezreader.py:
  1. Motor overhead (M1+M2) included in FFD/Gaze/TRT predictions.
     This keeps L1/L2 at actual processing-time scale (~50-80ms),
     which is what the discrete simulation expects.
  2. TRT is conditional on fixation (no skip weighting).
     Training masks reading-time losses to fixated words only.
  3. attention_shift excluded from predictions (exported in sim_params
     for simulation use, but doesn't affect fixation duration).

Why motor overhead matters:
  Without it, conditional FFD loss (pred vs human ~210ms) forces L1 to ~200ms
  to compensate.  With it, FFD = L1 + M1 + M2 ≈ 60 + 150 = 210ms, so L1
  stays at a realistic ~60ms — exactly what the simulation uses.

All parameter names match the simulation's model_parameters dict,
so get_sim_params() transfers directly to NeuralEZReaderSim.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableEZReaderConditional(nn.Module):

    def __init__(self):
        super().__init__()

        # Motor timing (simulation dict keys)
        self.saccade_programming = nn.Parameter(torch.tensor(125.0))  # M1 labile
        self.saccade_finishing = nn.Parameter(torch.tensor(25.0))     # M2 ballistic
        self.time_attention_shift = nn.Parameter(torch.tensor(25.0))  # between words

        # Eccentricity (linear, same formula as simulation)
        self.eccentricity = nn.Parameter(torch.tensor(0.1))

        # L2 leakage into FFD (small: L2 runs concurrently with M1, rarely exceeds it)
        self.l2_contribution = nn.Parameter(torch.tensor(0.3))

        # Integration failure (maps to simulation's per-word integration_failure)
        self.integration_threshold = nn.Parameter(torch.tensor(50.0))
        self.integration_sharpness = nn.Parameter(torch.tensor(0.1))

    def get_sim_params(self):
        """Export for NeuralEZReaderSim — includes attention_shift."""
        return {
            'saccade_programming': self.saccade_programming.item(),
            'saccade_finishing': self.saccade_finishing.item(),
            'time_attention_shift': self.time_attention_shift.item(),
            'eccentricity': self.eccentricity.item(),
            'integration_threshold': self.integration_threshold.item(),
            'integration_sharpness': self.integration_sharpness.item(),
        }

    def compute_integration_failure(self, L2):
        """Per-word integration failure probability from L2."""
        return torch.sigmoid(
            self.integration_sharpness * (L2 - self.integration_threshold)
        )

    def forward(self, L1, L2, skip_prob, word_lengths):
        """
        Args:
            L1:           (B, T) familiarity check time per word (ms)
            L2:           (B, T) lexical access time per word (ms)
            skip_prob:    (B, T) skip probability (0-1)
            word_lengths: (B, T) character count per word

        Returns dict with:
            trt_conditional:         (B, T) TRT given fixation (no skip weighting)
            total_reading_time:      (B, T) expected TRT = (1-skip) * conditional
            first_fixation:          (B, T) FFD (includes motor overhead)
            gaze_duration:           (B, T) Gaze (includes motor overhead)
            skip_prob:               (B, T) pass-through
            integration_failure_prob:(B, T) per-word integration failure
        """
        # Motor overhead: M1 + M2 (labile + ballistic saccade programming)
        # This is the minimum fixation duration floor — the eye can't move
        # until M2 completes, and M1 starts after L1.
        motor = self.saccade_programming + self.saccade_finishing

        # Eccentricity scaling on L1 (same formula as simulation)
        ecc_scale = 1.0 + self.eccentricity * (word_lengths - 4.0).clamp(min=0)
        L1_scaled = L1 * ecc_scale

        # --- FFD: L1 + motor + small L2 leakage ---
        # In simulation: FFD ≈ L1 + M1 + M2 because L2 runs concurrently
        # within the M1 window (L2 << M1 for typical words).
        # l2_contribution captures the rare case where L2 exceeds the motor window.
        first_fixation = L1_scaled + motor + F.softplus(self.l2_contribution) * L2

        # --- Gaze: L1 + L2 + motor ---
        # First-pass total.  Gaze > FFD by ~L2 because L2 determines
        # when the reader finishes processing and gaze accounts for
        # any additional within-word refixation time.
        gaze_duration = L1_scaled + L2 + motor

        # --- Integration failure → regression penalty ---
        integration_failure_prob = self.compute_integration_failure(L2)
        prev_gaze = torch.zeros_like(gaze_duration)
        prev_gaze[:, 1:] = gaze_duration[:, :-1]
        regression_penalty = integration_failure_prob * prev_gaze

        # --- TRT conditional on fixation (NO skip weighting) ---
        trt_conditional = gaze_duration + regression_penalty

        # --- Expected TRT (for comparison / backward compat) ---
        trt_expected = (1.0 - skip_prob) * trt_conditional

        return {
            'trt_conditional': trt_conditional,
            'total_reading_time': trt_expected,
            'first_fixation': first_fixation,
            'gaze_duration': gaze_duration,
            'skip_prob': skip_prob,
            'integration_failure_prob': integration_failure_prob,
        }
