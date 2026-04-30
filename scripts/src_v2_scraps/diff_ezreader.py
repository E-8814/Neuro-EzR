"""
Differentiable EZ Reader v2.

Changes from v1:
  - FFD formula: L1_scaled + softplus(l2_contribution) * L2_scaled
  - Explicit regression mechanism: sigmoid threshold on L2
  - Removed integration_cost parameter (from v1) and ffd_offset (caused L1 collapse)
  - New parameters: l2_contribution, regression_threshold,
    regression_sharpness, regression_cost_scale
"""

import torch
import torch.nn as nn


class DifferentiableEZReader(nn.Module):
    """
    Differentiable approximation of the EZ Reader simulation (v2).

    Takes L1/L2 predictions from a neural network and produces
    predicted reading time metrics using smooth, differentiable operations.

    v2 changes:
      - FFD includes L2 contribution and motor latency offset
      - Explicit regression probability based on L2 threshold
      - Regression cost scales with previous word's gaze duration
    """

    def __init__(self, ablation=None):
        super().__init__()
        self.ablation = ablation  # None, 'no_eccentricity', 'no_regressions', 'ffd_l1_only'

        # Learnable EZ Reader parameters (initialized from literature values)
        self.saccade_time = nn.Parameter(torch.tensor(150.0))     # M1 + M2 overhead (ms)
        self.attention_shift = nn.Parameter(torch.tensor(25.0))   # attention shift time (ms)
        self.skip_sharpness = nn.Parameter(torch.tensor(8.0))     # sigmoid steepness for skip
        self.eccentricity = nn.Parameter(torch.tensor(0.1))       # eccentricity scaling factor

        # v2 new parameters
        self.l2_contribution = nn.Parameter(torch.tensor(0.3))            # fraction of L2 in FFD
        self.regression_threshold = nn.Parameter(torch.tensor(50.0))      # L2 threshold for regression
        self.regression_sharpness = nn.Parameter(torch.tensor(0.1))       # sigmoid steepness
        self.regression_cost_scale = nn.Parameter(torch.tensor(1.0))      # cost multiplier

    def forward(self, L1, L2, skip_input, word_lengths, input_is_prob=False):
        """
        Forward pass: convert L1/L2 predictions into reading time estimates.

        Args:
            L1: (batch, seq_len) - familiarity check time per word (ms)
            L2: (batch, seq_len) - lexical access time per word (ms)
            skip_input: (batch, seq_len) - either cloze predictability or skip probability
            word_lengths: (batch, seq_len) - number of characters per word
            input_is_prob: (bool) - if True, treat skip_input as probability (0-1).
                                    if False, treat as predictability and apply sigmoid.

        Returns:
            dict with:
                'total_reading_time': (batch, seq_len) predicted TRT (ms)
                'first_fixation':     (batch, seq_len) predicted FFD (ms)
                'skip_prob':          (batch, seq_len) predicted skip probability
                'gaze_duration':      (batch, seq_len) predicted gaze duration (ms)
        """
        # --- 1. Soft skip probability ---
        if input_is_prob:
            skip_prob = skip_input
        else:
            # High predictability -> high skip probability
            skip_prob = torch.sigmoid(self.skip_sharpness * (skip_input - 0.5))

        # --- 2. Eccentricity scaling ---
        if self.ablation == 'no_eccentricity':
            ecc_scale = torch.ones_like(word_lengths)
        else:
            ecc_scale = 1.0 + self.eccentricity * (word_lengths - 4.0).clamp(min=0)

        # --- 3. Scaled processing times ---
        L1_scaled = L1 * ecc_scale
        L2_scaled = L2

        # --- 4. First fixation duration ---
        if self.ablation == 'ffd_l1_only':
            first_fixation = L1_scaled  # no L2 contribution
        else:
            first_fixation = L1_scaled + torch.nn.functional.softplus(self.l2_contribution) * L2_scaled

        # --- 5. Gaze duration (first-pass reading) ---
        gaze_duration = L1_scaled + L2_scaled

        # --- 6. Regression mechanism ---
        if self.ablation == 'no_regressions':
            regression_penalty = torch.zeros_like(gaze_duration)
        else:
            regression_prob = torch.sigmoid(
                self.regression_sharpness * (L2 - self.regression_threshold)
            )
            prev_gaze = torch.zeros_like(gaze_duration)
            prev_gaze[:, 1:] = gaze_duration[:, :-1]
            regression_penalty = regression_prob * torch.nn.functional.softplus(self.regression_cost_scale) * prev_gaze

        # --- 7. Total reading time ---
        fixate_prob = 1.0 - skip_prob
        overhead = self.saccade_time + self.attention_shift
        total_reading_time = fixate_prob * (gaze_duration + overhead + regression_penalty)

        return {
            'total_reading_time': total_reading_time,
            'first_fixation': first_fixation,
            'gaze_duration': gaze_duration,
            'skip_prob': skip_prob,
        }
