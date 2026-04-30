"""
Differentiable EZ Reader v3.

Changes from v2:
  - TRT/FFD/Gaze predict reading time CONDITIONAL on fixation (no skip multiplication)
  - Skip probability is a separate, independent output
  - This fixes the systematic under-prediction caused by conflating
    P(fixated) with reading time in v2's TRT = (1-skip) * reading_time

In the original E-Z Reader, skipping is a discrete event: you either skip
or you don't. If you don't skip, your reading time is determined by L1/L2/regressions.
v3 respects this separation.
"""

import torch
import torch.nn as nn


class DifferentiableEZReaderV3(nn.Module):
    """
    Differentiable approximation of the EZ Reader simulation (v3).

    Takes L1/L2 predictions from a neural network and produces
    predicted reading time metrics using smooth, differentiable operations.

    Key difference from v2: reading times are conditional on fixation.
    Skip probability is a separate output, not multiplied into TRT.
    """

    def __init__(self, ablation=None):
        super().__init__()
        self.ablation = ablation  # None, 'no_eccentricity', 'no_regressions', 'ffd_l1_only'

        # Learnable EZ Reader parameters (initialized from literature values)
        self.saccade_time = nn.Parameter(torch.tensor(150.0))     # M1 + M2 overhead (ms)
        self.attention_shift = nn.Parameter(torch.tensor(25.0))   # attention shift time (ms)
        self.skip_sharpness = nn.Parameter(torch.tensor(8.0))     # sigmoid steepness for skip
        self.eccentricity = nn.Parameter(torch.tensor(0.1))       # eccentricity scaling factor

        # v2 parameters carried forward
        self.l2_contribution = nn.Parameter(torch.tensor(0.3))            # fraction of L2 in FFD
        self.regression_threshold = nn.Parameter(torch.tensor(50.0))      # L2 threshold for regression
        self.regression_sharpness = nn.Parameter(torch.tensor(0.1))       # sigmoid steepness
        self.regression_cost_scale = nn.Parameter(torch.tensor(1.0))      # cost multiplier

    def forward(self, L1, L2, skip_input, word_lengths, input_is_prob=False):
        """
        Forward pass: convert L1/L2 predictions into reading time estimates.

        All reading times (TRT, FFD, Gaze) are CONDITIONAL on fixation --
        they represent how long you spend on a word IF you look at it.
        Skip probability is a separate output.

        Args:
            L1: (batch, seq_len) - familiarity check time per word (ms)
            L2: (batch, seq_len) - lexical access time per word (ms)
            skip_input: (batch, seq_len) - either cloze predictability or skip probability
            word_lengths: (batch, seq_len) - number of characters per word
            input_is_prob: (bool) - if True, treat skip_input as probability (0-1).
                                    if False, treat as predictability and apply sigmoid.

        Returns:
            dict with:
                'total_reading_time': (batch, seq_len) predicted TRT | fixated (ms)
                'first_fixation':     (batch, seq_len) predicted FFD | fixated (ms)
                'gaze_duration':      (batch, seq_len) predicted Gaze | fixated (ms)
                'skip_prob':          (batch, seq_len) predicted skip probability
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

        # --- 4. First fixation duration (conditional on fixation) ---
        if self.ablation == 'ffd_l1_only':
            first_fixation = L1_scaled  # no L2 contribution
        else:
            first_fixation = L1_scaled + torch.nn.functional.softplus(self.l2_contribution) * L2_scaled

        # --- 5. Gaze duration (first-pass reading, conditional on fixation) ---
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

        # --- 7. Total reading time (conditional on fixation) ---
        # v3: NO multiplication by fixate_prob. This is TRT given that
        # the word was fixated. Skip is handled separately in the loss.
        overhead = self.saccade_time + self.attention_shift
        total_reading_time = gaze_duration + overhead + regression_penalty

        return {
            'total_reading_time': total_reading_time,
            'first_fixation': first_fixation,
            'gaze_duration': gaze_duration,
            'skip_prob': skip_prob,
        }
