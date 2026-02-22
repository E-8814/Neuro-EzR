"""
Differentiable EZ Reader.

A simplified, fully-differentiable approximation of the EZ Reader model.
Replaces discrete events and random draws with smooth tensor operations.

Preserved from EZ Reader:
  - L1 (familiarity check) and L2 (lexical access) as separate stages
  - Word skipping based on predictability
  - Saccade overhead between words
  - Eccentricity scaling (further words take longer)
  - Integration difficulty contributing to total time

Replaced with smooth approximations:
  - Hard if/else skipping → sigmoid soft skip
  - Discrete regressions → expected re-reading penalty
  - Random motor noise → expected landing position
  - Simpy event queue → cumulative tensor sums
"""

import torch
import torch.nn as nn


class DifferentiableEZReader(nn.Module):
    """
    Differentiable approximation of the EZ Reader simulation.

    Takes L1/L2 predictions from a neural network and produces
    predicted reading time metrics using smooth, differentiable operations.

    All operations are standard PyTorch — backprop works end-to-end.
    """

    def __init__(self):
        super().__init__()

        # Learnable EZ Reader parameters (initialized from literature values)
        self.saccade_time = nn.Parameter(torch.tensor(150.0))     # M1 + M2 overhead (ms)
        self.attention_shift = nn.Parameter(torch.tensor(25.0))   # attention shift time (ms)
        self.skip_sharpness = nn.Parameter(torch.tensor(8.0))     # sigmoid steepness for skip
        self.eccentricity = nn.Parameter(torch.tensor(0.1))       # eccentricity scaling factor
        self.integration_cost = nn.Parameter(torch.tensor(0.08))  # expected regression cost factor

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
            # High predictability → high skip probability
            # sigmoid maps predictability to (0, 1) with steepness controlled by skip_sharpness
            skip_prob = torch.sigmoid(self.skip_sharpness * (skip_input - 0.5))

        # --- 2. Eccentricity scaling ---
        # Words further from fixation take longer to process
        # In the original, this scales L1 by eccentricity^distance
        # Here we use a soft approximation: longer words = slightly harder
        ecc_scale = 1.0 + self.eccentricity * (word_lengths - 4.0).clamp(min=0)

        # --- 3. Scaled processing times ---
        L1_scaled = L1 * ecc_scale
        L2_scaled = L2

        # --- 4. First fixation duration ---
        # FFD ≈ L1 conditional on fixation (not scaled by skip)
        first_fixation = L1_scaled

        # --- 5. Gaze duration (first-pass reading) ---
        # Gaze = L1 + L2 conditional on fixation
        gaze_duration = L1_scaled + L2_scaled

        # --- 6. Integration / regression penalty ---
        # Expected cost of re-reading due to integration difficulty
        # Harder words (higher L1 + L2) are more likely to cause regressions
        # Softplus ensures cost stays positive (can't subtract time)
        integration_penalty = torch.nn.functional.softplus(self.integration_cost) * (L1_scaled + L2_scaled)

        # --- 7. Total reading time ---
        # TRT is the *expected* time across trials: P(fixated) * (gaze + overhead + integration)
        # Skipped words contribute 0 to TRT
        fixate_prob = 1.0 - skip_prob
        overhead = self.saccade_time + self.attention_shift
        total_reading_time = fixate_prob * (gaze_duration + overhead + integration_penalty)

        return {
            'total_reading_time': total_reading_time,
            'first_fixation': first_fixation,
            'gaze_duration': gaze_duration,
            'skip_prob': skip_prob,
        }
