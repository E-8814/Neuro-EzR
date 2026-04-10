"""
Neural EZ Reader - Refined Model with Proper Cognitive Constraints.

Changes from model_llama_v2_delta.py:
  1. DiffEZReader has ZERO learnable parameters — all fixed from theory.
  2. Removed saccade_time and attention_shift from TRT (inter-fixation events,
     not fixation durations).
  3. Removed skip_sharpness (dead code when skip head outputs probability).
  4. Removed eccentricity (LLaMA already captures word length effects).
  5. Removed l2_contribution — replaced by refixation mechanism.
  6. FFD = L1 (first fixation = familiarity check, L2 hasn't completed).
  7. Gaze = L1 + refix_prob * L2, where refix_prob = sigmoid(sharpness * (L1 - threshold)).
     Harder words (high L1) get refixated; easy words don't. Matches EZ Reader theory.
  8. TRT = (1 - skip) * (Gaze + regression_penalty). No overhead.
  9. Regression parameters fixed as constants.
  10. No ablation branches — single clean path.
  11. l1_scale fixed at 50.
  12. delta remains the only learnable cognitive parameter (L2/L1 ratio).

All learning happens in: LLaMA layers + projection + L1 head + skip head + delta.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

import os
import sys


# --------------------------------------------------------------------------- #
#  Differentiable EZ Reader — zero learnable parameters
# --------------------------------------------------------------------------- #

class DifferentiableEZReader(nn.Module):
    """
    Pure deterministic function: (L1, L2, skip_prob, word_lengths) -> reading metrics.

    All parameters fixed from EZ Reader literature. No nn.Parameters.
    """

    # Fixed constants from literature / model convergence
    REFIX_THRESHOLD = 150.0   # L1 (ms) above which refixation becomes likely
    REFIX_SHARPNESS = 0.03    # sigmoid steepness for refixation

    REGRESSION_THRESHOLD = 50.0   # L2 (ms) above which regression is likely
    REGRESSION_SHARPNESS = 0.1    # sigmoid steepness for regression
    REGRESSION_COST_SCALE = 1.0   # cost multiplier

    def forward(self, L1, L2, skip_prob, word_lengths):
        """
        Args:
            L1:           (batch, seq_len) familiarity check time (ms)
            L2:           (batch, seq_len) lexical access time (ms)
            skip_prob:    (batch, seq_len) skip probability (0-1)
            word_lengths: (batch, seq_len) character counts (unused, kept for API compat)

        Returns:
            dict with total_reading_time, first_fixation, gaze_duration, skip_prob
        """
        # --- FFD: first fixation = L1 only ---
        first_fixation = L1

        # --- Refixation: hard words (high L1) get refixated ---
        refix_prob = torch.sigmoid(
            self.REFIX_SHARPNESS * (L1 - self.REFIX_THRESHOLD)
        )

        # --- Gaze: first-pass reading = L1 + refixation contribution ---
        gaze_duration = L1 + refix_prob * L2

        # --- Regression mechanism ---
        regression_prob = torch.sigmoid(
            self.REGRESSION_SHARPNESS * (L2 - self.REGRESSION_THRESHOLD)
        )
        prev_gaze = torch.zeros_like(gaze_duration)
        prev_gaze[:, 1:] = gaze_duration[:, :-1]
        regression_penalty = (
            regression_prob
            * F.softplus(torch.tensor(self.REGRESSION_COST_SCALE))
            * prev_gaze
        )

        # --- TRT: no overhead (saccade/attention are inter-fixation events) ---
        # Detach skip so TRT gradient doesn't fight skip head.
        # Skip learns from BCE loss only; TRT gradient flows through Gaze/regression only.
        total_reading_time = (1.0 - skip_prob.detach()) * (gaze_duration + regression_penalty)

        return {
            'total_reading_time': total_reading_time,
            'first_fixation': first_fixation,
            'gaze_duration': gaze_duration,
            'skip_prob': skip_prob,
        }


# --------------------------------------------------------------------------- #
#  Neural EZ Reader Model
# --------------------------------------------------------------------------- #

class NeuralEZReaderLLaMA(nn.Module):
    """
    End-to-end model:
        word tokens -> LLaMA (causal) -> word-level pooling -> L1 head
        -> L2 = delta * L1 (theory-constrained)
        -> skip head
        -> DifferentiableEZReader -> (TRT, FFD, Gaze, skip)

    Learnable: LLaMA top layers, projection, L1 head, skip head, delta.
    Fixed: l1_scale (50), all EZReader parameters.
    """

    L1_SCALE = 50.0  # fixed scaling: head output * 50 -> milliseconds

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B",
        freeze_layers: int = 12,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # --- LLaMA encoder ---
        self.llama = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.llama.config.pad_token_id = self.tokenizer.eos_token_id

        llama_dim = self.llama.config.hidden_size

        # Freeze lower layers
        if freeze_layers > 0:
            for param in self.llama.embed_tokens.parameters():
                param.requires_grad = False
            for layer_idx in range(min(freeze_layers, len(self.llama.layers))):
                for param in self.llama.layers[layer_idx].parameters():
                    param.requires_grad = False

        # --- Projection ---
        self.projection = nn.Sequential(
            nn.Linear(llama_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # --- L1 head ---
        self.l1_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )

        # --- Delta: L2 = delta * L1, constrained to (0, 1) via sigmoid ---
        self._delta_raw = nn.Parameter(torch.tensor(0.0))
        with torch.no_grad():
            self._delta_raw.fill_(torch.log(torch.tensor(0.34 / (1.0 - 0.34))).item())

        # --- Skip head ---
        self.skip_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # --- Differentiable EZ Reader (zero learnable parameters) ---
        self.ezreader = DifferentiableEZReader()

    @property
    def delta(self):
        return torch.sigmoid(self._delta_raw)

    def _tokenize_and_align(self, word_lists, device):
        encodings = self.tokenizer(
            word_lists,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        batch_word_maps = []
        max_words = 0

        for batch_idx in range(len(word_lists)):
            word_ids = encodings.word_ids(batch_index=batch_idx)
            word_map = {}

            for subword_idx, word_idx in enumerate(word_ids):
                if word_idx is None:
                    continue
                if word_idx not in word_map:
                    word_map[word_idx] = [subword_idx, subword_idx + 1]
                else:
                    word_map[word_idx][1] = subword_idx + 1

            n_words = len(word_lists[batch_idx])
            spans = []
            for w_idx in range(n_words):
                if w_idx in word_map:
                    spans.append(tuple(word_map[w_idx]))
                else:
                    spans.append((0, 1))

            batch_word_maps.append(spans)
            max_words = max(max_words, n_words)

        return input_ids, attention_mask, batch_word_maps, max_words

    def _pool_subwords_to_words(self, hidden_states, batch_word_maps, max_words, device):
        batch_size = hidden_states.size(0)
        hidden_dim = hidden_states.size(2)

        idx = torch.zeros(batch_size, max_words, dtype=torch.long)
        for b in range(batch_size):
            for w_idx, (start, end) in enumerate(batch_word_maps[b]):
                idx[b, w_idx] = end - 1
        idx = idx.to(device)

        word_repr = torch.gather(
            hidden_states, 1, idx.unsqueeze(-1).expand(-1, -1, hidden_dim)
        )
        return word_repr

    def forward(self, word_lists, predictability, word_lengths):
        """
        Args:
            word_lists:     list of list of str
            predictability: (batch, seq_len) float tensor (0-1)
            word_lengths:   (batch, seq_len) float tensor

        Returns:
            dict with predicted reading metrics + L1/L2/delta
        """
        device = predictability.device

        input_ids, attention_mask, word_maps, max_words = self._tokenize_and_align(
            word_lists, device
        )

        llama_out = self.llama(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

        word_repr = self._pool_subwords_to_words(
            llama_out, word_maps, max_words, device
        )

        projected = self.projection(word_repr)

        # --- L1 and L2 ---
        L1 = self.l1_head(projected).squeeze(-1) * self.L1_SCALE
        L1 = L1.clamp(min=1.0, max=500.0)
        L2 = self.delta * L1

        # --- Skip ---
        skip_prob = self.skip_head(projected).squeeze(-1)

        # Trim to match actual sequence lengths
        seq_len = predictability.size(1)
        L1 = L1[:, :seq_len]
        L2 = L2[:, :seq_len]
        skip_prob = skip_prob[:, :seq_len]

        # --- EZ Reader derives FFD, Gaze, TRT ---
        result = self.ezreader(L1, L2, skip_prob, word_lengths)

        result['L1'] = L1
        result['L2'] = L2
        result['delta'] = self.delta

        return result
