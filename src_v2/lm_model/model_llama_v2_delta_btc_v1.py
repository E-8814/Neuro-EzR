"""
Neural EZ Reader Model - v2 delta, Break The Ceiling v1.

Changes from model_llama_v2_delta.py:
  1. Default freeze_layers reduced from 75% to 50% — unfreezes more LLaMA layers
     so the LM can actually adapt its representations for eye-tracking prediction.
  2. Bigger projection and heads:
     - Projection: llama_dim -> 512 (was 256)
     - L1 head: 512 -> 512 -> 256 -> 1 (was 256 -> 128 -> 1)
     - Skip head: 512 -> 512 -> 256 -> 1 (was 256 -> 128 -> 1)
     More capacity so the heads don't saturate in 1-2 epochs.

Everything else unchanged: L2 = delta * L1, DiffEZReader v2, same forward logic.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from diff_ezreader import DifferentiableEZReader


class NeuralEZReaderLLaMA(nn.Module):
    """
    End-to-end model (BTC v1 — bigger heads, more unfrozen layers):
        word tokens -> LLaMA (causal) -> word-level pooling -> L1 head
        -> L2 = delta * L1 (theory-constrained)
        -> skip head
        -> DifferentiableEZReader v2 -> (TRT, FFD, Gaze, skip)
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B",
        freeze_layers: int = None,  # None = auto (50% of layers)
        hidden_dim: int = 512,      # was 256
        ablation: str = None,
    ):
        super().__init__()
        self.ablation = ablation

        # --- LLaMA encoder ---
        self.llama = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.llama.config.pad_token_id = self.tokenizer.eos_token_id

        llama_dim = self.llama.config.hidden_size
        n_layers = len(self.llama.layers)

        # Auto-determine freeze layers: 50% (was 75%)
        if freeze_layers is None:
            freeze_layers = n_layers // 2
        self.freeze_layers = freeze_layers

        # Freeze lower layers
        if freeze_layers > 0:
            for param in self.llama.embed_tokens.parameters():
                param.requires_grad = False
            for layer_idx in range(min(freeze_layers, n_layers)):
                for param in self.llama.layers[layer_idx].parameters():
                    param.requires_grad = False

        # --- Projection from LLaMA dim to internal hidden dim ---
        # Bigger projection: 2-layer with residual-friendly structure
        self.projection = nn.Sequential(
            nn.Linear(llama_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # --- L1 head: deeper (3 layers instead of 2) ---
        if ablation == 'no_two_stage':
            self.single_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softplus(),
            )
            self.l1_head = None
        else:
            self.l1_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softplus(),
            )
            self.single_head = None

        # --- Delta parameter: L2 = delta * L1 ---
        self._delta_raw = nn.Parameter(torch.tensor(0.0))
        with torch.no_grad():
            self._delta_raw.fill_(torch.log(torch.tensor(0.34 / (1.0 - 0.34))).item())

        # --- Skip head: deeper (3 layers instead of 2) ---
        if ablation == 'skip_from_l1':
            self.skip_head = None
        else:
            self.skip_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )

        # Scale to start in reasonable ms range
        self.l1_scale = nn.Parameter(torch.tensor(50.0))

        # --- Differentiable EZ Reader v2 ---
        ezr_ablation = ablation if ablation in ('no_eccentricity', 'no_regressions', 'ffd_l1_only') else None
        self.ezreader = DifferentiableEZReader(ablation=ezr_ablation)

    @property
    def delta(self):
        """The learned L2/L1 ratio, constrained to (0, 1)."""
        return torch.sigmoid(self._delta_raw)

    def _tokenize_and_align(self, word_lists, device):
        """
        Tokenize a batch of word lists and compute the mapping from
        subword tokens back to original words.
        """
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
        """Pool subword representations to word-level using last-subword strategy."""
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
        Full forward pass.

        Args:
            word_lists:     list of list of str -- raw word tokens per sentence
            predictability: (batch, seq_len) float tensor (0-1)
            word_lengths:   (batch, seq_len) float tensor (character counts)

        Returns:
            dict with predicted reading metrics + L1/L2/delta for inspection
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

        # --- Predict L1; derive L2 = delta * L1 ---
        if self.ablation == 'no_two_stage':
            L_total = self.single_head(projected).squeeze(-1) * self.l1_scale
            L_total = L_total.clamp(min=1.0, max=500.0)
            L1 = 0.6 * L_total
            L2 = 0.4 * L_total
        else:
            L1 = self.l1_head(projected).squeeze(-1) * self.l1_scale
            L1 = L1.clamp(min=1.0, max=500.0)
            L2 = self.delta * L1

        # --- Predict skip probability ---
        if self.ablation == 'skip_from_l1':
            skip_prob = torch.sigmoid(-0.05 * (L1 - 80.0))
        else:
            skip_prob = self.skip_head(projected).squeeze(-1)

        # Trim to match actual sequence lengths
        seq_len = predictability.size(1)
        L1 = L1[:, :seq_len]
        L2 = L2[:, :seq_len]
        skip_prob = skip_prob[:, :seq_len]

        result = self.ezreader(L1, L2, skip_prob, word_lengths, input_is_prob=True)

        result['L1'] = L1
        result['L2'] = L2
        result['delta'] = self.delta

        return result
