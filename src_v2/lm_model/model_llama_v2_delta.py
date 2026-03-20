"""
Neural EZ Reader Model (LLaMA + Differentiable EZ Reader) - v2 with delta constraint.

Same as model_llama.py (v2) but with one change:
  - L2 = delta * L1 instead of two independent heads.
    delta is a learned scalar in (0,1) via sigmoid, initialized at 0.34.

Everything else (DiffEZReader v2, skip head, eccentricity, ablations) is unchanged.
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
    End-to-end model:
        word tokens -> LLaMA (causal) -> word-level pooling -> L1 head
        -> L2 = delta * L1 (theory-constrained)
        -> skip head
        -> DifferentiableEZReader v2 -> (TRT, FFD, Gaze, skip)
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B",
        freeze_layers: int = 12,
        hidden_dim: int = 256,
        ablation: str = None,
    ):
        super().__init__()
        self.ablation = ablation  # None, 'no_two_stage', 'no_eccentricity',
                                  # 'no_regressions', 'skip_from_l1', 'ffd_l1_only'

        # --- LLaMA encoder ---
        self.llama = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # LLaMA tokenizers often lack a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.llama.config.pad_token_id = self.tokenizer.eos_token_id

        llama_dim = self.llama.config.hidden_size  # 2048 for LLaMA-3.2-1B

        # Freeze lower layers
        if freeze_layers > 0:
            for param in self.llama.embed_tokens.parameters():
                param.requires_grad = False
            for layer_idx in range(min(freeze_layers, len(self.llama.layers))):
                for param in self.llama.layers[layer_idx].parameters():
                    param.requires_grad = False

        # --- Projection from LLaMA dim to internal hidden dim ---
        self.projection = nn.Sequential(
            nn.Linear(llama_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # --- Predict L1 per word; L2 = delta * L1 ---
        if ablation == 'no_two_stage':
            # Single processing time head instead of separate L1/L2
            self.single_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softplus(),
            )
            self.l1_head = None
        else:
            self.l1_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softplus(),  # L1 > 0
            )
            self.single_head = None

        # --- Delta parameter: L2 = delta * L1 ---
        # Published values: 0.85 (1998), 0.50 (EZR-9), 0.25 (EZR-10), 0.34 (2012)
        # Stored in unconstrained space; sigmoid maps it to (0, 1) in forward
        self._delta_raw = nn.Parameter(torch.tensor(0.0))
        with torch.no_grad():
            self._delta_raw.fill_(torch.log(torch.tensor(0.34 / (1.0 - 0.34))).item())

        # --- Skip prediction head ---
        if ablation == 'skip_from_l1':
            self.skip_head = None  # derived from L1 in forward()
        else:
            self.skip_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )

        # Scale to start in reasonable ms range
        self.l1_scale = nn.Parameter(torch.tensor(50.0))

        # --- Differentiable EZ Reader v2 ---
        # Pass EZR-level ablations through
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
                    # Fallback: map to pad token position
                    spans.append((0, 1))

            batch_word_maps.append(spans)
            max_words = max(max_words, n_words)

        return input_ids, attention_mask, batch_word_maps, max_words

    def _pool_subwords_to_words(self, hidden_states, batch_word_maps, max_words, device):
        """Pool subword representations to word-level using last-subword strategy.

        For causal models, the last subword token has the most context
        (it has attended to all previous subwords of the same word).
        """
        batch_size = hidden_states.size(0)
        hidden_dim = hidden_states.size(2)

        # Use last subword index (end - 1) for causal models
        idx = torch.zeros(batch_size, max_words, dtype=torch.long)
        for b in range(batch_size):
            for w_idx, (start, end) in enumerate(batch_word_maps[b]):
                idx[b, w_idx] = end - 1  # last subword token
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

        # --- LLaMA encodes the sentence (causal: each word only sees left context) ---
        input_ids, attention_mask, word_maps, max_words = self._tokenize_and_align(
            word_lists, device
        )

        llama_out = self.llama(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state  # (B, subword_len, llama_dim)

        # --- Pool subwords -> word-level (last subword for causal models) ---
        word_repr = self._pool_subwords_to_words(
            llama_out, word_maps, max_words, device
        )  # (B, T, llama_dim)

        # --- Project to hidden dim ---
        projected = self.projection(word_repr)  # (B, T, hidden_dim)

        # --- Predict L1; derive L2 = delta * L1 ---
        if self.ablation == 'no_two_stage':
            # Single processing time, split into L1 and L2 with fixed ratio
            L_total = self.single_head(projected).squeeze(-1) * self.l1_scale
            L_total = L_total.clamp(min=1.0, max=500.0)
            L1 = 0.6 * L_total
            L2 = 0.4 * L_total
        else:
            L1 = self.l1_head(projected).squeeze(-1) * self.l1_scale   # (B, T)
            L1 = L1.clamp(min=1.0, max=500.0)
            L2 = self.delta * L1

        # --- Predict skip probability ---
        if self.ablation == 'skip_from_l1':
            # Derive skip from L1: easy words (low L1) get skipped
            skip_prob = torch.sigmoid(-0.05 * (L1 - 80.0))  # centered at 80ms
        else:
            skip_prob = self.skip_head(projected).squeeze(-1)  # (B, T)

        # Trim to match actual sequence lengths
        seq_len = predictability.size(1)
        L1 = L1[:, :seq_len]
        L2 = L2[:, :seq_len]
        skip_prob = skip_prob[:, :seq_len]

        # --- Differentiable EZ Reader v2 produces reading metrics ---
        result = self.ezreader(L1, L2, skip_prob, word_lengths, input_is_prob=True)

        # Add L1/L2/delta to result for logging
        result['L1'] = L1
        result['L2'] = L2
        result['delta'] = self.delta

        return result
