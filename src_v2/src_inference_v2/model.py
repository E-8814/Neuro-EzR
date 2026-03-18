"""
Neural E-Z Reader: LLaMA encoder + Differentiable E-Z Reader (aligned).

Architecture:
    Text -> LLaMA (causal, left-to-right)
         -> word-level pooling (last subword)
         -> projection
         -> L1 head (familiarity check time)
         -> L2 = delta * L1 (lexical access time, theory-constrained)
         -> skip head (skip probability)
         -> DiffEZReader (aligned with discrete simulation)
         -> predicted TRT, FFD, Gaze, Skip, integration_failure_prob

L2 = delta * L1 follows original E-Z Reader theory (Reichle et al., 1998-2012).
Published delta values: 0.85 (1998), 0.50 (EZR-9), 0.25 (EZR-10), 0.34 (2012).
"""

import os
import sys

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diff_ezreader import DifferentiableEZReader


class NeuralEZReader(nn.Module):

    def __init__(
        self,
        model_name="meta-llama/Llama-3.2-1B",
        freeze_layers=12,
        hidden_dim=256,
    ):
        super().__init__()

        # --- LLaMA encoder (causal: word N only sees words 1..N) ---
        self.llama = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.llama.config.pad_token_id = self.tokenizer.eos_token_id

        llama_dim = self.llama.config.hidden_size

        # Freeze lower layers (keep top layers trainable for fine-tuning)
        if freeze_layers > 0:
            for param in self.llama.embed_tokens.parameters():
                param.requires_grad = False
            for i in range(min(freeze_layers, len(self.llama.layers))):
                for param in self.llama.layers[i].parameters():
                    param.requires_grad = False

        # --- Projection from LLaMA dim to internal hidden dim ---
        self.projection = nn.Sequential(
            nn.Linear(llama_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # --- L1 prediction head (familiarity check) ---
        self.l1_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # L1 > 0
        )
        self.l1_scale = nn.Parameter(torch.tensor(50.0))

        # --- Delta: L2 = delta * L1 ---
        # Stored in unconstrained space; sigmoid maps to (0, 1) in forward
        # Initialize so sigmoid(_delta_raw) ~ 0.34 (latest published value)
        self._delta_raw = nn.Parameter(torch.tensor(0.0))
        with torch.no_grad():
            self._delta_raw.fill_(
                torch.log(torch.tensor(0.34 / (1.0 - 0.34))).item()
            )

        # --- Skip prediction head ---
        self.skip_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # --- Differentiable E-Z Reader (aligned with simulation) ---
        self.ezreader = DifferentiableEZReader()

    @property
    def delta(self):
        """The learned L2/L1 ratio, constrained to (0, 1)."""
        return torch.sigmoid(self._delta_raw)

    def get_sim_params(self):
        """Get all parameters needed by the discrete simulation.

        Returns a dict that can be passed directly to NeuralEZReaderSim.
        """
        params = self.ezreader.get_sim_params()
        params['delta'] = self.delta.item()
        return params

    def _tokenize_and_align(self, word_lists, device):
        """Tokenize a batch of word lists and compute subword->word mapping."""
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
                    spans.append((0, 1))  # fallback to pad position

            batch_word_maps.append(spans)
            max_words = max(max_words, n_words)

        return input_ids, attention_mask, batch_word_maps, max_words

    def _pool_subwords_to_words(self, hidden_states, batch_word_maps,
                                 max_words, device):
        """Pool subword representations to word-level (last subword).

        For causal models, the last subword has attended to all previous
        subwords of the same word, giving the richest representation.
        """
        batch_size = hidden_states.size(0)
        hidden_dim = hidden_states.size(2)

        idx = torch.zeros(batch_size, max_words, dtype=torch.long)
        for b in range(batch_size):
            for w_idx, (start, end) in enumerate(batch_word_maps[b]):
                idx[b, w_idx] = end - 1  # last subword
        idx = idx.to(device)

        return torch.gather(
            hidden_states, 1,
            idx.unsqueeze(-1).expand(-1, -1, hidden_dim),
        )

    def forward(self, word_lists, predictability, word_lengths):
        """
        Full forward pass.

        Args:
            word_lists:     list of list of str (raw word tokens per sentence)
            predictability: (B, T) float tensor (accepted for interface compat,
                            not used -- the LLM learns its own predictability)
            word_lengths:   (B, T) float tensor (character counts per word)

        Returns:
            dict with:
                total_reading_time, first_fixation, gaze_duration, skip_prob,
                integration_failure_prob, L1, L2, delta
        """
        device = predictability.device
        seq_len = predictability.size(1)

        # --- LLaMA forward (causal: each word sees only left context) ---
        input_ids, attn_mask, word_maps, max_words = \
            self._tokenize_and_align(word_lists, device)

        llama_out = self.llama(
            input_ids=input_ids,
            attention_mask=attn_mask,
        ).last_hidden_state

        # --- Pool subwords -> word-level ---
        word_repr = self._pool_subwords_to_words(
            llama_out, word_maps, max_words, device,
        )

        # --- Project to hidden dim ---
        projected = self.projection(word_repr)

        # --- Predict L1 (familiarity check) ---
        L1 = self.l1_head(projected).squeeze(-1) * self.l1_scale
        L1 = L1[:, :seq_len].clamp(min=1.0, max=500.0)

        # --- L2 = delta * L1 (theory-constrained) ---
        L2 = self.delta * L1

        # --- Predict skip probability ---
        skip_prob = self.skip_head(projected).squeeze(-1)[:, :seq_len]

        # --- Differentiable E-Z Reader (aligned) ---
        result = self.ezreader(L1, L2, skip_prob, word_lengths)

        # Add neural outputs for inspection and logging
        result['L1'] = L1
        result['L2'] = L2
        result['delta'] = self.delta

        return result
