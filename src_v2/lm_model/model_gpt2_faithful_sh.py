"""
Neural EZ Reader — GPT-2 engine variant with learned skip head.

Identical architecture to model_llama_faithful_sh.py but uses GPT-2 as the
causal language model encoder instead of LLaMA.

Supported models: gpt2 (124M), gpt2-medium (355M), gpt2-large (774M), gpt2-xl (1.5B)

Core EZ Reader cascade (unchanged from LLaMA version):
  L1 = neural_net(context)         replaces frequency formula
  L2 = δ × L1                     Reichle et al., δ ≈ 0.34
  FFD = L1                         first fixation ≈ familiarity check
  Gaze = L1 + L2                   first pass = both processing stages

Learnable parameters:
  Neural:    GPT-2 top layers, projection, L1 head, skip head
  Cognitive: delta (L2/L1 ratio), l1_scale (calibration)
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


# --------------------------------------------------------------------------- #
#  Faithful Differentiable EZ Reader — zero learnable parameters
# --------------------------------------------------------------------------- #

class FaithfulEZReader(nn.Module):
    """
    Maps (L1, L2, skip_prob) → reading metrics using published EZ Reader
    equations. Zero learnable parameters.
    """

    REGRESSION_SHARPNESS = 0.03
    REGRESSION_THRESHOLD = 100.0
    REGRESSION_COST_SCALE = 0.25

    def forward(self, L1, L2, skip_prob, word_lengths):
        first_fixation = L1
        gaze_duration = L1 + L2

        regression_prob = torch.sigmoid(
            self.REGRESSION_SHARPNESS * (L2 - self.REGRESSION_THRESHOLD)
        )
        prev_gaze = torch.zeros_like(gaze_duration)
        prev_gaze[:, 1:] = gaze_duration[:, :-1]
        regression_cost = regression_prob * self.REGRESSION_COST_SCALE * prev_gaze

        conditional_trt = gaze_duration + regression_cost
        total_reading_time = (1.0 - skip_prob.detach()) * conditional_trt

        return {
            'first_fixation': first_fixation,
            'gaze_duration': gaze_duration,
            'conditional_trt': conditional_trt,
            'total_reading_time': total_reading_time,
            'skip_prob': skip_prob,
        }


# --------------------------------------------------------------------------- #
#  Neural EZ Reader Model — GPT-2 engine
# --------------------------------------------------------------------------- #

class NeuralEZReaderGPT2(nn.Module):
    """
    word tokens → GPT-2 (causal) → word pooling → L1 head
    → L2 = delta × L1
    → skip head (learned, parallel to L1→L2 cascade)
    → FaithfulEZReader → (FFD, Gaze, TRT, skip)
    """

    def __init__(
        self,
        model_name: str = "gpt2-medium",
        freeze_layers: int = 16,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # --- GPT-2 encoder ---
        self.gpt2 = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.gpt2.config.pad_token_id = self.tokenizer.eos_token_id

        gpt2_dim = self.gpt2.config.n_embd

        # Freeze embeddings and lower layers
        if freeze_layers > 0:
            for param in self.gpt2.wte.parameters():
                param.requires_grad = False
            for param in self.gpt2.wpe.parameters():
                param.requires_grad = False
            for layer_idx in range(min(freeze_layers, len(self.gpt2.h))):
                for param in self.gpt2.h[layer_idx].parameters():
                    param.requires_grad = False

        # --- Projection ---
        self.projection = nn.Sequential(
            nn.Linear(gpt2_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # --- L1 head: predicts familiarity check time ---
        self.l1_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )

        # --- l1_scale: calibration parameter ---
        self.l1_scale = nn.Parameter(torch.tensor(50.0))

        # --- Delta: L2 = delta × L1, constrained to (0, 1) via sigmoid ---
        self._delta_raw = nn.Parameter(torch.tensor(0.0))
        with torch.no_grad():
            self._delta_raw.fill_(torch.log(torch.tensor(0.34 / (1.0 - 0.34))).item())

        # --- Skip head: learned parallel parafoveal process ---
        self.skip_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # --- Faithful EZ Reader (zero learnable parameters) ---
        self.ezreader = FaithfulEZReader()

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
            predictability: (batch, seq_len) float tensor (0-1) — unused, kept for interface compat
            word_lengths:   (batch, seq_len) float tensor
        """
        device = predictability.device

        input_ids, attention_mask, word_maps, max_words = self._tokenize_and_align(
            word_lists, device
        )

        gpt2_out = self.gpt2(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

        word_repr = self._pool_subwords_to_words(
            gpt2_out, word_maps, max_words, device
        )

        projected = self.projection(word_repr)

        # --- L1: neural prediction of familiarity check time ---
        L1 = self.l1_head(projected).squeeze(-1) * self.l1_scale
        L1 = L1.clamp(min=1.0, max=600.0)

        # --- L2 = delta × L1 (Reichle et al.) ---
        L2 = self.delta * L1

        # --- Skip: learned parallel parafoveal process ---
        skip_prob = self.skip_head(projected).squeeze(-1)

        # Trim to match actual sequence lengths
        seq_len = predictability.size(1)
        L1 = L1[:, :seq_len]
        L2 = L2[:, :seq_len]
        skip_prob = skip_prob[:, :seq_len]

        # --- EZ Reader: FFD, Gaze, TRT from theory ---
        result = self.ezreader(L1, L2, skip_prob, word_lengths)

        result['L1'] = L1
        result['L2'] = L2
        result['delta'] = self.delta

        return result
