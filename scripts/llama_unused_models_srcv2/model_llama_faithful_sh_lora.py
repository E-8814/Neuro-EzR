"""
Neural EZ Reader — Faithful EZReader + learned skip head + LoRA.

Same as model_llama_faithful_sh.py but uses LoRA adapters on ALL
LLaMA attention layers instead of unfreezing the top N layers.

LoRA (Low-Rank Adaptation):
  For each attention weight W (frozen), adds a trainable bypass:
    output = W·x + (A·B)·x    where A is (r×d), B is (d×r), r<<d
  Applied to: q_proj, k_proj, v_proj, o_proj in every layer.
  This lets all 22 layers adapt with ~2M params instead of
  unfreezing ~50M params in the top 6.

Faithful EZReader cascade (unchanged):
  L1 = neural_net(context)
  L2 = δ × L1
  FFD = L1, Gaze = L1 + L2, TRT = Gaze + regression
  Skip = learned head (parallel parafoveal process)

Learnable parameters:
  LoRA:      A and B matrices in all attention layers (~2M params)
  Neural:    projection, L1 head, skip head
  Cognitive: delta, l1_scale
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


# --------------------------------------------------------------------------- #
#  LoRA
# --------------------------------------------------------------------------- #

class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with low-rank adaptation."""

    def __init__(self, original_linear, rank=16, alpha=32, dropout=0.05):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.scaling = alpha / rank

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # A: kaiming init, B: zeros → LoRA starts as identity (no change)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Freeze original weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x):
        original_out = self.original(x)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return original_out + lora_out


def apply_lora(model, rank=16, alpha=32, dropout=0.05,
               target_modules=("q_proj", "k_proj", "v_proj", "o_proj")):
    """Replace target linear layers in model with LoRA wrappers."""
    replaced = 0
    for name, module in list(model.named_modules()):
        if any(name.endswith(t) for t in target_modules):
            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            original = getattr(parent, parts[-1])
            if isinstance(original, nn.Linear):
                setattr(parent, parts[-1],
                        LoRALinear(original, rank, alpha, dropout))
                replaced += 1
    return replaced


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
#  Neural EZ Reader Model with LoRA
# --------------------------------------------------------------------------- #

class NeuralEZReaderLLaMA(nn.Module):
    """
    word tokens → LLaMA (LoRA-adapted) → word pooling → L1 head
    → L2 = delta × L1
    → skip head (learned)
    → FaithfulEZReader → (FFD, Gaze, TRT, skip)
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B",
        hidden_dim: int = 256,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        super().__init__()

        # --- LLaMA encoder ---
        self.llama = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.llama.config.pad_token_id = self.tokenizer.eos_token_id

        llama_dim = self.llama.config.hidden_size

        # Freeze ALL LLaMA parameters
        for param in self.llama.parameters():
            param.requires_grad = False

        # Apply LoRA to all attention layers
        n_lora = apply_lora(self.llama, rank=lora_rank, alpha=lora_alpha,
                            dropout=lora_dropout)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.n_lora_modules = n_lora

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

        # --- l1_scale ---
        self.l1_scale = nn.Parameter(torch.tensor(50.0))

        # --- Delta ---
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

        # --- Faithful EZ Reader ---
        self.ezreader = FaithfulEZReader()

    @property
    def delta(self):
        return torch.sigmoid(self._delta_raw)

    def lora_params(self):
        """Return only LoRA parameters (for separate optimizer group)."""
        for name, param in self.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                yield param

    def head_params(self):
        """Return non-LoRA trainable parameters."""
        lora_ids = {id(p) for p in self.lora_params()}
        for param in self.parameters():
            if param.requires_grad and id(param) not in lora_ids:
                yield param

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

        L1 = self.l1_head(projected).squeeze(-1) * self.l1_scale
        L1 = L1.clamp(min=1.0, max=600.0)

        L2 = self.delta * L1

        skip_prob = self.skip_head(projected).squeeze(-1)

        seq_len = predictability.size(1)
        L1 = L1[:, :seq_len]
        L2 = L2[:, :seq_len]
        skip_prob = skip_prob[:, :seq_len]

        result = self.ezreader(L1, L2, skip_prob, word_lengths)

        result['L1'] = L1
        result['L2'] = L2
        result['delta'] = self.delta

        return result
