"""
Neural EZ Reader v3 — Conditional TRT training with minimal learnable EZR params.

Changes from model_llama_delta_rem_p_comp.py:
  1. l1_scale is learnable (was fixed at 50). Lets the model calibrate
     L1 to match human FFD scale (~200ms) instead of staying at ~120ms.
  2. refix_threshold is learnable (was fixed at 150ms). When L1 scales up
     to realistic values, the fixed 150ms threshold makes almost every
     word get refixated. A learnable threshold adapts to the actual L1
     distribution.
  3. Outputs conditional_trt = gaze + regression (before the (1-skip)
     multiplier). Training should use this for TRT loss because human TRT
     is conditional on fixation — the (1-skip) factor was creating a
     systematic underestimate.
  4. total_reading_time still computed with (1-skip) for eval compatibility.

Learnable parameters:
  - LLaMA top layers + projection + L1 head + skip head  (neural)
  - delta (L2/L1 ratio), l1_scale, refix_threshold       (cognitive)
Everything else in EZReader is fixed from theory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


# --------------------------------------------------------------------------- #
#  Differentiable EZ Reader — 1 learnable parameter (refix_threshold)
# --------------------------------------------------------------------------- #

class DifferentiableEZReader(nn.Module):
    """
    (L1, L2, skip_prob, word_lengths) -> reading metrics.

    Fixed from literature: refix_sharpness, regression params.
    Learnable: refix_threshold (adapts to actual L1 distribution).
    """

    REFIX_SHARPNESS = 0.03

    REGRESSION_THRESHOLD = 50.0
    REGRESSION_SHARPNESS = 0.1
    REGRESSION_COST_SCALE = 1.0

    def __init__(self, refix_threshold_init=200.0):
        super().__init__()
        self.refix_threshold = nn.Parameter(torch.tensor(refix_threshold_init))

    def forward(self, L1, L2, skip_prob, word_lengths):
        """
        Returns dict with:
            total_reading_time  — (1-skip) * (gaze + regression), for eval
            conditional_trt     — gaze + regression, for training loss
            first_fixation      — L1
            gaze_duration       — L1 + refix_prob * L2
            skip_prob           — pass-through
        """
        # --- FFD = L1 ---
        first_fixation = L1

        # --- Refixation: hard words (high L1) get refixated ---
        refix_prob = torch.sigmoid(
            self.REFIX_SHARPNESS * (L1 - self.refix_threshold)
        )

        # --- Gaze = L1 + selective refixation ---
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

        # --- Conditional TRT: what you'd observe given fixation ---
        conditional_trt = gaze_duration + regression_penalty

        # --- Expected TRT: marginalizing over skip ---
        total_reading_time = (1.0 - skip_prob.detach()) * conditional_trt

        return {
            'total_reading_time': total_reading_time,
            'conditional_trt': conditional_trt,
            'first_fixation': first_fixation,
            'gaze_duration': gaze_duration,
            'skip_prob': skip_prob,
        }


# --------------------------------------------------------------------------- #
#  Neural EZ Reader Model
# --------------------------------------------------------------------------- #

class NeuralEZReaderLLaMA(nn.Module):
    """
    word tokens -> LLaMA -> word pooling -> L1 head -> L2 = delta * L1
    -> skip head -> EZReader -> (conditional_trt, FFD, Gaze, skip)

    Learnable cognitive params: delta, l1_scale, refix_threshold.
    """

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

        # --- l1_scale: learnable, initialized at 50 ---
        self.l1_scale = nn.Parameter(torch.tensor(50.0))

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

        # --- EZ Reader (1 learnable param: refix_threshold) ---
        self.ezreader = DifferentiableEZReader(refix_threshold_init=200.0)

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
        L1 = self.l1_head(projected).squeeze(-1) * self.l1_scale
        L1 = L1.clamp(min=1.0, max=600.0)
        L2 = self.delta * L1

        # --- Skip ---
        skip_prob = self.skip_head(projected).squeeze(-1)

        # Trim to match actual sequence lengths
        seq_len = predictability.size(1)
        L1 = L1[:, :seq_len]
        L2 = L2[:, :seq_len]
        skip_prob = skip_prob[:, :seq_len]

        # --- EZ Reader ---
        result = self.ezreader(L1, L2, skip_prob, word_lengths)

        result['L1'] = L1
        result['L2'] = L2
        result['delta'] = self.delta

        return result
