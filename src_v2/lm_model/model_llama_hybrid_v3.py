"""
Neural EZ Reader Hybrid v3 — decoupled projection for the skip head.

Three architectural changes vs model_llama_hybrid_v2.py, driven by the
observation that v2 traded ~0.01 r_Gaze for +0.19 r_skip relative to v1
because skip BCE (after the loss-scale rebalance in v2's training recipe)
was reshaping the shared projection used by the L1 head.

  (1) Split projection. The skip head now has its own `skip_projection`
      from the LLaMA hidden states, independent of `self.projection` that
      feeds the L1 head. Skip BCE gradient reaches LLaMA only through
      `skip_projection`, so it can no longer reshape the features the
      L1 head depends on. The two tasks still compete through the top
      LLaMA layers, which have much more capacity than a single 256-dim
      projection and can carry both targets without collapse.

  (2) Smooth L1 floor. Replaces `.clamp(min=30)` with a softplus-based
      floor `L1 = 30 + softplus(L1_raw - 30)`. Identity far above 30,
      smooth and strictly positive near it, always differentiable. The
      hard floor in v1/v2 zeroed gradients on fast-fixation words and
      was one reason L1 variance kept growing over training (the model
      could not shrink small L1 values further, so it compensated by
      stretching large L1 values).

  (3) total_reading_time == conditional_trt. v2's multiplication by
      `(1 - skip_prob.detach())` shrank the reported TRT by the skip
      rate even though the training loss compares `conditional_trt`
      against fixated-only human TRT. The two were inconsistent and
      introduced a systematic negative bias in the reported TRT MAE.
      v3 aliases `total_reading_time` to `conditional_trt`. If you need
      the unconditional mean downstream, multiply by `(1 - skip_prob)`
      at evaluation time.

Length concat on the skip head (`[skip_projected, word_length/10]`) is
kept from v2 — it is a small, direct cue and is orthogonal to the split.
The rest of the Reichle cascade is unchanged.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


def _logit(y: float) -> float:
    return math.log(y / (1.0 - y))


# --------------------------------------------------------------------------- #
#  Reichle EZ Reader cascade
# --------------------------------------------------------------------------- #

class ReichleEZReader(nn.Module):
    L1_MIN = 30.0
    L1_MAX = 500.0

    def __init__(self):
        super().__init__()

        self._epsilon_raw = nn.Parameter(torch.tensor(_inv_softplus(0.15)))

        self._M1_raw = nn.Parameter(torch.tensor(_inv_softplus(125.0)))
        self._M2_raw = nn.Parameter(torch.tensor(_inv_softplus(25.0)))
        self._I_raw = nn.Parameter(torch.tensor(_inv_softplus(25.0)))

        self.lambda_refix = nn.Parameter(torch.tensor(0.4))
        self.refix_pivot = nn.Parameter(torch.tensor(8.0))

        self._pF_raw = nn.Parameter(torch.tensor(_logit(0.01)))

        self._reg_weight_raw = nn.Parameter(torch.tensor(_inv_softplus(0.5)))

    @property
    def epsilon(self):
        return 1.0 + F.softplus(self._epsilon_raw)

    @property
    def M1(self):
        return F.softplus(self._M1_raw)

    @property
    def M2(self):
        return F.softplus(self._M2_raw)

    @property
    def I(self):
        return F.softplus(self._I_raw)

    @property
    def pF(self):
        return torch.sigmoid(self._pF_raw)

    @property
    def reg_weight(self):
        return F.softplus(self._reg_weight_raw)

    def forward(self, base_L1, L2, skip_prob, word_lengths):
        ecc_exponent = (word_lengths - 1.0) / 2.0
        L1_raw = base_L1 * torch.pow(self.epsilon, ecc_exponent)

        # Smooth floor: approaches L1_MIN from above as L1_raw -> -inf,
        # identity as L1_raw grows. Always has a non-zero gradient unlike
        # the hard clamp used in v1 / v2.
        L1 = self.L1_MIN + F.softplus(L1_raw - self.L1_MIN)
        L1 = L1.clamp(max=self.L1_MAX)

        first_fixation = L1 + self.M1 + self.M2

        refix_prob = torch.sigmoid(
            self.lambda_refix * (word_lengths - self.refix_pivot)
        )

        refix_duration = L2 + self.M1 + self.M2
        gaze_duration = first_fixation + refix_prob * refix_duration

        prev_gaze = torch.cat(
            [torch.zeros_like(gaze_duration[:, :1]), gaze_duration[:, :-1]],
            dim=1,
        )
        regression_cost = self.pF * self.reg_weight * prev_gaze

        conditional_trt = gaze_duration + self.I + regression_cost

        # v3: no (1 - skip.detach()) factor. total_reading_time matches the
        # training loss target (fixated-only conditional_trt). Multiply by
        # (1 - skip_prob) downstream if unconditional TRT is needed.
        total_reading_time = conditional_trt

        return {
            'first_fixation': first_fixation,
            'gaze_duration': gaze_duration,
            'conditional_trt': conditional_trt,
            'total_reading_time': total_reading_time,
            'skip_prob': skip_prob,
            'L1': L1,
            'L2': L2,
            'refix_prob': refix_prob,
            'epsilon': self.epsilon.detach(),
            'M1': self.M1.detach(),
            'M2': self.M2.detach(),
            'I': self.I.detach(),
            'pF': self.pF.detach(),
            'reg_weight': self.reg_weight.detach(),
            'lambda_refix': self.lambda_refix.detach(),
            'refix_pivot': self.refix_pivot.detach(),
        }


# --------------------------------------------------------------------------- #
#  Neural EZ Reader Hybrid v3
# --------------------------------------------------------------------------- #

class NeuralEZReaderHybrid(nn.Module):
    """
    Independent projection paths for the L1 head and the skip head so that
    skip BCE gradient cannot reshape the features consumed by the L1 head.

        word tokens -> LLaMA encoder -> last-subword pooling -> word_repr

        word_repr -> projection       -> l1_head      -> base_L1
                  -> skip_projection  -> [+len/10]    -> skip_head -> skip_prob

        base_L1 -> (delta)  -> L2
               -> (cascade) -> FFD, Gaze, TRT

    Both projections still share the LLaMA top layers, which have enough
    capacity to serve both targets. Gradients from skip BCE only touch
    `skip_projection` and LLaMA; they never touch `self.projection` or
    `l1_head`, so L1 / gaze quality is protected from whatever loss
    rebalancing the training script chooses.
    """

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        freeze_layers: int = 12,
        hidden_dim: int = 256,
        skip_hidden_dim: int = 128,
    ):
        super().__init__()

        self.llama = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.llama.config.pad_token_id = self.tokenizer.eos_token_id

        llama_dim = self.llama.config.hidden_size
        self.llama_dim = llama_dim

        if freeze_layers > 0:
            for param in self.llama.embed_tokens.parameters():
                param.requires_grad = False
            for layer_idx in range(min(freeze_layers, len(self.llama.layers))):
                for param in self.llama.layers[layer_idx].parameters():
                    param.requires_grad = False

        # L1 / gaze path.
        self.projection = nn.Sequential(
            nn.Linear(llama_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.l1_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )

        self.l1_scale = nn.Parameter(torch.tensor(50.0))

        self._delta_raw = nn.Parameter(torch.tensor(_logit(0.34)))

        # Skip path: independent projection + length concat on the head.
        self.skip_projection = nn.Sequential(
            nn.Linear(llama_dim, skip_hidden_dim),
            nn.LayerNorm(skip_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.skip_head = nn.Sequential(
            nn.Linear(skip_hidden_dim + 1, skip_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(skip_hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.ezreader = ReichleEZReader()

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

    def forward(self, word_lists, word_lengths):
        device = word_lengths.device

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
        skip_projected = self.skip_projection(word_repr)

        base_L1 = self.l1_head(projected).squeeze(-1) * self.l1_scale

        seq_len = word_lengths.size(1)
        base_L1 = base_L1[:, :seq_len]
        skip_projected = skip_projected[:, :seq_len, :]

        length_feat = (word_lengths / 10.0).unsqueeze(-1)
        skip_input = torch.cat([skip_projected, length_feat], dim=-1)
        skip_prob = self.skip_head(skip_input).squeeze(-1)

        L2 = self.delta * base_L1

        result = self.ezreader(
            base_L1=base_L1,
            L2=L2,
            skip_prob=skip_prob,
            word_lengths=word_lengths,
        )

        result['base_L1'] = base_L1
        result['delta'] = self.delta.detach()
        result['l1_scale'] = self.l1_scale.detach()

        return result
