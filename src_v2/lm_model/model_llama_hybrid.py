"""
Neural EZ Reader Hybrid — neural L1 head + Reichle cascade.

Combines the neural L1 head from model_llama_faithful_sh_v2.py (which
replaces Reichle's L1 formula with a trainable projection over LLaMA
features) with the Reichle cascade from model_llama_reichle_v3.py
(explicit M1/M2/I motor stages, eccentricity, refixation gate, constant
pF integration failure).

Cognitive cascade:

    base_L1  =  softplus(L1_head(LLaMA_hidden)) * l1_scale
                    neural prediction of pre-eccentricity
                    familiarity-check time; no formula, no frequency
                    input, no surprisal input, no residual head.

    L1_ecc   =  base_L1 * epsilon^((wordlen - 1) / 2)
                    eccentricity exponent (clamped to [30, 500])

    L2       =  delta * base_L1
                    Reichle: L2 uses the pre-eccentricity base L1

    FFD      =  L1_ecc + M1 + M2
                    first fixation = L1 processing + labile + non-labile

    refix    =  sigmoid(lambda_refix * (wordlen - refix_pivot))

    Gaze     =  FFD + refix * (L2 + M1 + M2)

    TRT      =  Gaze + I + pF * reg_weight * prev_gaze

Skip is a learned parallel parafoveal head whose gradient is detached
inside TRT to keep the L1 cascade and the skip head training signals
independent.

Differences from model_llama_reichle_v3.py:
    - base_L1 comes from an L1 head (no alpha1/alpha2/alpha3 formula,
      no frequency log, no clipped surprisal, no residual head)
    - AutoModel instead of AutoModelForCausalLM (no logits needed)
    - forward() no longer takes `frequencies`

Differences from model_llama_faithful_sh_v2.py:
    - Reichle cascade replaces the simple FFD = L1, Gaze = L1 + L2
    - Adds eccentricity, explicit M1/M2/I motor stages, refixation gate,
      constant pF regression (all learnable)

Learnable parameters:
    Neural:    LLaMA top layers, projection, L1 head, skip head
    Cognitive: delta, l1_scale, epsilon, M1, M2, I, pF, reg_weight,
               lambda_refix, refix_pivot
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
    """
    Maps (base_L1, L2, skip_prob, word_lengths) to FFD / Gaze / TRT via the
    Reichle cascade. All cognitive constants are exposed as learnable
    parameters so they can be reported and compared to the literature.
    """

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
        L1 = base_L1 * torch.pow(self.epsilon, ecc_exponent)
        L1 = L1.clamp(min=self.L1_MIN, max=self.L1_MAX)

        first_fixation = L1 + self.M1 + self.M2

        refix_prob = torch.sigmoid(
            self.lambda_refix * (word_lengths - self.refix_pivot)
        )

        refix_duration = L2 + self.M1 + self.M2
        gaze_duration = first_fixation + refix_prob * refix_duration

        prev_gaze = torch.zeros_like(gaze_duration)
        prev_gaze[:, 1:] = gaze_duration[:, :-1]
        regression_cost = self.pF * self.reg_weight * prev_gaze

        conditional_trt = gaze_duration + self.I + regression_cost

        total_reading_time = (1.0 - skip_prob.detach()) * conditional_trt

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
#  Neural EZ Reader Hybrid
# --------------------------------------------------------------------------- #

class NeuralEZReaderHybrid(nn.Module):
    """
    word tokens -> LLaMA encoder
        -> last-subword pooling
        -> projection
        -> L1 head (softplus * l1_scale) -> base_L1
        -> L2 = delta * base_L1
        -> ReichleEZReader cascade
        -> FFD, Gaze, TRT, skip

    The L1 head is a neural network, not a formula. LLaMA hidden states
    implicitly carry context/predictability information; no explicit
    frequency or surprisal input is provided.

    Learnable:
        Neural:    LLaMA top layers, projection, L1 head, skip head
        Cognitive: delta, l1_scale, epsilon, M1, M2, I, pF, reg_weight,
                   lambda_refix, refix_pivot
    """

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        freeze_layers: int = 12,
        hidden_dim: int = 256,
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

        self.skip_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
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
        """
        Args:
            word_lists:   list of list of str
            word_lengths: (B, S) float tensor of character counts per word.

        Returns dict with FFD / Gaze / TRT / skip plus L1 / L2 / delta
        and all Reichle cascade parameters for logging.
        """
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

        base_L1 = self.l1_head(projected).squeeze(-1) * self.l1_scale
        skip_prob = self.skip_head(projected).squeeze(-1)

        seq_len = word_lengths.size(1)
        base_L1 = base_L1[:, :seq_len]
        skip_prob = skip_prob[:, :seq_len]

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
