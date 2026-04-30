"""
Neural EZ Reader Hybrid v4 — Reichle-skeleton L1 head + cognitively-
grounded (race-based) skip prediction.

Branches from model_llama_hybrid.py (v1), independently of v2 (which
added word_length to the skip head) and v3 (which split the projection
path between the L1 and skip heads). v4 is a different experiment on
the same base.

Two architectural changes vs v1, motivated by analysis of
train_hybrid_v2_geco.log:

  (1) L1 head is suppressed in v1/v2/v3. The softplus(MLP) * 50 head
      produces base_L1 with only ~20 ms of per-word std, far less than
      the ~50 ms needed to fit long-tail words. "ascendancy" was
      under-predicted by 150-200 ms in every epoch of v2 training.
      Two causes: (a) softplus saturates near 0 so initial output range
      is narrow; (b) LLaMA hidden states encode frequency only
      implicitly, so the L1 head has to rediscover -alpha2 * ln(freq)
      from scratch through a 256->128->1 MLP.

      Fix: reintroduce the Reichle skeleton as an explicit additive
      structure, with LLaMA providing only the context-dependent residual.

          base_L1 = l1_base_offset
                    + l1_freq_coef * (log(freq) - 10) / 5
                    + l1_neural_head(projected_LLaMA)
          base_L1 = 5 + softplus(base_L1 - 5)      # soft floor

      `l1_base_offset` (Reichle alpha1 analogue) is initialized to 60 ms.
      `l1_freq_coef` (Reichle alpha2 analogue, measured against the
      normalized log-frequency) is initialized to -17, which matches
      Reichle's -alpha2 * log(freq) ~ -3.4 * log(freq) after rescaling.
      `l1_neural_head` is a small MLP with near-zero init so the
      Reichle skeleton dominates at epoch 1 and the LLaMA residual only
      picks up context/predictability as training progresses.

      The hard [30, 500] L1 clamp in the cascade is also replaced with
      a soft floor via `5 + softplus(L1 - 5)`. No hard ceiling — if
      the data wants L1 = 400 ms for rare words, it can express it.

  (2) M1, M2, and I are not separately identifiable in v1/v2/v3. They
      appear only as additive constants, so the optimizer can't tell
      them apart. The v2 log confirms this: M2 and I both ended at
      16.6 ms (same to 0.06 ms).

      Fix: skip probability now emerges from a soft race between M1
      and the parafoveal L1 of word n+1, as in the original E-Z Reader
      simulation. M1 gets a unique gradient path through the skip BCE
      loss that M2 doesn't share, restoring M1's individual role. A
      small learned residual head catches non-cognitive skip patterns
      (end-of-line, function-word grammatical effects) that a pure
      race would miss.

Cognitive cascade:

    # L1: Reichle skeleton + small neural residual
    log_freq_norm = (log(freq) - 10) / 5
    base_L1_raw   = l1_base_offset
                    + l1_freq_coef * log_freq_norm
                    + l1_neural_head(projected_LLaMA)
    base_L1       = 5 + softplus(base_L1_raw - 5)          # soft floor

    # Reichle cascade with soft L1 floor (no hard ceiling)
    L1_ecc        = base_L1 * epsilon^((wordlen - 1) / 2)
    L1            = 5 + softplus(L1_ecc - 5)
    L2            = delta * base_L1
    FFD           = L1 + M1 + M2
    refix_prob    = sigmoid(lambda_refix * (wordlen - refix_pivot))
    Gaze          = FFD + refix_prob * (L2 + M1 + M2)
    TRT           = Gaze + I + pF * reg_weight * prev_gaze

    # Skip: parafoveal race + learned residual
    ecc_exp_next  = (wordlen_n / 2) + 1 + (wordlen_{n+1} - 1) / 2
    L1_next_para  = base_L1[n+1] * epsilon^ecc_exp_next
    race_logit    = (M1 - L1_next_para) / skip_temperature
    skip_prob     = sigmoid(race_logit + residual_skip_logit)

forward() signature:
    model(word_lists, frequencies, word_lengths)

Learnable parameters:
    Neural:    LLaMA top layers, projection, l1_neural_head,
               skip_residual_head
    Cognitive: l1_base_offset, l1_freq_coef, delta, epsilon, M1, M2, I,
               pF, reg_weight, lambda_refix, refix_pivot,
               skip_temperature
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
#  Reichle EZ Reader cascade with parafoveal-race skip
# --------------------------------------------------------------------------- #

class ReichleEZReader(nn.Module):
    """
    Maps (base_L1, L2, residual_skip_logit, word_lengths) to FFD / Gaze
    / TRT / skip via the Reichle cascade.

    Differences from model_llama_hybrid.ReichleEZReader:
      - Takes residual_skip_logit instead of skip_prob. Skip is computed
        internally from M1 vs parafoveal L1_next + residual logit.
      - Replaces hard L1 clamp [30, 500] with soft floor via
        `5 + softplus(L1 - 5)`. No hard ceiling.
      - Adds skip_temperature as a learnable cognitive parameter.
    """

    L1_SOFT_FLOOR = 5.0   # soft floor (ms) on eccentricity-adjusted L1

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

        # Skip race softness (ms). Larger = softer race. Init 30 ms.
        self._skip_temperature_raw = nn.Parameter(
            torch.tensor(_inv_softplus(30.0))
        )

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

    @property
    def skip_temperature(self):
        return 1.0 + F.softplus(self._skip_temperature_raw)

    def forward(self, base_L1, L2, residual_skip_logit, word_lengths):
        # --- L1 with eccentricity and soft floor ---
        ecc_exponent = (word_lengths - 1.0) / 2.0
        L1_ecc = base_L1 * torch.pow(self.epsilon, ecc_exponent)
        L1 = self.L1_SOFT_FLOOR + F.softplus(L1_ecc - self.L1_SOFT_FLOOR)

        # --- FFD and refixation ---
        first_fixation = L1 + self.M1 + self.M2

        refix_prob = torch.sigmoid(
            self.lambda_refix * (word_lengths - self.refix_pivot)
        )

        refix_duration = L2 + self.M1 + self.M2
        gaze_duration = first_fixation + refix_prob * refix_duration

        # --- TRT with integration + regression cost ---
        prev_gaze = torch.cat(
            [torch.zeros_like(gaze_duration[:, :1]), gaze_duration[:, :-1]],
            dim=1,
        )
        regression_cost = self.pF * self.reg_weight * prev_gaze

        conditional_trt = gaze_duration + self.I + regression_cost

        # --- Skip: parafoveal race + residual ---
        # base_L1 of word n+1, shifted. The last position has no next
        # word; assign a large L1_next so the race logit is strongly
        # negative and only the residual head matters at that position.
        base_L1_next = torch.cat(
            [base_L1[:, 1:], torch.full_like(base_L1[:, :1], 1000.0)],
            dim=1,
        )
        wordlen_next = torch.cat(
            [word_lengths[:, 1:], torch.zeros_like(word_lengths[:, :1])],
            dim=1,
        )

        # Parafoveal eccentricity exponent: from the center of word n
        # to the first letter of word n+1 is approximately
        # (wordlen_n / 2) + 1, plus (wordlen_{n+1} - 1) / 2 to reach the
        # center of word n+1.
        parafoveal_dist = word_lengths / 2.0 + 1.0
        ecc_exp_next = parafoveal_dist + (
            (wordlen_next - 1.0).clamp(min=0.0) / 2.0
        )
        L1_next_parafoveal = base_L1_next * torch.pow(
            self.epsilon, ecc_exp_next
        )

        race_logit = (self.M1 - L1_next_parafoveal) / self.skip_temperature
        skip_prob = torch.sigmoid(race_logit + residual_skip_logit)

        # v4 aligns total_reading_time with the training target
        # (conditional_trt). Multiply by (1 - skip_prob) downstream if
        # unconditional TRT is needed.
        total_reading_time = conditional_trt

        return {
            'first_fixation': first_fixation,
            'gaze_duration': gaze_duration,
            'conditional_trt': conditional_trt,
            'total_reading_time': total_reading_time,
            'skip_prob': skip_prob,
            'race_logit': race_logit,
            'residual_skip_logit': residual_skip_logit,
            'L1': L1,
            'L2': L2,
            'L1_next_parafoveal': L1_next_parafoveal,
            'refix_prob': refix_prob,
            'epsilon': self.epsilon.detach(),
            'M1': self.M1.detach(),
            'M2': self.M2.detach(),
            'I': self.I.detach(),
            'pF': self.pF.detach(),
            'reg_weight': self.reg_weight.detach(),
            'lambda_refix': self.lambda_refix.detach(),
            'refix_pivot': self.refix_pivot.detach(),
            'skip_temperature': self.skip_temperature.detach(),
        }


# --------------------------------------------------------------------------- #
#  Neural EZ Reader Hybrid v4
# --------------------------------------------------------------------------- #

class NeuralEZReaderHybrid(nn.Module):
    """
    (word tokens, frequencies) -> LLaMA encoder
        -> last-subword pooling
        -> projection
        -> base_L1 = l1_base_offset + l1_freq_coef * log_freq_norm
                     + l1_neural_head(projected)
        -> L2 = delta * base_L1
        -> skip_residual_head -> residual_skip_logit
        -> ReichleEZReader cascade (race-based skip)
        -> FFD, Gaze, TRT, skip

    Learnable:
        Neural:    LLaMA top layers, projection, l1_neural_head,
                   skip_residual_head
        Cognitive: l1_base_offset, l1_freq_coef, delta, epsilon,
                   M1, M2, I, pF, reg_weight, lambda_refix,
                   refix_pivot, skip_temperature
    """

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        freeze_layers: int = 12,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.llama = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.float32
        )
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

        # Reichle skeleton parameters.
        #   l1_base_offset = alpha1 analogue (Reichle 2003: 104 ms)
        #   l1_freq_coef   = alpha2 analogue on normalized log_freq.
        # With log_freq_norm = (log(freq) - 10) / 5, a coefficient of
        # -17 gives `-17 * (log(freq) - 10) / 5 = -3.4 * log(freq) + 34`
        # which matches Reichle's -alpha2 * log(freq) up to a constant
        # that l1_base_offset absorbs.
        self.l1_base_offset = nn.Parameter(torch.tensor(60.0))
        self.l1_freq_coef = nn.Parameter(torch.tensor(-17.0))

        # Neural residual on the Reichle skeleton: LLaMA context
        # provides context/predictability effects that the formula's
        # freq term cannot capture. Near-zero init so the Reichle
        # skeleton dominates at epoch 1.
        self.l1_neural_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.normal_(self.l1_neural_head[-1].weight, std=0.01)
        nn.init.zeros_(self.l1_neural_head[-1].bias)

        self._delta_raw = nn.Parameter(torch.tensor(_logit(0.34)))

        # Residual skip head: small correction on top of the race logit.
        # Near-zero init so the parafoveal race dominates at start.
        self.skip_residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.normal_(self.skip_residual_head[-1].weight, std=0.01)
        nn.init.zeros_(self.skip_residual_head[-1].bias)

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

    def _pool_subwords_to_words(
        self, hidden_states, batch_word_maps, max_words, device
    ):
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

    def forward(self, word_lists, frequencies, word_lengths):
        """
        Args:
            word_lists:   list of list of str
            frequencies:  (B, S) float tensor of SUBTLEX raw counts per word
            word_lengths: (B, S) float tensor of character counts per word

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

        seq_len = word_lengths.size(1)
        projected = projected[:, :seq_len, :]

        # Reichle skeleton: base + freq_coef * log_freq_norm + neural residual
        log_freq = torch.log(frequencies.clamp(min=1.0))
        log_freq_norm = (log_freq - 10.0) / 5.0

        base_L1_formula = (
            self.l1_base_offset + self.l1_freq_coef * log_freq_norm
        )
        base_L1_neural = self.l1_neural_head(projected).squeeze(-1)
        l1_raw = base_L1_formula + base_L1_neural

        # Soft floor at 5 ms; no hard ceiling.
        base_L1 = 5.0 + F.softplus(l1_raw - 5.0)

        residual_skip_logit = self.skip_residual_head(projected).squeeze(-1)

        L2 = self.delta * base_L1

        result = self.ezreader(
            base_L1=base_L1,
            L2=L2,
            residual_skip_logit=residual_skip_logit,
            word_lengths=word_lengths,
        )

        result['base_L1'] = base_L1
        result['base_L1_formula'] = base_L1_formula
        result['base_L1_neural'] = base_L1_neural
        result['delta'] = self.delta.detach()
        result['l1_base_offset'] = self.l1_base_offset.detach()
        result['l1_freq_coef'] = self.l1_freq_coef.detach()
        result['log_freq'] = log_freq

        return result
