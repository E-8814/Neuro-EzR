"""
Neural EZ Reader Hybrid v4b — v4 simplified.

This is v4 with one structural swap and one identifiability fix:

  (1) L1_NEURAL_HEAD REPLACED BY EXPLICIT PREDICTABILITY.
      v4's 33k-parameter MLP on LLaMA hidden states is removed.
      In its place is a single scalar coefficient on LLaMA's next-token
      probability:
          base_L1 = alpha1 + alpha2 * log_freq_norm
                           - alpha3 * exp(-surprisal)
      The MLP was absorbing *something* between frequency and
      "everything else" with no clean decomposition. v4b forces the
      predictability signal into a named, interpretable scalar and
      drops the unexplained residual pathway entirely.

      Tradeoffs vs v4:
        - 33,000 fewer parameters; 1 new scalar.
        - Every L1 term is a Reichle-canonical quantity — no MLP.
        - Likely costs 0.02-0.04 on r_TRT (the MLP was doing real work).
        - Gains clean parameter recovery for the paper table.

      Skip mechanism is unchanged — race + residual — because v6b
      showed that dropping the skip residual breaks r_skip entirely
      (pure race could not match empirical skip rates).

  (2) M2 AND I TIED.
      v4 had two separate parameters that drifted to the same value
      (24.1 ms each in the v4 log). They enter the cascade only as
      additive constants so their sum is the only identifiable
      quantity. v4b uses a single _M2I_raw parameter, reporting
      M2 and I as equal in the paper.

Reichle-unit accessors (alpha1_reichle, alpha2_reichle,
alpha3_reichle) are exposed as properties so the paper table can
report parameter values directly in Reichle's 2003 conventions
without mental arithmetic from the normalized log-frequency feature.

Unchanged from v4:
  - Cascade: FFD = L1 + M1 + M2, Gaze = FFD + refix_prob*(L2 + M1 + M2),
    TRT = Gaze + I + pF*reg_weight*prev_gaze.
  - Eccentricity: L1 = base_L1 * epsilon^((w-1)/2).
  - L2 = delta * base_L1.
  - Skip = sigmoid((M1 - L1_next_parafoveal)/tau + residual_skip_logit).
  - Soft L1 floor at 5 ms (no hard ceiling).
  - TinyLlama top-10 trainable, bottom-12 frozen.

forward() signature: (word_lists, frequencies, word_lengths).
The model now uses AutoModelForCausalLM internally so it can access
lm_head for surprisal computation.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


def _logit(y: float) -> float:
    return math.log(y / (1.0 - y))


# --------------------------------------------------------------------------- #
#  Reichle EZ Reader cascade (race + residual skip, tied M2 = I)
# --------------------------------------------------------------------------- #

class ReichleEZReader(nn.Module):
    L1_SOFT_FLOOR = 5.0

    def __init__(self):
        super().__init__()

        self._epsilon_raw = nn.Parameter(torch.tensor(_inv_softplus(0.15)))

        self._M1_raw = nn.Parameter(torch.tensor(_inv_softplus(125.0)))
        self._M2I_raw = nn.Parameter(torch.tensor(_inv_softplus(25.0)))

        self.lambda_refix = nn.Parameter(torch.tensor(0.4))
        self.refix_pivot = nn.Parameter(torch.tensor(8.0))

        self._pF_raw = nn.Parameter(torch.tensor(_logit(0.01)))
        self._reg_weight_raw = nn.Parameter(torch.tensor(_inv_softplus(0.5)))

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
        return F.softplus(self._M2I_raw)

    @property
    def I(self):
        return F.softplus(self._M2I_raw)

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
        ecc_exponent = (word_lengths - 1.0) / 2.0
        L1_ecc = base_L1 * torch.pow(self.epsilon, ecc_exponent)
        L1 = self.L1_SOFT_FLOOR + F.softplus(L1_ecc - self.L1_SOFT_FLOOR)

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

        # Skip: parafoveal race + residual (v4 mechanism, kept).
        base_L1_next = torch.cat(
            [base_L1[:, 1:], torch.full_like(base_L1[:, :1], 1000.0)],
            dim=1,
        )
        wordlen_next = torch.cat(
            [word_lengths[:, 1:], torch.zeros_like(word_lengths[:, :1])],
            dim=1,
        )

        parafoveal_dist = word_lengths / 2.0 + 1.0
        ecc_exp_next = parafoveal_dist + (
            (wordlen_next - 1.0).clamp(min=0.0) / 2.0
        )
        L1_next_parafoveal = base_L1_next * torch.pow(
            self.epsilon, ecc_exp_next
        )

        race_logit = (self.M1 - L1_next_parafoveal) / self.skip_temperature
        skip_prob = torch.sigmoid(race_logit + residual_skip_logit)

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
#  Neural EZ Reader Hybrid v4b
# --------------------------------------------------------------------------- #

class NeuralEZReaderHybrid(nn.Module):
    """
    (word tokens, frequencies) -> LLaMA (AutoModelForCausalLM)
        -> last-subword pooling + per-word surprisal from lm_head
        -> projection  (only used by the skip residual head)

        base_L1 = alpha1 + alpha2 * log_freq_norm
                         - alpha3 * exp(-surprisal)      (no neural L1 residual)

        base_L1 -> soft floor
                -> delta      -> L2
                -> cascade    -> FFD, Gaze, TRT
                -> race + skip_residual_head -> skip

    Learnable:
        Neural:    LLaMA top layers, projection, skip_residual_head
        Cognitive: l1_base_offset (alpha1), l1_freq_coef (alpha2),
                   l1_pred_coef  (alpha3), delta, epsilon, M1,
                   M2 = I (tied), pF, reg_weight, lambda_refix,
                   refix_pivot, skip_temperature

    Compared to v4:
      - l1_neural_head removed (33k params gone).
      - Explicit alpha3 * exp(-surprisal) term added (1 scalar).
      - M2 and I tied (-1 parameter).
      - AutoModel -> AutoModelForCausalLM (needed for lm_head).
      - Everything else identical.
    """

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        freeze_layers: int = 12,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.lm = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.lm.config.pad_token_id = self.tokenizer.eos_token_id

        llama_dim = self.lm.config.hidden_size
        self.llama_dim = llama_dim

        if freeze_layers > 0:
            base = self.lm.model
            for param in base.embed_tokens.parameters():
                param.requires_grad = False
            for layer_idx in range(min(freeze_layers, len(base.layers))):
                for param in base.layers[layer_idx].parameters():
                    param.requires_grad = False

        # projection is only consumed by skip_residual_head in v4b
        # (L1 path is formula-only). Retaining it keeps the skip head's
        # gradient path identical to v4.
        self.projection = nn.Sequential(
            nn.Linear(llama_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # --- Reichle skeleton scalars (L1 formula) ---
        self.l1_base_offset = nn.Parameter(torch.tensor(60.0))
        self.l1_freq_coef = nn.Parameter(torch.tensor(-17.0))
        # alpha3 analogue: ms per unit cloze probability. Feeds
        # p = exp(-surprisal) in [0, 1] directly, so already in
        # Reichle's units (no rescaling needed).
        self.l1_pred_coef = nn.Parameter(torch.tensor(40.0))

        self._delta_raw = nn.Parameter(torch.tensor(_logit(0.34)))

        # Skip residual head: preserved from v4. The parafoveal race
        # alone cannot match empirical skip rates (v6b experiment
        # showed r_skip collapsing to ~0.05 without the residual); the
        # residual is a structural necessity, not a performance
        # upgrade.
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

    # --- Paper-side aliases in Reichle's original units ---

    @property
    def alpha1(self):
        return self.l1_base_offset

    @property
    def alpha2(self):
        return self.l1_freq_coef

    @property
    def alpha3(self):
        return self.l1_pred_coef

    @property
    def alpha1_reichle(self):
        return self.l1_base_offset - 2.0 * self.l1_freq_coef

    @property
    def alpha2_reichle(self):
        return -self.l1_freq_coef / 5.0

    @property
    def alpha3_reichle(self):
        return self.l1_pred_coef

    # ------------------------------------------------------------------ #
    # Tokenization, pooling, surprisal
    # ------------------------------------------------------------------ #

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

    def _compute_word_surprisal(
        self, logits, input_ids, batch_word_maps, max_words, device
    ):
        B = logits.size(0)
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        token_log_probs = log_probs.gather(
            -1, input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)

        token_log_probs = torch.cat(
            [torch.zeros(B, 1, device=device), token_log_probs], dim=1
        )

        surprisal = torch.zeros(B, max_words, device=device)
        for b in range(B):
            for w_idx, (start, end) in enumerate(batch_word_maps[b]):
                surprisal[b, w_idx] = -token_log_probs[b, start:end].sum()
        return surprisal

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(self, word_lists, frequencies, word_lengths):
        device = word_lengths.device

        input_ids, attention_mask, word_maps, max_words = (
            self._tokenize_and_align(word_lists, device)
        )

        base_out = self.lm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = base_out.last_hidden_state
        logits = self.lm.lm_head(hidden)

        word_repr = self._pool_subwords_to_words(
            hidden, word_maps, max_words, device
        )

        surprisal = self._compute_word_surprisal(
            logits, input_ids, word_maps, max_words, device
        )
        # Detach: LLaMA is a fixed predictability oracle; alpha3 learns.
        surprisal = surprisal.detach()

        seq_len = word_lengths.size(1)
        word_repr = word_repr[:, :seq_len, :]
        surprisal = surprisal[:, :seq_len]

        # projection is still used for the skip residual head even
        # though the L1 path is formula-only.
        projected = self.projection(word_repr)
        residual_skip_logit = self.skip_residual_head(projected).squeeze(-1)

        # --- Pure-formula L1: alpha1 + alpha2 * log_freq_norm - alpha3 * p ---
        log_freq = torch.log(frequencies.clamp(min=1.0))
        log_freq_norm = (log_freq - 10.0) / 5.0
        predictability = torch.exp(-surprisal).clamp(max=1.0)

        base_L1_raw = (
            self.l1_base_offset
            + self.l1_freq_coef * log_freq_norm
            - self.l1_pred_coef * predictability
        )
        base_L1 = 5.0 + F.softplus(base_L1_raw - 5.0)

        L2 = self.delta * base_L1

        result = self.ezreader(
            base_L1=base_L1,
            L2=L2,
            residual_skip_logit=residual_skip_logit,
            word_lengths=word_lengths,
        )

        result['base_L1'] = base_L1
        result['base_L1_formula'] = base_L1_raw
        result['word_surprisal'] = surprisal
        result['word_predictability'] = predictability
        result['log_freq'] = log_freq
        result['delta'] = self.delta.detach()
        result['l1_base_offset'] = self.l1_base_offset.detach()
        result['l1_freq_coef'] = self.l1_freq_coef.detach()
        result['l1_pred_coef'] = self.l1_pred_coef.detach()
        result['alpha1_reichle'] = self.alpha1_reichle.detach()
        result['alpha2_reichle'] = self.alpha2_reichle.detach()
        result['alpha3_reichle'] = self.alpha3_reichle.detach()

        return result
