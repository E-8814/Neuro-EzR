"""
Neural EZ Reader Hybrid v4c — v4 with three paper-hygiene changes.

v4c is v4 with cosmetic cleanup only. No architectural changes, no
training recipe changes, no change to what's being modeled. The
purpose is to make the paper table and reviewer story cleaner.

Changes from v4:

  (1) M2 AND I TIED TO ONE PARAMETER.
      v4 had `_M2_raw` and `_I_raw` as two separate learnable scalars.
      They enter the cascade only as additive constants in FFD and
      TRT respectively, and the v4 training log shows they drifted to
      the same value (M2 = I = 24.1 ms at convergence). The data
      cannot identify them separately — only their sum enters the
      prediction. v4c ties them through a single `_M2I_raw` parameter,
      which is honest about the identifiability constraint. The paper
      reports a single "M2 = I" value rather than two values that
      happen to agree.

  (2) REICHLE-UNIT PARAMETER ALIASES.
      v4 parameterizes log-frequency through the normalized feature
      `(log f - 10) / 5`, so `l1_freq_coef = -17` is Reichle's
      alpha2 = 3.4 after the conversion alpha2_R = -coef / 5. The
      paper table should report Reichle-unit values directly so
      readers can compare to the 2003 literature without mental
      arithmetic. v4c exposes two properties:

          alpha1_reichle = l1_base_offset - 2 * l1_freq_coef
          alpha2_reichle = -l1_freq_coef / 5

      These map our normalized parameterization to Reichle's
      unnormalized formula (L1 = alpha1_R - alpha2_R * log f).

  (3) RENAMED l1_neural_head -> ctx_head.
      v4's MLP-on-LLaMA-hidden-state was misleadingly named — it
      suggested the head is doing something opaque to L1. A clearer
      name for the paper: `ctx_head` makes the role explicit. It is
      a learned correction that captures context-dependent effects
      LLaMA's hidden state can express beyond what frequency alone
      captures in the formula skeleton.

No architectural, hyperparameter, or training-recipe change. Trained
checkpoints from v4 and v4c will NOT be interchangeable because
(a) the state dict now has `_M2I_raw` instead of `_M2_raw` / `_I_raw`
and (b) the head's parameter names are prefixed `ctx_head.*` instead
of `l1_neural_head.*`. v4c expects to be trained fresh; performance
should match v4's within seed-level noise.

forward() signature unchanged: model(word_lists, frequencies, word_lengths).
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
#  Reichle EZ Reader cascade (M2 = I tied, race + residual skip)
# --------------------------------------------------------------------------- #

class ReichleEZReader(nn.Module):
    """
    Maps (base_L1, L2, residual_skip_logit, word_lengths) to FFD / Gaze /
    TRT / skip via the Reichle cascade. Identical to v4's cascade except
    that M2 and I are now a single tied parameter.
    """

    L1_SOFT_FLOOR = 5.0

    def __init__(self):
        super().__init__()

        self._epsilon_raw = nn.Parameter(torch.tensor(_inv_softplus(0.15)))

        self._M1_raw = nn.Parameter(torch.tensor(_inv_softplus(125.0)))
        # M2 = I tied: a single parameter, used in both FFD and TRT.
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
#  Neural EZ Reader Hybrid v4c
# --------------------------------------------------------------------------- #

class NeuralEZReaderHybrid(nn.Module):
    """
    (word tokens, frequencies) -> LLaMA encoder -> last-subword pooling
        -> projection
        -> base_L1 = l1_base_offset + l1_freq_coef * log_freq_norm
                                    + ctx_head(projected)
        -> soft floor
        -> delta     -> L2
        -> cascade   -> FFD, Gaze, TRT
        -> race + skip_residual_head -> skip

    Learnable:
        Neural:    LLaMA top layers, projection, ctx_head, skip_residual_head
        Cognitive: l1_base_offset (alpha1), l1_freq_coef (alpha2),
                   delta, epsilon, M1, M2 = I (tied), pF, reg_weight,
                   lambda_refix, refix_pivot, skip_temperature

    Reichle-unit accessors (alpha1_reichle, alpha2_reichle) convert
    from the normalized log-frequency parameterization to Reichle's
    2003 conventions.
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

        # --- Reichle skeleton (L1 formula) ---
        self.l1_base_offset = nn.Parameter(torch.tensor(60.0))
        self.l1_freq_coef = nn.Parameter(torch.tensor(-17.0))

        # --- Context-effects residual head (renamed from l1_neural_head) ---
        # Same architecture and init as v4's l1_neural_head. Reads only
        # the projected LLaMA state so its role is "effects beyond the
        # explicit formula skeleton."
        self.ctx_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.normal_(self.ctx_head[-1].weight, std=0.01)
        nn.init.zeros_(self.ctx_head[-1].bias)

        self._delta_raw = nn.Parameter(torch.tensor(_logit(0.34)))

        # Skip residual head: preserved from v4. v6b ablation showed
        # the pure parafoveal race cannot match empirical skip rates
        # (r_skip collapses to ~0.05 without the residual).
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

    # --- Paper-side aliases in Reichle 2003 units ---
    #
    # Our parameterization uses a centered, scaled log-frequency feature:
    #     log_freq_norm = (log f - 10) / 5
    #     L1 += l1_base_offset + l1_freq_coef * log_freq_norm
    # Reichle 2003 uses:
    #     L1 = alpha1_R - alpha2_R * log f
    # Matching linear and constant parts:
    #     alpha1_R = l1_base_offset - 2 * l1_freq_coef
    #     alpha2_R = -l1_freq_coef / 5
    # These are properties (not Parameters), so they do not appear in
    # state_dict but can be read in logs, reports, and checkpoints.

    @property
    def alpha1(self):
        return self.l1_base_offset

    @property
    def alpha2(self):
        return self.l1_freq_coef

    @property
    def alpha1_reichle(self):
        return self.l1_base_offset - 2.0 * self.l1_freq_coef

    @property
    def alpha2_reichle(self):
        return -self.l1_freq_coef / 5.0

    # ------------------------------------------------------------------ #
    # Tokenization + pooling (unchanged from v4)
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

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(self, word_lists, frequencies, word_lengths):
        device = word_lengths.device

        input_ids, attention_mask, word_maps, max_words = (
            self._tokenize_and_align(word_lists, device)
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

        log_freq = torch.log(frequencies.clamp(min=1.0))
        log_freq_norm = (log_freq - 10.0) / 5.0

        base_L1_formula = (
            self.l1_base_offset + self.l1_freq_coef * log_freq_norm
        )
        ctx = self.ctx_head(projected).squeeze(-1)
        l1_raw = base_L1_formula + ctx

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
        result['ctx'] = ctx
        result['log_freq'] = log_freq
        result['delta'] = self.delta.detach()
        result['l1_base_offset'] = self.l1_base_offset.detach()
        result['l1_freq_coef'] = self.l1_freq_coef.detach()
        result['alpha1_reichle'] = self.alpha1_reichle.detach()
        result['alpha2_reichle'] = self.alpha2_reichle.detach()

        return result
