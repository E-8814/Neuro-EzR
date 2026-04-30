"""
Neural EZ Reader Hybrid v6b — v6 with three small additions driven by
paper-hygiene review.

Changes from model_llama_hybrid_v6.py:

  (1) alpha3_reichle property (paper-table symmetry).
      v6 had alpha1_reichle / alpha2_reichle but no alpha3_reichle,
      even though alpha3 is already in Reichle's units. v6b adds the
      alias so every Reichle coefficient has a *_reichle accessor and
      the paper table has a consistent naming convention.

  (2) no_ctx ablation flag.
      Setting no_ctx=True in the constructor zeros the ctx_head output
      at forward time (the module still exists so checkpoints match,
      but its output is replaced by zeros tensor). This is the
      Claim A / Claim B discriminator: train v6b twice, once with
      and once without ctx, and compare r_TRT.

      (Implemented as a flag rather than a separate file so both runs
      share the same code path and any future model changes propagate
      automatically to both conditions.)

  (3) base_L1_formula is already exported in v6's result dict — no
      model change needed. The training script collects it for the
      ctx / formula magnitude ratio.

No other architectural changes vs v6. Pure parafoveal race skip,
explicit alpha3 * exp(-surprisal) predictability term, tied M2 = I,
ctx_head with near-zero init, everything else unchanged.

forward() signature unchanged: model(word_lists, frequencies, word_lengths).
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
#  Reichle EZ Reader cascade with pure parafoveal-race skip
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

    def forward(self, base_L1, L2, word_lengths):
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
        skip_prob = torch.sigmoid(race_logit)

        total_reading_time = conditional_trt

        return {
            'first_fixation': first_fixation,
            'gaze_duration': gaze_duration,
            'conditional_trt': conditional_trt,
            'total_reading_time': total_reading_time,
            'skip_prob': skip_prob,
            'race_logit': race_logit,
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
#  Neural EZ Reader Hybrid v6b
# --------------------------------------------------------------------------- #

class NeuralEZReaderHybrid(nn.Module):
    """
    Same structure as v6. `no_ctx=True` zeros ctx_head output for the
    Claim A / Claim B ablation. Adds alpha3_reichle property.
    """

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        freeze_layers: int = 12,
        hidden_dim: int = 256,
        no_ctx: bool = False,
    ):
        super().__init__()

        self.no_ctx = no_ctx

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

        self.projection = nn.Sequential(
            nn.Linear(llama_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.l1_base_offset = nn.Parameter(torch.tensor(60.0))
        self.l1_freq_coef = nn.Parameter(torch.tensor(-17.0))
        self.l1_pred_coef = nn.Parameter(torch.tensor(40.0))

        self.ctx_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.normal_(self.ctx_head[-1].weight, std=0.01)
        nn.init.zeros_(self.ctx_head[-1].bias)

        # If ablating ctx, freeze its weights — they won't be used but
        # shouldn't accumulate gradient noise through dropout either.
        if self.no_ctx:
            for param in self.ctx_head.parameters():
                param.requires_grad = False

        self._delta_raw = nn.Parameter(torch.tensor(_logit(0.34)))

        self.ezreader = ReichleEZReader()

    @property
    def delta(self):
        return torch.sigmoid(self._delta_raw)

    # --- Paper-side aliases ---

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
        """alpha1 in Reichle's original units (ms baseline at log f = 0)."""
        return self.l1_base_offset - 2.0 * self.l1_freq_coef

    @property
    def alpha2_reichle(self):
        """alpha2 in Reichle's original units (ms per unit natural log-frequency)."""
        return -self.l1_freq_coef / 5.0

    @property
    def alpha3_reichle(self):
        """
        alpha3 in Reichle's units (ms per unit cloze probability).
        Our parameterization already feeds p = exp(-surprisal) in [0, 1]
        directly into -alpha3 * p, so no transform is needed. Exposed
        as an alias for symmetry with alpha1_reichle / alpha2_reichle.
        """
        return self.l1_pred_coef

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
        surprisal = surprisal.detach()

        seq_len = word_lengths.size(1)
        word_repr = word_repr[:, :seq_len, :]
        surprisal = surprisal[:, :seq_len]

        projected = self.projection(word_repr)

        log_freq = torch.log(frequencies.clamp(min=1.0))
        log_freq_norm = (log_freq - 10.0) / 5.0

        predictability = torch.exp(-surprisal).clamp(max=1.0)

        base_L1_formula = (
            self.l1_base_offset
            + self.l1_freq_coef * log_freq_norm
            - self.l1_pred_coef * predictability
        )

        if self.no_ctx:
            ctx = torch.zeros_like(base_L1_formula)
        else:
            ctx = self.ctx_head(projected).squeeze(-1)

        l1_raw = base_L1_formula + ctx
        base_L1 = 5.0 + F.softplus(l1_raw - 5.0)

        L2 = self.delta * base_L1

        result = self.ezreader(
            base_L1=base_L1,
            L2=L2,
            word_lengths=word_lengths,
        )

        result['base_L1'] = base_L1
        result['base_L1_formula'] = base_L1_formula
        result['ctx'] = ctx
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
