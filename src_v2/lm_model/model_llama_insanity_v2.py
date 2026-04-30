"""
Neural EZ Reader Insanity v2 — structured L1 head + larger residual init.

Diagnosis of insanity v1 (see logs/train_insanity_geco.log epoch 1):

  base_L1 = 34 +/- 8, much narrower than the ~30-40 ms std needed to
  fit long-tail words. Motor parameters (M1, M2, I) stayed at their
  Reichle priors under NLL — a theoretical win — but the L1 head
  itself did not develop per-word variance, and the sigma heads
  drifted upward (0.30 -> 0.40-0.50) absorbing prediction error that
  L1 should have been reducing.

  Root cause: v1's L1 head is `exp(l1_head(state) + log(50))` where
  `l1_head` is initialized near zero (std=0.01). At init every word
  gets L1 ~= 50 ms. Under NLL with heteroscedastic sigma, the
  optimizer finds that widening sigma is an easier path than teaching
  the L1 head to discriminate between words, so L1 stays flat.

v2 changes (structural, not regularization):

  (1) STRUCTURED L1 HEAD.
      L1 is no longer `exp(neural_head(state))`. Instead it has an
      explicit Reichle-style lexical anchor driven by the features
      we already compute:

          L1_lex = l1_bias
                   + l1_freq_coef * log_freq_norm
                   + l1_len_coef  * log_len_norm
                   + l1_surp_coef * surprisal_norm

          base_L1 = 5 + softplus(L1_lex + l1_residual(state) - 5)

      The coefficients are learnable scalars, initialized to values
      that give real per-word variance from step 1:

          l1_bias       =  60 ms
          l1_freq_coef  = -15 ms / unit log_freq_norm
          l1_len_coef   =   3 ms / unit log_len_norm
          l1_surp_coef  =   4 ms / unit surprisal_norm

      These were chosen so that common short predictable words
      (high freq, low len, low surp) get ~30 ms and rare long
      unpredictable words get ~80-100 ms pre-eccentricity. The
      eccentricity multiplier then stretches the range further.

      This mirrors what v4 did with l1_base_offset + l1_freq_coef but
      adds length and surprisal as explicit features — surprisal in
      particular, because v4 had to rediscover predictability through
      the neural head (and the neural head fought the frequency term).
      With surprisal as a direct input, the neural residual has much
      less reason to fight the lex anchor.

  (2) NEURAL RESIDUAL WITH LARGER INIT.
      `l1_residual_head` now initializes with weight std = 0.05
      (vs v1's 0.01 on the whole l1_head). It still reads from the
      GRU state and contributes context-dependent corrections in ms
      space. Bias is zero. At init the residual has ~+/- 5 ms spread,
      which adds to the lex anchor's variance without dominating it.

  (3) L1 WORKS IN LINEAR MS SPACE, NOT LOG SPACE.
      v1 computed `log_L1 -> L1 = exp(log_L1)`, so the L1 head output
      was an exponent. v2 computes L1 directly in ms via
      `softplus(lex + residual - 5) + 5`. This is more interpretable
      and the coefficients have direct ms units.

Everything else is identical to v1: GRU reader state, sigma heads,
fixation head, Reichle cascade (means). The training script handles
the sigma regularization side of the fix.

forward() signature is unchanged:
    model(word_lists, frequencies, word_lengths)
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
#  Reichle motor cascade (means only) — unchanged from v1
# --------------------------------------------------------------------------- #

class ReichleCascadeMeans(nn.Module):
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

        self._delta_raw = nn.Parameter(torch.tensor(_logit(0.34)))

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
    def delta(self):
        return torch.sigmoid(self._delta_raw)

    def forward(self, base_L1, word_lengths):
        ecc_exponent = (word_lengths - 1.0) / 2.0
        L1_ecc = base_L1 * torch.pow(self.epsilon, ecc_exponent)

        L2 = self.delta * base_L1

        ffd_mean = L1_ecc + self.M1 + self.M2

        refix_prob = torch.sigmoid(
            self.lambda_refix * (word_lengths - self.refix_pivot)
        )
        refix_duration = L2 + self.M1 + self.M2
        gaze_mean = ffd_mean + refix_prob * refix_duration

        prev_gaze = torch.cat(
            [torch.zeros_like(gaze_mean[:, :1]), gaze_mean[:, :-1]],
            dim=1,
        )
        regression_cost = self.pF * self.reg_weight * prev_gaze
        trt_mean = gaze_mean + self.I + regression_cost

        return {
            'ffd_mean': ffd_mean,
            'gaze_mean': gaze_mean,
            'trt_mean': trt_mean,
            'L1_ecc': L1_ecc,
            'L2': L2,
            'refix_prob': refix_prob,
        }


# --------------------------------------------------------------------------- #
#  Neural EZ Reader Insanity v2
# --------------------------------------------------------------------------- #

class NeuralEZReaderInsanity(nn.Module):
    """
    v2 structural changes vs v1:
      - L1 head has explicit lexical coefficients (freq, len, surprisal)
        initialized to Reichle-like effect sizes. The neural residual on
        the GRU state contributes ms-space corrections with std=0.05 init.
      - L1 is computed in linear ms space with a soft floor, not as
        exp(log_L1).

    Everything else (LLaMA encoder + lm_head surprisal, feature fusion,
    GRU reader state, sigma heads, fixation head, Reichle cascade) is
    identical to v1.
    """

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        freeze_layers: int = 12,
        hidden_dim: int = 256,
        gru_hidden: int = 256,
        sigma_init: float = 0.30,
        sigma_floor: float = 0.05,
        fixation_prior: float = 0.65,
        # L1 head lex coefficients (in ms, per unit normalized feature).
        l1_bias_init: float = 60.0,
        l1_freq_coef_init: float = -15.0,
        l1_len_coef_init: float = 3.0,
        l1_surp_coef_init: float = 4.0,
        l1_residual_std: float = 0.05,
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
        self.hidden_dim = hidden_dim
        self.gru_hidden = gru_hidden
        self.sigma_floor = sigma_floor

        if freeze_layers > 0:
            base = self.lm.model
            for param in base.embed_tokens.parameters():
                param.requires_grad = False
            for layer_idx in range(min(freeze_layers, len(base.layers))):
                for param in base.layers[layer_idx].parameters():
                    param.requires_grad = False

        # ---- Feature pipeline ----
        self.llama_projection = nn.Sequential(
            nn.Linear(llama_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.reader_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )

        state_dim = hidden_dim + gru_hidden

        # ---- L1 head: structured lex anchor + small neural residual ----
        # Lex coefficients are scalar learnable parameters in ms space.
        self.l1_bias = nn.Parameter(torch.tensor(l1_bias_init))
        self.l1_freq_coef = nn.Parameter(torch.tensor(l1_freq_coef_init))
        self.l1_len_coef = nn.Parameter(torch.tensor(l1_len_coef_init))
        self.l1_surp_coef = nn.Parameter(torch.tensor(l1_surp_coef_init))

        # Neural residual: context-dependent ms-space correction on top
        # of the lex anchor. Larger init std than v1 so it contributes
        # from epoch 1, but bias is zero so it starts centered.
        self.l1_residual_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.normal_(self.l1_residual_head[-1].weight, std=l1_residual_std)
        nn.init.zeros_(self.l1_residual_head[-1].bias)

        # ---- Variance heads (same as v1) ----
        sigma_bias_init = _inv_softplus(max(sigma_init - sigma_floor, 1e-3))

        def make_sigma_head():
            head = nn.Sequential(
                nn.Linear(state_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1),
            )
            nn.init.normal_(head[-1].weight, std=0.01)
            nn.init.constant_(head[-1].bias, sigma_bias_init)
            return head

        self.sigma_ffd_head = make_sigma_head()
        self.sigma_gaze_head = make_sigma_head()
        self.sigma_trt_head = make_sigma_head()

        # ---- Fixation head (same as v1) ----
        self.fixation_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.normal_(self.fixation_head[-1].weight, std=0.01)
        nn.init.constant_(self.fixation_head[-1].bias, _logit(fixation_prior))

        # ---- Reichle cascade ----
        self.cascade = ReichleCascadeMeans()

    # ------------------------------------------------------------------ #
    # Tokenization / pooling / surprisal — unchanged from v1
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
        surprisal = surprisal.detach()

        seq_len = word_lengths.size(1)
        word_repr = word_repr[:, :seq_len, :]
        surprisal = surprisal[:, :seq_len]

        # ---- Feature fusion ----
        projected = self.llama_projection(word_repr)

        log_freq = torch.log(frequencies.clamp(min=1.0))
        log_freq_norm = (log_freq - 10.0) / 5.0
        log_len_norm = torch.log(word_lengths.clamp(min=1.0)) - 1.5
        surprisal_norm = (surprisal - 5.0) / 5.0

        scalar_feats = torch.stack(
            [log_freq_norm, log_len_norm, surprisal_norm], dim=-1
        )

        fused = self.feature_fusion(
            torch.cat([projected, scalar_feats], dim=-1)
        )

        # ---- Sequential reader state ----
        reader_state, _ = self.reader_gru(fused)
        state = torch.cat([fused, reader_state], dim=-1)

        # ---- L1 head (v2: structured lex + residual, linear ms space) ----
        l1_lex = (
            self.l1_bias
            + self.l1_freq_coef * log_freq_norm
            + self.l1_len_coef * log_len_norm
            + self.l1_surp_coef * surprisal_norm
        )
        l1_residual = self.l1_residual_head(state).squeeze(-1)

        l1_raw = l1_lex + l1_residual
        # Soft floor at 5 ms; no hard ceiling.
        L1 = 5.0 + F.softplus(l1_raw - 5.0)

        # ---- Variance heads ----
        raw_sigma_ffd = self.sigma_ffd_head(state).squeeze(-1)
        raw_sigma_gaze = self.sigma_gaze_head(state).squeeze(-1)
        raw_sigma_trt = self.sigma_trt_head(state).squeeze(-1)

        sigma_ffd = self.sigma_floor + F.softplus(raw_sigma_ffd)
        sigma_gaze = self.sigma_floor + F.softplus(raw_sigma_gaze)
        sigma_trt = self.sigma_floor + F.softplus(raw_sigma_trt)

        # ---- Fixation head ----
        fixation_logit = self.fixation_head(state).squeeze(-1)
        fixation_prob = torch.sigmoid(fixation_logit)
        skip_prob = 1.0 - fixation_prob

        # ---- Reichle cascade ----
        cascade_out = self.cascade(base_L1=L1, word_lengths=word_lengths)
        ffd_mean = cascade_out['ffd_mean']
        gaze_mean = cascade_out['gaze_mean']
        trt_mean = cascade_out['trt_mean']

        ffd_mu_log = torch.log(ffd_mean.clamp(min=1.0))
        gaze_mu_log = torch.log(gaze_mean.clamp(min=1.0))
        trt_mu_log = torch.log(trt_mean.clamp(min=1.0))

        return {
            # Distribution parameters.
            'ffd_mu_log': ffd_mu_log,
            'ffd_sigma': sigma_ffd,
            'gaze_mu_log': gaze_mu_log,
            'gaze_sigma': sigma_gaze,
            'trt_mu_log': trt_mu_log,
            'trt_sigma': sigma_trt,

            # Cascade medians.
            'ffd_mean': ffd_mean,
            'gaze_mean': gaze_mean,
            'trt_mean': trt_mean,

            # Fixation / skip.
            'fixation_logit': fixation_logit,
            'fixation_prob': fixation_prob,
            'skip_prob': skip_prob,

            # Cognitive intermediates.
            'base_L1': L1,
            'l1_lex': l1_lex,
            'l1_residual': l1_residual,
            'L1_ecc': cascade_out['L1_ecc'],
            'L2': cascade_out['L2'],
            'refix_prob': cascade_out['refix_prob'],

            # Input features.
            'surprisal': surprisal,
            'log_freq': log_freq,

            # Detached cascade parameters.
            'epsilon': self.cascade.epsilon.detach(),
            'M1': self.cascade.M1.detach(),
            'M2': self.cascade.M2.detach(),
            'I': self.cascade.I.detach(),
            'pF': self.cascade.pF.detach(),
            'reg_weight': self.cascade.reg_weight.detach(),
            'delta': self.cascade.delta.detach(),
            'lambda_refix': self.cascade.lambda_refix.detach(),
            'refix_pivot': self.cascade.refix_pivot.detach(),

            # Detached L1 head lex coefficients (for logging).
            'l1_bias_param': self.l1_bias.detach(),
            'l1_freq_coef_param': self.l1_freq_coef.detach(),
            'l1_len_coef_param': self.l1_len_coef.detach(),
            'l1_surp_coef_param': self.l1_surp_coef.detach(),
        }


# --------------------------------------------------------------------------- #
#  Loss helper: Gaussian NLL on log(observation) — same as v1
# --------------------------------------------------------------------------- #

def log_normal_nll(mu_log, sigma, observed_ms, mask=None):
    observed_ms = observed_ms.clamp(min=1.0)
    log_obs = torch.log(observed_ms)

    residual = (log_obs - mu_log) / sigma
    nll = (
        0.5 * residual * residual
        + torch.log(sigma)
        + log_obs
        + 0.5 * math.log(2.0 * math.pi)
    )

    if mask is not None:
        nll = nll[mask]

    return nll.mean()
