"""
Neural EZ Reader Hybrid v4c_full — v4c_saccade_noise + landing-driven
refixation + explicit integration / regression mechanics.

This file implements the four largest deviations between v4c and the
Reichle 2003 simulation engine, in differentiable closed form:

  (1) Mψ (saccade execution noise) — same as v4c_saccade_noise.
      Adds omega1, omega2, eta1, eta2 (Reichle 2003 defaults).
      Mψ enters L1 via the closed-form expectation of eps^d under the
      Gaussian landing distribution.

  (2) Refixation from landing error — REPLACES v4c's length-driven
      refix_prob. Reichle: P_refix = lambda * |landing - target_center|.
      We use the analytic mean of the absolute Gaussian (folded normal):
          E[|X|] = sigma * sqrt(2/π) * exp(-delta^2/(2 sigma^2))
                 + delta * erf(delta / (sigma * sqrt(2)))
      where delta = sys_err and the random error has variance sigma^2.
      P_refix = clamp(lambda * E[|landing - center|], 0, 0.95).

  (3) Integration stage — splits v4c's tied I (= M2) into three
      separate Reichle-named parameters:
          M2  = saccade finishing  (Reichle: 25 ms)
          I   = integration_time   (Reichle: 25 ms)
          A   = attention_shift    (Reichle: 25 ms)
      v4c tied M2 = I; v4c_full unties them. attention_shift only
      appears during regression-to-different-word (Reichle's
      _attend_again triggers it only when the attended word changes).

  (4) Stochastic integration failure + regression direction —
      replaces v4c's `pF * reg_weight * prev_gaze` linear term with
      Reichle's branching mechanic:
          p_fail = sigmoid(ifail_offset + ifail_coef * log_freq_norm)
          (frequency-based proxy for context-dependent surprisal;
           higher freq -> lower fail probability)
          p_correct_regression = sigmoid(_pcr_raw)  (Reichle: 0.6)

      On integration failure, the eye regresses either to the current
      word (prob p_correct, attention stays) or the previous word
      (prob 1 - p_correct, attention shifts). The reread cost uses
      Reichle's predictability_repeated_attention shortcut: with prob
      0.9 the second-pass L1 is zero, so E[L1'] = 0.1 * L1.

          cost_curr = M1 + M2 + 0.1*L1 + L2 + I
          cost_prev = M1 + M2 + 0.1*L1[n-1] + L2[n-1] + I + A

          regression_cost = p_correct * cost_curr +
                            (1 - p_correct) * cost_prev

          TRT = Gaze + I + p_fail * regression_cost

  Depth-1 truncation: we assume the reread succeeds. Reichle's engine
  can have nested regressions; we ignore that to keep the closed form
  tractable.

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


OPTIMAL_SACCADE_LENGTH = 7.0
PREDICTABILITY_REPEATED_ATTENTION = 0.9  # Reichle 2003 fixed constant
SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
SQRT_2 = math.sqrt(2.0)


# --------------------------------------------------------------------------- #
#  Reichle EZ Reader cascade — full Mψ + refix + integration + regression
# --------------------------------------------------------------------------- #

class ReichleEZReaderFull(nn.Module):
    """
    v4c_saccade_noise cascade extended with:
      - landing-error-driven refixation (replaces length-driven refix)
      - separate integration_time, attention_shift, M2 parameters
      - stochastic integration failure with surprisal-proxy gating
      - regression direction mixture
    """

    L1_SOFT_FLOOR = 5.0
    SIGMA_FLOOR = 1e-3  # avoid div-by-zero in folded-normal formula

    def __init__(self):
        super().__init__()

        self._epsilon_raw = nn.Parameter(torch.tensor(_inv_softplus(0.15)))

        self._M1_raw = nn.Parameter(torch.tensor(_inv_softplus(125.0)))
        # Untied: M2, I, A are now three separate scalars.
        self._M2_raw = nn.Parameter(torch.tensor(_inv_softplus(25.0)))
        self._I_raw = nn.Parameter(torch.tensor(_inv_softplus(25.0)))
        self._A_raw = nn.Parameter(torch.tensor(_inv_softplus(25.0)))

        # Mψ (saccade execution noise) — Reichle 2003 defaults.
        self._omega1_raw = nn.Parameter(torch.tensor(_inv_softplus(6.0)))
        self._omega2_raw = nn.Parameter(torch.tensor(_inv_softplus(3.0)))
        self._eta1_raw = nn.Parameter(torch.tensor(_inv_softplus(0.5)))
        self._eta2_raw = nn.Parameter(torch.tensor(_inv_softplus(0.15)))

        # Refixation (landing-error driven). Reichle: lambda = 0.16.
        self._lambda_refix_raw = nn.Parameter(torch.tensor(_inv_softplus(0.16)))

        # Integration failure: surprisal-proxy gating.
        # Init: ifail_offset such that baseline p_fail ~ 0.05;
        # ifail_coef negative so high freq -> low p_fail.
        self.ifail_offset = nn.Parameter(torch.tensor(_logit(0.05)))
        self.ifail_coef = nn.Parameter(torch.tensor(-0.5))

        # Regression direction: Reichle p_correct = 0.6.
        self._p_correct_raw = nn.Parameter(torch.tensor(_logit(0.6)))

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
    def A(self):
        return F.softplus(self._A_raw)

    @property
    def omega1(self):
        return F.softplus(self._omega1_raw)

    @property
    def omega2(self):
        return F.softplus(self._omega2_raw) + 1e-3

    @property
    def eta1(self):
        return F.softplus(self._eta1_raw)

    @property
    def eta2(self):
        return F.softplus(self._eta2_raw)

    @property
    def lambda_refix(self):
        return F.softplus(self._lambda_refix_raw)

    @property
    def p_correct(self):
        return torch.sigmoid(self._p_correct_raw)

    @property
    def skip_temperature(self):
        return 1.0 + F.softplus(self._skip_temperature_raw)

    def _intended_saccade_lengths(self, word_lengths):
        prev_w = torch.cat(
            [torch.zeros_like(word_lengths[:, :1]), word_lengths[:, :-1]],
            dim=1,
        )
        intended = prev_w / 2.0 + 1.0 + word_lengths / 2.0
        intended = intended.clone()
        intended[:, 0] = 0.0
        return intended

    def _msi_correction(self, intended_len, launch_dur):
        log_launch = torch.log(launch_dur.clamp(min=1.0))
        sys_err = (
            (OPTIMAL_SACCADE_LENGTH - intended_len)
            * (self.omega1 - log_launch)
            / self.omega2
        )
        sigma = self.eta1 + self.eta2 * intended_len

        first_word_mask = (intended_len > 1e-6).to(sys_err.dtype)
        sys_err = sys_err * first_word_mask
        sigma = sigma * first_word_mask
        return sys_err, sigma

    @staticmethod
    def _folded_normal_mean(delta, sigma):
        """
        E[|X|] for X ~ N(delta, sigma^2):
            = sigma * sqrt(2/pi) * exp(-delta^2 / (2 sigma^2))
              + delta * erf(delta / (sigma * sqrt(2)))
        Robust at sigma -> 0: tends to |delta|.
        """
        sigma_safe = sigma.clamp(min=ReichleEZReaderFull.SIGMA_FLOOR)
        term1 = sigma_safe * SQRT_2_OVER_PI * torch.exp(
            -(delta ** 2) / (2.0 * sigma_safe ** 2)
        )
        term2 = delta * torch.erf(delta / (sigma_safe * SQRT_2))
        return term1 + term2

    def forward(
        self,
        base_L1,
        L2,
        residual_skip_logit,
        word_lengths,
        log_freq_norm,
    ):
        # --- Step 1: pre-Mψ FFD estimate (used as launch_dur proxy) ---
        ecc_exponent = (word_lengths - 1.0) / 2.0
        L1_ecc_pre = base_L1 * torch.pow(self.epsilon, ecc_exponent)
        L1_pre = self.L1_SOFT_FLOOR + F.softplus(L1_ecc_pre - self.L1_SOFT_FLOOR)
        ffd_pre = L1_pre + self.M1 + self.M2

        launch_dur = torch.cat(
            [torch.full_like(ffd_pre[:, :1], 250.0), ffd_pre[:, :-1]],
            dim=1,
        )

        # --- Step 2: Mψ corrections per word ---
        intended_len = self._intended_saccade_lengths(word_lengths)
        sys_err, sigma = self._msi_correction(intended_len, launch_dur)

        # --- Step 3: closed-form expected L1 with Mψ ---
        ln_eps = torch.log(self.epsilon)
        ecc_exp_full = ecc_exponent + sys_err
        variance_factor = torch.exp(0.5 * (sigma ** 2) * (ln_eps ** 2))
        L1_ecc = base_L1 * torch.pow(self.epsilon, ecc_exp_full) * variance_factor

        L1 = self.L1_SOFT_FLOOR + F.softplus(L1_ecc - self.L1_SOFT_FLOOR)

        first_fixation = L1 + self.M1 + self.M2

        # --- Step 4: refixation from landing error (replaces v4c's
        #     length-driven refix). Reichle: P_refix = lambda * |landing -
        #     target_center|. We use the folded-normal mean.
        e_abs_landing = self._folded_normal_mean(sys_err, sigma)
        refix_prob = (self.lambda_refix * e_abs_landing).clamp(max=0.95)

        refix_duration = L2 + self.M1 + self.M2
        gaze_duration = first_fixation + refix_prob * refix_duration

        # --- Step 5: integration + regression mechanic ---
        # p_fail per word: frequency-based surprisal proxy.
        p_fail = torch.sigmoid(self.ifail_offset + self.ifail_coef * log_freq_norm)

        # Reread cost — depth-1 truncation, predictability_repeated_attention
        # shortcut baked in as 0.1 multiplier on L1' (Reichle: 0.9 chance L1' = 0).
        repeated_L1_factor = 1.0 - PREDICTABILITY_REPEATED_ATTENTION  # = 0.1
        cost_curr = (
            self.M1 + self.M2
            + repeated_L1_factor * L1
            + L2
            + self.I
        )

        # Previous-word reread requires attention shift to a different word.
        L1_prev = torch.cat(
            [torch.zeros_like(L1[:, :1]), L1[:, :-1]],
            dim=1,
        )
        L2_prev = torch.cat(
            [torch.zeros_like(L2[:, :1]), L2[:, :-1]],
            dim=1,
        )
        cost_prev = (
            self.M1 + self.M2
            + repeated_L1_factor * L1_prev
            + L2_prev
            + self.I
            + self.A
        )

        regression_cost = (
            self.p_correct * cost_curr + (1.0 - self.p_correct) * cost_prev
        )

        conditional_trt = gaze_duration + self.I + p_fail * regression_cost

        # --- Step 6: skip race ---
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
            'sys_err': sys_err,
            'sigma_landing': sigma,
            'intended_sac_len': intended_len,
            'launch_dur': launch_dur,
            'e_abs_landing': e_abs_landing,
            'p_fail': p_fail,
            'regression_cost': regression_cost,
            'epsilon': self.epsilon.detach(),
            'M1': self.M1.detach(),
            'M2': self.M2.detach(),
            'I': self.I.detach(),
            'A': self.A.detach(),
            'omega1': self.omega1.detach(),
            'omega2': self.omega2.detach(),
            'eta1': self.eta1.detach(),
            'eta2': self.eta2.detach(),
            'lambda_refix': self.lambda_refix.detach(),
            'p_correct': self.p_correct.detach(),
            'ifail_offset': self.ifail_offset.detach(),
            'ifail_coef': self.ifail_coef.detach(),
            'skip_temperature': self.skip_temperature.detach(),
        }


# --------------------------------------------------------------------------- #
#  Neural EZ Reader Hybrid v4c_full
# --------------------------------------------------------------------------- #

class NeuralEZReaderHybrid(nn.Module):
    """
    Same neural backbone as v4c. Only the cascade module changes
    (ReichleEZReader → ReichleEZReaderFull). Forward now also passes
    log_freq_norm into the cascade so the integration-failure head
    can use it as a surprisal proxy.

    Learnable:
        Neural:    LLaMA top layers, projection, ctx_head, skip_residual_head
        Cognitive: l1_base_offset (alpha1), l1_freq_coef (alpha2), delta,
                   epsilon, M1, M2, I, A,
                   omega1, omega2, eta1, eta2,
                   lambda_refix,
                   ifail_offset, ifail_coef, p_correct,
                   skip_temperature
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

        self.l1_base_offset = nn.Parameter(torch.tensor(60.0))
        self.l1_freq_coef = nn.Parameter(torch.tensor(-17.0))

        self.ctx_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.normal_(self.ctx_head[-1].weight, std=0.01)
        nn.init.zeros_(self.ctx_head[-1].bias)

        self._delta_raw = nn.Parameter(torch.tensor(_logit(0.34)))

        self.skip_residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.normal_(self.skip_residual_head[-1].weight, std=0.01)
        nn.init.zeros_(self.skip_residual_head[-1].bias)

        self.ezreader = ReichleEZReaderFull()

    @property
    def delta(self):
        return torch.sigmoid(self._delta_raw)

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
            log_freq_norm=log_freq_norm,
        )

        result['base_L1'] = base_L1
        result['base_L1_formula'] = base_L1_formula
        result['ctx'] = ctx
        result['log_freq'] = log_freq
        result['log_freq_norm'] = log_freq_norm
        result['delta'] = self.delta.detach()
        result['l1_base_offset'] = self.l1_base_offset.detach()
        result['l1_freq_coef'] = self.l1_freq_coef.detach()
        result['alpha1_reichle'] = self.alpha1_reichle.detach()
        result['alpha2_reichle'] = self.alpha2_reichle.detach()

        return result
