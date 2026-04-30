"""
Neural EZ Reader Hybrid v4c_v2_saccade_noise — v4c_v2 with Reichle's Mψ stage.

Combines v4c_v2's training and cascade hygiene with v4c_saccade_noise's
Mψ (saccade execution noise) cognitive stage.

From v4c_v2:
  - TRT = Gaze + I (drops the dead `pF * reg_weight * prev_gaze` term).
    Lesion analysis showed Δr ≈ -0.001 from this term; it's been removed.
  - First-word skip floor: skip_prob[:, 0] = 1e-6 (anatomically forced;
    no parafoveal preview before the first fixation).
  - M2 = I tied via single _M2I_raw parameter.

From v4c_saccade_noise:
  - 4 new Mψ parameters with Reichle 2003 defaults:
        omega1 = 6,  omega2 = 3   (systematic landing error)
        eta1   = 0.5, eta2  = 0.15 (random landing error)
  - OPTIMAL_SACCADE_LENGTH = 7 (Reichle constant).
  - Per-word systematic_error and sigma_landing computed from
    [intended_saccade_length, launch_duration].  Inputs are oculomotor
    only — no LLaMA in Mψ.
  - L1 enters via the closed-form Gaussian expectation:
        E[L1_ecc] = base_L1 * eps^(sys_err + (w-1)/2)
                          * exp(0.5 * sigma^2 * (ln eps)^2)
    Reduces to v4c_v2's L1 when sys_err = sigma = 0 (first word).

Preserved from both:
  - ctx_head(LLaMA_hidden) — LM-substituted predictability slot.
  - skip_residual_head(LLaMA_hidden) — learned skip adjustment.
  - alpha1_reichle, alpha2_reichle property aliases.
  - lambda_refix, refix_pivot length-driven refix probability.
  - skip_temperature for the race sigmoid.

Dropped (vs v4c):
  - pF, reg_weight (lesion-confirmed dead).

Cognitive parameters (13 total):
    alpha1_reichle, alpha2_reichle, delta, epsilon,
    M1, M2 = I (tied),
    omega1, omega2, eta1, eta2,            <- new (Mψ)
    lambda_refix, refix_pivot, skip_temperature.

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


# --------------------------------------------------------------------------- #
#  Reichle EZ Reader cascade — v4c_v2 hygiene + Mψ cognitive stage
# --------------------------------------------------------------------------- #

class ReichleEZReaderV2SaccadeNoise(nn.Module):
    """
    v4c_v2 cascade (no regression term, first-word skip mask) extended
    with the Mψ landing-error stage. Mψ adjusts the eccentricity factor
    in L1 via the closed-form expectation over the Gaussian landing
    distribution.
    """

    L1_SOFT_FLOOR = 5.0
    FIRST_WORD_SKIP_FLOOR = 1.0e-6

    def __init__(self):
        super().__init__()

        self._epsilon_raw = nn.Parameter(torch.tensor(_inv_softplus(0.15)))

        self._M1_raw = nn.Parameter(torch.tensor(_inv_softplus(125.0)))
        # M2 = I tied: a single parameter, used in both FFD and TRT.
        self._M2I_raw = nn.Parameter(torch.tensor(_inv_softplus(25.0)))

        # --- Mψ (saccade execution noise) parameters ---
        # Initialized at Reichle 2003 published values.
        self._omega1_raw = nn.Parameter(torch.tensor(_inv_softplus(6.0)))
        self._omega2_raw = nn.Parameter(torch.tensor(_inv_softplus(3.0)))
        self._eta1_raw = nn.Parameter(torch.tensor(_inv_softplus(0.5)))
        self._eta2_raw = nn.Parameter(torch.tensor(_inv_softplus(0.15)))

        self.lambda_refix = nn.Parameter(torch.tensor(0.4))
        self.refix_pivot = nn.Parameter(torch.tensor(8.0))

        # NB: pF and reg_weight removed (lesion showed dead term).

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
    def skip_temperature(self):
        return 1.0 + F.softplus(self._skip_temperature_raw)

    def _intended_saccade_lengths(self, word_lengths):
        """
        Intended saccade length in chars for the saccade INTO each word.
        Approximation: from middle of word n-1 to middle of word n,
        plus 1 char for the inter-word space:
            len = w_{n-1}/2 + 1 + w_n/2
        First word: length = 0 (no preceding saccade) → Mψ no-op.
        """
        prev_w = torch.cat(
            [torch.zeros_like(word_lengths[:, :1]), word_lengths[:, :-1]],
            dim=1,
        )
        intended = prev_w / 2.0 + 1.0 + word_lengths / 2.0
        intended = intended.clone()
        intended[:, 0] = 0.0
        return intended

    def _msi_correction(self, intended_len, launch_dur):
        """
        Returns (sys_err, sigma_landing) tensors with shape == intended_len.

        sys_err  = (7 - intended_len) * (omega1 - log(launch_dur)) / omega2
        sigma    = eta1 + eta2 * intended_len

        For the first word (intended_len = 0), sys_err and sigma are
        masked to zero so Mψ becomes a no-op there.
        """
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

    def forward(self, base_L1, L2, residual_skip_logit, word_lengths):
        # --- Step 1: pre-Mψ FFD estimate (used as launch_dur proxy) ---
        # For the saccade INTO word n, launch_dur is roughly the duration
        # of the previous fixation. We approximate this with a v4c-style
        # FFD without Mψ correction: L1_ecc(no Mψ) + M1 + M2.
        ecc_exponent = (word_lengths - 1.0) / 2.0
        L1_ecc_pre = base_L1 * torch.pow(self.epsilon, ecc_exponent)
        L1_pre = self.L1_SOFT_FLOOR + F.softplus(L1_ecc_pre - self.L1_SOFT_FLOOR)
        ffd_pre = L1_pre + self.M1 + self.M2  # (B, T)

        # launch_dur for saccade INTO word n = predicted FFD on word n-1.
        launch_dur = torch.cat(
            [torch.full_like(ffd_pre[:, :1], 250.0), ffd_pre[:, :-1]],
            dim=1,
        )

        # --- Step 2: Mψ corrections per word ---
        intended_len = self._intended_saccade_lengths(word_lengths)
        sys_err, sigma = self._msi_correction(intended_len, launch_dur)

        # --- Step 3: closed-form expected L1 with Mψ ---
        # E[eps^d] for d ~ N(sys_err, sigma^2)
        #   = exp(sys_err * ln eps + 0.5 * sigma^2 * (ln eps)^2)
        # E[L1_ecc] = base_L1 * eps^((w-1)/2 + sys_err) *
        #             exp(0.5 * sigma^2 * (ln eps)^2)
        ln_eps = torch.log(self.epsilon)
        ecc_exp_full = ecc_exponent + sys_err
        variance_factor = torch.exp(0.5 * (sigma ** 2) * (ln_eps ** 2))
        L1_ecc = base_L1 * torch.pow(self.epsilon, ecc_exp_full) * variance_factor

        L1 = self.L1_SOFT_FLOOR + F.softplus(L1_ecc - self.L1_SOFT_FLOOR)

        first_fixation = L1 + self.M1 + self.M2

        # --- Step 4: refixation (length-driven, same as v4c) ---
        refix_prob = torch.sigmoid(
            self.lambda_refix * (word_lengths - self.refix_pivot)
        )
        refix_duration = L2 + self.M1 + self.M2
        gaze_duration = first_fixation + refix_prob * refix_duration

        # --- Step 5: TRT (v4c_v2: no regression term) ---
        conditional_trt = gaze_duration + self.I

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

        # First-word skip mask: anatomically forced to ~0.
        first_word_mask_skip = torch.zeros_like(skip_prob)
        first_word_mask_skip[:, 0] = 1.0
        skip_prob = (
            skip_prob * (1.0 - first_word_mask_skip)
            + self.FIRST_WORD_SKIP_FLOOR * first_word_mask_skip
        )

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
            'epsilon': self.epsilon.detach(),
            'M1': self.M1.detach(),
            'M2': self.M2.detach(),
            'I': self.I.detach(),
            'omega1': self.omega1.detach(),
            'omega2': self.omega2.detach(),
            'eta1': self.eta1.detach(),
            'eta2': self.eta2.detach(),
            'lambda_refix': self.lambda_refix.detach(),
            'refix_pivot': self.refix_pivot.detach(),
            'skip_temperature': self.skip_temperature.detach(),
        }


# --------------------------------------------------------------------------- #
#  Neural EZ Reader Hybrid v4c_v2_saccade_noise
# --------------------------------------------------------------------------- #

class NeuralEZReaderHybrid(nn.Module):
    """
    Same neural backbone as v4c. New cascade
    (ReichleEZReaderV2SaccadeNoise) drops the regression term, masks
    first-word skip, and adds Mψ.

    Learnable:
        Neural:    LLaMA top layers, projection, ctx_head, skip_residual_head
        Cognitive: l1_base_offset (alpha1), l1_freq_coef (alpha2),
                   delta, epsilon, M1, M2 = I (tied),
                   omega1, omega2, eta1, eta2 (Mψ),
                   lambda_refix, refix_pivot,
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

        self.ezreader = ReichleEZReaderV2SaccadeNoise()

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
