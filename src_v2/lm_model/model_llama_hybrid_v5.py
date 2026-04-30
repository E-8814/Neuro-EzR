"""
Neural EZ Reader Hybrid v5 — Reichle-faithful L1 with explicit predictability
and LLaMA-conditioned context-effects residual.

Branches from model_llama_hybrid_v4.py. The central design claim of v5 is a
clean two-way split of what LLaMA contributes to each word:

  (a) PREDICTABILITY ORACLE. LLaMA's next-token probability
        p = exp(-surprisal)
      substitutes for Reichle's human-cloze term. This enters the L1
      formula as an explicit scalar coefficient alpha3, matching the
      structure of time_familiarity_check() in the original simulation:
        tL1 = alpha1 - alpha2*ln(freq) - alpha3 * p
      alpha3 is a single learnable Reichle-canonical parameter with ms units.

  (b) CONTEXT-EFFECTS ENCODER. A small MLP ('ctx_head') reads LLaMA's
      hidden state and emits a per-word additive correction in ms.
      It sees only the hidden state (not freq, not surprisal, not length)
      so its role is well-defined: supra-lexical effects that are not
      captured by frequency or cloze (syntactic integration difficulty,
      discourse binding, morphological complexity).

Architectural changes vs v4:

  (1) EXPLICIT PREDICTABILITY TERM.
      v4's base_L1 = offset + freq_coef * log_freq + l1_neural_head(LLaMA)
      had NO explicit predictability term — the neural head had to
      rediscover cloze-like effects mixed with syntax, semantics, etc.
      v5 adds an explicit term:
        base_L1 = alpha1 + alpha2 * log_freq_norm
                         - alpha3 * p
                         + ctx_head(LLaMA_hidden)
      This separates lexical predictability (alpha3 knob) from
      supra-lexical context effects (ctx_head).

  (2) SURPRISAL COMPUTATION.
      Switches the LLaMA backbone from AutoModel to AutoModelForCausalLM
      so we can access lm_head and compute per-word surprisal from
      next-token log probabilities. Subword surprisals are summed per
      word (same as insanity_v2). Surprisal is detached before feeding
      into the cascade — the language model is a fixed predictability
      oracle, and the cognitive parameter alpha3 learns the scaling.

  (3) M2 AND I ARE TIED.
      In v4 both are additive constants
        FFD = L1 + M1 + M2,   TRT = Gaze + I + regression
      so the optimizer cannot distinguish them (the v4 log showed
      M2 == I == 24.1 after training — they drifted together). v5
      uses a single parameter for both, reflecting that only their
      sum is identifiable from aggregated eye-tracking means. The
      paper reports one "M2/I" value.

  (4) L1 NEURAL HEAD RENAMED TO ctx_head.
      Architecture unchanged (Linear -> GELU -> Dropout -> Linear,
      last layer small-init). New name matches the paper's
      "context-effects encoder" framing.

  (5) SKIP RESIDUAL HEAD KEPT.
      The parafoveal race alone gave r_skip ~ 0.53 in v4 evaluation;
      with the residual it improved to ~0.67. Keeping the residual
      (L2-penalized) trades a small amount of story purity for ~0.1
      gain on skip correlation. The paper frames it as "race + learned
      correction" and reports the ablation.

Reichle-literature parameter analogues (what you report in the paper):

      Reichle            v5 name            Init (ms)     Role
      ------------------ ------------------ ------------- ----------------------
      alpha1             l1_base_offset     60.0          base familiarity time
      alpha2             l1_freq_coef       -17.0*        frequency coefficient
      alpha3             l1_pred_coef       40.0          predictability coef.
      delta              delta (via _raw)   0.34          L2 / L1 ratio
      epsilon            epsilon (via _raw) 1.15          eccentricity base
      M1                 M1 (via _raw)      125.0         labile saccade prog.
      M2 = I             M2I (via _raw)     25.0          (tied) motor/attn

  * l1_freq_coef is defined against normalized log_freq ((ln f - 10) / 5),
    so the unnormalized Reichle equivalent is alpha2_Reichle = -coef / 5
    (initial value -17 corresponds to Reichle's 3.4).

Other:

  - pF, reg_weight, lambda_refix, refix_pivot, skip_temperature:
    unchanged from v4.
  - L1 soft floor at 5 ms via softplus: unchanged from v4.
  - Parafoveal race (M1 - L1_next_parafoveal) / tau: unchanged from v4.
  - Frequency normalization scheme: unchanged from v4.

forward() signature unchanged: model(word_lists, frequencies, word_lengths).

The returned dict includes 'word_surprisal' and 'word_predictability' so
downstream logging / evaluation can inspect the LLaMA oracle.
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
#  Reichle EZ Reader cascade with tied M2 = I and parafoveal-race skip
# --------------------------------------------------------------------------- #

class ReichleEZReader(nn.Module):
    """
    Maps (base_L1, L2, residual_skip_logit, word_lengths) to FFD / Gaze /
    TRT / skip via the Reichle cascade.

    Differences from model_llama_hybrid_v4.ReichleEZReader:
      - M2 and I are a single tied parameter. Only their sum is
        identifiable from aggregated means, and the v4 run showed them
        drifting to the same value anyway.
      - Skip cascade logic (parafoveal race + residual) unchanged.
      - Soft L1 floor at 5 ms unchanged.
    """

    L1_SOFT_FLOOR = 5.0

    def __init__(self):
        super().__init__()

        self._epsilon_raw = nn.Parameter(torch.tensor(_inv_softplus(0.15)))

        self._M1_raw = nn.Parameter(torch.tensor(_inv_softplus(125.0)))
        # M2 and I tied: one parameter, used in both FFD and TRT.
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
        # Tied to M2.
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

        # I is tied to M2 — see class docstring.
        conditional_trt = gaze_duration + self.I + regression_cost

        # --- Skip: parafoveal race + residual (unchanged from v4) ---
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
#  Neural EZ Reader Hybrid v5
# --------------------------------------------------------------------------- #

class NeuralEZReaderHybrid(nn.Module):
    """
    (word tokens, frequencies) -> LLaMA (AutoModelForCausalLM)
        -> last-subword pooling + per-word surprisal
        -> projection

        base_L1 = alpha1 + alpha2 * log_freq_norm
                         - alpha3 * exp(-surprisal)       # <-- NEW in v5
                         + ctx_head(projected)

        base_L1 -> soft floor
                -> delta      -> L2
                -> cascade    -> FFD, Gaze, TRT
                -> parafoveal race + residual -> skip

    Learnable:
        Neural:    LLaMA top layers, projection, ctx_head,
                   skip_residual_head
        Cognitive: l1_base_offset (alpha1),
                   l1_freq_coef  (alpha2),
                   l1_pred_coef  (alpha3),        # NEW
                   delta, epsilon, M1, M2=I, pF, reg_weight,
                   lambda_refix, refix_pivot, skip_temperature
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

        self.projection = nn.Sequential(
            nn.Linear(llama_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # --- Reichle skeleton scalars ---
        # Names match v4 for checkpoint / script compatibility; the
        # Reichle-canonical alpha labels appear in the paper table.
        # alpha1 analogue: base familiarity time (ms).
        self.l1_base_offset = nn.Parameter(torch.tensor(60.0))
        # alpha2 analogue: frequency coefficient on normalized log_freq
        # (ln f - 10) / 5. Unnormalized Reichle equivalent is -coef / 5.
        self.l1_freq_coef = nn.Parameter(torch.tensor(-17.0))
        # alpha3 analogue (NEW in v5): predictability coefficient (ms),
        # applied as `- l1_pred_coef * exp(-surprisal)`. Initialized at
        # 40 ms, close to Reichle 2003's alpha3 = 39 ms/cloze.
        self.l1_pred_coef = nn.Parameter(torch.tensor(40.0))

        # Context-effects encoder (formerly l1_neural_head). Reads only
        # the projected LLaMA state, so its role is purely supra-lexical:
        # effects beyond what frequency or predictability can explain.
        # Near-zero init so the Reichle skeleton dominates at epoch 1.
        self.ctx_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.normal_(self.ctx_head[-1].weight, std=0.01)
        nn.init.zeros_(self.ctx_head[-1].bias)

        self._delta_raw = nn.Parameter(torch.tensor(_logit(0.34)))

        # Skip residual head (kept from v4): a small learned correction
        # on top of the parafoveal race logit. L2-penalized during
        # training so the race carries most of the skip signal.
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

    # Convenience aliases for paper-side reporting.
    @property
    def alpha1(self):
        return self.l1_base_offset

    @property
    def alpha2(self):
        return self.l1_freq_coef

    @property
    def alpha3(self):
        return self.l1_pred_coef

    # ------------------------------------------------------------------ #
    # Tokenization, subword pooling, surprisal
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
        """Sum per-subword surprisal into per-word surprisal, in nats."""
        B = logits.size(0)
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        token_log_probs = log_probs.gather(
            -1, input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)

        # Align token_log_probs[i] with input_ids[:, i]: pad leading 0 so
        # that subword index `sub` has log_prob `token_log_probs[b, sub]`.
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
        """
        Args:
            word_lists:   list of list of str
            frequencies:  (B, S) SUBTLEX raw counts per word
            word_lengths: (B, S) character counts per word

        Returns the cascade result dict plus base_L1, ctx, surprisal,
        predictability, and the three Reichle L1-formula coefficients.
        """
        device = word_lengths.device

        input_ids, attention_mask, word_maps, max_words = (
            self._tokenize_and_align(word_lists, device)
        )

        # Single forward pass gets both hidden states (for ctx_head) and
        # logits (for surprisal). Using AutoModelForCausalLM so lm_head
        # is available.
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
        # Detach surprisal — LLaMA is a fixed predictability oracle
        # and alpha3 learns the scaling.
        surprisal = surprisal.detach()

        seq_len = word_lengths.size(1)
        word_repr = word_repr[:, :seq_len, :]
        surprisal = surprisal[:, :seq_len]

        projected = self.projection(word_repr)

        # --- Reichle skeleton with explicit alpha3 * predictability ---
        log_freq = torch.log(frequencies.clamp(min=1.0))
        log_freq_norm = (log_freq - 10.0) / 5.0

        # p in [0, 1]: LLaMA's next-token probability as cloze substitute.
        predictability = torch.exp(-surprisal).clamp(max=1.0)

        base_L1_formula = (
            self.l1_base_offset
            + self.l1_freq_coef * log_freq_norm
            - self.l1_pred_coef * predictability
        )

        # --- Context-effects residual (supra-lexical only) ---
        ctx = self.ctx_head(projected).squeeze(-1)
        l1_raw = base_L1_formula + ctx

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
        result['ctx'] = ctx
        result['word_surprisal'] = surprisal
        result['word_predictability'] = predictability
        result['log_freq'] = log_freq
        result['delta'] = self.delta.detach()
        result['l1_base_offset'] = self.l1_base_offset.detach()
        result['l1_freq_coef'] = self.l1_freq_coef.detach()
        result['l1_pred_coef'] = self.l1_pred_coef.detach()

        return result
