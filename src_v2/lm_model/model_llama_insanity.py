"""
Neural EZ Reader Insanity — full-distribution, surprisal-conditioned,
sequential reader model.

Hybrid v1–v4 all predict point estimates of FFD / Gaze / TRT and train
with MSE on expected values. Reading times are stochastic: MSE on
means throws away heteroscedastic variance, forces the long tail to
compress toward the mean, and leaves every motor scalar (M1, M2, I)
fungible because they all enter the cascade additively. "Insanity"
changes the observation model, the conditioning features, and the
sequential structure simultaneously. The Reichle cascade itself is
unchanged — it becomes the *median* of a log-normal distribution
rather than a deterministic prediction of the mean.

Five departures from hybrid v1–v4:

  (1) DISTRIBUTIONAL OUTPUT.
      Each observable (FFD, Gaze, TRT) is a log-normal distribution.
      The Reichle cascade mean is interpreted as the MEDIAN of the
      log-normal; a heteroscedastic variance head predicts the
      per-word log-space sigma. Loss is Gaussian NLL on log(observed).
      Rare words can have large predicted sigma without compressing
      the fit on common words — this is the structural fix for the
      `ascendancy` / `invalided` under-prediction failure in v1–v4.

  (2) SURPRISAL AS AN EXPLICIT INPUT.
      The LM head already computes P(word_n | word_<n). v1–v4
      expected the frozen LLaMA hidden state to implicitly encode
      this (Reichle's alpha3 / predictability term), and the L1 heads
      have been silently rediscovering it through a 256→128→1 MLP.
      Insanity reads per-word surprisal directly off the lm_head
      logits (summed over the word's subwords) and concatenates it
      as a scalar feature. The computation is `detach()`ed — it is a
      fixed input, not a gradient path through the LM head.

  (3) SEQUENTIAL READER STATE.
      A unidirectional GRU runs over per-word features to maintain a
      reader state representing cumulative cognitive load and
      parafoveal carry-over. v1–v4 process each word independently;
      the real E-Z Reader simulation explicitly does not.

  (4) HETEROSCEDASTIC VARIANCE HEADS.
      Three independent sigma heads (one each for FFD, Gaze, TRT)
      predict per-word log-space standard deviation. A soft floor
      prevents sigma from collapsing to a delta. This is the
      uncertainty quantification that point-estimate models cannot
      express.

  (5) FIXATION HEAD FED BY GRU STATE.
      Skip is still a sigmoid P(fixated) head (not a race, not a
      parallel projection) but it reads from the GRU reader state,
      so it can use cumulative context. A categorical landing
      distribution over {0, +1, +2, +3} relative positions is the
      cleaner formulation but requires per-word landing labels that
      are not yet plumbed through the data loaders — left as future
      work.

Architecture:

    word_lists, frequencies, word_lengths
        │
        ▼
    LlamaForCausalLM
        ├─ last_hidden_state  → per-word repr (last-subword pooling)
        └─ lm_head(hidden)    → per-word surprisal (detached)
        │
        ▼
    [projected_repr | log_freq_norm, log_len_norm, surprisal_norm]
        │
        ▼
    feature_fusion
        │
        ▼
    reader_gru → reader_state
        │
        ▼
    state = [fused | reader_state]
        │
        ├─ l1_head          → log_L1 → L1 = exp(log_L1)
        ├─ sigma_ffd_head   → sigma_ffd   (softplus + floor)
        ├─ sigma_gaze_head  → sigma_gaze
        ├─ sigma_trt_head   → sigma_trt
        └─ fixation_head    → P(fixated)
        │
        ▼
    ReichleCascadeMeans(L1, word_lengths)
        → ffd_mean, gaze_mean, trt_mean     (these are the medians)

    log-normal observation model:
        log FFD  ~ N(log ffd_mean,  sigma_ffd)
        log Gaze ~ N(log gaze_mean, sigma_gaze)
        log TRT  ~ N(log trt_mean,  sigma_trt)

forward() signature:
    model(word_lists, frequencies, word_lengths)

The loss (training-side, not model-side) combines:
    - Gaussian NLL on log(observed_FFD / Gaze / TRT)
    - BCE on fixation_prob against fixation indicator
    - Small regularization toward Reichle priors on the cascade params

See `log_normal_nll` at the bottom of this file for the NLL helper.
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
#  Reichle motor cascade (means only)
# --------------------------------------------------------------------------- #

class ReichleCascadeMeans(nn.Module):
    """
    Maps per-word L1 to per-word FFD / Gaze / TRT *medians* via the
    Reichle motor cascade. Same reparameterization as hybrid v1–v4.

    Parameters (all learnable):
        epsilon, M1, M2, I, delta, pF, reg_weight, lambda_refix, refix_pivot.

    The cascade here produces only the means (interpreted as log-normal
    medians downstream). Variance is handled by heads on the main model.
    """

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
#  Neural EZ Reader Insanity
# --------------------------------------------------------------------------- #

class NeuralEZReaderInsanity(nn.Module):
    """
    Distributional, surprisal-conditioned, sequential reader model. See
    the module docstring for the full rationale.

    Key differences from `NeuralEZReaderHybrid` (any version):
      * Uses `AutoModelForCausalLM` so the LM head is available for
        surprisal extraction.
      * Adds a GRU over per-word features for reader state.
      * Emits `(mu_log, sigma)` per observable, not a point estimate.
      * L1 head outputs log_L1 (no softplus * scale dance).
      * No parafoveal race; fixation is a sigmoid head on the GRU state.
    """

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        freeze_layers: int = 12,
        hidden_dim: int = 256,
        gru_hidden: int = 256,
        l1_init_ms: float = 50.0,
        sigma_init: float = 0.30,
        sigma_floor: float = 0.05,
        fixation_prior: float = 0.65,
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

        # Freeze the bottom of the LLaMA stack. The LM head is tied to
        # embed_tokens in TinyLlama, so freezing the embedding also
        # effectively freezes the LM head — which is fine since we
        # `detach()` surprisal anyway.
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

        # Fuse projected word_repr with 3 scalar features:
        # log_freq_norm, log_len_norm, surprisal_norm.
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # ---- Sequential reader state ----
        # Unidirectional: real readers do not have full-sentence preview.
        self.reader_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )

        state_dim = hidden_dim + gru_hidden

        # ---- L1 head (outputs log_L1 directly) ----
        # Init bias so exp(bias) = l1_init_ms, giving base_L1 ~= 50 ms
        # at start, which with M1 + M2 = 150 ms puts FFD_mean ~= 200 ms.
        self.l1_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.normal_(self.l1_head[-1].weight, std=0.01)
        nn.init.constant_(self.l1_head[-1].bias, math.log(l1_init_ms))

        # ---- Variance heads (heteroscedastic log-space sigmas) ----
        # sigma = sigma_floor + softplus(head(state) + init_bias).
        # Pick init_bias so softplus(init_bias) = sigma_init - sigma_floor.
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

        # ---- Fixation head (P(word is fixated)) ----
        self.fixation_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.normal_(self.fixation_head[-1].weight, std=0.01)
        nn.init.constant_(self.fixation_head[-1].bias, _logit(fixation_prior))

        # ---- Reichle motor cascade ----
        self.cascade = ReichleCascadeMeans()

    # ------------------------------------------------------------------ #
    # Tokenization / pooling / surprisal
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
        """
        Sum per-subword negative log P(subword_k | context_<k) across
        each word's subwords. Returns (B, max_words) in nats.

        Convention: `logits[:, t, :]` predicts the token at position
        t+1, so `log P(token_k) = log_softmax(logits[:, k-1, :])[token_k]`
        for k >= 1. Position 0 is BOS and is never a word subword.
        """
        B = logits.size(0)
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # (B, T-1, V)
        token_log_probs = log_probs.gather(
            -1, input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)  # (B, T-1)

        # Pad at position 0 so that token_log_probs[:, k] corresponds to
        # log P(token at position k). Position 0 is BOS and is never
        # indexed by a word subword start.
        token_log_probs = torch.cat(
            [torch.zeros(B, 1, device=device), token_log_probs], dim=1
        )  # (B, T)

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
            word_lists:   list[list[str]] of length B, each a sentence
                          split into words.
            frequencies:  (B, S) float tensor, SUBTLEX raw counts per word.
            word_lengths: (B, S) float tensor, character count per word.

        Returns a dict with:
            ffd_mu_log, ffd_sigma,
            gaze_mu_log, gaze_sigma,
            trt_mu_log, trt_sigma,
            ffd_mean, gaze_mean, trt_mean,           # == log-normal medians
            fixation_logit, fixation_prob, skip_prob,
            base_L1, log_L1, L1_ecc, L2, refix_prob,
            surprisal, log_freq,
            epsilon, M1, M2, I, pF, reg_weight, delta,
            lambda_refix, refix_pivot                # all detached
        """
        device = word_lengths.device

        input_ids, attention_mask, word_maps, max_words = (
            self._tokenize_and_align(word_lists, device)
        )

        # One forward pass; pull hidden states and logits out separately
        # so we do not allocate every intermediate hidden state.
        base_out = self.lm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = base_out.last_hidden_state   # (B, T, D_llama)
        logits = self.lm.lm_head(hidden)      # (B, T, V)

        word_repr = self._pool_subwords_to_words(
            hidden, word_maps, max_words, device
        )  # (B, max_words, D_llama)

        surprisal = self._compute_word_surprisal(
            logits, input_ids, word_maps, max_words, device
        )  # (B, max_words)

        # Treat surprisal as a fixed input feature. We do not want
        # downstream losses to reshape the LM head through this path.
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
        )  # (B, S, 3)

        fused = self.feature_fusion(
            torch.cat([projected, scalar_feats], dim=-1)
        )  # (B, S, hidden_dim)

        # ---- Sequential reader state ----
        reader_state, _ = self.reader_gru(fused)  # (B, S, gru_hidden)
        state = torch.cat([fused, reader_state], dim=-1)  # (B, S, state_dim)

        # ---- L1 head ----
        log_L1 = self.l1_head(state).squeeze(-1)
        L1 = torch.exp(log_L1)

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

        # ---- Reichle cascade means ----
        cascade_out = self.cascade(base_L1=L1, word_lengths=word_lengths)
        ffd_mean = cascade_out['ffd_mean']
        gaze_mean = cascade_out['gaze_mean']
        trt_mean = cascade_out['trt_mean']

        # The cascade mean is the MEDIAN of the log-normal, i.e.,
        # exp(mu_log) == ffd_mean. So mu_log == log(ffd_mean).
        ffd_mu_log = torch.log(ffd_mean.clamp(min=1.0))
        gaze_mu_log = torch.log(gaze_mean.clamp(min=1.0))
        trt_mu_log = torch.log(trt_mean.clamp(min=1.0))

        return {
            # Distribution parameters (log-normal in log-ms space).
            'ffd_mu_log': ffd_mu_log,
            'ffd_sigma': sigma_ffd,
            'gaze_mu_log': gaze_mu_log,
            'gaze_sigma': sigma_gaze,
            'trt_mu_log': trt_mu_log,
            'trt_sigma': sigma_trt,

            # Cascade arithmetic means (== log-normal medians).
            'ffd_mean': ffd_mean,
            'gaze_mean': gaze_mean,
            'trt_mean': trt_mean,

            # Fixation / skip.
            'fixation_logit': fixation_logit,
            'fixation_prob': fixation_prob,
            'skip_prob': skip_prob,

            # Cognitive intermediates.
            'base_L1': L1,
            'log_L1': log_L1,
            'L1_ecc': cascade_out['L1_ecc'],
            'L2': cascade_out['L2'],
            'refix_prob': cascade_out['refix_prob'],

            # Input features (for logging).
            'surprisal': surprisal,
            'log_freq': log_freq,

            # Detached Reichle parameters (for logging / regularization).
            'epsilon': self.cascade.epsilon.detach(),
            'M1': self.cascade.M1.detach(),
            'M2': self.cascade.M2.detach(),
            'I': self.cascade.I.detach(),
            'pF': self.cascade.pF.detach(),
            'reg_weight': self.cascade.reg_weight.detach(),
            'delta': self.cascade.delta.detach(),
            'lambda_refix': self.cascade.lambda_refix.detach(),
            'refix_pivot': self.cascade.refix_pivot.detach(),
        }


# --------------------------------------------------------------------------- #
#  Loss helper: Gaussian NLL on log(observation), with Jacobian term
# --------------------------------------------------------------------------- #

def log_normal_nll(mu_log, sigma, observed_ms, mask=None):
    """
    Negative log-likelihood of `observed_ms` under a log-normal with
    location `mu_log` and log-space scale `sigma`:

        log X ~ N(mu_log, sigma)
        -log p(x) = 0.5 * ((log x - mu_log) / sigma)^2
                    + log(sigma)
                    + log(x)                   # Jacobian of log
                    + 0.5 * log(2 pi)

    The `log(x)` and `0.5 log(2 pi)` terms do not depend on the model
    parameters but are included so that the reported NLL is the true
    negative log density — comparable across runs and checkpoints.

    Args:
        mu_log:       (B, S) log-space median (log of cascade mean).
        sigma:        (B, S) log-space standard deviation.
        observed_ms:  (B, S) in ms, > 0.
        mask:         (B, S) bool, True where observation is valid.

    Returns:
        Scalar mean NLL across valid entries.
    """
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
