"""
Neural EZ Reader Hybrid v4c_v2 with RANDOMIZED initialization
(parameter-recovery experiment).

Architecturally identical to v4c_v2:
  - L1 = alpha1 + alpha2*log_freq_norm + ctx_head(LLaMA)
  - L2 = delta * L1
  - FFD = L1_ecc + M1 + M2
  - Gaze = FFD + P_refix(length) * (L2 + M1 + M2)
  - TRT = Gaze + I  (M2 = I tied)
  - Skip = sigmoid((M1 - L1_next)/tau + skip_residual)
  - First-word skip floor

The ONLY difference: cognitive scalar parameters are randomly perturbed
from Reichle 2003 values at initialization time, by a factor uniformly
sampled in [1 - JITTER, 1 + JITTER], with JITTER = 0.5 by default.

Purpose: test whether end-to-end training on GECO causes parameters to
*converge* toward Reichle 2003 published values (a real recovery claim),
as opposed to staying near init (a stability claim).

Run with multiple seeds. If parameters converge near Reichle 2003 values
across seeds despite different starting points, that's evidence for
genuine recovery.

Cognitive scalars perturbed:
    l1_base_offset (60.0)        -> alpha1_norm
    l1_freq_coef (-17.0)         -> alpha2_norm
    delta (0.34)
    epsilon - 1.0 (0.15)
    M1 (125.0)
    M2 = I tied (25.0)
    lambda_refix (0.4)
    refix_pivot (8.0)
    skip_temperature - 1.0 (30.0)

Neural net parameters (LLaMA, projection, ctx_head, skip_residual_head)
are NOT perturbed - they always use their default initialization.

forward() signature unchanged: model(word_lists, frequencies, word_lengths).
"""

import math
import random as _random

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


def _logit(y: float) -> float:
    return math.log(y / (1.0 - y))


# Reichle 2003 reference values (also v4c_v2 default inits).
REICHLE_INIT = {
    'l1_base_offset': 60.0,        # gives alpha1_reichle = 94 at init
    'l1_freq_coef': -17.0,         # gives alpha2_reichle = 3.4 at init
    'delta': 0.34,
    'epsilon_minus_1': 0.15,        # epsilon = 1.0 + softplus(_eps_raw)
    'M1': 125.0,
    'M2I': 25.0,
    'lambda_refix': 0.4,
    'refix_pivot': 8.0,
    'skip_temperature_minus_1': 30.0,
}

# Default jitter range: ±50% (sample uniformly in [0.5, 1.5] times Reichle).
DEFAULT_JITTER = 0.5

# Bounds enforced after sampling (to avoid pathological inits).
DELTA_BOUND_MIN = 0.10
DELTA_BOUND_MAX = 0.50


def sample_init(init_seed=None, jitter=DEFAULT_JITTER):
    """
    Sample randomized cognitive-parameter initial values within
    [1-jitter, 1+jitter] of Reichle 2003 defaults.

    Returns dict mapping parameter name -> sampled value.
    """
    rng = _random.Random(init_seed)
    lo, hi = 1.0 - jitter, 1.0 + jitter

    def scale():
        return rng.uniform(lo, hi)

    sampled = {
        'l1_base_offset': REICHLE_INIT['l1_base_offset'] * scale(),
        'l1_freq_coef': REICHLE_INIT['l1_freq_coef'] * scale(),
        'delta': max(
            DELTA_BOUND_MIN,
            min(DELTA_BOUND_MAX, REICHLE_INIT['delta'] * scale()),
        ),
        'epsilon_minus_1': REICHLE_INIT['epsilon_minus_1'] * scale(),
        'M1': REICHLE_INIT['M1'] * scale(),
        'M2I': REICHLE_INIT['M2I'] * scale(),
        'lambda_refix': REICHLE_INIT['lambda_refix'] * scale(),
        'refix_pivot': REICHLE_INIT['refix_pivot'] * scale(),
        'skip_temperature_minus_1': (
            REICHLE_INIT['skip_temperature_minus_1'] * scale()
        ),
    }
    return sampled


# --------------------------------------------------------------------------- #
#  Reichle EZ Reader cascade (v4c_v2: no regression term, first-word mask)
#  with externally-injected init values for cognitive scalars
# --------------------------------------------------------------------------- #

class ReichleEZReader(nn.Module):
    """Same cascade as v4c_v2's ReichleEZReader; init values are passed in."""

    L1_SOFT_FLOOR = 5.0
    FIRST_WORD_SKIP_FLOOR = 1.0e-6

    def __init__(self, sampled_init):
        super().__init__()

        self._epsilon_raw = nn.Parameter(
            torch.tensor(_inv_softplus(sampled_init['epsilon_minus_1']))
        )
        self._M1_raw = nn.Parameter(
            torch.tensor(_inv_softplus(sampled_init['M1']))
        )
        self._M2I_raw = nn.Parameter(
            torch.tensor(_inv_softplus(sampled_init['M2I']))
        )

        self.lambda_refix = nn.Parameter(
            torch.tensor(sampled_init['lambda_refix'])
        )
        self.refix_pivot = nn.Parameter(
            torch.tensor(sampled_init['refix_pivot'])
        )

        self._skip_temperature_raw = nn.Parameter(
            torch.tensor(_inv_softplus(sampled_init['skip_temperature_minus_1']))
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

        conditional_trt = gaze_duration + self.I

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

        first_word_mask = torch.zeros_like(skip_prob)
        first_word_mask[:, 0] = 1.0
        skip_prob = (
            skip_prob * (1.0 - first_word_mask)
            + self.FIRST_WORD_SKIP_FLOOR * first_word_mask
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
            'epsilon': self.epsilon.detach(),
            'M1': self.M1.detach(),
            'M2': self.M2.detach(),
            'I': self.I.detach(),
            'lambda_refix': self.lambda_refix.detach(),
            'refix_pivot': self.refix_pivot.detach(),
            'skip_temperature': self.skip_temperature.detach(),
        }


# --------------------------------------------------------------------------- #
#  Neural EZ Reader Hybrid v4c_v2_randinit
# --------------------------------------------------------------------------- #

class NeuralEZReaderHybrid(nn.Module):
    """
    Same neural backbone as v4c_v2. Cognitive scalars initialized with
    ±jitter random perturbation around Reichle 2003 values.

    The sampled initial values are stored in `self.sampled_init` for
    inspection.

    Args:
        init_seed: RNG seed for the random init. If None, uses Python's
            global RNG state (NOT recommended for reproducibility).
        jitter: half-width of the multiplicative perturbation interval.
            Default 0.5 → sample uniformly in [0.5, 1.5] * Reichle value.
    """

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        freeze_layers: int = 12,
        hidden_dim: int = 256,
        init_seed: int = None,
        jitter: float = DEFAULT_JITTER,
    ):
        super().__init__()

        # --- Sample randomized cognitive parameter inits ---
        sampled = sample_init(init_seed=init_seed, jitter=jitter)
        self.sampled_init = sampled
        self.jitter = jitter

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

        # --- L1 formula (alpha1, alpha2): randomized init ---
        self.l1_base_offset = nn.Parameter(
            torch.tensor(sampled['l1_base_offset'])
        )
        self.l1_freq_coef = nn.Parameter(
            torch.tensor(sampled['l1_freq_coef'])
        )

        # --- ctx_head: NOT randomized (always default init) ---
        self.ctx_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.normal_(self.ctx_head[-1].weight, std=0.01)
        nn.init.zeros_(self.ctx_head[-1].bias)

        # --- delta: randomized init ---
        self._delta_raw = nn.Parameter(
            torch.tensor(_logit(sampled['delta']))
        )

        # --- skip_residual_head: NOT randomized ---
        self.skip_residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.normal_(self.skip_residual_head[-1].weight, std=0.01)
        nn.init.zeros_(self.skip_residual_head[-1].bias)

        # --- Cascade scalars: randomized init ---
        self.ezreader = ReichleEZReader(sampled_init=sampled)

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

    def get_init_summary(self):
        """Return a dict mapping cog param name -> (sampled_init, current_value)."""
        return {
            'l1_base_offset': (
                self.sampled_init['l1_base_offset'],
                self.l1_base_offset.item(),
            ),
            'l1_freq_coef': (
                self.sampled_init['l1_freq_coef'],
                self.l1_freq_coef.item(),
            ),
            'alpha1_reichle': (
                self.sampled_init['l1_base_offset']
                - 2.0 * self.sampled_init['l1_freq_coef'],
                self.alpha1_reichle.item(),
            ),
            'alpha2_reichle': (
                -self.sampled_init['l1_freq_coef'] / 5.0,
                self.alpha2_reichle.item(),
            ),
            'delta': (
                self.sampled_init['delta'],
                self.delta.item(),
            ),
            'epsilon': (
                1.0 + self.sampled_init['epsilon_minus_1'],
                self.ezreader.epsilon.item(),
            ),
            'M1': (
                self.sampled_init['M1'],
                self.ezreader.M1.item(),
            ),
            'M2_eq_I': (
                self.sampled_init['M2I'],
                self.ezreader.M2.item(),
            ),
            'lambda_refix': (
                self.sampled_init['lambda_refix'],
                self.ezreader.lambda_refix.item(),
            ),
            'refix_pivot': (
                self.sampled_init['refix_pivot'],
                self.ezreader.refix_pivot.item(),
            ),
            'skip_temperature': (
                1.0 + self.sampled_init['skip_temperature_minus_1'],
                self.ezreader.skip_temperature.item(),
            ),
        }

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
