"""
Neural EZ Reader Hybrid v4c_v2_surp (Q2 = "full replace") — v4c_v2 with
ctx_head replaced by `−α3 · TinyLlama_surprisal`.

This is the head-to-head ablation for exp07: tests whether the LLM
hidden state provides anything beyond the model's own surprisal.

Key differences from v4c_v2:
  - No ctx_head (the LLaMA-conditioned MLP).
  - New parameter α3 (init = 39.0, Reichle 2003 published value).
  - Surprisal is provided as an extra forward-pass argument
    (computed and cached separately by precompute_surprisal.py for speed).
  - L1 formula: L1 = α1 + α2*log_freq_norm − α3*surprisal_per_word

The cascade itself (epsilon, M1, M2 = I tied, refix, skip residual,
first-word skip mask, TRT = Gaze + I) is identical to v4c_v2.

forward() signature CHANGES: now requires an extra `surprisals` argument.
    model(word_lists, frequencies, word_lengths, surprisals)
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
#  Reichle EZ Reader cascade (identical to v4c_v2's)
# --------------------------------------------------------------------------- #


class ReichleEZReader(nn.Module):
    """v4c_v2 cascade: TRT = Gaze + I, first-word skip floor, M2 = I tied."""

    L1_SOFT_FLOOR = 5.0
    FIRST_WORD_SKIP_FLOOR = 1.0e-6

    def __init__(self):
        super().__init__()
        self._epsilon_raw = nn.Parameter(torch.tensor(_inv_softplus(0.15)))
        self._M1_raw = nn.Parameter(torch.tensor(_inv_softplus(125.0)))
        self._M2I_raw = nn.Parameter(torch.tensor(_inv_softplus(25.0)))
        self.lambda_refix = nn.Parameter(torch.tensor(0.4))
        self.refix_pivot = nn.Parameter(torch.tensor(8.0))
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
            [base_L1[:, 1:], torch.full_like(base_L1[:, :1], 1000.0)], dim=1,
        )
        wordlen_next = torch.cat(
            [word_lengths[:, 1:], torch.zeros_like(word_lengths[:, :1])], dim=1,
        )
        parafoveal_dist = word_lengths / 2.0 + 1.0
        ecc_exp_next = parafoveal_dist + (
            (wordlen_next - 1.0).clamp(min=0.0) / 2.0
        )
        L1_next_parafoveal = base_L1_next * torch.pow(
            self.epsilon, ecc_exp_next,
        )

        race_logit = (self.M1 - L1_next_parafoveal) / self.skip_temperature
        skip_prob = torch.sigmoid(race_logit + residual_skip_logit)

        first_word_mask = torch.zeros_like(skip_prob)
        first_word_mask[:, 0] = 1.0
        skip_prob = (
            skip_prob * (1.0 - first_word_mask)
            + self.FIRST_WORD_SKIP_FLOOR * first_word_mask
        )

        return {
            'first_fixation': first_fixation,
            'gaze_duration': gaze_duration,
            'conditional_trt': conditional_trt,
            'total_reading_time': conditional_trt,
            'skip_prob': skip_prob,
            'race_logit': race_logit,
            'residual_skip_logit': residual_skip_logit,
            'L1': L1, 'L2': L2,
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
#  Neural backbone with surprisal-driven L1 (no ctx_head)
# --------------------------------------------------------------------------- #


class NeuralEZReaderHybrid(nn.Module):
    """
    Same neural backbone as v4c_v2 but ctx_head is REMOVED. Predictability
    comes from a per-word surprisal scalar (provided as forward-pass arg).

    L1 formula:
        base_L1 = α1 + α2 * log_freq_norm − α3 * surprisal

    Where α3 is a learnable Reichle-style scalar (init 39.0, the published
    value).

    skip_residual_head is RETAINED — it's not surprisal-related and we
    don't want to confound the comparison.
    """

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        freeze_layers: int = 12,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.llama = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.float32,
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

        # Reichle L1 formula scalars
        self.l1_base_offset = nn.Parameter(torch.tensor(60.0))
        self.l1_freq_coef = nn.Parameter(torch.tensor(-17.0))
        # alpha3: predictability coefficient. Reichle 2003 = 39.
        # In our normalized log_freq formulation, a comparable scale is ~10.
        # Init at 10 to start more conservative; will train.
        self.alpha3 = nn.Parameter(torch.tensor(10.0))

        self._delta_raw = nn.Parameter(torch.tensor(_logit(0.34)))

        # skip_residual_head retained — same as v4c_v2.
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
        self, hidden_states, batch_word_maps, max_words, device,
    ):
        batch_size = hidden_states.size(0)
        hidden_dim = hidden_states.size(2)
        idx = torch.zeros(batch_size, max_words, dtype=torch.long)
        for b in range(batch_size):
            for w_idx, (start, end) in enumerate(batch_word_maps[b]):
                idx[b, w_idx] = end - 1
        idx = idx.to(device)
        word_repr = torch.gather(
            hidden_states, 1, idx.unsqueeze(-1).expand(-1, -1, hidden_dim),
        )
        return word_repr

    def forward(self, word_lists, frequencies, word_lengths, surprisals):
        """
        Args:
            surprisals: Tensor of shape (batch, seq_len), per-word surprisal
                from TinyLlama (or whichever LM was used to compute them).
                Same shape as frequencies / word_lengths. Must be precomputed.
        """
        device = word_lengths.device

        input_ids, attention_mask, word_maps, max_words = (
            self._tokenize_and_align(word_lists, device)
        )

        llama_out = self.llama(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

        word_repr = self._pool_subwords_to_words(
            llama_out, word_maps, max_words, device,
        )

        projected = self.projection(word_repr)

        seq_len = word_lengths.size(1)
        projected = projected[:, :seq_len, :]

        log_freq = torch.log(frequencies.clamp(min=1.0))
        log_freq_norm = (log_freq - 10.0) / 5.0

        # Normalize surprisal (mean 0 ish, scale ~1) — surprisal in nats has
        # a wide range. Centering at the per-batch mean makes alpha3 stable.
        # We DON'T normalize globally — we let alpha3 absorb the magnitude.
        # surprisal: raw nats (per-word).

        # Predictability term: − α3 · surprisal (Reichle's α3 has + sign,
        # but in our parameterization the negative absorbs it: high
        # surprisal → larger L1 → slower processing).
        base_L1_formula = (
            self.l1_base_offset
            + self.l1_freq_coef * log_freq_norm
            + self.alpha3 * surprisals    # NB: surprisal is positive (nats), α3 is positive
        )
        # Note: this gives base_L1 ↑ when surprisal ↑ (high surprise → slow).

        l1_raw = base_L1_formula

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
        result['surprisals'] = surprisals
        result['log_freq'] = log_freq
        result['delta'] = self.delta.detach()
        result['l1_base_offset'] = self.l1_base_offset.detach()
        result['l1_freq_coef'] = self.l1_freq_coef.detach()
        result['alpha1_reichle'] = self.alpha1_reichle.detach()
        result['alpha2_reichle'] = self.alpha2_reichle.detach()
        result['alpha3'] = self.alpha3.detach()

        return result
