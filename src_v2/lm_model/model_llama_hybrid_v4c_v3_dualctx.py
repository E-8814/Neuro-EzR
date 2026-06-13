"""
Neural EZ Reader Hybrid v4c_v3_dualctx — v4c_v2_dualctx without the
first-word skip clamp.

Identical to model_llama_hybrid_v4c_v2_dualctx in every respect except
one: the FIRST_WORD_SKIP_FLOOR clamp is removed from the cascade.

Why the clamp is removed
------------------------
v4c_v2-family models hard-set skip_prob[:, 0] = 1e-6. That mirrors the
original E-Z Reader convention (the simulation starts with the eyes on
word 1, which therefore cannot be skipped), but in GECO the aggregated
human skip rate for sentence-initial words is ~0.54 — readers arrive
mid-line from the previous sentence and often jump past word 1. The
clamp therefore forces a maximally-wrong constant prediction on ~9% of
test words, with zero gradient (it is a constant), and depresses skip
Pearson r from ~0.51 to ~0.39 on GECO test.

The v3 stance: the cascade does not model boundary skips, so the first
word should be *excluded from skip supervision and evaluation* rather
than clamped to a constant that is then scored. The exclusion lives in
the trainer (see train_hybrid_v4c_v3_dualctx_geco.py); the model simply
returns the raw race output for every position.

Skip-race row alignment (handled in the trainer, not here)
----------------------------------------------------------
The race at row i is computed from the NEXT word's preview difficulty:
    skip_prob[i] = sigma((M1 - base_L1_skip[i+1] * eps^ecc) / tau + r_i)
i.e. the quantity at row i is "P(the word after row i is skipped)".
v4c_v2-family training supervised this against h_skip[i] (the row's own
word). The v3 trainer makes the alignment an explicit choice:
  --skip_align same : supervise row i against h_skip[i]   (legacy)
  --skip_align next : supervise row i against h_skip[i+1] (race-faithful)
The forward pass is identical for both; nothing in this file depends on
the choice.

Everything else (two ctx heads, cog scalars, L1/L2/refix/time paths,
forward() signature, result keys) is byte-for-byte the same as
v4c_v2_dualctx, so downstream eval utilities work unchanged.
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
#  Reichle EZ Reader cascade — v4c_v2_dualctx minus the first-word clamp
# --------------------------------------------------------------------------- #


class ReichleEZReaderDualCtx(nn.Module):
    """
    Same cascade math as v4c_v2_dualctx's ReichleEZReaderDualCtx, with
    the FIRST_WORD_SKIP_FLOOR clamp removed: skip_prob is the raw race
    output at every position, including position 0. Position 0 is
    excluded from supervision/eval by the trainer instead.
    """

    L1_SOFT_FLOOR = 5.0

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
    def epsilon(self): return 1.0 + F.softplus(self._epsilon_raw)
    @property
    def M1(self): return F.softplus(self._M1_raw)
    @property
    def M2(self): return F.softplus(self._M2I_raw)
    @property
    def I(self): return F.softplus(self._M2I_raw)
    @property
    def skip_temperature(self): return 1.0 + F.softplus(self._skip_temperature_raw)

    def forward(self, base_L1_FFD, base_L1_skip, L2, residual_skip_logit,
                word_lengths):
        # --- Current-word path: uses base_L1_FFD ---
        ecc_exponent = (word_lengths - 1.0) / 2.0
        L1_ecc = base_L1_FFD * torch.pow(self.epsilon, ecc_exponent)
        L1 = self.L1_SOFT_FLOOR + F.softplus(L1_ecc - self.L1_SOFT_FLOOR)

        first_fixation = L1 + self.M1 + self.M2

        refix_prob = torch.sigmoid(
            self.lambda_refix * (word_lengths - self.refix_pivot)
        )
        refix_duration = L2 + self.M1 + self.M2
        gaze_duration = first_fixation + refix_prob * refix_duration

        conditional_trt = gaze_duration + self.I

        # --- Parafoveal-preview path: uses base_L1_skip ---
        base_L1_skip_next = torch.cat(
            [base_L1_skip[:, 1:], torch.full_like(base_L1_skip[:, :1], 1000.0)],
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
        L1_next_parafoveal = base_L1_skip_next * torch.pow(
            self.epsilon, ecc_exp_next,
        )

        race_logit = (self.M1 - L1_next_parafoveal) / self.skip_temperature
        skip_prob = torch.sigmoid(race_logit + residual_skip_logit)

        # v3: no first-word clamp. The trainer excludes position 0 from
        # skip supervision and evaluation instead.

        return {
            'first_fixation': first_fixation,
            'gaze_duration': gaze_duration,
            'conditional_trt': conditional_trt,
            'total_reading_time': conditional_trt,
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
#  Neural EZ Reader Hybrid v4c_v3_dualctx
# --------------------------------------------------------------------------- #


class NeuralEZReaderHybrid(nn.Module):
    """
    Same neural backbone and heads as v4c_v2_dualctx. Only the cascade
    differs (no first-word skip clamp).

    Learnable:
        Neural:    LLaMA top layers, projection,
                   ctx_head_FFD, ctx_head_skip,
                   skip_residual_head
        Cognitive: l1_base_offset (alpha1), l1_freq_coef (alpha2),
                   delta, epsilon, M1, M2 = I (tied),
                   lambda_refix, refix_pivot, skip_temperature
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

        # Reichle L1 formula scalars (shared between both base_L1s)
        self.l1_base_offset = nn.Parameter(torch.tensor(60.0))
        self.l1_freq_coef = nn.Parameter(torch.tensor(-17.0))

        # --- Two specialized ctx heads ---
        self.ctx_head_FFD = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.normal_(self.ctx_head_FFD[-1].weight, std=0.01)
        nn.init.zeros_(self.ctx_head_FFD[-1].bias)

        self.ctx_head_skip = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.normal_(self.ctx_head_skip[-1].weight, std=0.01)
        nn.init.zeros_(self.ctx_head_skip[-1].bias)

        self._delta_raw = nn.Parameter(torch.tensor(_logit(0.34)))

        self.skip_residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.normal_(self.skip_residual_head[-1].weight, std=0.01)
        nn.init.zeros_(self.skip_residual_head[-1].bias)

        self.ezreader = ReichleEZReaderDualCtx()

    @property
    def delta(self): return torch.sigmoid(self._delta_raw)
    @property
    def alpha1(self): return self.l1_base_offset
    @property
    def alpha2(self): return self.l1_freq_coef
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
            padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        batch_word_maps = []
        max_words = 0
        for batch_idx in range(len(word_lists)):
            word_ids = encodings.word_ids(batch_index=batch_idx)
            word_map = {}
            for subword_idx, word_idx in enumerate(word_ids):
                if word_idx is None: continue
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

    def forward(self, word_lists, frequencies, word_lengths):
        device = word_lengths.device

        input_ids, attention_mask, word_maps, max_words = (
            self._tokenize_and_align(word_lists, device)
        )

        llama_out = self.llama(
            input_ids=input_ids, attention_mask=attention_mask,
        ).last_hidden_state

        word_repr = self._pool_subwords_to_words(
            llama_out, word_maps, max_words, device,
        )

        projected = self.projection(word_repr)

        seq_len = word_lengths.size(1)
        projected = projected[:, :seq_len, :]

        log_freq = torch.log(frequencies.clamp(min=1.0))
        log_freq_norm = (log_freq - 10.0) / 5.0

        # Shared formula skeleton
        base_L1_formula = (
            self.l1_base_offset + self.l1_freq_coef * log_freq_norm
        )

        # Two specialized neural corrections
        ctx_FFD = self.ctx_head_FFD(projected).squeeze(-1)
        ctx_skip = self.ctx_head_skip(projected).squeeze(-1)

        # Two base_L1 paths (shared formula + specialized correction)
        l1_FFD_raw = base_L1_formula + ctx_FFD
        l1_skip_raw = base_L1_formula + ctx_skip
        base_L1_FFD = 5.0 + F.softplus(l1_FFD_raw - 5.0)
        base_L1_skip = 5.0 + F.softplus(l1_skip_raw - 5.0)

        residual_skip_logit = self.skip_residual_head(projected).squeeze(-1)

        # L2 = δ · base_L1_FFD (current word property; uses FFD-side L1)
        L2 = self.delta * base_L1_FFD

        result = self.ezreader(
            base_L1_FFD=base_L1_FFD,
            base_L1_skip=base_L1_skip,
            L2=L2,
            residual_skip_logit=residual_skip_logit,
            word_lengths=word_lengths,
        )

        result['base_L1_FFD'] = base_L1_FFD
        result['base_L1_skip'] = base_L1_skip
        result['base_L1_formula'] = base_L1_formula
        result['ctx_FFD'] = ctx_FFD
        result['ctx_skip'] = ctx_skip
        result['log_freq'] = log_freq
        result['delta'] = self.delta.detach()
        result['l1_base_offset'] = self.l1_base_offset.detach()
        result['l1_freq_coef'] = self.l1_freq_coef.detach()
        result['alpha1_reichle'] = self.alpha1_reichle.detach()
        result['alpha2_reichle'] = self.alpha2_reichle.detach()

        # Backward-compat alias for downstream eval scripts that read 'base_L1' / 'ctx'
        result['base_L1'] = base_L1_FFD
        result['ctx'] = ctx_FFD

        return result
