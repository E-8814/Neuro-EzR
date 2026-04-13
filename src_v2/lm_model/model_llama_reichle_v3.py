"""
Neural EZ Reader v3 — Reichle-faithful with fixes for FFD compression.

Changes from model_llama_reichle_v2.py:

  1. L1 clamp floor lowered from 60 -> 30 ms.
     In v2 the floor created a hard FFD lower bound at ~200 ms (60 + M1 +
     M2). Many human FFDs in aggregated GECO data are below 200 ms (120-180
     for quickly-fixated function words), so the floor compressed predicted
     FFD variance and killed the FFD correlation. 30 ms is below Reichle's
     typical L1 range but is necessary to give the model room to express
     fast-fixation words without clipping.

  2. Surprisal is clipped at SURPRISAL_CLIP = 12 nats before entering
     the L1 formula (but NOT before logging).
     In v2, word_surprisal had a right-skewed distribution with a tail
     reaching ~25 nats. Rare words (ascendancy, invalided) had alpha3 *
     surprisal contributions of 70-85 ms to base_L1 — much larger than
     Reichle's max alpha3 * cloze contribution of 39 ms. The residual head
     was then spending most of its capacity on negative corrections (-60
     to -65 ms) to undo this over-prediction. Clipping at 12 nats caps the
     formula's per-word predictability contribution at alpha3 * 12 ~ 40
     ms, matching Reichle's magnitude, and frees the residual to learn
     real per-word patterns.

     The unclipped surprisal is still returned in result['word_surprisal']
     for logging and surprisal-decomposition analysis.

Everything else is identical to v2:
  - epsilon = 1 + softplus(raw) constraint
  - small-init residual head (std=0.01)
  - explicit M1 / M2 / I motor stages
  - refixation gate via word length
  - constant pF regression
  - parallel parafoveal skip head with detach inside TRT

Class name is still NeuralEZReaderReichle so the interface is unchanged.
v3 checkpoints are NOT compatible with v1 or v2 (the state dict structure
is the same as v2, but the value distributions will differ).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


SURPRISAL_CLIP = 12.0  # nats; caps alpha3 * surprisal contribution near Reichle's range.


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


def _logit(y: float) -> float:
    return math.log(y / (1.0 - y))


# --------------------------------------------------------------------------- #
#  Reichle EZ Reader cascade
# --------------------------------------------------------------------------- #

class ReichleEZReader(nn.Module):
    """
    Maps (base_L1, L2, skip_prob, word_lengths) to FFD / Gaze / TRT via the
    Reichle cascade. All cognitive constants are exposed as learnable
    parameters so they can be reported and compared to the literature.
    """

    L1_MIN = 30.0   # v3: lowered from 60 to free up FFD variance for short-fixation words
    L1_MAX = 500.0

    def __init__(self):
        super().__init__()

        # Eccentricity: epsilon = 1 + softplus(raw) is always > 1.
        self._epsilon_raw = nn.Parameter(torch.tensor(_inv_softplus(0.15)))

        # Motor stages in ms.
        self._M1_raw = nn.Parameter(torch.tensor(_inv_softplus(125.0)))
        self._M2_raw = nn.Parameter(torch.tensor(_inv_softplus(25.0)))
        self._I_raw = nn.Parameter(torch.tensor(_inv_softplus(25.0)))

        # Refixation gate driven by word length (proxy for saccadic landing error).
        self.lambda_refix = nn.Parameter(torch.tensor(0.4))
        self.refix_pivot = nn.Parameter(torch.tensor(8.0))

        # Integration failure probability (Reichle 2009: pF ~ 0.01 constant).
        self._pF_raw = nn.Parameter(torch.tensor(_logit(0.01)))

        # Regression cost multiplier on previous word's gaze.
        self._reg_weight_raw = nn.Parameter(torch.tensor(_inv_softplus(0.5)))

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

    def forward(self, base_L1, L2, skip_prob, word_lengths):
        ecc_exponent = (word_lengths - 1.0) / 2.0
        L1 = base_L1 * torch.pow(self.epsilon, ecc_exponent)
        L1 = L1.clamp(min=self.L1_MIN, max=self.L1_MAX)

        first_fixation = L1 + self.M1 + self.M2

        refix_prob = torch.sigmoid(
            self.lambda_refix * (word_lengths - self.refix_pivot)
        )

        refix_duration = L2 + self.M1 + self.M2
        gaze_duration = first_fixation + refix_prob * refix_duration

        prev_gaze = torch.zeros_like(gaze_duration)
        prev_gaze[:, 1:] = gaze_duration[:, :-1]
        regression_cost = self.pF * self.reg_weight * prev_gaze

        conditional_trt = gaze_duration + self.I + regression_cost

        total_reading_time = (1.0 - skip_prob.detach()) * conditional_trt

        return {
            'first_fixation': first_fixation,
            'gaze_duration': gaze_duration,
            'conditional_trt': conditional_trt,
            'total_reading_time': total_reading_time,
            'skip_prob': skip_prob,
            'L1': L1,
            'L2': L2,
            'refix_prob': refix_prob,
            'epsilon': self.epsilon.detach(),
            'M1': self.M1.detach(),
            'M2': self.M2.detach(),
            'I': self.I.detach(),
            'pF': self.pF.detach(),
            'reg_weight': self.reg_weight.detach(),
            'lambda_refix': self.lambda_refix.detach(),
            'refix_pivot': self.refix_pivot.detach(),
        }


# --------------------------------------------------------------------------- #
#  Neural EZ Reader v3
# --------------------------------------------------------------------------- #

class NeuralEZReaderReichle(nn.Module):
    """
    word tokens -> LLaMA (causal LM)
        -> per-word surprisal from next-token log probabilities
        -> per-word hidden representation from last-subword pooling
        -> base_L1 = alpha1 - alpha2 * ln(freq)
                             + alpha3 * clip(surprisal, max=SURPRISAL_CLIP)
                             + residual
        -> L2 = delta * base_L1
        -> ReichleEZReader cascade (L1 clamp min=30 ms)
        -> FFD, Gaze, TRT, skip

    Learnable:
        Neural:    LLaMA top layers, projection, residual head, skip head
        Cognitive: alpha1, alpha2, alpha3, delta, epsilon, M1, M2, I,
                   pF, reg_weight, lambda_refix, refix_pivot
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B",
        freeze_layers: int = 12,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.llama = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.llama.config.pad_token_id = self.tokenizer.eos_token_id

        llama_dim = self.llama.config.hidden_size
        self.llama_dim = llama_dim

        if freeze_layers > 0:
            for p in self.llama.get_input_embeddings().parameters():
                p.requires_grad = False
            base = self.llama.model
            for layer_idx in range(min(freeze_layers, len(base.layers))):
                for p in base.layers[layer_idx].parameters():
                    p.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(llama_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Residual L1 correction. Small-init so gradient flows to the
        # projection from the first batch (see v2 docstring).
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.normal_(self.residual_head[-1].weight, std=0.01)
        nn.init.zeros_(self.residual_head[-1].bias)

        self.skip_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Reichle alpha coefficients at 2003 literature init.
        # alpha3 is ms/nat (surprisal) rather than Reichle's ms/cloze.
        self.alpha1 = nn.Parameter(torch.tensor(104.0))
        self.alpha2 = nn.Parameter(torch.tensor(3.5))
        self.alpha3 = nn.Parameter(torch.tensor(4.0))

        self._delta_raw = nn.Parameter(torch.tensor(_logit(0.34)))

        self.ezreader = ReichleEZReader()

    @property
    def delta(self):
        return torch.sigmoid(self._delta_raw)

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

    def _pool_and_surprisal(
        self, logits, hidden, input_ids, batch_word_maps, max_words
    ):
        device = hidden.device
        B, T, D = hidden.shape

        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        targets = input_ids[:, 1:].unsqueeze(-1)
        token_log_probs = log_probs.gather(-1, targets).squeeze(-1)
        token_surprisal = -token_log_probs  # (B, T - 1), nats.

        word_surprisal = torch.zeros(B, max_words, device=device)
        word_hidden = torch.zeros(B, max_words, D, device=device)

        for b in range(B):
            for w_idx, (start, end) in enumerate(batch_word_maps[b]):
                word_hidden[b, w_idx] = hidden[b, end - 1]
                total = word_hidden.new_zeros(())
                for sub in range(start, end):
                    if 1 <= sub <= token_surprisal.size(1):
                        total = total + token_surprisal[b, sub - 1]
                word_surprisal[b, w_idx] = total

        return word_surprisal, word_hidden

    def forward(self, word_lists, frequencies, word_lengths):
        device = frequencies.device

        input_ids, attention_mask, word_maps, max_words = self._tokenize_and_align(
            word_lists, device
        )

        outputs = self.llama(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        logits = outputs.logits
        hidden = outputs.hidden_states[-1]

        word_surprisal, word_hidden = self._pool_and_surprisal(
            logits, hidden, input_ids, word_maps, max_words
        )

        projected = self.projection(word_hidden)
        residual = self.residual_head(projected).squeeze(-1)
        skip_prob = self.skip_head(projected).squeeze(-1)

        seq_len = frequencies.size(1)
        word_surprisal = word_surprisal[:, :seq_len]
        residual = residual[:, :seq_len]
        skip_prob = skip_prob[:, :seq_len]

        # v3: clip surprisal before feeding into the formula so alpha3's
        # per-word contribution stays comparable to Reichle's cloze range.
        # The unclipped version is still returned for logging below.
        surprisal_for_formula = word_surprisal.clamp(max=SURPRISAL_CLIP)

        log_freq = torch.log(frequencies.clamp(min=1.0))
        base_L1_formula = (
            self.alpha1
            - self.alpha2 * log_freq
            + self.alpha3 * surprisal_for_formula
        )
        base_L1 = base_L1_formula + residual
        base_L1 = base_L1.clamp(min=1.0)

        L2 = self.delta * base_L1

        result = self.ezreader(
            base_L1=base_L1,
            L2=L2,
            skip_prob=skip_prob,
            word_lengths=word_lengths,
        )

        result['base_L1'] = base_L1
        result['base_L1_formula'] = base_L1_formula
        result['residual'] = residual
        result['word_surprisal'] = word_surprisal           # unclipped, for logging
        result['surprisal_clipped'] = surprisal_for_formula  # what actually entered L1
        result['alpha1'] = self.alpha1.detach()
        result['alpha2'] = self.alpha2.detach()
        result['alpha3'] = self.alpha3.detach()
        result['delta'] = self.delta.detach()

        return result
