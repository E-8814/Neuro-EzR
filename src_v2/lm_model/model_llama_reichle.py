"""
Neural EZ Reader — Reichle-faithful variant with hybrid formula + LLM residual.

Goal: stay as close as possible to Reichle, Rayner & Pollatsek (2003) and
Reichle, Warren & McConnell (2009) while using a pretrained causal LM to
provide the predictability signal.

Cognitive cascade:

    base_L1  =  alpha1 - alpha2 * ln(freq) + alpha3 * surprisal_LLM + residual_LLM
                    (sign on alpha3 is flipped relative to Reichle because
                     surprisal = -log(cloze); the residual is zero-initialized
                     so the model starts at the pure Reichle formula)

    L1       =  base_L1 * epsilon^((wordlen - 1) / 2)
                    (eccentricity exponent; wordlen is a proxy for the
                     mean absolute character distance used by Reichle; we
                     do not have saccadic landing sites in a batched
                     word-level setting)

    L2       =  delta * base_L1
                    (Reichle's L2 uses the pre-eccentricity base L1)

    FFD      =  L1 + M1 + M2
                    (first fixation = L1 processing + labile + non-labile
                     saccade programming)

    refix    =  sigmoid(lambda_refix * (wordlen - refix_pivot))
                    (proxy for saccadic landing error; longer word -> more
                     likely to land off the optimal viewing position and
                     trigger a corrective refixation)

    Gaze     =  FFD + refix * (L2 + M1 + M2)
                    (refixation continues lexical access then programs the
                     saccade off the word)

    TRT      =  Gaze + I + pF * reg_weight * prev_gaze
                    (integration time I is a per-word constant; pF is the
                     constant integration-failure probability from Reichle
                     2009; regression cost is approximated against the
                     previous word's gaze)

Reportable cognitive parameters (all learnable, all init at literature values):
    alpha1, alpha2, alpha3, delta, epsilon, M1, M2, I,
    pF, reg_weight, lambda_refix, refix_pivot

Skip is treated as a parallel parafoveal process implemented as a learned
head over LLaMA representations. Its gradient is detached inside TRT to keep
the L1->L2 cascade training signal independent of the skip training signal.

Required simplifications in the batch-parallel, word-level setting:
    - No saccadic landing site  -> refixation is driven by word length only.
    - No actual regression trajectory -> regression cost is a learned scalar
      times the previous word's gaze.
    - Causal attention only -> no parafoveal preview across the word.
    - Integration time is per-word constant; no sentence-level state.

Inputs (changed from model_llama_faithful_sh.py):
    word_lists:    list of list of str
    frequencies:   (batch, seq_len) float tensor of raw word frequencies
                   (e.g. SUBTLEX counts). These MUST be passed in by the
                   caller. Missing / OOV words should get a small default
                   like 1.0.
    word_lengths:  (batch, seq_len) float tensor of character counts per word.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def _inv_softplus(y: float) -> float:
    # Inverse of softplus so that softplus(_inv_softplus(y)) == y.
    return math.log(math.expm1(y))


def _logit(y: float) -> float:
    return math.log(y / (1.0 - y))


# --------------------------------------------------------------------------- #
#  Reichle EZ Reader cascade — all cognitive constants are learnable
# --------------------------------------------------------------------------- #

class ReichleEZReader(nn.Module):
    """
    Maps (base_L1, L2, skip_prob, word_lengths) to FFD / Gaze / TRT via the
    Reichle cascade. Everything that is a fixed constant in the published
    model is exposed here as a learnable parameter so it can be reported
    and compared to the literature.
    """

    def __init__(self):
        super().__init__()

        # Eccentricity exponent base; Reichle default 1.15.
        self.epsilon = nn.Parameter(torch.tensor(1.15))

        # Motor stages, all in ms. softplus keeps them positive.
        self._M1_raw = nn.Parameter(torch.tensor(_inv_softplus(125.0)))
        self._M2_raw = nn.Parameter(torch.tensor(_inv_softplus(25.0)))
        self._I_raw = nn.Parameter(torch.tensor(_inv_softplus(25.0)))

        # Refixation gate driven by word length (proxy for saccadic landing error).
        # sigmoid(lambda_refix * (wordlen - refix_pivot)):
        #   wordlen = pivot -> 50% refix rate, shorter words less, longer more.
        self.lambda_refix = nn.Parameter(torch.tensor(0.4))
        self.refix_pivot = nn.Parameter(torch.tensor(8.0))

        # Integration failure probability (Reichle 2009: pF ~ 0.01 constant).
        self._pF_raw = nn.Parameter(torch.tensor(_logit(0.01)))

        # Regression cost multiplier (applied to previous word's gaze).
        self._reg_weight_raw = nn.Parameter(torch.tensor(_inv_softplus(0.5)))

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
        """
        Args:
            base_L1:      (B, S) pre-eccentricity L1 (ms).
            L2:           (B, S) lexical access time (ms). Reichle: delta * base_L1.
            skip_prob:    (B, S) skip probability from the parallel skip head.
            word_lengths: (B, S) character count per word, used for eccentricity
                          and as a proxy for saccadic landing error.

        Returns dict with FFD / Gaze / TRT / skip plus intermediate values.
        """
        # Eccentricity: apply epsilon^((len - 1) / 2) only to L1, not L2.
        # Reichle's formula uses the mean absolute distance from fixation to
        # each letter; with no launch site available we use (len - 1) / 2,
        # which is the exponent you get assuming fixation at the first letter.
        ecc_exponent = (word_lengths - 1.0) / 2.0
        L1 = base_L1 * torch.pow(self.epsilon, ecc_exponent)
        L1 = L1.clamp(min=60.0, max=500.0)

        # First fixation = L1 processing + saccade programming.
        first_fixation = L1 + self.M1 + self.M2

        # Refixation driven by word length (stand-in for saccadic landing error).
        refix_prob = torch.sigmoid(
            self.lambda_refix * (word_lengths - self.refix_pivot)
        )

        # Gaze = FFD + optional refixation (second fixation finishes L2 + motor).
        refix_duration = L2 + self.M1 + self.M2
        gaze_duration = first_fixation + refix_prob * refix_duration

        # Regression cost: constant pF (Reichle 2009) times previous word gaze.
        # This replaces the data-dependent sigmoid-on-L2 from diff_ezreader.
        prev_gaze = torch.zeros_like(gaze_duration)
        prev_gaze[:, 1:] = gaze_duration[:, :-1]
        regression_cost = self.pF * self.reg_weight * prev_gaze

        # TRT conditional on fixation: gaze + integration + expected regression.
        conditional_trt = gaze_duration + self.I + regression_cost

        # Detach skip inside TRT so the L1/L2 cascade trains independently of
        # the skip head, preserving the "parallel parafoveal process" structure.
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
#  Neural EZ Reader with hybrid Reichle formula + LLM residual
# --------------------------------------------------------------------------- #

class NeuralEZReaderReichle(nn.Module):
    """
    word tokens -> LLaMA (causal LM)
        -> per-word surprisal from next-token log probabilities
        -> per-word hidden representation from last-subword pooling
        -> base_L1 = alpha1 - alpha2 * ln(freq) + alpha3 * surprisal + residual
        -> L2 = delta * base_L1
        -> ReichleEZReader cascade
        -> FFD, Gaze, TRT, skip

    The residual head is zero-initialized, so at t=0 the model is literally
    the Reichle formula with alpha values at their published init. The
    residual grows only if the LLM representation carries information
    beyond what freq and surprisal capture; otherwise the model stays pure.

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

        # Causal LM so we can get both logits (for surprisal) and hidden states.
        self.llama = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.llama.config.pad_token_id = self.tokenizer.eos_token_id

        llama_dim = self.llama.config.hidden_size
        self.llama_dim = llama_dim

        # Freeze lower layers. The LLaMA base model lives at self.llama.model
        # for a LlamaForCausalLM wrapper.
        if freeze_layers > 0:
            for p in self.llama.get_input_embeddings().parameters():
                p.requires_grad = False
            base = self.llama.model
            for layer_idx in range(min(freeze_layers, len(base.layers))):
                for p in base.layers[layer_idx].parameters():
                    p.requires_grad = False

        # Projection applied to pooled word hidden states for both heads.
        self.projection = nn.Sequential(
            nn.Linear(llama_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Residual L1 correction. Zero-initialized final layer so the model
        # starts at the pure Reichle formula and only drifts from it if the
        # data require LLM-specific structure beyond surprisal + frequency.
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

        # Skip head: learned parallel parafoveal process.
        self.skip_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Reichle alpha coefficients. Init at 2003 literature values for
        # alpha1 and alpha2. alpha3 is smaller than Reichle's 39 because
        # our predictability signal is surprisal in nats rather than cloze
        # in [0, 1], so the per-unit contribution is different.
        self.alpha1 = nn.Parameter(torch.tensor(104.0))
        self.alpha2 = nn.Parameter(torch.tensor(3.5))
        self.alpha3 = nn.Parameter(torch.tensor(4.0))

        # L2/L1 ratio, sigmoid-constrained to (0, 1), init at Reichle's 0.34.
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
        """
        In one pass over the word map:
          - pool hidden states by taking the last subword of each word,
          - compute per-word surprisal by summing next-token surprisals
            across all subwords of that word (standard in RT prediction).
        """
        device = hidden.device
        B, T, D = hidden.shape

        # log P(input_ids[:, t+1] | input_ids[:, :t+1])
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        targets = input_ids[:, 1:].unsqueeze(-1)
        token_log_probs = log_probs.gather(-1, targets).squeeze(-1)
        token_surprisal = -token_log_probs  # shape (B, T - 1), in nats.

        word_surprisal = torch.zeros(B, max_words, device=device)
        word_hidden = torch.zeros(B, max_words, D, device=device)

        for b in range(B):
            for w_idx, (start, end) in enumerate(batch_word_maps[b]):
                word_hidden[b, w_idx] = hidden[b, end - 1]
                total = word_hidden.new_zeros(())
                for sub in range(start, end):
                    # token_surprisal[b, k] covers predicting input_ids[b, k+1],
                    # so the surprisal of the subword at position `sub` is
                    # token_surprisal[b, sub - 1] for sub >= 1.
                    if 1 <= sub <= token_surprisal.size(1):
                        total = total + token_surprisal[b, sub - 1]
                word_surprisal[b, w_idx] = total

        return word_surprisal, word_hidden

    def forward(self, word_lists, frequencies, word_lengths):
        """
        Args:
            word_lists:   list of list of str
            frequencies:  (B, S) raw word frequencies (e.g. SUBTLEX counts,
                          fallback to 1.0 for OOV).
            word_lengths: (B, S) character count per word.

        Returns dict with FFD / Gaze / TRT / skip plus all cognitive
        parameters and intermediates for logging and analysis.
        """
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

        # Trim to the provided sequence length (frequencies / word_lengths
        # may be shorter than max_words if the tokenizer padded the batch
        # to accommodate longer sentences than this one).
        seq_len = frequencies.size(1)
        word_surprisal = word_surprisal[:, :seq_len]
        residual = residual[:, :seq_len]
        skip_prob = skip_prob[:, :seq_len]

        # Reichle L1 formula with LLM surprisal as the predictability signal.
        # Sign on alpha3 is + because surprisal = -log(cloze); high surprisal
        # is low predictability, which in Reichle's formula corresponds to
        # -alpha3 * cloze with cloze small, i.e. a positive contribution to L1.
        log_freq = torch.log(frequencies.clamp(min=1.0))
        base_L1_formula = (
            self.alpha1
            - self.alpha2 * log_freq
            + self.alpha3 * word_surprisal
        )
        base_L1 = base_L1_formula + residual
        base_L1 = base_L1.clamp(min=1.0)

        # Reichle's L2 uses the pre-eccentricity base L1, not the scaled L1.
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
        result['word_surprisal'] = word_surprisal
        result['alpha1'] = self.alpha1.detach()
        result['alpha2'] = self.alpha2.detach()
        result['alpha3'] = self.alpha3.detach()
        result['delta'] = self.delta.detach()

        return result
