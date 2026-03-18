"""
Sequential EZ Reader: a model that reads step-by-step with a visual span.

Unlike the v2 model (which processes the full sentence in one pass and applies
EZ Reader parameters on top), this model simulates actual reading:

  1. Fixate on a word
  2. See nearby words through a Gaussian visual span (degraded by distance)
  3. Compute L1/L2 processing times for the fixated word
  4. Predict where to look next (saccade target)
  5. Move eyes and repeat

The model maintains a reading state (GRU) that accumulates information across
fixations, like a human reader building sentence comprehension over time.

Training uses teacher forcing on human scanpaths.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizerFast


class SequentialEZReader(nn.Module):

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        hidden_dim: int = 256,
        sigma_init: float = 4.0,
        freeze_bert_layers: int = 8,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # --- BERT encoder (run once per sentence) ---
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
        bert_dim = self.bert.config.hidden_size

        if freeze_bert_layers > 0:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            for i in range(min(freeze_bert_layers, len(self.bert.encoder.layer))):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = False

        # --- Project BERT dim -> hidden dim ---
        self.projection = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # --- Reader GRU: tracks accumulated reading state ---
        self.reader_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # --- Duration prediction heads ---
        self.l1_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )
        self.l2_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )

        # --- Saccade prediction (attention over word positions) ---
        self.saccade_query = nn.Linear(hidden_dim, hidden_dim)
        self.saccade_key = nn.Linear(hidden_dim, hidden_dim)

        # --- EZ Reader parameters ---
        self.l1_scale = nn.Parameter(torch.tensor(120.0))   # higher init to match human FFD ~210ms
        self.l2_scale = nn.Parameter(torch.tensor(60.0))
        self.eccentricity = nn.Parameter(torch.tensor(0.1))
        self.l2_contribution = nn.Parameter(torch.tensor(0.3))

        # --- Visual span (learnable sigma) ---
        self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma_init)))

    # ------------------------------------------------------------------ #
    #  Tokenization and subword -> word pooling (same as model_bert.py)
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
                    spans.append((1, 2))
            batch_word_maps.append(spans)
            max_words = max(max_words, n_words)

        return input_ids, attention_mask, batch_word_maps, max_words

    def _pool_subwords_to_words(self, bert_output, batch_word_maps, max_words, device):
        batch_size = bert_output.size(0)
        bert_dim = bert_output.size(2)

        idx = torch.zeros(batch_size, max_words, dtype=torch.long)
        for b in range(batch_size):
            for w_idx, (start, end) in enumerate(batch_word_maps[b]):
                idx[b, w_idx] = start
        idx = idx.to(device)

        word_repr = torch.gather(
            bert_output, 1, idx.unsqueeze(-1).expand(-1, -1, bert_dim)
        )
        return word_repr

    # ------------------------------------------------------------------ #
    #  Get word embeddings (run BERT once)
    # ------------------------------------------------------------------ #

    def get_word_embeddings(self, word_lists, device):
        """Run BERT once, return projected word-level embeddings."""
        input_ids, attention_mask, word_maps, max_words = self._tokenize_and_align(
            word_lists, device
        )
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

        word_repr = self._pool_subwords_to_words(bert_out, word_maps, max_words, device)
        projected = self.projection(word_repr)
        return projected  # (B, T, hidden_dim)

    # ------------------------------------------------------------------ #
    #  Visual span mask
    # ------------------------------------------------------------------ #

    def visual_mask(self, fixation_pos, num_words, device):
        """
        Gaussian mask centered on fixation position.

        Args:
            fixation_pos: (B,) float tensor, current fixation word index
            num_words: int, sentence length
            device: torch device

        Returns:
            mask: (B, T) attention weights based on distance from fixation
        """
        sigma = self.log_sigma.exp()
        positions = torch.arange(num_words, device=device, dtype=torch.float32)
        # (B, T): distance from fixation for each word
        dist_sq = (positions.unsqueeze(0) - fixation_pos.unsqueeze(1)) ** 2
        mask = torch.exp(-dist_sq / (2 * sigma ** 2))
        return mask

    # ------------------------------------------------------------------ #
    #  Single fixation step
    # ------------------------------------------------------------------ #

    def fixation_step(self, word_embeddings, word_lengths, fixation_pos, reader_state, word_mask=None):
        """
        Process one fixation.

        Args:
            word_embeddings: (B, T, D) projected word embeddings
            word_lengths:    (B, T) character counts per word
            fixation_pos:    (B,) current fixation position (float, 0-indexed)
            reader_state:    (1, B, D) GRU hidden state
            word_mask:       (B, T) 1.0 for valid words, 0.0 for padding

        Returns:
            ffd:             (B,) predicted first fixation duration
            L1:              (B,) predicted L1 (before eccentricity)
            L2:              (B,) predicted L2
            saccade_logits:  (B, T) logits over next fixation positions
            new_state:       (1, B, D) updated GRU state
        """
        B, T, D = word_embeddings.shape
        device = word_embeddings.device

        # 1. Visual span mask
        vmask = self.visual_mask(fixation_pos, T, device)  # (B, T)
        if word_mask is not None:
            vmask = vmask * word_mask

        # 2. Weighted sum of embeddings (soft foveal attention)
        vmask_norm = vmask / (vmask.sum(dim=-1, keepdim=True) + 1e-8)
        fixation_repr = (word_embeddings * vmask_norm.unsqueeze(-1)).sum(dim=1)  # (B, D)

        # 3. Update reading state
        output, new_state = self.reader_gru(fixation_repr.unsqueeze(1), reader_state)
        output = output.squeeze(1)  # (B, D)

        # 4. Predict L1, L2
        L1 = self.l1_head(output).squeeze(-1) * self.l1_scale  # (B,)
        L2 = self.l2_head(output).squeeze(-1) * self.l2_scale  # (B,)
        L1 = L1.clamp(min=1.0, max=500.0)
        L2 = L2.clamp(min=1.0, max=500.0)

        # 5. Eccentricity scaling based on fixated word length
        fix_idx = fixation_pos.long().clamp(0, T - 1)
        fix_word_len = torch.gather(word_lengths, 1, fix_idx.unsqueeze(1)).squeeze(1)
        ecc_scale = 1.0 + self.eccentricity * (fix_word_len - 4.0).clamp(min=0)
        L1_scaled = L1 * ecc_scale

        # 6. FFD = L1 (eccentricity-scaled) + fraction of L2
        ffd = L1_scaled + F.softplus(self.l2_contribution) * L2

        # 7. Saccade prediction (scaled dot-product attention)
        query = self.saccade_query(output)       # (B, D)
        keys = self.saccade_key(word_embeddings)  # (B, T, D)
        saccade_logits = torch.bmm(
            query.unsqueeze(1), keys.transpose(1, 2)
        ).squeeze(1) / math.sqrt(D)  # (B, T)

        # 8. Forward bias: add a gentle prior that favors positions ahead of current fixation
        #    Human saccades are 99.7% forward (mean +1.7 words)
        positions = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0)  # (1, T)
        offset = positions - fixation_pos.unsqueeze(1)  # (B, T) signed distance from current
        # Favor +1 to +3 words ahead, penalize backward and current position
        forward_bias = -0.5 * (offset - 1.7) ** 2 / (2.0 ** 2)  # Gaussian centered at +1.7
        saccade_logits = saccade_logits + forward_bias

        # Mask padding positions
        if word_mask is not None:
            saccade_logits = saccade_logits.masked_fill(word_mask == 0, float('-inf'))

        return ffd, L1, L2, saccade_logits, new_state

    # ------------------------------------------------------------------ #
    #  Forward: teacher forcing (training)
    # ------------------------------------------------------------------ #

    def forward_teacher_forcing(
        self,
        word_lists,
        word_lengths,
        fix_positions,
        fix_durations,
        fix_mask,
        saccade_targets,
        saccade_mask,
    ):
        """
        Forward pass with teacher forcing on human scanpaths.

        Args:
            word_lists:      list of list of str
            word_lengths:    (B, T) character counts
            fix_positions:   (B, max_fix) word index at each fixation step
            fix_durations:   (B, max_fix) human FFD at each fixation
            fix_mask:        (B, max_fix) 1.0 for valid fixation steps
            saccade_targets: (B, max_fix) next fixation position (target for saccade)
            saccade_mask:    (B, max_fix) 1.0 for steps with valid saccade target

        Returns:
            dict with losses and predictions
        """
        device = word_lengths.device
        B = word_lengths.size(0)
        T = word_lengths.size(1)
        max_fix = fix_positions.size(1)

        # Word-level mask (1 for valid words, 0 for padding)
        word_mask = (word_lengths > 0).float()

        # Get word embeddings (BERT runs once)
        with torch.amp.autocast("cuda", enabled=False):
            word_embeddings = self.get_word_embeddings(word_lists, device)
        word_embeddings = word_embeddings[:, :T, :]

        # Initialize reader state
        reader_state = torch.zeros(1, B, self.hidden_dim, device=device)

        # Step through fixations with teacher forcing
        all_ffd_pred = []
        all_saccade_logits = []
        all_L1 = []
        all_L2 = []

        for step in range(max_fix):
            fix_pos = fix_positions[:, step].float()

            ffd, L1, L2, saccade_logits, reader_state = self.fixation_step(
                word_embeddings, word_lengths, fix_pos, reader_state, word_mask
            )

            all_ffd_pred.append(ffd)
            all_saccade_logits.append(saccade_logits)
            all_L1.append(L1)
            all_L2.append(L2)

        pred_ffd = torch.stack(all_ffd_pred, dim=1)           # (B, max_fix)
        pred_saccade = torch.stack(all_saccade_logits, dim=1)  # (B, max_fix, T)
        pred_L1 = torch.stack(all_L1, dim=1)                   # (B, max_fix)
        pred_L2 = torch.stack(all_L2, dim=1)                   # (B, max_fix)

        # Duration loss (MSE on FFD, masked)
        dur_error = (pred_ffd - fix_durations) ** 2 * fix_mask
        duration_loss = dur_error.sum() / fix_mask.sum().clamp(min=1)

        # Saccade loss (cross-entropy, masked)
        # Reshape for cross_entropy: (B*max_fix, T) vs (B*max_fix,)
        sac_logits_flat = pred_saccade.view(-1, T)
        sac_targets_flat = saccade_targets.view(-1)
        sac_mask_flat = saccade_mask.view(-1)

        sac_loss_per_step = F.cross_entropy(
            sac_logits_flat, sac_targets_flat, reduction='none'
        )
        saccade_loss = (sac_loss_per_step * sac_mask_flat).sum() / sac_mask_flat.sum().clamp(min=1)

        return {
            'duration_loss': duration_loss,
            'saccade_loss': saccade_loss,
            'pred_ffd': pred_ffd,
            'pred_saccade': pred_saccade,
            'L1': pred_L1,
            'L2': pred_L2,
        }

    # ------------------------------------------------------------------ #
    #  Forward: free-running inference
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def forward_free(self, word_lists, word_lengths, max_fixations=50):
        """
        Free-running inference: model picks its own fixation targets.

        Prevents getting stuck by suppressing the current fixation position
        in saccade logits (forces the model to move).

        Returns per-word FFD and total fixation time (approximate TRT).
        """
        device = word_lengths.device
        B = word_lengths.size(0)
        T = word_lengths.size(1)

        word_mask = (word_lengths > 0).float()
        word_embeddings = self.get_word_embeddings(word_lists, device)
        word_embeddings = word_embeddings[:, :T, :]

        reader_state = torch.zeros(1, B, self.hidden_dim, device=device)

        # Per-word accumulators
        per_word_first_dur = torch.zeros(B, T, device=device)
        per_word_total_dur = torch.zeros(B, T, device=device)
        per_word_fixcount = torch.zeros(B, T, device=device)
        per_word_skipped = torch.ones(B, T, device=device)  # assume skipped until fixated

        # Track fixation sequence for analysis
        scanpath_positions = []
        scanpath_durations = []

        fix_pos = torch.zeros(B, device=device)  # start at word 0
        active = torch.ones(B, dtype=torch.bool, device=device)  # still reading

        # Compute sentence lengths per batch item
        sent_lengths = word_mask.sum(dim=1)  # (B,)

        for step in range(max_fixations):
            if not active.any():
                break

            ffd, L1, L2, saccade_logits, reader_state = self.fixation_step(
                word_embeddings, word_lengths, fix_pos, reader_state, word_mask
            )

            # Record fixation
            fix_idx = fix_pos.long().clamp(0, T - 1)
            for b in range(B):
                if active[b]:
                    idx = fix_idx[b].item()
                    if per_word_fixcount[b, idx] == 0:
                        per_word_first_dur[b, idx] = ffd[b]
                    per_word_total_dur[b, idx] += ffd[b]
                    per_word_fixcount[b, idx] += 1
                    per_word_skipped[b, idx] = 0.0

            scanpath_positions.append(fix_idx.clone())
            scanpath_durations.append(ffd.clone())

            # Suppress current position to prevent getting stuck
            suppressed_logits = saccade_logits.clone()
            for b in range(B):
                suppressed_logits[b, fix_idx[b]] = float('-inf')

            # Pick next position (greedy argmax on suppressed logits)
            next_pos = suppressed_logits.argmax(dim=-1).float()

            # Deactivate if moved past sentence end or all logits are -inf
            for b in range(B):
                if next_pos[b] >= sent_lengths[b] or suppressed_logits[b].max() == float('-inf'):
                    active[b] = False

            fix_pos = next_pos.clamp(0, T - 1)

        return {
            'first_fixation': per_word_first_dur,
            'total_reading_time': per_word_total_dur,
            'fixation_count': per_word_fixcount,
            'skip_prob': per_word_skipped,  # 1 = skipped, 0 = fixated
            'scanpath_positions': torch.stack(scanpath_positions, dim=1) if scanpath_positions else None,
            'scanpath_durations': torch.stack(scanpath_durations, dim=1) if scanpath_durations else None,
        }
