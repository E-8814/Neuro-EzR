"""
Sequential EZ Reader with LLaMA 3.2-1B backbone (v2).

Same architecture as sequential_reader.py but uses a causal LM instead of BERT:
  - LLaMA is cognitively more plausible (left-to-right, like human reading)
  - Uses last-subword pooling (causal models accumulate context left-to-right)
  - Much larger model (1.1B params) so we freeze more layers

v2 improvements:
  - Duration heads receive explicit word-level features (word_length, log_freq, position, visited)
  - Log-space duration loss for better relative error weighting
  - Stop head with reading progress features (position/length, step/expected)
  - exp() activation for duration heads (wider dynamic range than Softplus)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class SequentialEZReaderLLaMA(nn.Module):

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        hidden_dim: int = 256,
        sigma_init: float = 4.0,
        freeze_layers: int = 14,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # --- LLaMA encoder (run once per sentence) ---
        self.llama = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # LLaMA tokenizers often lack a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.llama.config.pad_token_id = self.tokenizer.eos_token_id

        llama_dim = self.llama.config.hidden_size  # 2048 for LLaMA-3.2-1B

        # Freeze lower layers
        if freeze_layers > 0:
            for param in self.llama.embed_tokens.parameters():
                param.requires_grad = False
            for layer_idx in range(min(freeze_layers, len(self.llama.layers))):
                for param in self.llama.layers[layer_idx].parameters():
                    param.requires_grad = False

        # --- Project LLaMA dim -> hidden dim ---
        self.projection = nn.Sequential(
            nn.Linear(llama_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # --- Reader GRU: tracks accumulated reading state ---
        self.reader_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # --- Duration prediction heads ---
        # Input: GRU output (hidden_dim) + word_length (1) + log_freq (1)
        #        + word_position (1) + previously_visited (1)
        dur_input_dim = hidden_dim + 4
        self.l1_head = nn.Sequential(
            nn.Linear(dur_input_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            # No final activation -- we apply exp() manually for wider dynamic range
        )
        self.l2_head = nn.Sequential(
            nn.Linear(dur_input_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

        # --- Saccade prediction (attention over word positions) ---
        self.saccade_query = nn.Linear(hidden_dim, hidden_dim)
        self.saccade_key = nn.Linear(hidden_dim, hidden_dim)

        # --- Stop head: predicts when reading is done ---
        # Input: GRU output (hidden_dim) + reading_progress (1) + step_progress (1)
        self.stop_head = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        # --- EZ Reader parameters ---
        self.l1_scale = nn.Parameter(torch.tensor(120.0))
        self.l2_scale = nn.Parameter(torch.tensor(60.0))
        self.eccentricity = nn.Parameter(torch.tensor(0.1))
        self.l2_contribution = nn.Parameter(torch.tensor(0.3))

        # --- Visual span (learnable sigma) ---
        self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma_init)))

    # ------------------------------------------------------------------ #
    #  Tokenization and subword -> word pooling (last-subword for causal)
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
                    spans.append((0, 1))  # fallback to first token
            batch_word_maps.append(spans)
            max_words = max(max_words, n_words)

        return input_ids, attention_mask, batch_word_maps, max_words

    def _pool_subwords_to_words(self, hidden_states, batch_word_maps, max_words, device):
        """Pool subword representations to word-level using last-subword strategy.

        For causal models, the last subword token has the most context
        (it has attended to all previous subwords of the same word).
        """
        batch_size = hidden_states.size(0)
        hidden_dim = hidden_states.size(2)

        idx = torch.zeros(batch_size, max_words, dtype=torch.long)
        for b in range(batch_size):
            for w_idx, (start, end) in enumerate(batch_word_maps[b]):
                idx[b, w_idx] = end - 1  # last subword token
        idx = idx.to(device)

        word_repr = torch.gather(
            hidden_states, 1, idx.unsqueeze(-1).expand(-1, -1, hidden_dim)
        )
        return word_repr

    # ------------------------------------------------------------------ #
    #  Get word embeddings (run LLaMA once)
    # ------------------------------------------------------------------ #

    def get_word_embeddings(self, word_lists, device):
        """Run LLaMA once, return projected word-level embeddings."""
        input_ids, attention_mask, word_maps, max_words = self._tokenize_and_align(
            word_lists, device
        )
        llama_out = self.llama(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

        word_repr = self._pool_subwords_to_words(llama_out, word_maps, max_words, device)
        projected = self.projection(word_repr)
        return projected  # (B, T, hidden_dim)

    # ------------------------------------------------------------------ #
    #  Visual span mask
    # ------------------------------------------------------------------ #

    def visual_mask(self, fixation_pos, num_words, device):
        sigma = self.log_sigma.exp()
        positions = torch.arange(num_words, device=device, dtype=torch.float32)
        dist_sq = (positions.unsqueeze(0) - fixation_pos.unsqueeze(1)) ** 2
        mask = torch.exp(-dist_sq / (2 * sigma ** 2))
        return mask

    # ------------------------------------------------------------------ #
    #  Single fixation step
    # ------------------------------------------------------------------ #

    def fixation_step(self, word_embeddings, word_lengths, word_log_freqs,
                      fixation_pos, reader_state, word_mask=None,
                      step_number=0, sentence_length=None, visited_mask=None):
        B, T, D = word_embeddings.shape
        device = word_embeddings.device

        if sentence_length is None:
            sentence_length = word_mask.sum(dim=1) if word_mask is not None else torch.full((B,), T, device=device)

        # 1. Visual span mask
        vmask = self.visual_mask(fixation_pos, T, device)
        if word_mask is not None:
            vmask = vmask * word_mask

        # 2. Weighted sum of embeddings (soft foveal attention)
        vmask_norm = vmask / (vmask.sum(dim=-1, keepdim=True) + 1e-8)
        fixation_repr = (word_embeddings * vmask_norm.unsqueeze(-1)).sum(dim=1)

        # 3. Update reading state
        output, new_state = self.reader_gru(fixation_repr.unsqueeze(1), reader_state)
        output = output.squeeze(1)

        # 4. Predict L1, L2 (with explicit word features + position + visited)
        fix_idx = fixation_pos.long().clamp(0, T - 1)
        fix_word_len = torch.gather(word_lengths, 1, fix_idx.unsqueeze(1)).squeeze(1)  # (B,)
        fix_log_freq = torch.gather(word_log_freqs, 1, fix_idx.unsqueeze(1)).squeeze(1)  # (B,)

        # Word position in sentence (0→1)
        word_position = fix_idx.float() / (sentence_length.float() - 1).clamp(min=1)  # (B,)

        # Whether the word was previously visited
        if visited_mask is not None:
            previously_visited = torch.gather(visited_mask.float(), 1, fix_idx.unsqueeze(1)).squeeze(1)
        else:
            previously_visited = torch.zeros(B, device=device)

        dur_features = torch.stack([
            fix_word_len / 10.0, fix_log_freq / 6.0,
            word_position, previously_visited
        ], dim=-1)  # (B, 4)
        dur_input = torch.cat([output, dur_features], dim=-1)  # (B, hidden_dim + 4)

        # exp() activation for wider dynamic range than Softplus
        L1 = self.l1_head(dur_input).squeeze(-1).exp() * self.l1_scale
        L2 = self.l2_head(dur_input).squeeze(-1).exp() * self.l2_scale
        L1 = L1.clamp(min=1.0, max=500.0)
        L2 = L2.clamp(min=1.0, max=500.0)

        # 5. Eccentricity scaling based on fixated word length
        ecc_scale = 1.0 + self.eccentricity * (fix_word_len - 4.0).clamp(min=0)
        L1_scaled = L1 * ecc_scale

        # 6. FFD = L1 (eccentricity-scaled) + fraction of L2
        ffd = L1_scaled + F.softplus(self.l2_contribution) * L2

        # 7. Saccade prediction (scaled dot-product attention)
        query = self.saccade_query(output)
        keys = self.saccade_key(word_embeddings)
        saccade_logits = torch.bmm(
            query.unsqueeze(1), keys.transpose(1, 2)
        ).squeeze(1) / math.sqrt(D)

        # 8. Forward bias
        positions = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0)
        offset = positions - fixation_pos.unsqueeze(1)
        forward_bias = -0.5 * (offset - 1.7) ** 2 / (2.0 ** 2)
        saccade_logits = saccade_logits + forward_bias

        # Mask padding positions
        if word_mask is not None:
            saccade_logits = saccade_logits.masked_fill(word_mask == 0, float('-inf'))

        # 9. Stop prediction with reading progress features
        reading_progress = fixation_pos / sentence_length.float().clamp(min=1)  # (B,)
        step_progress = torch.full((B,), step_number / 20.0, device=device)  # normalized
        stop_input = torch.cat([output, reading_progress.unsqueeze(-1),
                                step_progress.unsqueeze(-1)], dim=-1)  # (B, hidden_dim + 2)
        stop_logit = self.stop_head(stop_input).squeeze(-1)  # (B,)

        return ffd, L1, L2, saccade_logits, stop_logit, new_state

    # ------------------------------------------------------------------ #
    #  Forward: teacher forcing (training)
    # ------------------------------------------------------------------ #

    def forward_teacher_forcing(
        self,
        word_lists,
        word_lengths,
        word_log_freqs,
        fix_positions,
        fix_durations,
        fix_mask,
        saccade_targets,
        saccade_mask,
        stop_targets,
    ):
        device = word_lengths.device
        B = word_lengths.size(0)
        T = word_lengths.size(1)
        max_fix = fix_positions.size(1)

        word_mask = (word_lengths > 0).float()

        with torch.amp.autocast("cuda", enabled=False):
            word_embeddings = self.get_word_embeddings(word_lists, device)
        word_embeddings = word_embeddings[:, :T, :]

        reader_state = torch.zeros(1, B, self.hidden_dim, device=device)
        sentence_length = word_mask.sum(dim=1)  # (B,)

        # Track visited positions for the "previously visited" feature
        visited_mask = torch.zeros(B, T, device=device)

        all_ffd_pred = []
        all_saccade_logits = []
        all_stop_logits = []
        all_L1 = []
        all_L2 = []

        for step in range(max_fix):
            fix_pos = fix_positions[:, step].float()

            ffd, L1, L2, saccade_logits, stop_logit, reader_state = self.fixation_step(
                word_embeddings, word_lengths, word_log_freqs, fix_pos, reader_state, word_mask,
                step_number=step, sentence_length=sentence_length, visited_mask=visited_mask,
            )

            # Update visited mask AFTER using it (current fixation is "new" on first visit)
            fix_idx = fix_pos.long().clamp(0, T - 1)
            for b in range(B):
                visited_mask[b, fix_idx[b]] = 1.0

            all_ffd_pred.append(ffd)
            all_saccade_logits.append(saccade_logits)
            all_stop_logits.append(stop_logit)
            all_L1.append(L1)
            all_L2.append(L2)

        pred_ffd = torch.stack(all_ffd_pred, dim=1)
        pred_saccade = torch.stack(all_saccade_logits, dim=1)
        pred_stop = torch.stack(all_stop_logits, dim=1)  # (B, max_fix)
        pred_L1 = torch.stack(all_L1, dim=1)
        pred_L2 = torch.stack(all_L2, dim=1)

        # Duration loss (log-space MSE for better relative error weighting)
        log_pred = torch.log(pred_ffd.clamp(min=1.0))
        log_target = torch.log(fix_durations.clamp(min=1.0))
        dur_error = (log_pred - log_target) ** 2 * fix_mask
        duration_loss = dur_error.sum() / fix_mask.sum().clamp(min=1)

        # Saccade loss (cross-entropy, masked)
        sac_logits_flat = pred_saccade.view(-1, T)
        sac_targets_flat = saccade_targets.view(-1)
        sac_mask_flat = saccade_mask.view(-1)

        sac_loss_per_step = F.cross_entropy(
            sac_logits_flat, sac_targets_flat, reduction='none'
        )
        saccade_loss = (sac_loss_per_step * sac_mask_flat).sum() / sac_mask_flat.sum().clamp(min=1)

        # Stop loss (binary cross-entropy, masked by fix_mask)
        stop_loss_per_step = F.binary_cross_entropy_with_logits(
            pred_stop, stop_targets, reduction='none'
        )
        stop_loss = (stop_loss_per_step * fix_mask).sum() / fix_mask.sum().clamp(min=1)

        return {
            'duration_loss': duration_loss,
            'saccade_loss': saccade_loss,
            'stop_loss': stop_loss,
            'pred_ffd': pred_ffd,
            'pred_saccade': pred_saccade,
            'pred_stop': pred_stop,
            'L1': pred_L1,
            'L2': pred_L2,
        }

    # ------------------------------------------------------------------ #
    #  Forward: free-running inference
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def forward_free(self, word_lists, word_lengths, word_log_freqs, max_fixations=50):
        device = word_lengths.device
        B = word_lengths.size(0)
        T = word_lengths.size(1)

        word_mask = (word_lengths > 0).float()
        word_embeddings = self.get_word_embeddings(word_lists, device)
        word_embeddings = word_embeddings[:, :T, :]

        reader_state = torch.zeros(1, B, self.hidden_dim, device=device)
        sent_lengths = word_mask.sum(dim=1)

        per_word_first_dur = torch.zeros(B, T, device=device)
        per_word_total_dur = torch.zeros(B, T, device=device)
        per_word_fixcount = torch.zeros(B, T, device=device)
        per_word_skipped = torch.ones(B, T, device=device)

        # Track ALL visited positions (for suppression and as a feature)
        visited_mask = torch.zeros(B, T, device=device)

        scanpath_positions = []
        scanpath_durations = []

        fix_pos = torch.zeros(B, device=device)
        active = torch.ones(B, dtype=torch.bool, device=device)

        for step in range(max_fixations):
            if not active.any():
                break

            ffd, L1, L2, saccade_logits, stop_logit, reader_state = self.fixation_step(
                word_embeddings, word_lengths, word_log_freqs, fix_pos, reader_state, word_mask,
                step_number=step, sentence_length=sent_lengths, visited_mask=visited_mask,
            )

            fix_idx = fix_pos.long().clamp(0, T - 1)
            stop_prob = torch.sigmoid(stop_logit)

            for b in range(B):
                if active[b]:
                    idx = fix_idx[b].item()
                    if per_word_fixcount[b, idx] == 0:
                        per_word_first_dur[b, idx] = ffd[b]
                    per_word_total_dur[b, idx] += ffd[b]
                    per_word_fixcount[b, idx] += 1
                    per_word_skipped[b, idx] = 0.0
                    # Mark as visited
                    visited_mask[b, idx] = 1.0

            scanpath_positions.append(fix_idx.clone())
            scanpath_durations.append(ffd.clone())

            # Check if model wants to stop reading
            for b in range(B):
                if active[b] and stop_prob[b] > 0.5:
                    active[b] = False

            if not active.any():
                break

            # Soft-suppress visited positions (penalty, not -inf) to allow regressions
            suppressed_logits = saccade_logits.clone()
            suppressed_logits = suppressed_logits - 3.0 * visited_mask

            next_pos = suppressed_logits.argmax(dim=-1).float()

            for b in range(B):
                if next_pos[b] >= sent_lengths[b] or suppressed_logits[b].max() == float('-inf'):
                    active[b] = False

            fix_pos = next_pos.clamp(0, T - 1)

        return {
            'first_fixation': per_word_first_dur,
            'total_reading_time': per_word_total_dur,
            'fixation_count': per_word_fixcount,
            'skip_prob': per_word_skipped,
            'scanpath_positions': torch.stack(scanpath_positions, dim=1) if scanpath_positions else None,
            'scanpath_durations': torch.stack(scanpath_durations, dim=1) if scanpath_durations else None,
        }
