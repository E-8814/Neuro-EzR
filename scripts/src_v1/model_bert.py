"""
Neural EZ Reader Model (BERT + Differentiable EZ Reader).

Full pipeline: sentence → BERT → L1/L2 predictions → Differentiable EZ Reader → reading times.

Replaces the LSTM encoder with a pretrained BERT model.
BERT's contextual representations should capture richer linguistic features
(syntax, semantics, surprisal) that influence reading behavior.
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizerFast

from diff_ezreader import DifferentiableEZReader


class NeuralEZReaderBERT(nn.Module):
    """
    End-to-end model:
        word tokens → BERT (subword) → word-level pooling → (L1, L2)
        → DifferentiableEZReader → (TRT, FFD, skip)

    BERT encodes each sentence into contextual subword representations.
    Subword representations are pooled back to word-level, then projection
    heads predict L1/L2 per word.  The differentiable EZ Reader converts
    those into observable reading metrics.
    """

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        freeze_bert_layers: int = 8,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # --- BERT encoder ---
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
        bert_dim = self.bert.config.hidden_size  # 768 for bert-base

        # Freeze lower BERT layers to save memory / speed up training
        if freeze_bert_layers > 0:
            # Freeze embeddings
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            # Freeze first N encoder layers
            for layer_idx in range(min(freeze_bert_layers, len(self.bert.encoder.layer))):
                for param in self.bert.encoder.layer[layer_idx].parameters():
                    param.requires_grad = False

        # --- Projection from BERT dim to internal hidden dim ---
        self.projection = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # --- Predict L1 and L2 per word ---
        self.l1_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # L1 > 0
        )
        self.l2_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # L2 > 0
        )

        # Scale + bias to start in reasonable ms range
        self.l1_scale = nn.Parameter(torch.tensor(50.0))
        self.l2_scale = nn.Parameter(torch.tensor(30.0))

        # --- Differentiable EZ Reader (the "body") ---
        self.ezreader = DifferentiableEZReader()

    def _tokenize_and_align(self, word_lists, device):
        """
        Tokenize a batch of word lists with BERT tokenizer and compute
        the mapping from subword tokens back to original words.

        Args:
            word_lists: list of list of str — each inner list is one sentence
                        as a sequence of words.
            device: torch device

        Returns:
            input_ids:      (batch, max_subword_len) int tensor
            attention_mask: (batch, max_subword_len) int tensor
            word_to_subword: list of list of (start, end) — for each sentence,
                             the subword index range for each word.
            max_words: max number of words across the batch
        """
        # Tokenize with word-level alignment info
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

        # Build word → subword mapping for each sentence
        batch_word_maps = []
        max_words = 0

        for batch_idx in range(len(word_lists)):
            word_ids = encodings.word_ids(batch_index=batch_idx)
            word_map = {}  # word_idx -> (first_subword, last_subword+1)

            for subword_idx, word_idx in enumerate(word_ids):
                if word_idx is None:
                    continue  # [CLS], [SEP], [PAD]
                if word_idx not in word_map:
                    word_map[word_idx] = [subword_idx, subword_idx + 1]
                else:
                    word_map[word_idx][1] = subword_idx + 1

            # Convert to ordered list
            n_words = len(word_lists[batch_idx])
            spans = []
            for w_idx in range(n_words):
                if w_idx in word_map:
                    spans.append(tuple(word_map[w_idx]))
                else:
                    # Fallback: word was truncated, use last valid token
                    spans.append((1, 2))  # [CLS] token as fallback

            batch_word_maps.append(spans)
            max_words = max(max_words, n_words)

        return input_ids, attention_mask, batch_word_maps, max_words

    def _pool_subwords_to_words(self, bert_output, batch_word_maps, max_words, device):
        """
        Pool subword representations to word-level using first-subword strategy.

        Args:
            bert_output: (batch, subword_len, bert_dim) — BERT last hidden state
            batch_word_maps: from _tokenize_and_align
            max_words: max words in batch
            device: torch device

        Returns:
            word_repr: (batch, max_words, bert_dim)
        """
        batch_size = bert_output.size(0)
        bert_dim = bert_output.size(2)

        # Build index tensor on CPU then move once — avoids per-element GPU calls
        idx = torch.zeros(batch_size, max_words, dtype=torch.long)
        for b in range(batch_size):
            for w_idx, (start, end) in enumerate(batch_word_maps[b]):
                idx[b, w_idx] = start
        idx = idx.to(device)

        # gather: pick bert_output[b, idx[b,w], :] for every (b, w)
        word_repr = torch.gather(
            bert_output, 1, idx.unsqueeze(-1).expand(-1, -1, bert_dim)
        )
        return word_repr

    def forward(self, word_lists, predictability, word_lengths):
        """
        Full forward pass.

        Args:
            word_lists:     list of list of str — raw word tokens per sentence
            predictability: (batch, seq_len) float tensor (0-1)
            word_lengths:   (batch, seq_len) float tensor (character counts)

        Returns:
            dict with predicted reading metrics + L1/L2 for inspection
        """
        device = predictability.device

        # --- BERT encodes the sentence ---
        input_ids, attention_mask, word_maps, max_words = self._tokenize_and_align(
            word_lists, device
        )

        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state  # (B, subword_len, 768)

        # --- Pool subwords → word-level ---
        word_repr = self._pool_subwords_to_words(
            bert_out, word_maps, max_words, device
        )  # (B, T, 768)

        # --- Project to hidden dim ---
        projected = self.projection(word_repr)  # (B, T, hidden_dim)

        # --- Predict L1 and L2 ---
        L1 = self.l1_head(projected).squeeze(-1) * self.l1_scale   # (B, T)
        L2 = self.l2_head(projected).squeeze(-1) * self.l2_scale   # (B, T)

        # Clamp to reasonable range (1ms - 500ms)
        L1 = L1.clamp(min=1.0, max=500.0)
        L2 = L2.clamp(min=1.0, max=500.0)

        # Trim to match actual sequence lengths
        seq_len = predictability.size(1)
        L1 = L1[:, :seq_len]
        L2 = L2[:, :seq_len]

        # --- Differentiable EZ Reader produces reading metrics ---
        result = self.ezreader(L1, L2, predictability, word_lengths)

        # Add L1/L2 to result for logging
        result['L1'] = L1
        result['L2'] = L2

        return result
