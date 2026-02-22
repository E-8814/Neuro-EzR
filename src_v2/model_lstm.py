"""
Neural EZ Reader Model (Differentiable) - v2.

Changes from v1:
  - Uses DifferentiableEZReader v2 (improved FFD formula + explicit regressions)
  - LSTM predicts skip probability (not computed from predictability alone)
  - L2 has a distinct role via regression mechanism
  - Gaze duration loss forces L2 to be meaningful
"""

import torch
import torch.nn as nn

from diff_ezreader import DifferentiableEZReader


class NeuralEZReader(nn.Module):
    """
    End-to-end model:
        tokens + predictability -> LSTM -> (L1, L2, skip_prob) -> DiffEZReader v2 -> reading times
    """

    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128):
        super().__init__()

        # --- Neural Core ---
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim,
            batch_first=True, num_layers=2, dropout=0.1,
        )

        # L1 head (familiarity check - early stage)
        self.l1_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )

        # L2 head (lexical access - later stage)
        self.l2_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )

        # Skip head (predicts skip probability from context)
        self.skip_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Scale parameters
        self.l1_scale = nn.Parameter(torch.tensor(8.0))
        self.l2_scale = nn.Parameter(torch.tensor(8.0))

        # --- Differentiable EZ Reader v2 ---
        self.ezreader = DifferentiableEZReader()

    def forward(self, token_ids, predictability, word_lengths):
        """
        Args:
            token_ids:      (batch, seq_len) int tensor
            predictability: (batch, seq_len) float tensor (0-1)
            word_lengths:   (batch, seq_len) float tensor
        """
        embedded = self.embedding(token_ids)
        lstm_out, _ = self.lstm(embedded)

        # Predict L1, L2, skip
        L1 = self.l1_head(lstm_out).squeeze(-1) * self.l1_scale
        L2 = self.l2_head(lstm_out).squeeze(-1) * self.l2_scale
        skip_prob = self.skip_head(lstm_out).squeeze(-1)

        # Soft clamp to reasonable ranges (gradient-friendly)
        L1 = torch.nn.functional.softplus(L1 - 1.0) + 1.0   # floor at ~1ms, no ceiling death
        L2 = torch.nn.functional.softplus(L2 - 1.0) + 1.0

        # Run differentiable EZ Reader v2
        result = self.ezreader(L1, L2, skip_prob, word_lengths, input_is_prob=True)

        # Add internals for logging
        result['L1'] = L1
        result['L2'] = L2
        result['skip_prob'] = skip_prob

        return result


class Vocabulary:
    """Simple word-to-index mapping."""

    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.frozen = False

    def __len__(self):
        return len(self.word2idx)

    def add_word(self, word):
        w = word.lower()
        if w not in self.word2idx:
            if self.frozen:
                return self.word2idx["<UNK>"]
            self.word2idx[w] = len(self.word2idx)
        return self.word2idx[w]

    def encode(self, word):
        return self.word2idx.get(word.lower(), self.word2idx["<UNK>"])

    def encode_sentence(self, tokens):
        return torch.tensor([self.encode(t) for t in tokens], dtype=torch.long)

    def build_from_sentences(self, all_tokens):
        for tokens in all_tokens:
            for t in tokens:
                self.add_word(t)

    def freeze(self):
        self.frozen = True
