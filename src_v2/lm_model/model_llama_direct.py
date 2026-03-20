"""
Direct Regression Model (LLaMA / causal LM -> FFD, Gaze, TRT, Skip).

No E-Z Reader cognitive architecture. The LLM directly predicts all four
reading measures from contextual word representations. This serves as a
baseline to test whether the E-Z Reader structure adds value beyond what
a powerful language model can learn on its own.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class DirectRegressionLLaMA(nn.Module):
    """
    End-to-end model:
        word tokens -> LLaMA (causal) -> word-level pooling
        -> 4 independent heads -> (TRT, FFD, Gaze, Skip)
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B",
        freeze_layers: int = 12,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # --- LLaMA encoder ---
        self.llama = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.llama.config.pad_token_id = self.tokenizer.eos_token_id

        llama_dim = self.llama.config.hidden_size

        # Freeze lower layers
        if freeze_layers > 0:
            for param in self.llama.embed_tokens.parameters():
                param.requires_grad = False
            for layer_idx in range(min(freeze_layers, len(self.llama.layers))):
                for param in self.llama.layers[layer_idx].parameters():
                    param.requires_grad = False

        # --- Projection ---
        self.projection = nn.Sequential(
            nn.Linear(llama_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # --- Direct regression heads (no EZ Reader) ---
        self.trt_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # TRT > 0
        )
        self.ffd_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # FFD > 0
        )
        self.gaze_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # Gaze > 0
        )
        self.skip_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Learnable scales to start in reasonable ms range
        self.trt_scale = nn.Parameter(torch.tensor(100.0))
        self.ffd_scale = nn.Parameter(torch.tensor(100.0))
        self.gaze_scale = nn.Parameter(torch.tensor(100.0))

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

    def _pool_subwords_to_words(self, hidden_states, batch_word_maps, max_words, device):
        """Pool subword representations to word-level using last-subword strategy."""
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

    def forward(self, word_lists, predictability, word_lengths):
        """
        Args:
            word_lists:     list of list of str
            predictability: (batch, seq_len) float tensor — accepted but NOT used
                            by the heads (the LLM implicitly captures predictability)
            word_lengths:   (batch, seq_len) float tensor — accepted but NOT used

        Returns:
            dict with: total_reading_time, first_fixation, gaze_duration, skip_prob
        """
        device = predictability.device
        seq_len = predictability.size(1)

        input_ids, attention_mask, word_maps, max_words = self._tokenize_and_align(
            word_lists, device
        )

        llama_out = self.llama(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

        word_repr = self._pool_subwords_to_words(
            llama_out, word_maps, max_words, device
        )

        projected = self.projection(word_repr)

        # Direct predictions
        trt = self.trt_head(projected).squeeze(-1) * self.trt_scale
        ffd = self.ffd_head(projected).squeeze(-1) * self.ffd_scale
        gaze = self.gaze_head(projected).squeeze(-1) * self.gaze_scale
        skip = self.skip_head(projected).squeeze(-1)

        # Clamp to reasonable range
        trt = trt[:, :seq_len].clamp(min=1.0, max=1500.0)
        ffd = ffd[:, :seq_len].clamp(min=1.0, max=1000.0)
        gaze = gaze[:, :seq_len].clamp(min=1.0, max=1500.0)
        skip = skip[:, :seq_len]

        return {
            'total_reading_time': trt,
            'first_fixation': ffd,
            'gaze_duration': gaze,
            'skip_prob': skip,
        }
