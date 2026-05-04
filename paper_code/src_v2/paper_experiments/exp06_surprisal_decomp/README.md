# exp06 — Surprisal decomposition (Table 3)

## What this produces

Three statistics demonstrating that L1 (the cognitive lexical-processing
output of the paper model) captures BOTH surprisal and unique variance
beyond surprisal:

1. `r(L1, surprisal)` — does the model's L1 correlate with the LM's
   per-word surprisal? Higher = L1 has learned a surprisal-like signal.
2. `partial r(L1, h_TRT | surprisal + log_freq + word_length)` — does L1
   predict human reading time after controlling for surprisal and
   word-level controls? Higher = L1 captures unique variance.
3. `ΔR² L1 beyond surprisal` — hierarchical regression: how much R² does
   adding L1 give over a model with only surprisal + controls?

If (2) and (3) are positive, the cognitive structure provides
information that surprisal alone cannot.

## Files

| File | Purpose |
|---|---|
| `compute_surprisal_decomp.py` | computes per-word L1 + TinyLlama surprisal + controls; computes the three statistics; emits CSV |

## Outputs

- `surprisal_decomp_results.csv` — long form: `(corpus, statistic, value, p_value)` (p_value optional)
- `per_word_features.csv` — per-word: `(corpus, sentence_idx, word_position, word, L1, surprisal, log_freq, word_length, h_TRT, h_FFD)`

## Run individually

```bash
python compute_surprisal_decomp.py
```

## Prerequisites

- Trained paper model (1 seed minimum).
- TinyLlama causal LM (loaded via HuggingFace, same as backbone).

## Compute

~1 hour on GPU (TinyLlama forward pass over GECO + Provo).
