# Neural EZ Reader — Project Status

## What This Project Is

A differentiable approximation of the E-Z Reader cognitive model of reading,
coupled with pretrained language models (LLaMA/BERT/LSTM), trained end-to-end
on human eye-tracking data.

The architecture:
```
Text → LLM encoder → word representations → L1, L2, Skip heads
    → Differentiable EZ Reader (cognitive simulation layer)
    → predicted FFD, Gaze Duration, TRT, Skip probability
    → loss against human eye-tracking data (GECO / Provo corpora)
```

---

## What Has Been Done

### Core System (complete)

- **Differentiable EZ Reader v2** (`src_v2/diff_ezreader.py`)
  Smooth, differentiable approximation of E-Z Reader with learnable parameters:
  saccade time, attention shift, eccentricity scaling, L2 contribution to FFD,
  regression mechanism (threshold + cost). Supports ablation flags.

- **LLaMA + EZ Reader model** (`src_v2/model_llama.py`)
  TinyLlama-1.1B encoder → subword-to-word pooling (last subword for causal) →
  projection → L1 head, L2 head, skip head → DiffEZReader.
  Supports 5 ablation modes.

- **Direct regression baseline** (`src_v2/model_llama_direct.py`)
  Same TinyLlama encoder but 4 independent heads directly predicting
  TRT/FFD/Gaze/Skip. No cognitive architecture. Key control model.

- **BERT + EZ Reader** (`src_v2/model_bert.py`) — bidirectional encoder variant.

- **LSTM + EZ Reader** (`src_v2/model_lstm.py`) — lightweight baseline.

### Training (complete)

All models trained on GECO corpus (5,285 sentences, 14 participants):

| Model | Checkpoint | Best r_TRT (val) |
|-------|-----------|------------------|
| LLaMA + EZR | `geco_TinyLlama_.../best_model.pt` | 0.467 |
| Direct LLaMA | `geco_direct_TinyLlama_.../best_model.pt` | 0.466 |
| BERT + EZR | `geco_bert/best_model_bert.pt` | ~0.43 |
| LSTM + EZR | `geco_lstm/best_model_lstm.pt` | ~0.43 |
| 5 ablation variants | `geco_..._ablation_*/best_model.pt` | 0.462-0.466 |

### Evaluation Scripts (complete)

- **`compare_geco_provo.py`** — Full model comparison:
  word-level correlations, 16 psycholinguistic effects (frequency, predictability,
  word length, Freq×Pred interaction, content/function words).
  All models: LSTM, BERT, LLaMA, Direct, Orig EZ Reader, Diff EZ Reader.

- **`eval_binned.py`** — Bin-level evaluation (E-Z Reader style):
  5 bins by frequency/predictability/length, RMSD and correlation across bins.
  Fair comparison to how E-Z Reader was originally validated.

- **`eval_ablations.py`** — Detailed ablation analysis:
  word-level, bin-level, effects, effect magnitude (% of human), L1/L2 distributions,
  component contributions, L1/L2 correlations with psycholinguistic variables,
  stratified analyses, ablation-specific diagnostics, error analysis.

---

## Key Results

### Model Comparison (Provo, cross-corpus)

| Model | r_TRT | r_FFD | r_Skip | Bin r (mean) | Effects |
|-------|-------|-------|--------|-------------|---------|
| Orig EZ Reader | 0.357 | 0.160 | 0.391 | 0.978 | — |
| Diff EZ (formula) | 0.521 | 0.286 | 0.338 | 0.942 | — |
| LSTM + EZR | 0.524 | 0.275 | 0.717 | 0.968 | 16/16 |
| BERT + EZR | 0.562 | 0.280 | -0.304 | 0.490 | broken |
| **LLaMA + EZR** | **0.615** | **0.298** | **0.832** | **0.990** | **15/15** |
| Direct LLaMA | 0.603 | 0.311 | 0.841 | 0.990 | 15/15 |

### Key Findings So Far

1. **LLaMA + EZR is the best model** — highest word-level correlations and bin-level fit.
   Generalizes to unseen corpus (Provo) without retraining.

2. **Direct LLaMA matches LLaMA+EZR on prediction** — the cognitive architecture
   doesn't improve accuracy. Its value is interpretability (L1/L2/skip decomposition).

3. **BERT is cognitively broken** — negative skip correlations (-0.304 word-level,
   -0.870 bin-level). Bidirectional attention is incompatible with incremental reading.

4. **Original EZ Reader is good at bin-level** (0.978) — poor word-level r (0.357)
   is expected; it was never designed for word-level prediction.

5. **Ablations show the LLM dominates** — removing any single EZ Reader component
   barely changes performance. The LLM compensates. But:
   - Eccentricity is needed for Freq×Pred interactions on unseen data (13/15 without it)
   - The model independently discovers FFD ≈ L1 (l2_contribution goes negative)
   - Removing regressions decorrelates L1/L2 (r drops 0.962 → 0.189)
   - Skip needs its own head (r_Skip drops 0.832 → 0.805 without it)

---

## Current Trajectory

### The Paper As-Is

**"A Differentiable E-Z Reader: Bridging Cognitive Models and Neural Language Models"**

Story: We made E-Z Reader differentiable, coupled it with an LLM, trained end-to-end.
It matches bin-level performance of the original (faithful approximation) while adding
word-level prediction the original can't do. The cognitive architecture provides
interpretable decomposition at no accuracy cost.

This is a solid, publishable methods paper. Targets: ACL/EMNLP (psycholinguistics track),
CogSci, Journal of Memory and Language.

Expected citation impact: modest (5-15 citations/3 years). The reading/eye-tracking
prediction community is small.

### Possible Extensions (3 months available)

#### Option 1: Surprisal Decomposition (highest impact, ~1 month)

The biggest debate in reading research: surprisal theory (Levy 2008, 2000+ citations)
vs cognitive process models (E-Z Reader, 1000+ citations). Our model uniquely combines
both — the LLM computes surprisal internally, and E-Z Reader decomposes it into L1/L2.

Experiment: Extract per-word surprisal from the LLM. Ask:
- Does L1 ≈ surprisal? Or does it capture something different?
- Does L2 capture variance beyond surprisal?
- Can we empirically decompose reading time into surprisal-driven vs architecture-driven?

If L1 maps onto surprisal but L2 captures an independent process → we've reconciled
two competing theories. That would be cited by everyone in both camps.

#### Option 2: Individual Reader Differences (~1 month)

GECO has 14 participants. Train per-participant models.
Do different readers have different learned EZ Reader parameters?
Do slow readers have higher L1? Do skilled readers have lower L2/L1 ratios?

This extends E-Z Reader theory to individual differences — discussed theoretically
but never demonstrated computationally.

#### Option 3: Cognitive Architecture as Structured Output Layer (AI framing)

Reframe for NeurIPS/ICLR: cognitive inductive biases as structured prediction layers
for LLMs. Extend the skip/L1/L2 mechanism to efficient inference (early exit),
document triage, or interpretable uncertainty decomposition.

#### Option 4: Cross-linguistic Transfer (~1 month)

GECO has Dutch data. Train on English, test on Dutch.
Which EZ Reader parameters transfer (universal cognition) vs which don't
(language-specific)?

### Recommended Next Step

**Option 1 (Surprisal Decomposition)** — it's the fastest, uses the existing codebase,
and addresses the highest-impact theoretical question. Everything else is already built.

---

## File Inventory

### Source (`src_v2/`)
| File | Purpose |
|------|---------|
| `diff_ezreader.py` | Differentiable EZ Reader v2 (core module) |
| `model_llama.py` | LLaMA + EZR (main model, supports ablations) |
| `model_llama_direct.py` | Direct regression baseline (no EZR) |
| `model_bert.py` | BERT + EZR |
| `model_lstm.py` | LSTM + EZR |
| `data_loader.py` | Provo corpus loader + aggregation |
| `geco_loader.py` | GECO corpus loader |
| `train_geco_llama.py` | LLaMA training (supports --ablation) |
| `train_direct_llama.py` | Direct LLaMA training |
| `train_geco_bert.py` | BERT training |
| `train_geco_lstm.py` | LSTM training |
| `compare_geco_provo.py` | Full model comparison + effects |
| `eval_binned.py` | Bin-level evaluation |
| `eval_ablations.py` | Detailed ablation evaluation |

### Data (`data/`)
- `Geco_MonolingualReadingData.csv` — GECO eye-tracking data
- `Geco_EnglishMaterial.csv` — GECO stimuli
- `geco_predictability.pkl` — precomputed cloze predictability
- `Provo_Corpus-Eyetracking_Data.csv` — Provo eye-tracking data
- `SUBTLEXus.txt` — word frequency norms

### Results (`results/`)
- `comparison_geco_v2_results.txt` — main model comparison
- `eval_binned_results.txt` — bin-level evaluation
- `eval_ablations_results.txt` — ablation study results
