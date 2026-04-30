# Neural EZ Reader

## Overview

When you read a sentence, your eyes don't scan every word equally. You spend more time on rare or surprising words ("ephemeral"), and you skip right over common, predictable ones ("the", "is"). These eye movement patterns tell us a lot about how the brain processes language.

The **E-Z Reader** is a well-known cognitive model that explains these patterns. It says word reading happens in two stages: a quick familiarity check (L1) and a deeper meaning retrieval (L2). Together, these determine how long you look at each word and whether you skip it.

The problem is that E-Z Reader needs hand-tuned formulas and precomputed word statistics to work. It can't just read a new sentence on its own.

**Our solution:** We take the E-Z Reader's cognitive architecture and make it trainable. We plug in a neural network (LSTM or BERT) that learns to predict how hard each word is to process, directly from text. The whole system — neural network + cognitive model — is trained end-to-end on real human eye-tracking data. The result is a model that can read any sentence and predict where a human reader would look, how long they'd fixate, and which words they'd skip.

## What Are We Trying To Do

### Background

When people read, their eyes don't move smoothly across the page. Instead, they make rapid jumps called **saccades**, landing on words for brief periods called **fixations**. Some words are fixated for a long time (difficult, rare words like "ephemeral"), some are barely glanced at, and some are skipped entirely (short, predictable words like "the" or "is"). These patterns are not random — they reflect the underlying cognitive processes of word recognition and language comprehension.

The **E-Z Reader model** (Reichle, Pollatsek, Fisher & Rayner, 1998) is one of the most influential computational models that explains these eye movement patterns. At its core, E-Z Reader proposes two sequential stages of word processing:

- **L1 (Familiarity Check):** A fast initial assessment of whether a word is recognizable. This stage is influenced by how frequently the word appears in language (word frequency) and how predictable it is from context (cloze predictability). When L1 completes, the eye movement system begins programming a saccade to the next word.

- **L2 (Lexical Access):** A slower, deeper stage where the word's meaning is fully retrieved. When L2 completes, attention shifts to the next word. If L2 takes too long (integration difficulty), the reader may regress back to re-read previous words.

Together, these two stages determine the key eye-tracking measures that psycholinguists study:

- **First Fixation Duration (FFD):** How long the eyes first land on a word. Primarily driven by L1.
- **Gaze Duration:** Total first-pass reading time on a word before moving forward. Driven by L1 + L2.
- **Total Reading Time (TRT):** All time spent on a word, including re-reading. Includes regressions.
- **Skip Rate:** Probability that a word is never fixated at all. Highly predictable or short words are skipped more often.

### The Problem

The original E-Z Reader uses hand-tuned formulas to compute L1 and L2 from word frequency and predictability:

```
L1 = f1 - f2 * ln(frequency) - f3 * predictability
L2 = f4 - f5 * ln(frequency) - f6 * predictability
```

These formulas work reasonably well, but they have fundamental limitations:

1. **They require precomputed norms.** To process any sentence, you need a frequency database (like SUBTLEXus) and cloze predictability norms (collected from dozens of human participants filling in blanks). This means E-Z Reader can't just read arbitrary text.

2. **The formulas are too simple.** Real word processing is influenced by far more than just frequency and predictability — morphological structure, orthographic regularity, semantic context, syntactic expectations, and the identity of surrounding words all matter. A linear function of two variables can't capture this.

3. **The parameters are hand-tuned.** The model's ~15 parameters (saccade timing, attention shift duration, etc.) are typically set by hand to fit a specific dataset, making it hard to generalize across corpora.

4. **No contextual modeling.** The original E-Z Reader processes each word independently. It doesn't know that "bank" means something different in "river bank" vs "bank account." Predictability norms partially capture this, but they're a crude summary of context.

### Our Approach

We make the E-Z Reader **differentiable** — meaning we rewrite its equations using smooth, differentiable operations so that gradients can flow through the entire model. Then we replace the hand-tuned input formulas with neural networks:

- An **LSTM** (Long Short-Term Memory network) that reads the sentence word by word and predicts L1, L2, and skip probability for each word based on the sequence so far.

- A **BERT** (Bidirectional Encoder Representations from Transformers) model that reads the entire sentence at once and produces contextual representations for each word, which are then projected to L1, L2, and skip probability.

The neural network's predictions are fed into the differentiable EZ Reader, which computes predicted FFD, gaze duration, TRT, and skip rate using the same cognitive architecture as the original model. The whole system is trained end-to-end on real human eye-tracking data, with the loss function comparing predicted reading times and skip rates against actual human measurements.

This means:
- The model learns what makes words easy or hard to process directly from eye-tracking data, rather than relying on hand-crafted formulas
- The EZ Reader's own parameters (saccade time, attention shift, regression threshold, etc.) are learned jointly with the neural network
- BERT provides rich contextual representations, effectively replacing precomputed predictability norms with a learned, context-sensitive model of word processing difficulty
- The cognitive architecture constrains the neural network to produce interpretable, cognitively motivated predictions — it can't just memorize reading times, it has to explain them through the L1/L2/skip mechanism

### The Differentiable EZ Reader

The core of the system is a differentiable approximation of the EZ Reader simulation. Instead of running a discrete event simulation (which can't be backpropagated through), we use continuous, smooth functions:

- **Skip probability:** A sigmoid function of predictability (or learned from the neural network directly), controlling whether a word is fixated or skipped.
- **First Fixation Duration:** Combines L1 processing time with a fraction of L2 and a motor latency baseline. The L2 contribution is learned — the model figures out how much of the late processing stage "leaks" into the first fixation.
- **Gaze Duration:** L1 + L2 processing time (first-pass reading before moving on).
- **Regression mechanism:** When L2 exceeds a learned threshold, there is a probability of regressing back. The regression cost scales with how long the previous word took to process (because you regress back to where you had trouble).
- **Total Reading Time:** Combines gaze duration, saccade/attention overhead, and regression penalties, weighted by the probability that the word was actually fixated (1 - skip probability).

All parameters in this module (saccade time, attention shift, regression threshold, regression sharpness, L2 contribution weight, etc.) are learnable and trained alongside the neural network.

## Datasets

We use two major eye-tracking corpora:

### Provo Corpus
55 short English passages (news articles, encyclopedia entries) with 2,654 words. Eye-tracking data from ~84 participants, plus cloze predictability norms from a separate group of participants who filled in each word given the preceding context. Provo is small but very clean — many participants per word give stable averages, and the predictability norms are high quality.

### GECO Corpus (Ghent Eye-tracking Corpus)
The English portion of GECO contains ~5,300 sentences from a complete novel (*The Mysterious Affair at Styles* by Agatha Christie), with 54,000+ words read by 14 participants. GECO is much larger than Provo but noisier — fewer participants per word, and the text comes from a single literary genre. Predictability norms were computed separately using a language model.

### SUBTLEXus
A word frequency database with 74,286 entries derived from American English movie subtitles. Used as the frequency input for the original (non-neural) EZ Reader baseline.

## Models

We compare four models, each representing a different level of complexity:

### 1. Original EZ Reader
The classic formula-based model. L1 and L2 are computed from SUBTLEXus word frequencies and cloze predictability using the original hand-tuned equations. This is the baseline — it represents what the field has been using for decades. It can predict FFD and TRT but has no mechanism for predicting skip rates in this implementation.

### 2. Differentiable EZ Reader (formula-based)
Our differentiable rewrite of the EZ Reader, but still using the same frequency-based formulas as input (no neural network). The EZ Reader parameters are learnable but the L1/L2 inputs come from the same formulas as the original. This isolates the effect of making the EZ Reader differentiable and learning its parameters from data.

### 3. Neural EZ Reader (LSTM)
An LSTM network replaces the formula-based inputs. The LSTM reads the sentence as a sequence of word embeddings and predicts L1, L2, and skip probability for each word. These predictions are fed into the differentiable EZ Reader. The LSTM has access to explicit word features (length, position) through its input representation. Trained with supervision on per-participant eye-tracking measurements.

### 4. Neural EZ Reader (BERT)
BERT (bert-base-uncased) replaces the LSTM. BERT's contextual representations provide rich information about each word's role in the sentence, effectively learning its own version of predictability from the eye-tracking supervision signal. The first 8 BERT layers are frozen and only the top 4 layers are fine-tuned (with a lower learning rate) to prevent catastrophic forgetting. Separate projection heads map BERT's hidden states to L1, L2, and skip probability.

## Training

Models are trained on per-participant eye-tracking data (not averaged). The loss function combines four components:

- **MSE on Total Reading Time** — matches predicted TRT to human TRT
- **MSE on First Fixation Duration** — matches predicted FFD to human FFD
- **MSE on Gaze Duration** — matches predicted gaze to human gaze (critical for preventing L2 from collapsing)
- **BCE on Skip Probability** — matches predicted skip rate to human skip/fixate binary decisions

Additional regularizers constrain the model to find cognitively plausible solutions:
- **Skip prior:** Penalizes the mean skip rate for deviating too far from the corpus average (~0.45)
- **L1 bounds:** Soft floor and ceiling on L1 values to keep them in a cognitively plausible range
- **TRT scale matching:** Penalizes systematic gaps between mean predicted and mean human TRT

## Evaluation

We evaluate models on two dimensions:

### 1. Correlation with Human Data
How well do predicted reading times correlate with actual human reading times? Measured with Pearson r on Total Reading Time, First Fixation Duration, and Skip Rate. Higher is better.

### 2. Psycholinguistic Effects
Does the model reproduce the well-established effects from the reading literature? These are qualitative tests — we check whether the model shows the correct direction and sufficient magnitude for each effect:

- **Frequency effect:** Low-frequency words should have longer reading times and lower skip rates than high-frequency words
- **Predictability effect:** Unpredictable words should have longer reading times and lower skip rates than predictable words
- **Word length effect:** Longer words should have longer reading times and lower skip rates than shorter words
- **Frequency x Predictability interaction:** The frequency effect should be larger for unpredictable words than predictable words
- **Content vs Function words:** Content words (nouns, verbs, adjectives) should have longer reading times than function words (the, is, of)

A model that achieves high correlations but fails to reproduce these effects is not a good cognitive model — it might be fitting noise rather than capturing the underlying cognitive processes.

### 3. Cross-Corpus Generalization
Models trained on GECO are evaluated on the entire Provo corpus (which they never saw during training). This tests whether the model learned general reading patterns or just memorized corpus-specific quirks.

## Key Results

### Correlation with Human Data

On both GECO (in-distribution) and Provo (cross-corpus), the neural models substantially outperform the formula-based baselines:

- **BERT achieves the best TRT correlation** across both corpora. Contextual representations capture word processing difficulty better than sequential (LSTM) or formula-based approaches.
- **LSTM achieves the best skip rate prediction** by a wide margin. It learns strong skip behavior from explicit word features (length, frequency patterns) that BERT's contextual embeddings don't encode as strongly.
- **Cross-corpus generalization works.** Both neural models achieve *higher* correlations on Provo (unseen corpus) than on GECO (training corpus). This is likely because Provo has more participants per word (giving cleaner averages) and shorter, more controlled passages. The key finding is that the models are not overfitting to GECO — they learn transferable reading patterns.

### Psycholinguistic Effects

- **LSTM reproduces 100% of tested effects** on both corpora. Every frequency, predictability, word length, and content/function word effect goes in the right direction with sufficient magnitude.
- **BERT reproduces ~80-83% of effects.** It fails specifically on skip rate effects for frequency and word length — it predicts skip rates but doesn't differentiate enough between conditions. BERT's skip predictions are too flat across word types.
- **All models reproduce the frequency x predictability interaction** in the correct direction, showing that the differentiable EZ Reader architecture captures this important non-linear relationship.

## Known Issues and Open Problems

- **L1/L2 imbalance in LSTM.** The LSTM learns very low L1 values (~18-20ms) and compensates with high L2 (~112-115ms). Cognitively, L1 (familiarity check) shouldn't be nearly instantaneous — it should take 50-150ms. The model finds a degenerate but mathematically valid solution. Stronger constraints or a different parameterization may be needed.

- **Absolute scale under-prediction.** Both neural models predict mean TRT ~100ms below human averages, and FFD ~60-70ms below. The correlations (relative ordering) are good, but the absolute predictions are systematically too low. This suggests the model isn't capturing some baseline component of fixation duration (perhaps motor planning time, or re-reading that isn't modeled).

- **BERT skip rate is too flat.** BERT predicts similar skip probabilities across word types (~0.37-0.50) while humans show a much wider range (~0.19-0.69). BERT's contextual embeddings may not encode surface-level features like word length and orthographic frequency as directly as LSTM's explicit feature inputs.

- **Small validation sets.** Even with the improved 70/15/15 split, Provo gives only ~23 validation sentences. Correlation estimates on this few data points have wide confidence intervals.

## Project Structure

```
Neuro_EZR/
├── ez_reader/                     # Core EZ Reader engine and data loading
│   ├── ez_reader_engine.py        # Original EZ Reader simulation
│   ├── ez_wrapper.py              # Wrapper for running simulations
│   ├── utilities.py               # L1/L2 formula functions
│   ├── data_loader.py             # Provo corpus loader
│   ├── geco_loader.py             # GECO corpus loader
│   └── example1/2/3.py            # Usage examples
├── src_v1/                        # v1 neural models (preserved)
├── src_v2/                        # v2 neural models (current)
│   ├── diff_ezreader.py           # Differentiable EZ Reader module
│   ├── data_loader.py             # Data loader with 70/15/15 split
│   ├── model_lstm.py              # LSTM-based neural EZ Reader
│   ├── model_bert.py              # BERT-based neural EZ Reader
│   ├── train_provo_lstm.py        # Train LSTM on Provo corpus
│   ├── train_provo_bert.py        # Train BERT on Provo corpus
│   ├── train_geco_lstm.py         # Train LSTM on GECO corpus
│   ├── train_geco_bert.py         # Train BERT on GECO corpus
│   ├── compare_geco_provo.py      # Cross-corpus evaluation + effects
│   └── evaluate_provo_effects.py  # Provo-only effects analysis
├── checkpoints_v1/                # v1 trained model weights
├── checkpoints_v2/                # v2 trained model weights
├── results/                       # Evaluation output files
├── scripts/                       # One-off utilities
├── data/                          # Corpora (not tracked)
├── CLUSTER_GUIDE.md               # Compute cluster usage guide
└── requirements.txt
```

## How to Run

### Setup

```bash
pip install -r requirements.txt
```

### Training

```bash
# Train LSTM on GECO (GPU 0)
python src_v2/train_geco_lstm.py --gpu 0

# Train BERT on GECO (GPU 1)
CUDA_VISIBLE_DEVICES=1 python src_v2/train_geco_bert.py

# Train on Provo instead
python src_v2/train_provo_lstm.py
python src_v2/train_provo_bert.py
```

### Evaluation

```bash
# Cross-corpus comparison (GECO test + Provo) with effects analysis
python src_v2/compare_geco_provo.py

# Provo-only effects analysis
python src_v2/evaluate_provo_effects.py
```

### Data

The data files are not tracked in git. Place them in `data/`:
- `Provo_Corpus-Eyetracking_Data.csv`
- `Provo_Corpus-Predictability_Norms.csv`
- `Geco_MonolingualReadingData.csv`
- `Geco_EnglishMaterial.csv`
- `geco_predictability.pkl`
- `SUBTLEXus.txt`


