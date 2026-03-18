"""
Baseline 2: GPT-2 Surprisal Regression.

Uses GPT-2 to compute per-word surprisal (negative log probability), then
combines surprisal with frequency, predictability, and word length in a
linear regression to predict reading times.

This is the standard psycholinguistic approach from Smith & Levy (2013),
Goodkind & Bicknell (2018), etc.

Trained on GECO, evaluated on GECO test + full Provo.

Usage:
    python3 -u previous_implementations_of_word_level_predictions/gpt2_surprisal.py
"""

import os
import sys
import csv
import math
import time
import numpy as np
from collections import defaultdict

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src_v2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'original_ezreader'))

from data_loader import load_provo, aggregate_by_sentence
from geco_loader import load_geco, split_geco


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def load_subtlexus(path):
    freq = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            freq[row['Word'].lower()] = int(row['FREQcount'])
    return freq


def get_log_freq(word, subtlex):
    w = word.lower().strip(".,;:!?\"'()[]{}").replace("\u2019", "'")
    if w in subtlex:
        return math.log(max(1, subtlex[w]))
    for variant in [w.replace("'", ""), w.split("'")[0], w.split("-")[0]]:
        if variant in subtlex:
            return math.log(max(1, subtlex[variant]))
    length = len(w)
    if length <= 3:   return math.log(50000)
    elif length <= 5: return math.log(10000)
    elif length <= 7: return math.log(2000)
    else:             return math.log(500)


# --------------------------------------------------------------------------- #
#  GPT-2 Surprisal Computation
# --------------------------------------------------------------------------- #

class GPT2SurprisalComputer:
    """Compute per-word surprisal using GPT-2."""

    def __init__(self, model_name="gpt2", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Loading GPT-2 ({model_name})...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print(f"  GPT-2 loaded on {self.device}")

    @torch.no_grad()
    def compute_surprisal(self, tokens):
        """
        Compute per-word surprisal for a list of word tokens.

        Returns: list of surprisal values (bits), one per word.
        """
        text = " ".join(tokens)
        encoding = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = encoding["input_ids"][0]

        outputs = self.model(input_ids=input_ids.unsqueeze(0))
        logits = outputs.logits[0]  # (seq_len, vocab_size)

        # Surprisal of each token = -log2(P(token | context))
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Map subword tokens back to words
        word_surprisals = []
        char_to_word = []
        char_pos = 0
        for w_idx, token in enumerate(tokens):
            for _ in token:
                char_to_word.append(w_idx)
                char_pos += 1
            # Space between words
            char_to_word.append(w_idx)
            char_pos += 1

        # Compute per-subword surprisal and aggregate to word level
        subword_surprisals = {}  # word_idx -> list of surprisals

        for i in range(len(tokens)):
            subword_surprisals[i] = []

        # Re-tokenize to get word-to-subword mapping
        word_texts = []
        for i, token in enumerate(tokens):
            prefix = " " if i > 0 else ""
            word_texts.append(prefix + token)

        subword_offset = 0
        for w_idx, wtext in enumerate(word_texts):
            sub_ids = self.tokenizer.encode(wtext, add_special_tokens=False)
            n_subs = len(sub_ids)
            for j in range(n_subs):
                pos = subword_offset + j
                if pos < len(input_ids) - 1:
                    # Surprisal of next token given context up to pos
                    next_id = input_ids[pos + 1] if pos + 1 < len(input_ids) else input_ids[pos]
                    # Use conditional probability: P(token_{pos} | tokens_{<pos})
                    if pos > 0:
                        surp = -log_probs[pos - 1, input_ids[pos]].item() / math.log(2)
                    else:
                        surp = -log_probs[0, input_ids[0]].item() / math.log(2)
                    subword_surprisals[w_idx].append(max(0.0, surp))
            subword_offset += n_subs

        # Aggregate: sum surprisal across subwords for each word
        result = []
        for w_idx in range(len(tokens)):
            subs = subword_surprisals[w_idx]
            if subs:
                result.append(sum(subs))
            else:
                result.append(10.0)  # default high surprisal

        return result


# --------------------------------------------------------------------------- #
#  Linear Regression (same as baseline 1 but with more features)
# --------------------------------------------------------------------------- #

class LinearRegressionModel:
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        X_b = np.column_stack([X, np.ones(len(X))])
        lam = 1e-6
        XtX = X_b.T @ X_b + lam * np.eye(X_b.shape[1])
        Xty = X_b.T @ y
        self.weights = np.linalg.solve(XtX, Xty)

    def predict(self, X):
        X_b = np.column_stack([X, np.ones(len(X))])
        return X_b @ self.weights


class LogisticRegressionModel:
    def __init__(self, lr=0.01, epochs=500):
        self.weights = None
        self.lr = lr
        self.epochs = epochs

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        X_b = np.column_stack([X, np.ones(len(X))])
        self.weights = np.zeros(X_b.shape[1])
        for _ in range(self.epochs):
            pred = self._sigmoid(X_b @ self.weights)
            grad = X_b.T @ (pred - y) / len(y)
            self.weights -= self.lr * grad

    def predict(self, X):
        X_b = np.column_stack([X, np.ones(len(X))])
        return self._sigmoid(X_b @ self.weights)


# --------------------------------------------------------------------------- #
#  Feature extraction
# --------------------------------------------------------------------------- #

def extract_features_with_surprisal(sentences, subtlex, surprisal_computer):
    """Extract per-word features including GPT-2 surprisal."""
    features = []
    targets_ffd = []
    targets_gaze = []
    targets_trt = []
    targets_skip = []
    tokens_list = []

    total = len(sentences)
    for s_idx, agg in enumerate(sentences):
        if (s_idx + 1) % 50 == 0:
            print(f"    Sentence {s_idx+1}/{total}...", flush=True)

        # Compute surprisal for this sentence
        surprisals = surprisal_computer.compute_surprisal(agg.tokens)

        for i, token in enumerate(agg.tokens):
            lf = get_log_freq(token, subtlex)
            pred = agg.predictabilities[i]
            wlen = len(token)
            surp = surprisals[i]

            features.append([lf, pred, wlen, surp])
            targets_ffd.append(agg.mean_ffd[i])
            targets_gaze.append(agg.mean_gaze[i])
            targets_trt.append(agg.mean_trt[i])
            targets_skip.append(agg.skip_rate[i])
            tokens_list.append(token)

    return (np.array(features), np.array(targets_ffd), np.array(targets_gaze),
            np.array(targets_trt), np.array(targets_skip), tokens_list)


# --------------------------------------------------------------------------- #
#  Evaluation
# --------------------------------------------------------------------------- #

def pearson_r(a, b):
    a, b = np.array(a), np.array(b)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return np.corrcoef(a, b)[0, 1]


def evaluate(name, features, targets_ffd, targets_gaze, targets_trt, targets_skip,
             ffd_model, gaze_model, trt_model, skip_model):
    pred_ffd = ffd_model.predict(features)
    pred_gaze = gaze_model.predict(features)
    pred_trt = trt_model.predict(features)
    pred_skip = skip_model.predict(features)

    r_ffd = pearson_r(pred_ffd, targets_ffd)
    r_gaze = pearson_r(pred_gaze, targets_gaze)
    r_trt = pearson_r(pred_trt, targets_trt)
    r_skip = pearson_r(pred_skip, targets_skip)

    mae_ffd = np.mean(np.abs(pred_ffd - targets_ffd))
    mae_trt = np.mean(np.abs(pred_trt - targets_trt))

    print(f"\n  {name} ({len(targets_trt)} words)")
    print(f"    r_FFD  = {r_ffd:.3f}   MAE_FFD = {mae_ffd:.1f}ms")
    print(f"    r_Gaze = {r_gaze:.3f}")
    print(f"    r_TRT  = {r_trt:.3f}   MAE_TRT = {mae_trt:.1f}ms")
    print(f"    r_Skip = {r_skip:.3f}")

    return {'r_ffd': r_ffd, 'r_gaze': r_gaze, 'r_trt': r_trt, 'r_skip': r_skip}


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    print("=" * 80)
    print("BASELINE 2: GPT-2 Surprisal + Linear Regression")
    print("  Features: log_freq, predictability, word_length, GPT-2 surprisal")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load SUBTLEXus
    print("\nLoading SUBTLEXus...")
    subtlex = load_subtlexus(os.path.join(data_dir, 'SUBTLEXus.txt'))

    # Load GPT-2
    print("\nLoading GPT-2...")
    surp = GPT2SurprisalComputer(device=device)

    # Load GECO
    print("\nLoading GECO...")
    reading_path = os.path.join(data_dir, "Geco_MonolingualReadingData.csv")
    material_path = os.path.join(data_dir, "Geco_EnglishMaterial.csv")
    pred_path = os.path.join(data_dir, "geco_predictability.pkl")
    geco_raw = load_geco(reading_path, material_path, pred_path)
    train_raw, val_raw, test_raw = split_geco(geco_raw)

    geco_agg = aggregate_by_sentence(geco_raw, min_participants=5)
    train_ids = set(sd.text_id for sd in train_raw)
    val_ids = set(sd.text_id for sd in val_raw)
    train_agg = [a for a in geco_agg if a.text_id in train_ids]
    val_agg = [a for a in geco_agg if a.text_id in val_ids]
    test_agg = [a for a in geco_agg if a.text_id not in train_ids and a.text_id not in val_ids]

    print(f"  Train: {len(train_agg)} | Val: {len(val_agg)} | Test: {len(test_agg)} sentences")

    # Extract features (with GPT-2 surprisal)
    print("\nExtracting features (train)...")
    X_train, y_ffd_train, y_gaze_train, y_trt_train, y_skip_train, _ = \
        extract_features_with_surprisal(train_agg, subtlex, surp)

    print("\nExtracting features (val)...")
    X_val, y_ffd_val, y_gaze_val, y_trt_val, y_skip_val, _ = \
        extract_features_with_surprisal(val_agg, subtlex, surp)

    print("\nExtracting features (test)...")
    X_test, y_ffd_test, y_gaze_test, y_trt_test, y_skip_test, _ = \
        extract_features_with_surprisal(test_agg, subtlex, surp)

    # Train models
    print("\nTraining regressions...")
    t0 = time.time()

    ffd_model = LinearRegressionModel()
    ffd_model.fit(X_train, y_ffd_train)

    gaze_model = LinearRegressionModel()
    gaze_model.fit(X_train, y_gaze_train)

    trt_model = LinearRegressionModel()
    trt_model.fit(X_train, y_trt_train)

    skip_model = LogisticRegressionModel(lr=0.1, epochs=1000)
    skip_model.fit(X_train, y_skip_train)

    elapsed = time.time() - t0
    print(f"  Training completed in {elapsed:.1f}s")

    # Print weights
    print("\n  Learned weights (TRT): log_freq={:.2f}, pred={:.2f}, wlen={:.2f}, surprisal={:.2f}, bias={:.2f}".format(
        *trt_model.weights))
    print("  Learned weights (FFD): log_freq={:.2f}, pred={:.2f}, wlen={:.2f}, surprisal={:.2f}, bias={:.2f}".format(
        *ffd_model.weights))

    # Evaluate
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    evaluate("GECO Val", X_val, y_ffd_val, y_gaze_val, y_trt_val, y_skip_val,
             ffd_model, gaze_model, trt_model, skip_model)

    evaluate("GECO Test", X_test, y_ffd_test, y_gaze_test, y_trt_test, y_skip_test,
             ffd_model, gaze_model, trt_model, skip_model)

    # Cross-corpus: Provo
    print("\nLoading Provo for cross-corpus evaluation...")
    et_path = os.path.join(data_dir, "Provo_Corpus-Eyetracking_Data.csv")
    provo_raw = load_provo(et_path)
    provo_agg = aggregate_by_sentence(provo_raw, min_participants=10)

    print("\nExtracting features (Provo)...")
    X_provo, y_ffd_provo, y_gaze_provo, y_trt_provo, y_skip_provo, _ = \
        extract_features_with_surprisal(provo_agg, subtlex, surp)

    evaluate("Provo (cross-corpus)", X_provo, y_ffd_provo, y_gaze_provo,
             y_trt_provo, y_skip_provo,
             ffd_model, gaze_model, trt_model, skip_model)

    print("\nDone!")


if __name__ == "__main__":
    main()
