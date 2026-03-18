"""
Baseline 1: Linear Regression (freq + predictability + word_length).

The simplest psycholinguistic baseline. Uses three features per word:
  - log word frequency (from SUBTLEXus)
  - cloze predictability
  - word length (characters)

Trains separate linear regressions for FFD, TRT, Gaze Duration, and Skip.
Trained on GECO, evaluated on GECO test + full Provo.

Usage:
    python3 -u previous_implementations_of_word_level_predictions/linear_regression.py
"""

import os
import sys
import csv
import math
import time
import numpy as np
from collections import defaultdict

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


class LinearRegressionModel:
    """Simple closed-form OLS linear regression."""

    def __init__(self):
        self.weights = None  # (n_features+1,) includes bias

    def fit(self, X, y):
        """Fit using normal equation: w = (X^T X)^{-1} X^T y."""
        # Add bias column
        X_b = np.column_stack([X, np.ones(len(X))])
        # Normal equation with ridge regularization for stability
        lam = 1e-6
        XtX = X_b.T @ X_b + lam * np.eye(X_b.shape[1])
        Xty = X_b.T @ y
        self.weights = np.linalg.solve(XtX, Xty)

    def predict(self, X):
        X_b = np.column_stack([X, np.ones(len(X))])
        return X_b @ self.weights


class LogisticRegressionModel:
    """Simple logistic regression via gradient descent for skip prediction."""

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
#  Extract features from sentences
# --------------------------------------------------------------------------- #

def extract_features(sentences, subtlex):
    """Extract per-word features and targets from aggregated sentences."""
    features = []  # (log_freq, predictability, word_length)
    targets_ffd = []
    targets_gaze = []
    targets_trt = []
    targets_skip = []
    tokens_list = []

    for agg in sentences:
        for i, token in enumerate(agg.tokens):
            lf = get_log_freq(token, subtlex)
            pred = agg.predictabilities[i]
            wlen = len(token)

            features.append([lf, pred, wlen])
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

    return {'r_ffd': r_ffd, 'r_gaze': r_gaze, 'r_trt': r_trt, 'r_skip': r_skip,
            'mae_ffd': mae_ffd, 'mae_trt': mae_trt}


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    print("=" * 80)
    print("BASELINE 1: Linear Regression (log_freq + predictability + word_length)")
    print("=" * 80)

    # Load SUBTLEXus
    print("\nLoading SUBTLEXus...")
    subtlex = load_subtlexus(os.path.join(data_dir, 'SUBTLEXus.txt'))
    print(f"  {len(subtlex):,} entries")

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
    test_ids = set(sd.text_id for sd in test_raw)
    train_agg = [a for a in geco_agg if a.text_id in train_ids]
    val_agg = [a for a in geco_agg if a.text_id in val_ids]
    test_agg = [a for a in geco_agg if a.text_id not in train_ids and a.text_id not in val_ids]

    print(f"  Train: {len(train_agg)} sentences | Val: {len(val_agg)} | Test: {len(test_agg)}")

    # Extract features
    print("\nExtracting features...")
    X_train, y_ffd_train, y_gaze_train, y_trt_train, y_skip_train, _ = extract_features(train_agg, subtlex)
    X_val, y_ffd_val, y_gaze_val, y_trt_val, y_skip_val, _ = extract_features(val_agg, subtlex)
    X_test, y_ffd_test, y_gaze_test, y_trt_test, y_skip_test, _ = extract_features(test_agg, subtlex)
    print(f"  Train: {len(X_train)} words | Val: {len(X_val)} | Test: {len(X_test)}")

    # Train models
    print("\nTraining...")
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

    # Print learned weights
    print("\n  Learned weights (FFD): log_freq={:.2f}, pred={:.2f}, wlen={:.2f}, bias={:.2f}".format(
        *ffd_model.weights))
    print("  Learned weights (TRT): log_freq={:.2f}, pred={:.2f}, wlen={:.2f}, bias={:.2f}".format(
        *trt_model.weights))

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
    X_provo, y_ffd_provo, y_gaze_provo, y_trt_provo, y_skip_provo, _ = extract_features(provo_agg, subtlex)

    evaluate("Provo (cross-corpus)", X_provo, y_ffd_provo, y_gaze_provo,
             y_trt_provo, y_skip_provo,
             ffd_model, gaze_model, trt_model, skip_model)

    print("\nDone!")


if __name__ == "__main__":
    main()
