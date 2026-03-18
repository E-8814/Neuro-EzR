"""
Baseline 3: LightGBM / Gradient Boosting with handcrafted features.

Inspired by the CMCL 2021 Shared Task winner (Bestgen 2021) who used
gradient boosting with extensive feature engineering:
  - Word frequency (log, from SUBTLEXus)
  - Cloze predictability
  - Word length
  - GPT-2 surprisal
  - Previous word features (freq, length, surprisal)
  - Next word features (freq, length)
  - Sentence position
  - Word n-gram frequency proxies

Trained on GECO, evaluated on GECO test + full Provo.

Usage:
    python3 -u previous_implementations_of_word_level_predictions/lightgbm_baseline.py
"""

import os
import sys
import csv
import math
import time
import pickle
import numpy as np
from collections import defaultdict

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src_v2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'original_ezreader'))

from data_loader import load_provo, aggregate_by_sentence
from geco_loader import load_geco, split_geco

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier


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
#  GPT-2 Surprisal (reused from baseline 2)
# --------------------------------------------------------------------------- #

class GPT2SurprisalComputer:
    def __init__(self, model_name="gpt2", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def compute_surprisal(self, tokens):
        text = " ".join(tokens)
        encoding = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = encoding["input_ids"][0]

        outputs = self.model(input_ids=input_ids.unsqueeze(0))
        log_probs = torch.nn.functional.log_softmax(outputs.logits[0], dim=-1)

        subword_surprisals = {i: [] for i in range(len(tokens))}

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
                if pos < len(input_ids):
                    if pos > 0:
                        surp = -log_probs[pos - 1, input_ids[pos]].item() / math.log(2)
                    else:
                        surp = -log_probs[0, input_ids[0]].item() / math.log(2)
                    subword_surprisals[w_idx].append(max(0.0, surp))
            subword_offset += n_subs

        return [sum(subword_surprisals[i]) if subword_surprisals[i] else 10.0
                for i in range(len(tokens))]


# --------------------------------------------------------------------------- #
#  Rich feature extraction (CMCL 2021 style)
# --------------------------------------------------------------------------- #

FEATURE_NAMES = [
    'log_freq', 'predictability', 'word_length', 'surprisal',
    'prev_log_freq', 'prev_word_length', 'prev_surprisal',
    'next_log_freq', 'next_word_length',
    'sentence_position', 'sentence_position_norm',
    'freq_x_len', 'surp_x_len',
    'is_capitalized', 'has_punctuation',
    'n_vowels', 'n_consonants',
]


def extract_rich_features(sentences, subtlex, surprisal_computer):
    """Extract rich per-word features for gradient boosting."""
    features = []
    targets_ffd = []
    targets_gaze = []
    targets_trt = []
    targets_skip = []

    total = len(sentences)
    for s_idx, agg in enumerate(sentences):
        if (s_idx + 1) % 100 == 0:
            print(f"    Sentence {s_idx+1}/{total}...", flush=True)

        tokens = agg.tokens
        n = len(tokens)

        # Compute surprisal for this sentence
        surprisals = surprisal_computer.compute_surprisal(tokens)

        # Per-word features
        log_freqs = [get_log_freq(t, subtlex) for t in tokens]
        wlens = [len(t) for t in tokens]
        preds = agg.predictabilities

        for i in range(n):
            lf = log_freqs[i]
            pred = preds[i]
            wl = wlens[i]
            surp = surprisals[i]

            # Previous word features
            prev_lf = log_freqs[i - 1] if i > 0 else lf
            prev_wl = wlens[i - 1] if i > 0 else wl
            prev_surp = surprisals[i - 1] if i > 0 else surp

            # Next word features
            next_lf = log_freqs[i + 1] if i < n - 1 else lf
            next_wl = wlens[i + 1] if i < n - 1 else wl

            # Position features
            sent_pos = i
            sent_pos_norm = i / max(1, n - 1)

            # Interaction features
            freq_x_len = lf * wl
            surp_x_len = surp * wl

            # Surface features
            is_cap = 1.0 if tokens[i][0].isupper() else 0.0
            has_punct = 1.0 if any(c in tokens[i] for c in '.,;:!?') else 0.0

            # Character composition
            vowels = sum(1 for c in tokens[i].lower() if c in 'aeiou')
            consonants = sum(1 for c in tokens[i].lower() if c.isalpha() and c not in 'aeiou')

            feat = [
                lf, pred, wl, surp,
                prev_lf, prev_wl, prev_surp,
                next_lf, next_wl,
                sent_pos, sent_pos_norm,
                freq_x_len, surp_x_len,
                is_cap, has_punct,
                vowels, consonants,
            ]
            features.append(feat)
            targets_ffd.append(agg.mean_ffd[i])
            targets_gaze.append(agg.mean_gaze[i])
            targets_trt.append(agg.mean_trt[i])
            targets_skip.append(agg.skip_rate[i])

    return (np.array(features), np.array(targets_ffd), np.array(targets_gaze),
            np.array(targets_trt), np.array(targets_skip))


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
    pred_skip_raw = skip_model.predict(features)
    # For LightGBM classifier, predict returns class; for regressor-as-skip, clamp
    if hasattr(skip_model, 'predict_proba'):
        pred_skip = skip_model.predict_proba(features)[:, 1]
    else:
        pred_skip = np.clip(pred_skip_raw, 0, 1)

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
    save_dir = os.path.join(os.path.dirname(__file__), "checkpoints_lgbm")
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 80)
    print("BASELINE 3: Gradient Boosting (CMCL 2021 Style)")
    if HAS_LIGHTGBM:
        print("  Backend: LightGBM")
    else:
        print("  Backend: sklearn GradientBoosting (LightGBM not installed)")
    print(f"  Features: {len(FEATURE_NAMES)} features per word")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load SUBTLEXus
    print("\nLoading SUBTLEXus...")
    subtlex = load_subtlexus(os.path.join(data_dir, 'SUBTLEXus.txt'))

    # Load GPT-2 for surprisal
    print("\nLoading GPT-2 for surprisal...")
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

    # Extract features
    print("\nExtracting features (train)...")
    X_train, y_ffd_train, y_gaze_train, y_trt_train, y_skip_train = \
        extract_rich_features(train_agg, subtlex, surp)

    print("\nExtracting features (val)...")
    X_val, y_ffd_val, y_gaze_val, y_trt_val, y_skip_val = \
        extract_rich_features(val_agg, subtlex, surp)

    print("\nExtracting features (test)...")
    X_test, y_ffd_test, y_gaze_test, y_trt_test, y_skip_test = \
        extract_rich_features(test_agg, subtlex, surp)

    print(f"\n  Feature shape: {X_train.shape}")

    # Train models
    print("\nTraining gradient boosting models...")
    t0 = time.time()

    if HAS_LIGHTGBM:
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 63,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbose': -1,
        }

        print("  Training FFD model...")
        ffd_model = lgb.LGBMRegressor(**lgb_params)
        ffd_model.fit(X_train, y_ffd_train,
                      eval_set=[(X_val, y_ffd_val)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])

        print("  Training Gaze model...")
        gaze_model = lgb.LGBMRegressor(**lgb_params)
        gaze_model.fit(X_train, y_gaze_train,
                       eval_set=[(X_val, y_gaze_val)],
                       callbacks=[lgb.early_stopping(50, verbose=False)])

        print("  Training TRT model...")
        trt_model = lgb.LGBMRegressor(**lgb_params)
        trt_model.fit(X_train, y_trt_train,
                      eval_set=[(X_val, y_trt_val)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])

        print("  Training Skip model...")
        skip_params = lgb_params.copy()
        skip_params['objective'] = 'binary'
        skip_params['metric'] = 'binary_logloss'
        # Binarize skip targets for classification (threshold at 0.5)
        y_skip_binary = (y_skip_train > 0.5).astype(float)
        y_skip_val_binary = (y_skip_val > 0.5).astype(float)
        skip_model = lgb.LGBMClassifier(**skip_params)
        skip_model.fit(X_train, y_skip_binary,
                       eval_set=[(X_val, y_skip_val_binary)],
                       callbacks=[lgb.early_stopping(50, verbose=False)])

    else:
        # Fallback to sklearn
        print("  Training FFD model (sklearn)...")
        ffd_model = GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=20)
        ffd_model.fit(X_train, y_ffd_train)

        print("  Training Gaze model (sklearn)...")
        gaze_model = GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=20)
        gaze_model.fit(X_train, y_gaze_train)

        print("  Training TRT model (sklearn)...")
        trt_model = GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=20)
        trt_model.fit(X_train, y_trt_train)

        print("  Training Skip model (sklearn)...")
        y_skip_binary = (y_skip_train > 0.5).astype(int)
        skip_model = GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=20)
        skip_model.fit(X_train, y_skip_binary)

    elapsed = time.time() - t0
    print(f"  Training completed in {elapsed:.1f}s")

    # Feature importance
    if HAS_LIGHTGBM:
        print("\n  Feature importance (TRT model):")
        importances = trt_model.feature_importances_
        for fname, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1]):
            print(f"    {fname:<25s} {imp:>6d}")
    else:
        print("\n  Feature importance (TRT model):")
        importances = trt_model.feature_importances_
        for fname, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1]):
            print(f"    {fname:<25s} {imp:>8.4f}")

    # Save models
    with open(os.path.join(save_dir, "lgbm_models.pkl"), "wb") as f:
        pickle.dump({
            'ffd': ffd_model, 'gaze': gaze_model,
            'trt': trt_model, 'skip': skip_model,
        }, f)
    print(f"\n  Models saved to {save_dir}")

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
    X_provo, y_ffd_provo, y_gaze_provo, y_trt_provo, y_skip_provo = \
        extract_rich_features(provo_agg, subtlex, surp)

    evaluate("Provo (cross-corpus)", X_provo, y_ffd_provo, y_gaze_provo,
             y_trt_provo, y_skip_provo,
             ffd_model, gaze_model, trt_model, skip_model)

    print("\nDone!")


if __name__ == "__main__":
    main()
