"""
Adapter: Run the Ohio State CMCL 2021 RoBERTa model on GECO/Provo.

This uses the EXACT model architecture from:
    Byung-Doh Oh & Shayan Fazeli (2021)
    "Team Ohio State at CMCL 2021 Shared Task:
     Fine-Tuned RoBERTa for Eye-Tracking Data Prediction"
    https://github.com/byungdoh/cmcl21_st

The original code was designed for the ZuCo corpus. This adapter:
  1. Loads GECO data using our data loaders
  2. Converts to the format expected by their RobertaForGazePrediction model
  3. Trains separate models for FFD, Gaze, TRT, and Skip
  4. Evaluates on GECO test + full Provo (cross-corpus)

Architecture (from model.py):
    RoBERTa → first-subword selection → concat(repr, wlen, prop)
    → Dropout(0.1) → Linear(input_dim+2, 385) → ReLU → Dropout(0.1)
    → Linear(385, 1)

Usage:
    # Default: roberta-base (faster, fits in GPU memory)
    python3 -u previous_implementations_of_word_level_predictions/run_ohio_state_on_geco.py

    # Original paper config: roberta-large (slower, better)
    python3 -u previous_implementations_of_word_level_predictions/run_ohio_state_on_geco.py --model roberta-large

    # Specific metric only
    python3 -u previous_implementations_of_word_level_predictions/run_ohio_state_on_geco.py --metric trt
"""

import os
import sys
import time
import math
import random
import argparse
import logging

import torch
import numpy as np
from torch.nn import MSELoss, BCELoss
from transformers import RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cmcl21_st'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src_v2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'original_ezreader'))

from model import RobertaForGazePrediction
from data_loader import load_provo, aggregate_by_sentence
from geco_loader import load_geco, split_geco


# --------------------------------------------------------------------------- #
#  Logger
# --------------------------------------------------------------------------- #

class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def isatty(self):
        return self.terminal.isatty()


# --------------------------------------------------------------------------- #
#  Data conversion: our AggregatedSentence → Ohio State format
# --------------------------------------------------------------------------- #

def replace_bytes(s):
    """From original preprocess.py"""
    s = s.replace("Ġ", "")
    s = s.replace("âĢĵ", "–")
    return s


def convert_to_ohio_format(sentences, tokenizer):
    """
    Convert list of AggregatedSentence to Ohio State format.

    Returns:
        sent_strings: list of str (raw sentence strings)
        sent_first_idx: list of list of int (first subword index per word)
        sent_wlen: list of list of float (word lengths)
        sent_prop: list of list of float (sentence position proportions)
        sent_ffd: list of list of float
        sent_gaze: list of list of float
        sent_trt: list of list of float
        sent_skip: list of list of float
        sent_words: list of list of str
    """
    sent_strings = []
    sent_first_idx = []
    sent_wlen = []
    sent_prop = []
    sent_ffd = []
    sent_gaze = []
    sent_trt = []
    sent_skip = []
    sent_words = []
    skipped = 0

    for agg in sentences:
        tokens = agg.tokens
        string = " ".join(tokens)
        n = len(tokens)

        # Compute first-subword indices (exact logic from original preprocess.py)
        tokenized_ids = tokenizer(string)["input_ids"]
        tokenized_list = tokenizer.convert_ids_to_tokens(tokenized_ids)

        curr_idx = 0
        first_idx = []
        failed = False

        for word in tokens:
            found = False
            start_idx = curr_idx
            while curr_idx < len(tokenized_list):
                try:
                    if word.startswith(replace_bytes(tokenized_list[curr_idx])):
                        first_idx.append(curr_idx)
                        curr_idx += 1
                        found = True
                        break
                except Exception:
                    pass
                curr_idx += 1

            if not found:
                # Alignment failed for this sentence
                failed = True
                break

        if failed or len(first_idx) != n:
            skipped += 1
            continue

        # Word lengths
        wlens = [float(len(t)) for t in tokens]

        # Sentence position proportions (matching original: word_id / max_word_id)
        props = [float(i) / max(1, n - 1) for i in range(n)]

        sent_strings.append(string)
        sent_first_idx.append(first_idx)
        sent_wlen.append(wlens)
        sent_prop.append(props)
        sent_ffd.append(list(agg.mean_ffd))
        sent_gaze.append(list(agg.mean_gaze))
        sent_trt.append(list(agg.mean_trt))
        sent_skip.append(list(agg.skip_rate))
        sent_words.append(tokens)

    if skipped > 0:
        print(f"    Skipped {skipped} sentences due to tokenization alignment failures")

    return (sent_strings, sent_first_idx, sent_wlen, sent_prop,
            sent_ffd, sent_gaze, sent_trt, sent_skip, sent_words)


# --------------------------------------------------------------------------- #
#  Per-participant data conversion (for training on raw observations)
# --------------------------------------------------------------------------- #

def convert_raw_to_ohio_format(raw_data, tokenizer):
    """
    Convert list of SentenceData (per-participant) to Ohio State format.
    Uses per-participant reading times (not averaged).
    """
    sent_strings = []
    sent_first_idx = []
    sent_wlen = []
    sent_prop = []
    sent_ffd = []
    sent_gaze = []
    sent_trt = []
    sent_skip = []
    sent_words = []
    skipped = 0

    # Cache tokenization by sentence string to avoid re-tokenizing
    alignment_cache = {}

    for sd in raw_data:
        tokens = sd.tokens
        string = " ".join(tokens)
        n = len(tokens)

        # Check cache
        if string in alignment_cache:
            first_idx = alignment_cache[string]
        else:
            tokenized_ids = tokenizer(string)["input_ids"]
            tokenized_list = tokenizer.convert_ids_to_tokens(tokenized_ids)

            curr_idx = 0
            first_idx = []
            failed = False

            for word in tokens:
                found = False
                while curr_idx < len(tokenized_list):
                    try:
                        if word.startswith(replace_bytes(tokenized_list[curr_idx])):
                            first_idx.append(curr_idx)
                            curr_idx += 1
                            found = True
                            break
                    except Exception:
                        pass
                    curr_idx += 1

                if not found:
                    failed = True
                    break

            if failed or len(first_idx) != n:
                alignment_cache[string] = None
                skipped += 1
                continue

            alignment_cache[string] = first_idx

        if first_idx is None:
            skipped += 1
            continue

        wlens = [float(len(t)) for t in tokens]
        props = [float(i) / max(1, n - 1) for i in range(n)]

        sent_strings.append(string)
        sent_first_idx.append(first_idx)
        sent_wlen.append(wlens)
        sent_prop.append(props)
        sent_ffd.append([w.first_fixation_duration for w in sd.words])
        sent_gaze.append([w.gaze_duration for w in sd.words])
        sent_trt.append([w.total_reading_time for w in sd.words])
        sent_skip.append([1.0 if w.was_skipped else 0.0 for w in sd.words])
        sent_words.append(tokens)

    if skipped > 0:
        print(f"    Skipped {skipped} observations due to tokenization alignment")

    return (sent_strings, sent_first_idx, sent_wlen, sent_prop,
            sent_ffd, sent_gaze, sent_trt, sent_skip, sent_words)


# --------------------------------------------------------------------------- #
#  Training loop (adapted from original main.py)
# --------------------------------------------------------------------------- #

def train_one_metric(
    metric_name,
    train_data,
    val_data,
    pretrained_model_name,
    save_dir,
    num_epochs=32,
    batch_size=4,
    learning_rate=5e-5,
    hidden_dim=385,
    dropout_1=0.1,
    dropout_2=0.1,
    activation="relu",
    max_grad_norm=1.0,
    warmup_prop=0.1,
    weight_decay=0.01,
    seed=42,
    device=None,
):
    """
    Train the Ohio State RobertaForGazePrediction model for a single metric.
    This follows the original training code exactly.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    random.seed(seed)

    (sent_strings, sent_first_idx, sent_wlen, sent_prop,
     sent_ffd, sent_gaze, sent_trt, sent_skip, sent_words) = train_data

    (d_sent_strings, d_sent_first_idx, d_sent_wlen, d_sent_prop,
     d_sent_ffd, d_sent_gaze, d_sent_trt, d_sent_skip, d_sent_words) = val_data

    # Select the target metric
    metric_map = {'ffd': sent_ffd, 'gaze': sent_gaze, 'trt': sent_trt, 'skip': sent_skip}
    d_metric_map = {'ffd': d_sent_ffd, 'gaze': d_sent_gaze, 'trt': d_sent_trt, 'skip': d_sent_skip}
    dv = list(metric_map[metric_name])
    d_dv = list(d_metric_map[metric_name])

    # Build model (exact architecture from Ohio State)
    input_size = {"roberta-base": 768, "roberta-large": 1024}
    roberta = RobertaModel.from_pretrained(pretrained_model_name)
    tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name)
    model = RobertaForGazePrediction(
        pretrained=roberta,
        input_dim=input_size[pretrained_model_name],
        dropout_1=dropout_1,
        hidden_dim=hidden_dim,
        activation=activation,
        dropout_2=dropout_2,
    )
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Model: {n_params:,} params ({n_trainable:,} trainable)")

    # Optimizer (exact same as original)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = MSELoss()

    # Linear warmup schedule (exact same as original)
    num_train_steps = math.ceil(len(sent_strings) / batch_size) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, round(warmup_prop * num_train_steps), num_train_steps
    )

    os.makedirs(save_dir, exist_ok=True)
    lowest_mse = 1e8
    best_epoch = 0

    for epoch in range(num_epochs):
        t0 = time.time()

        # Shuffle training data (exact same as original)
        combined = list(zip(sent_strings, sent_first_idx, sent_wlen, sent_prop, dv))
        random.shuffle(combined)
        sent_strings_s, sent_first_idx_s, sent_wlen_s, sent_prop_s, dv_s = zip(*combined)

        model.train()
        train_loss = 0.0
        n_batches = 0

        for j in range(0, len(sent_strings_s), batch_size):
            batch_sentences = sent_strings_s[j:j+batch_size]
            first_idx = sent_first_idx_s[j:j+batch_size]
            flat_wlen = [w for sublist in sent_wlen_s[j:j+batch_size] for w in sublist]
            flat_prop = [p for sublist in sent_prop_s[j:j+batch_size] for p in sublist]
            flat_dv = [v for sublist in dv_s[j:j+batch_size] for v in sublist]

            encoded_inputs = tokenizer(list(batch_sentences), padding=True,
                                       truncation=True, return_tensors="pt")
            wlen_tensor = torch.FloatTensor(flat_wlen).view(-1, 1).to(device)
            prop_tensor = torch.FloatTensor(flat_prop).view(-1, 1).to(device)
            dv_tensor = torch.FloatTensor(flat_dv).view(-1, 1).to(device)
            encoded_inputs = encoded_inputs.to(device)

            optimizer.zero_grad()
            outputs = model(**encoded_inputs, first_idx=first_idx,
                          wlen=wlen_tensor, prop=prop_tensor)
            loss = criterion(outputs, dv_tensor)
            train_loss += loss.item()
            n_batches += 1

            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

        avg_train_loss = train_loss / max(n_batches, 1)

        # Validation
        model.eval()
        dev_loss = 0.0
        dev_points = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for j in range(0, len(d_sent_strings), 20):  # dev_batch_size=20 (original default)
                d_batch = d_sent_strings[j:j+20]
                d_first = d_sent_first_idx[j:j+20]
                d_flat_wlen = [w for sublist in d_sent_wlen[j:j+20] for w in sublist]
                d_flat_prop = [p for sublist in d_sent_prop[j:j+20] for p in sublist]
                d_flat_dv = [v for sublist in d_dv[j:j+20] for v in sublist]

                d_enc = tokenizer(list(d_batch), padding=True, truncation=True,
                                  return_tensors="pt").to(device)
                d_wlen = torch.FloatTensor(d_flat_wlen).view(-1, 1).to(device)
                d_prop = torch.FloatTensor(d_flat_prop).view(-1, 1).to(device)
                d_dv_t = torch.FloatTensor(d_flat_dv).view(-1, 1).to(device)

                outputs = model(**d_enc, first_idx=d_first, wlen=d_wlen, prop=d_prop)
                loss = criterion(outputs, d_dv_t)
                dev_loss += loss.item() * len(d_flat_dv)
                dev_points += len(d_flat_dv)

                all_preds.extend(outputs.cpu().numpy().flatten().tolist())
                all_targets.extend(d_flat_dv)

        avg_dev_mse = dev_loss / max(dev_points, 1)

        # Compute correlation
        preds_arr = np.array(all_preds)
        targets_arr = np.array(all_targets)
        if np.std(preds_arr) > 0 and np.std(targets_arr) > 0:
            r = np.corrcoef(preds_arr, targets_arr)[0, 1]
        else:
            r = 0.0
        mae = np.mean(np.abs(preds_arr - targets_arr))

        elapsed = time.time() - t0

        is_best = avg_dev_mse < lowest_mse
        marker = " ** BEST" if is_best else ""
        print(f"    Epoch {epoch+1:3d}/{num_epochs} | {elapsed:.1f}s | "
              f"train_MSE={avg_train_loss:.1f} | val_MSE={avg_dev_mse:.1f} | "
              f"r={r:.3f} | MAE={mae:.1f}{marker}")

        if is_best:
            lowest_mse = avg_dev_mse
            best_epoch = epoch + 1
            model_path = os.path.join(save_dir, f"best_model_{metric_name}.pth")
            torch.save(model.state_dict(), model_path)

    print(f"    Best epoch: {best_epoch}, best val MSE: {lowest_mse:.2f}")
    return model, lowest_mse


# --------------------------------------------------------------------------- #
#  Evaluation
# --------------------------------------------------------------------------- #

def evaluate_model(model, data, metric_name, tokenizer, device):
    """Evaluate a trained model and return predictions + targets."""
    (sent_strings, sent_first_idx, sent_wlen, sent_prop,
     sent_ffd, sent_gaze, sent_trt, sent_skip, sent_words) = data

    metric_map = {'ffd': sent_ffd, 'gaze': sent_gaze, 'trt': sent_trt, 'skip': sent_skip}
    dv = metric_map[metric_name]

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for j in range(0, len(sent_strings), 20):
            batch = sent_strings[j:j+20]
            first_idx = sent_first_idx[j:j+20]
            flat_wlen = [w for sublist in sent_wlen[j:j+20] for w in sublist]
            flat_prop = [p for sublist in sent_prop[j:j+20] for p in sublist]
            flat_dv = [v for sublist in dv[j:j+20] for v in sublist]

            enc = tokenizer(list(batch), padding=True, truncation=True,
                           return_tensors="pt").to(device)
            wlen_t = torch.FloatTensor(flat_wlen).view(-1, 1).to(device)
            prop_t = torch.FloatTensor(flat_prop).view(-1, 1).to(device)

            outputs = model(**enc, first_idx=first_idx, wlen=wlen_t, prop=prop_t)
            all_preds.extend(outputs.cpu().numpy().flatten().tolist())
            all_targets.extend(flat_dv)

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    if np.std(preds) > 0 and np.std(targets) > 0:
        r = np.corrcoef(preds, targets)[0, 1]
    else:
        r = 0.0
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))

    return r, mae, rmse, preds, targets


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="roberta-base",
                        choices=["roberta-base", "roberta-large"],
                        help="Pretrained model (default: roberta-base)")
    parser.add_argument("--epochs", type=int, default=32,
                        help="Training epochs per metric (default: 32, same as original)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size (default: 4, same as original)")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate (default: 5e-5, same as original)")
    parser.add_argument("--metric", type=str, default="all",
                        choices=["all", "ffd", "gaze", "trt", "skip"],
                        help="Which metric to train (default: all)")
    parser.add_argument("--use-raw", action="store_true",
                        help="Train on per-participant data (like original) instead of aggregated")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override save_dir (default: checkpoints_ohio_state_<model>)")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    save_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__),
        f"checkpoints_ohio_state_{args.model.replace('-', '_')}",
    )
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Redirect stdout
    sys.stdout = DualLogger(os.path.join(save_dir, "training_log.txt"))

    print("=" * 90)
    print("Ohio State CMCL 2021: RobertaForGazePrediction")
    print(f"  Original paper: Oh & Fazeli (2021)")
    print(f"  Code: https://github.com/byungdoh/cmcl21_st")
    print(f"  Model: {args.model}")
    print(f"  Adapted to train on GECO, evaluate on GECO test + Provo")
    print("=" * 90)

    print(f"\nDevice: {device}")

    # Load tokenizer
    print(f"\nLoading RoBERTa tokenizer ({args.model})...")
    tokenizer = RobertaTokenizer.from_pretrained(args.model)

    # Load GECO
    print("\nLoading GECO...")
    reading_path = os.path.join(data_dir, "Geco_MonolingualReadingData.csv")
    material_path = os.path.join(data_dir, "Geco_EnglishMaterial.csv")
    pred_path = os.path.join(data_dir, "geco_predictability.pkl")
    geco_raw = load_geco(reading_path, material_path, pred_path)
    train_raw, val_raw, test_raw = split_geco(geco_raw)

    if args.use_raw:
        # Train on per-participant data (more data, noisier)
        print("\n  Converting training data (per-participant)...")
        train_data = convert_raw_to_ohio_format(train_raw, tokenizer)
        print(f"    Train: {len(train_data[0])} sentence-observations")

        print("  Converting val data (per-participant)...")
        val_data = convert_raw_to_ohio_format(val_raw, tokenizer)
        print(f"    Val: {len(val_data[0])} sentence-observations")
    else:
        # Train on aggregated data (cleaner targets)
        geco_agg = aggregate_by_sentence(geco_raw, min_participants=5)
        train_ids = set(sd.text_id for sd in train_raw)
        val_ids = set(sd.text_id for sd in val_raw)
        train_agg = [a for a in geco_agg if a.text_id in train_ids]
        val_agg = [a for a in geco_agg if a.text_id in val_ids]

        print("\n  Converting training data (aggregated)...")
        train_data = convert_to_ohio_format(train_agg, tokenizer)
        print(f"    Train: {len(train_data[0])} sentences")

        print("  Converting val data (aggregated)...")
        val_data = convert_to_ohio_format(val_agg, tokenizer)
        print(f"    Val: {len(val_data[0])} sentences")

    # Prepare test data (always aggregated for clean evaluation)
    geco_agg_all = aggregate_by_sentence(geco_raw, min_participants=5)
    test_ids = set(sd.text_id for sd in test_raw)
    test_agg = [a for a in geco_agg_all if a.text_id in test_ids]

    print("\n  Converting GECO test data...")
    test_data = convert_to_ohio_format(test_agg, tokenizer)
    print(f"    Test: {len(test_data[0])} sentences")

    # Load Provo
    print("\n  Loading Provo for cross-corpus evaluation...")
    et_path = os.path.join(data_dir, "Provo_Corpus-Eyetracking_Data.csv")
    provo_raw = load_provo(et_path)
    provo_agg = aggregate_by_sentence(provo_raw, min_participants=10)

    print("  Converting Provo data...")
    provo_data = convert_to_ohio_format(provo_agg, tokenizer)
    print(f"    Provo: {len(provo_data[0])} sentences")

    # Determine which metrics to train
    if args.metric == "all":
        metrics = ["ffd", "gaze", "trt", "skip"]
    else:
        metrics = [args.metric]

    # Train and evaluate each metric
    results = {}

    for metric in metrics:
        print(f"\n{'=' * 90}")
        print(f"  Training {metric.upper()} model")
        print(f"{'=' * 90}")

        model, best_mse = train_one_metric(
            metric_name=metric,
            train_data=train_data,
            val_data=val_data,
            pretrained_model_name=args.model,
            save_dir=save_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            hidden_dim=385,          # original default
            dropout_1=0.1,           # original default
            dropout_2=0.1,           # original default
            activation="relu",       # original default
            max_grad_norm=1.0,       # original default
            warmup_prop=0.1,         # original default
            weight_decay=0.01,       # original default
            seed=args.seed,
            device=device,
        )

        # Reload best model
        input_size = {"roberta-base": 768, "roberta-large": 1024}
        roberta = RobertaModel.from_pretrained(args.model)
        best_model = RobertaForGazePrediction(
            pretrained=roberta,
            input_dim=input_size[args.model],
            dropout_1=0.1, hidden_dim=385, activation="relu", dropout_2=0.1,
        ).to(device)
        best_model.load_state_dict(
            torch.load(os.path.join(save_dir, f"best_model_{metric}.pth"),
                       map_location=device, weights_only=False))

        # Evaluate on GECO test
        r_test, mae_test, rmse_test, _, _ = evaluate_model(
            best_model, test_data, metric, tokenizer, device)

        # Evaluate on Provo
        r_provo, mae_provo, rmse_provo, _, _ = evaluate_model(
            best_model, provo_data, metric, tokenizer, device)

        results[metric] = {
            'geco_test': {'r': r_test, 'mae': mae_test, 'rmse': rmse_test},
            'provo': {'r': r_provo, 'mae': mae_provo, 'rmse': rmse_provo},
        }

        print(f"\n    {metric.upper()} Results:")
        print(f"      GECO test:  r={r_test:.3f}  MAE={mae_test:.1f}  RMSE={rmse_test:.1f}")
        print(f"      Provo:      r={r_provo:.3f}  MAE={mae_provo:.1f}  RMSE={rmse_provo:.1f}")

        # Clean up GPU memory before next metric
        del model, best_model, roberta
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Final summary
    print(f"\n{'=' * 90}")
    print("FINAL SUMMARY: Ohio State RobertaForGazePrediction")
    print(f"  Model: {args.model} | Epochs: {args.epochs} | LR: {args.lr}")
    print(f"  Training: {'per-participant' if args.use_raw else 'aggregated'} GECO data")
    print(f"{'=' * 90}")

    header = f"  {'Metric':<10s} {'GECO test r':>12s} {'GECO MAE':>10s} {'Provo r':>10s} {'Provo MAE':>10s}"
    print(header)
    print(f"  {'-' * 55}")
    for metric in metrics:
        r = results[metric]
        print(f"  {metric.upper():<10s} "
              f"{r['geco_test']['r']:>12.3f} {r['geco_test']['mae']:>10.1f} "
              f"{r['provo']['r']:>10.3f} {r['provo']['mae']:>10.1f}")

    print(f"\nCheckpoints saved to: {save_dir}")
    print("\nDone!")


if __name__ == "__main__":
    main()
