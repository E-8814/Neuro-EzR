"""
Adapter: Run the Toronto CL CMCL 2021 RoBERTa model on GECO/Provo.

This uses the EXACT model architecture from:
    Mathew Jain et al. (2021)
    "TorontoCL at CMCL 2021 Shared Task: RoBERTa with Multi-Stage Fine-Tuning
     for Eye-Tracking Prediction"
    https://github.com/SPOClab-ca/cmcl21-torontocl (3rd place)

Architecture:
    RoBERTa-base (768) -> first-subword selection -> Linear(768, 5)
    Predicts: nFix, FFD, GPT (=Gaze), TRT, fixProp (=1-skip)

This adapter:
  1. Loads GECO data using our data loaders
  2. Converts to the DataFrame format expected by Toronto CL code
  3. Trains the model on GECO
  4. Evaluates on GECO test + full Provo (cross-corpus)

Usage:
    python3 -u archive/baselines/run_toronto_on_geco.py
    python3 -u archive/baselines/run_toronto_on_geco.py --epochs 150 --ensemble 3
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch

# --- Path setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TORONTO_DIR = os.path.join(SCRIPT_DIR, 'cmcl21_torontocl')
ROOT_DIR = os.path.join(SCRIPT_DIR, '..', '..')
SRC_V2_DIR = os.path.join(ROOT_DIR, 'src_v2')
EZR_DIR = os.path.join(ROOT_DIR, 'archive', 'original_ezreader')

sys.path.insert(0, TORONTO_DIR)
sys.path.insert(0, SRC_V2_DIR)
sys.path.insert(0, EZR_DIR)

import src.model as toronto_model
import src.dataloader as toronto_dataloader
import src.eval_metric as toronto_eval
from data_loader import load_provo, aggregate_by_sentence
from geco_loader import load_geco, split_geco

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --------------------------------------------------------------------------- #
#  Monkey-patch Toronto CL dataloader to fix Ġ detection for newer transformers
# --------------------------------------------------------------------------- #

_OrigGetItem = toronto_dataloader.EyeTrackingCSV.__getitem__

def _patched_getitem(self, ix):
    """Use word_ids() instead of fragile Ġ character detection."""
    input_ids = self.ids['input_ids'][ix]
    attention_mask = self.ids['attention_mask'][ix]
    input_tokens = [self.tokenizer.convert_ids_to_tokens(x) for x in input_ids]

    # Robust first-subword detection using word_ids()
    word_ids = self.ids.word_ids(ix)
    seen_words = set()
    is_first_subword = []
    for wid in word_ids:
        if wid is not None and wid not in seen_words:
            is_first_subword.append(True)
            seen_words.add(wid)
        else:
            is_first_subword.append(False)

    features = -torch.ones((len(input_ids), 5))
    features[is_first_subword] = torch.Tensor(
        self.df[self.df.sentence_id == ix][toronto_dataloader.FEATURES_NAMES].to_numpy()
    )

    return (
        input_tokens,
        torch.LongTensor(input_ids),
        torch.LongTensor(attention_mask),
        features,
    )

toronto_dataloader.EyeTrackingCSV.__getitem__ = _patched_getitem


# --------------------------------------------------------------------------- #
#  Monkey-patch Toronto CL ModelTrainer.train to add checkpointing + early stop
# --------------------------------------------------------------------------- #

import random as _random
import src.eval_metric as _eval_metric

def _patched_train(self, train_df, valid_df=None, num_epochs=5, lr=5e-5,
                   batch_size=16, feature_ids=[0, 1, 2, 3, 4],
                   save_dir=None, patience=15):
    """Patched train with mid-training checkpoint saving and early stopping."""
    train_data = toronto_dataloader.EyeTrackingCSV(train_df, model_name=self.model_name)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()

    best_mae = float('inf')
    epochs_without_improvement = 0

    self.model.train()
    for epoch in range(num_epochs):
        for X_tokens, X_ids, X_attns, Y_true in train_loader:
            opt.zero_grad()
            X_ids = X_ids.to(toronto_model.device)
            X_attns = X_attns.to(toronto_model.device)
            Y_true = Y_true.to(toronto_model.device)
            predict_mask = torch.sum(Y_true, axis=2) >= 0
            Y_pred = self.model(X_ids, X_attns, predict_mask)
            loss = mse(Y_true[:, :, feature_ids], Y_pred[:, :, feature_ids])
            loss.backward()
            opt.step()

        print(f'Epoch: {epoch + 1}')

        if valid_df is not None:
            predict_df = self.predict(valid_df)
            self.model.train()  # predict() sets eval mode
            overall_mae = _eval_metric.evaluate(predict_df, valid_df)

            if overall_mae < best_mae:
                best_mae = overall_mae
                epochs_without_improvement = 0
                if save_dir:
                    ckpt_path = os.path.join(save_dir, "best_model.pt")
                    torch.save(self.model.state_dict(), ckpt_path)
                    print(f'  ** NEW BEST (MAE={best_mae:.4f}) — saved to {ckpt_path}')
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f'  Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)')
                    print(f'  Best MAE: {best_mae:.4f}')
                    # Reload best checkpoint
                    if save_dir:
                        best_path = os.path.join(save_dir, "best_model.pt")
                        if os.path.exists(best_path):
                            self.model.load_state_dict(torch.load(best_path, map_location=toronto_model.device, weights_only=False))
                            print(f'  Reloaded best checkpoint from {best_path}')
                    break

    print(f'  Training complete. Best MAE: {best_mae:.4f}')

toronto_model.ModelTrainer.train = _patched_train


# --------------------------------------------------------------------------- #
#  Logger
# --------------------------------------------------------------------------- #

class Logger(object):
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
        return False


# --------------------------------------------------------------------------- #
#  Convert aggregated data to Toronto CL DataFrame format
# --------------------------------------------------------------------------- #

def aggregated_to_dataframe(agg_data, start_sentence_id=0):
    """
    Convert list of AggregatedSentence to a pandas DataFrame in Toronto CL format.

    Columns: sentence_id, word_id, word, nFix, FFD, GPT, TRT, fixProp

    Mapping:
      - FFD = mean_ffd (ms)
      - GPT = mean_gaze (ms) [GPT = gaze pass time in CMCL terminology]
      - TRT = mean_trt (ms)
      - fixProp = (1 - skip_rate) * 100 [percentage of participants who fixated]
      - nFix = approximated as mean_trt / mean_ffd for fixated words
    """
    rows = []
    sid = start_sentence_id
    for s_idx, agg in enumerate(agg_data):
        # Filter out empty-string words (tokenizer skips them, causing shape mismatch)
        valid_indices = [i for i, t in enumerate(agg.tokens) if t.strip() != '']
        if len(valid_indices) == 0:
            continue
        for new_wid, w_idx in enumerate(valid_indices):
            ffd = agg.mean_ffd[w_idx]
            gaze = agg.mean_gaze[w_idx]
            trt = agg.mean_trt[w_idx]
            skip = agg.skip_rate[w_idx]
            fix_prop = (1.0 - skip) * 100.0

            # Approximate nFix: for fixated words, nFix ≈ TRT / FFD
            if ffd > 0 and trt > 0:
                nfix = trt / ffd
            else:
                nfix = 0.0

            rows.append({
                'sentence_id': sid,
                'word_id': new_wid,
                'word': agg.tokens[w_idx],
                'nFix': nfix,
                'FFD': ffd,
                'GPT': gaze,
                'TRT': trt,
                'fixProp': fix_prop,
            })
        sid += 1

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  Evaluate model on a dataset
# --------------------------------------------------------------------------- #

def evaluate_model(model_trainer, agg_data, dataset_name, start_sid=0):
    """Evaluate trained Toronto CL model on aggregated data."""
    df = aggregated_to_dataframe(agg_data, start_sentence_id=start_sid)
    predict_df = model_trainer.predict(df)

    # Compute Pearson r and MAE for each metric
    results = {}
    for col, our_name in [('FFD', 'ffd'), ('GPT', 'gaze'), ('TRT', 'trt'), ('fixProp', 'skip')]:
        pred = predict_df[col].values
        human = df[col].values

        # Filter valid (non-zero for reading times)
        if col in ('FFD', 'GPT', 'TRT'):
            mask = human > 0
        else:
            mask = np.ones(len(human), dtype=bool)

        pred_valid = pred[mask]
        human_valid = human[mask]

        if len(pred_valid) > 2 and np.std(pred_valid) > 0 and np.std(human_valid) > 0:
            r = np.corrcoef(pred_valid, human_valid)[0, 1]
        else:
            r = 0.0

        mae = np.mean(np.abs(pred_valid - human_valid))
        rmse = np.sqrt(np.mean((pred_valid - human_valid) ** 2))
        bias = np.mean(pred_valid) - np.mean(human_valid)

        results[our_name] = {'r': r, 'mae': mae, 'rmse': rmse, 'bias': bias}

    # Convert fixProp metrics to skip-rate scale for comparison with other models
    # fixProp is in [0, 100], skip_rate is in [0, 1]
    # Our other models report skip in [0, 1] scale
    # For fair comparison: convert fixProp MAE to skip scale: MAE / 100
    results['skip_01'] = {
        'r': results['skip']['r'],  # correlation is scale-invariant
        'mae': results['skip']['mae'] / 100.0,
        'rmse': results['skip']['rmse'] / 100.0,
        'bias': results['skip']['bias'] / 100.0,
    }

    print(f"\n  {dataset_name} Results:")
    print(f"    {'Metric':<10s} {'r':>8s} {'MAE':>10s} {'RMSE':>10s} {'Bias':>10s}")
    print(f"    {'─'*10} {'─'*8} {'─'*10} {'─'*10} {'─'*10}")
    for col, name in [('ffd', 'FFD'), ('gaze', 'Gaze(GPT)'), ('trt', 'TRT')]:
        m = results[col]
        print(f"    {name:<10s} {m['r']:>8.3f} {m['mae']:>9.1f}ms {m['rmse']:>9.1f}ms {m['bias']:>+9.1f}ms")
    m = results['skip_01']
    print(f"    {'Skip':<10s} {m['r']:>8.3f} {m['mae']:>10.4f} {m['rmse']:>10.4f} {m['bias']:>+10.4f}")

    return results


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150,
                        help="Training epochs (Toronto CL used 150 for dev)")
    parser.add_argument("--ensemble", type=int, default=1,
                        help="Number of ensemble models")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override save_dir (default: checkpoints_toronto_<model>)")
    args = parser.parse_args()

    import numpy as _np
    torch.manual_seed(args.seed)
    _np.random.seed(args.seed)
    _random.seed(args.seed)

    save_dir = args.output_dir or os.path.join(
        SCRIPT_DIR, f"checkpoints_toronto_{args.model_name.replace('/', '_')}"
    )
    os.makedirs(save_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(save_dir, "training_log.txt"))

    print("=" * 90)
    print("Toronto CL CMCL 2021: RobertaRegressionModel")
    print("  Original paper: Jain et al. (2021)")
    print("  Code: https://github.com/SPOClab-ca/cmcl21-torontocl (3rd place)")
    print(f"  Model: {args.model_name}")
    print("  Adapted to train on GECO, evaluate on GECO test + Provo")
    print("=" * 90)

    print(f"\nDevice: {device}")

    # ---- Load data ----
    data_dir = os.path.join(ROOT_DIR, "data")

    print("\nLoading GECO...")
    geco_raw = load_geco(
        os.path.join(data_dir, "Geco_MonolingualReadingData.csv"),
        os.path.join(data_dir, "Geco_EnglishMaterial.csv"),
        os.path.join(data_dir, "geco_predictability.pkl"),
    )
    train_raw, val_raw, test_raw = split_geco(geco_raw)

    aggregated = aggregate_by_sentence(geco_raw, min_participants=5)
    train_text_ids = set(sd.text_id for sd in train_raw)
    val_text_ids = set(sd.text_id for sd in val_raw)
    train_agg = [a for a in aggregated if a.text_id in train_text_ids]
    val_agg = [a for a in aggregated if a.text_id in val_text_ids]
    test_agg = [a for a in aggregated if a.text_id not in train_text_ids and a.text_id not in val_text_ids]

    print(f"  Aggregated: {len(train_agg)} train | {len(val_agg)} val | {len(test_agg)} test")

    # Convert to Toronto CL DataFrames
    print("\n  Converting to Toronto CL DataFrame format...")
    train_df = aggregated_to_dataframe(train_agg, start_sentence_id=0)
    val_df = aggregated_to_dataframe(val_agg, start_sentence_id=len(train_agg))
    test_df = aggregated_to_dataframe(test_agg, start_sentence_id=len(train_agg) + len(val_agg))
    print(f"    Train: {len(train_df)} word rows ({train_df.sentence_id.nunique()} sentences)")
    print(f"    Val:   {len(val_df)} word rows ({val_df.sentence_id.nunique()} sentences)")
    print(f"    Test:  {len(test_df)} word rows ({test_df.sentence_id.nunique()} sentences)")

    # Load Provo for cross-corpus evaluation
    print("\n  Loading Provo for cross-corpus evaluation...")
    provo_raw = load_provo(os.path.join(data_dir, "Provo_Corpus-Eyetracking_Data.csv"))
    # min_participants=10 matches the convention used by all other Provo
    # evaluations (lightgbm, linear_regression, eval_ohio_state, etc).
    # Previously this was 5, which evaluated on a different (larger but
    # noisier) word set than other baselines.
    provo_agg = aggregate_by_sentence(provo_raw, min_participants=10)
    provo_df = aggregated_to_dataframe(provo_agg, start_sentence_id=0)
    print(f"    Provo: {len(provo_df)} word rows ({provo_df.sentence_id.nunique()} sentences)")

    # ---- Print data statistics ----
    print(f"\n  GECO train data statistics:")
    print(f"    FFD:     mean={train_df.FFD.mean():.1f}ms  std={train_df.FFD.std():.1f}ms")
    print(f"    GPT:     mean={train_df.GPT.mean():.1f}ms  std={train_df.GPT.std():.1f}ms")
    print(f"    TRT:     mean={train_df.TRT.mean():.1f}ms  std={train_df.TRT.std():.1f}ms")
    print(f"    fixProp: mean={train_df.fixProp.mean():.1f}%  std={train_df.fixProp.std():.1f}%")

    # ---- Train (with optional ensemble) ----
    all_geco_results = []
    all_provo_results = []

    # Train on features [1,2,3,4] = FFD, GPT, TRT, fixProp (skip nFix since it's approximated)
    feature_ids = [1, 2, 3, 4]

    for ens_idx in range(args.ensemble):
        print(f"\n{'=' * 90}")
        print(f"  Training model {ens_idx + 1}/{args.ensemble}")
        print(f"  Epochs: {args.epochs} | LR: {args.lr} | Batch: {args.batch_size}")
        print(f"  Features: FFD, GPT, TRT, fixProp (excluding nFix)")
        print(f"{'=' * 90}")

        # Override device in the Toronto CL code
        toronto_model.device = device

        model_trainer = toronto_model.ModelTrainer(model_name=args.model_name)

        ens_save_dir = os.path.join(save_dir, f"ens{ens_idx}") if args.ensemble > 1 else save_dir
        os.makedirs(ens_save_dir, exist_ok=True)

        t0 = time.time()
        model_trainer.train(
            train_df, val_df,
            num_epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            feature_ids=feature_ids,
            save_dir=ens_save_dir,
            patience=15,
        )
        elapsed = time.time() - t0
        print(f"\n  Training completed in {elapsed:.0f}s ({elapsed/60:.1f}min)")

        # Evaluate
        geco_results = evaluate_model(model_trainer, test_agg, "GECO test",
                                       start_sid=len(train_agg) + len(val_agg))
        provo_results = evaluate_model(model_trainer, provo_agg, "Provo (cross-corpus)",
                                        start_sid=0)

        all_geco_results.append(geco_results)
        all_provo_results.append(provo_results)

    # ---- Final summary ----
    print(f"\n{'=' * 90}")
    print(f"FINAL SUMMARY: Toronto CL RobertaRegressionModel")
    print(f"  Model: {args.model_name} | Epochs: {args.epochs} | Ensemble: {args.ensemble}")
    print(f"  Training: aggregated GECO data (raw ms scale)")
    print(f"{'=' * 90}")

    # Average over ensemble
    for dataset_name, all_results in [("GECO test", all_geco_results), ("Provo", all_provo_results)]:
        print(f"\n  {dataset_name}:")
        print(f"    {'Metric':<10s} {'r':>8s} {'MAE':>10s} {'RMSE':>10s}")
        print(f"    {'─'*10} {'─'*8} {'─'*10} {'─'*10}")

        for metric, name in [('ffd', 'FFD'), ('gaze', 'Gaze'), ('trt', 'TRT'), ('skip_01', 'Skip')]:
            rs = [res[metric]['r'] for res in all_results]
            maes = [res[metric]['mae'] for res in all_results]
            rmses = [res[metric]['rmse'] for res in all_results]

            mean_r = np.mean(rs)
            mean_mae = np.mean(maes)
            mean_rmse = np.mean(rmses)

            if metric == 'skip_01':
                print(f"    {name:<10s} {mean_r:>8.3f} {mean_mae:>10.4f} {mean_rmse:>10.4f}")
            else:
                print(f"    {name:<10s} {mean_r:>8.3f} {mean_mae:>9.1f}ms {mean_rmse:>9.1f}ms")

    print(f"\nCheckpoints saved to: {save_dir}")
    print("\nDone!")


if __name__ == "__main__":
    main()
