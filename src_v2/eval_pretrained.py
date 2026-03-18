"""
Evaluate a pre-trained LM + DiffEZReader with NO fine-tuning.

This shows how much performance comes from the pre-trained representations
alone (with randomly initialized heads) vs what fine-tuning adds.

Usage:
  python3 -u src_v2/eval_pretrained.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
  python3 -u src_v2/eval_pretrained.py --model meta-llama/Llama-3.2-1B
  python3 -u src_v2/eval_pretrained.py --model bert-base-uncased
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ez_reader'))

from data_loader import aggregate_by_sentence, split_aggregated
from geco_loader import load_geco, split_geco
from torch.nn.utils.rnn import pad_sequence


def collate_aggregated(batch, device):
    word_lists = [a.tokens for a in batch]
    pred_vals = pad_sequence(
        [torch.tensor(a.predictabilities, dtype=torch.float32) for a in batch],
        batch_first=True,
    ).to(device)
    wlens = pad_sequence(
        [torch.tensor([len(t) for t in a.tokens], dtype=torch.float32) for a in batch],
        batch_first=True,
    ).to(device)
    h_trt = pad_sequence(
        [torch.tensor(a.mean_trt, dtype=torch.float32) for a in batch],
        batch_first=True,
    ).to(device)
    h_ffd = pad_sequence(
        [torch.tensor(a.mean_ffd, dtype=torch.float32) for a in batch],
        batch_first=True,
    ).to(device)
    h_gaze = pad_sequence(
        [torch.tensor(a.mean_gaze, dtype=torch.float32) for a in batch],
        batch_first=True,
    ).to(device)
    h_skip = pad_sequence(
        [torch.tensor(a.skip_rate, dtype=torch.float32) for a in batch],
        batch_first=True,
    ).to(device)
    return word_lists, pred_vals, wlens, h_trt, h_ffd, h_gaze, h_skip


def evaluate(model, agg_data, device, batch_size=8):
    model.eval()
    all_pred_trt, all_human_trt = [], []
    all_pred_ffd, all_human_ffd = [], []
    all_pred_gaze, all_human_gaze = [], []
    all_pred_skip, all_human_skip = [], []
    all_pred_l1, all_pred_l2 = [], []

    with torch.no_grad():
        for i in range(0, len(agg_data), batch_size):
            batch = agg_data[i:i + batch_size]
            word_lists, pred_vals, wlens, h_trt, h_ffd, h_gaze, h_skip = collate_aggregated(batch, device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(word_lists, pred_vals, wlens)

            for b in range(len(batch)):
                seq_len = len(batch[b].tokens)
                all_pred_trt.extend(pred['total_reading_time'][b, :seq_len].float().cpu().tolist())
                all_human_trt.extend(batch[b].mean_trt)
                all_pred_ffd.extend(pred['first_fixation'][b, :seq_len].float().cpu().tolist())
                all_human_ffd.extend(batch[b].mean_ffd)
                all_pred_gaze.extend(pred['gaze_duration'][b, :seq_len].float().cpu().tolist())
                all_human_gaze.extend(batch[b].mean_gaze)
                all_pred_skip.extend(pred['skip_prob'][b, :seq_len].float().cpu().tolist())
                all_human_skip.extend(batch[b].skip_rate)
                all_pred_l1.extend(pred['L1'][b, :seq_len].float().cpu().tolist())
                all_pred_l2.extend(pred['L2'][b, :seq_len].float().cpu().tolist())

    def corr(a, b):
        a, b = np.array(a), np.array(b)
        if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0:
            return np.corrcoef(a, b)[0, 1]
        return 0.0

    return {
        'r_trt': corr(all_pred_trt, all_human_trt),
        'r_ffd': corr(all_pred_ffd, all_human_ffd),
        'r_gaze': corr(all_pred_gaze, all_human_gaze),
        'r_skip': corr(all_pred_skip, all_human_skip),
        'mean_pred_trt': np.mean(all_pred_trt),
        'mean_human_trt': np.mean(all_human_trt),
        'mean_pred_ffd': np.mean(all_pred_ffd),
        'mean_human_ffd': np.mean(all_human_ffd),
        'mean_l1': np.mean(all_pred_l1),
        'std_l1': np.std(all_pred_l1),
        'mean_l2': np.mean(all_pred_l2),
        'std_l2': np.std(all_pred_l2),
        'mean_skip': np.mean(all_pred_skip),
        'mean_human_skip': np.mean(all_human_skip),
        'n_words': len(all_pred_trt),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--freeze", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of random seeds to average over (heads are random)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Seeds: {args.seeds}")

    # ---- Load data ----
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    reading_path = os.path.join(data_dir, "Geco_MonolingualReadingData.csv")
    material_path = os.path.join(data_dir, "Geco_EnglishMaterial.csv")
    pred_path = os.path.join(data_dir, "geco_predictability.pkl")

    print("Loading GECO Corpus...")
    raw_dataset = load_geco(reading_path, material_path, pred_path)
    train_raw, val_raw, test_raw = split_geco(raw_dataset)

    aggregated = aggregate_by_sentence(raw_dataset, min_participants=5)
    train_text_ids = set(sd.text_id for sd in train_raw)
    val_text_ids = set(sd.text_id for sd in val_raw)
    test_agg = [a for a in aggregated if a.text_id not in train_text_ids and a.text_id not in val_text_ids]
    val_agg = [a for a in aggregated if a.text_id in val_text_ids]

    # Use val + test for evaluation (no training happening)
    eval_agg = val_agg + test_agg
    print(f"  Evaluating on {len(eval_agg)} aggregated sentences")

    # ---- Determine freeze layers ----
    if args.freeze is not None:
        freeze_layers = args.freeze
    else:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model)
        n_layers = cfg.num_hidden_layers
        freeze_layers = int(n_layers * 0.75)

    # ---- Detect model type ----
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model)
    is_causal = cfg.model_type in ("llama", "gpt2", "gpt_neo", "gpt_neox", "mistral", "qwen2", "phi")

    # ---- Run multiple seeds (heads are randomly initialized) ----
    all_results = []

    for seed in range(args.seeds):
        torch.manual_seed(seed)
        print(f"\n--- Seed {seed} ---")

        if is_causal:
            from model_llama import NeuralEZReaderLLaMA
            model = NeuralEZReaderLLaMA(
                model_name=args.model,
                freeze_layers=freeze_layers,
                hidden_dim=256,
            ).to(device)
        else:
            from model_bert import NeuralEZReaderBERT
            model = NeuralEZReaderBERT(
                bert_model_name=args.model,
                freeze_bert_layers=freeze_layers,
                hidden_dim=256,
            ).to(device)

        metrics = evaluate(model, eval_agg, device)
        all_results.append(metrics)

        print(f"  r_TRT={metrics['r_trt']:.3f}  r_FFD={metrics['r_ffd']:.3f}  "
              f"r_Gaze={metrics['r_gaze']:.3f}  r_skip={metrics['r_skip']:.3f}")
        print(f"  mean_TRT={metrics['mean_pred_trt']:.0f}ms (human={metrics['mean_human_trt']:.0f}ms)  "
              f"mean_FFD={metrics['mean_pred_ffd']:.0f}ms (human={metrics['mean_human_ffd']:.0f}ms)")
        print(f"  L1={metrics['mean_l1']:.0f}+/-{metrics['std_l1']:.0f}ms  "
              f"L2={metrics['mean_l2']:.0f}+/-{metrics['std_l2']:.0f}ms  "
              f"skip={metrics['mean_skip']:.2f} (human={metrics['mean_human_skip']:.2f})")

        # Free memory
        del model
        torch.cuda.empty_cache()

    # ---- Summary ----
    print("\n" + "=" * 70)
    print(f"SUMMARY: {args.model} — NO fine-tuning ({args.seeds} seeds)")
    print("=" * 70)

    for key in ['r_trt', 'r_ffd', 'r_gaze', 'r_skip']:
        vals = [r[key] for r in all_results]
        print(f"  {key:8s}: {np.mean(vals):+.3f} +/- {np.std(vals):.3f}  "
              f"(range: {min(vals):.3f} to {max(vals):.3f})")

    print(f"\n  Note: correlations are from random heads (no training).")
    print(f"  Any non-zero correlation comes from structure in the")
    print(f"  pre-trained representations + the EZ Reader equations.")


if __name__ == "__main__":
    main()
