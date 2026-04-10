"""
Baseline evaluation: pretrained LLaMA + random heads, NO training.

This measures how much signal is already in the pretrained LLaMA
representations before any fine-tuning. If r_TRT is already high
(e.g., 0.35+), it means most of the model's performance comes from
pretrained features and there's limited headroom for training.

Runs model_llama_faithful_sh (faithful EZReader + learned skip head)
with freshly initialized (random) heads on GECO test set.

Usage:
  python3 -u src_v2/evaluation/eval_pretrained_baseline.py
  python3 -u src_v2/evaluation/eval_pretrained_baseline.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

import os
import sys
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lm_model'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'archive', 'original_ezreader'))

from model_llama_faithful_sh import NeuralEZReaderLLaMA
from data_loader import aggregate_by_sentence
from geco_loader import load_geco, split_geco


# --------------------------------------------------------------------------- #
#  Collate
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
#  Evaluation
# --------------------------------------------------------------------------- #

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
                all_pred_trt.extend(pred['conditional_trt'][b, :seq_len].cpu().tolist())
                all_human_trt.extend(batch[b].mean_trt)
                all_pred_ffd.extend(pred['first_fixation'][b, :seq_len].cpu().tolist())
                all_human_ffd.extend(batch[b].mean_ffd)
                all_pred_gaze.extend(pred['gaze_duration'][b, :seq_len].cpu().tolist())
                all_human_gaze.extend(batch[b].mean_gaze)
                all_pred_skip.extend(pred['skip_prob'][b, :seq_len].cpu().tolist())
                all_human_skip.extend(batch[b].skip_rate)
                all_pred_l1.extend(pred['L1'][b, :seq_len].cpu().tolist())
                all_pred_l2.extend(pred['L2'][b, :seq_len].cpu().tolist())

    def corr(a, b):
        a, b = np.array(a), np.array(b)
        if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0:
            return np.corrcoef(a, b)[0, 1]
        return 0.0

    pred_trt = np.array(all_pred_trt)
    pred_ffd = np.array(all_pred_ffd)
    pred_gaze = np.array(all_pred_gaze)
    pred_skip = np.array(all_pred_skip)
    human_trt = np.array(all_human_trt)
    human_ffd = np.array(all_human_ffd)
    human_gaze = np.array(all_human_gaze)
    human_skip = np.array(all_human_skip)

    return {
        'r_trt': corr(pred_trt, human_trt),
        'r_ffd': corr(pred_ffd, human_ffd),
        'r_gaze': corr(pred_gaze, human_gaze),
        'r_skip': corr(pred_skip, human_skip),
        'mae_trt': np.mean(np.abs(pred_trt - human_trt)),
        'mae_ffd': np.mean(np.abs(pred_ffd - human_ffd)),
        'mae_gaze': np.mean(np.abs(pred_gaze - human_gaze)),
        'bias_trt': np.mean(pred_trt) - np.mean(human_trt),
        'bias_ffd': np.mean(pred_ffd) - np.mean(human_ffd),
        'bias_gaze': np.mean(pred_gaze) - np.mean(human_gaze),
        'mean_pred_trt': np.mean(pred_trt),
        'mean_human_trt': np.mean(human_trt),
        'mean_l1': np.mean(all_pred_l1),
        'std_l1': np.std(all_pred_l1),
        'mean_l2': np.mean(all_pred_l2),
        'std_l2': np.std(all_pred_l2),
        'mean_skip': np.mean(all_pred_skip),
        'std_skip': np.std(all_pred_skip),
    }


def print_samples(model, agg_data, device, n_sentences=5, n_words=10):
    model.eval()
    with torch.no_grad():
        for s_idx in range(min(n_sentences, len(agg_data))):
            s = agg_data[s_idx]
            word_list = [s.tokens]
            pv = torch.tensor(
                s.predictabilities, dtype=torch.float32
            ).unsqueeze(0).to(device)
            wl = torch.tensor(
                [len(t) for t in s.tokens], dtype=torch.float32
            ).unsqueeze(0).to(device)
            p = model(word_list, pv, wl)

            title = ' '.join(s.tokens[:6]) + ('...' if len(s.tokens) > 6 else '')
            print(f"\n  Sentence {s_idx+1}: \"{title}\"")
            print(f"  {'word':<14s} {'L1':>5s} {'L2':>5s} | "
                  f"{'cTRT':>5s} {'hTRT':>5s} {'err':>5s} | "
                  f"{'pFFD':>5s} {'hFFD':>5s} | "
                  f"{'pGaze':>5s} {'hGaze':>5s} | "
                  f"{'skip':>5s} {'hSkip':>5s}")
            print(f"  {'-'*95}")

            for i in range(min(n_words, len(s.tokens))):
                l1 = p['L1'][0, i].item()
                l2 = p['L2'][0, i].item()
                ct = p['conditional_trt'][0, i].item()
                pf = p['first_fixation'][0, i].item()
                pg = p['gaze_duration'][0, i].item()
                ps = p['skip_prob'][0, i].item()
                ht = s.mean_trt[i]
                hf = s.mean_ffd[i]
                hg = s.mean_gaze[i]
                hs = s.skip_rate[i]
                err = ct - ht
                print(
                    f"  {s.tokens[i]:<14s} {l1:5.0f} {l2:5.0f} | "
                    f"{ct:5.0f} {ht:5.0f} {err:+5.0f} | "
                    f"{pf:5.0f} {hf:5.0f} | "
                    f"{pg:5.0f} {hg:5.0f} | "
                    f"{ps:5.2f} {hs:5.2f}"
                )


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main(model_name="meta-llama/Llama-3.2-1B", freeze_layers=12, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load GECO data ----
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    reading_path = os.path.join(data_dir, "Geco_MonolingualReadingData.csv")
    material_path = os.path.join(data_dir, "Geco_EnglishMaterial.csv")
    pred_path = os.path.join(data_dir, "geco_predictability.pkl")

    print("Loading GECO Corpus...")
    raw_dataset = load_geco(reading_path, material_path, pred_path)
    print(f"  Raw observations: {len(raw_dataset):,}")

    train_raw, val_raw, test_raw = split_geco(raw_dataset)

    aggregated = aggregate_by_sentence(raw_dataset, min_participants=5)
    train_text_ids = set(sd.text_id for sd in train_raw)
    val_text_ids = set(sd.text_id for sd in val_raw)
    test_agg = [a for a in aggregated if a.text_id not in train_text_ids and a.text_id not in val_text_ids]
    val_agg = [a for a in aggregated if a.text_id in val_text_ids]
    print(f"  Test sentences: {len(test_agg)} ({sum(len(a.tokens) for a in test_agg)} words)")
    print(f"  Val sentences:  {len(val_agg)} ({sum(len(a.tokens) for a in val_agg)} words)")

    # ---- Load model (NO training — pretrained LLaMA + random heads) ----
    print(f"\nLoading UNTRAINED model: {model_name}")
    print(f"  Freezing first {freeze_layers} layers")
    print(f"  Heads: randomly initialized (no checkpoint loaded)")
    model = NeuralEZReaderLLaMA(
        model_name=model_name,
        freeze_layers=freeze_layers,
        hidden_dim=256,
    ).to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Total params: {n_total:,} | Frozen: {n_frozen:,}")
    print(f"  delta = {model.delta.item():.4f} (init)")
    print(f"  l1_scale = {model.l1_scale.item():.1f} (init)")

    # ---- Run multiple seeds for robustness ----
    seeds = [42, 123, 456]
    all_results = []

    for s in seeds:
        torch.manual_seed(s)
        # Re-init heads with different random seed
        model_s = NeuralEZReaderLLaMA(
            model_name=model_name,
            freeze_layers=freeze_layers,
            hidden_dim=256,
        ).to(device)

        print(f"\n{'='*70}")
        print(f"  Seed {s}: Evaluating on test set (no training)")
        print(f"{'='*70}")

        test_results = evaluate(model_s, test_agg, device)
        val_results = evaluate(model_s, val_agg, device)

        print(f"\n  TEST SET:")
        print(f"    r_TRT  = {test_results['r_trt']:.4f}")
        print(f"    r_FFD  = {test_results['r_ffd']:.4f}")
        print(f"    r_Gaze = {test_results['r_gaze']:.4f}")
        print(f"    r_skip = {test_results['r_skip']:.4f}")
        print(f"    MAE_TRT  = {test_results['mae_trt']:.1f}ms")
        print(f"    MAE_FFD  = {test_results['mae_ffd']:.1f}ms")
        print(f"    MAE_Gaze = {test_results['mae_gaze']:.1f}ms")
        print(f"    Bias_TRT  = {test_results['bias_trt']:+.1f}ms")
        print(f"    Bias_FFD  = {test_results['bias_ffd']:+.1f}ms")
        print(f"    Bias_Gaze = {test_results['bias_gaze']:+.1f}ms")
        print(f"    L1 = {test_results['mean_l1']:.0f} +/- {test_results['std_l1']:.0f}ms")
        print(f"    L2 = {test_results['mean_l2']:.0f} +/- {test_results['std_l2']:.0f}ms")
        print(f"    mean_skip = {test_results['mean_skip']:.3f} +/- {test_results['std_skip']:.3f}")

        print(f"\n  VAL SET:")
        print(f"    r_TRT  = {val_results['r_trt']:.4f}")
        print(f"    r_FFD  = {val_results['r_ffd']:.4f}")
        print(f"    r_Gaze = {val_results['r_gaze']:.4f}")
        print(f"    r_skip = {val_results['r_skip']:.4f}")

        all_results.append(test_results)

        if s == seeds[0]:
            print("\n  Sample predictions (seed 42):")
            print_samples(model_s, test_agg, device, n_sentences=5, n_words=10)

        del model_s
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # ---- Summary across seeds ----
    print(f"\n{'='*70}")
    print(f"  SUMMARY: Pretrained baseline (no training) across {len(seeds)} seeds")
    print(f"{'='*70}")
    for metric in ['r_trt', 'r_ffd', 'r_gaze', 'r_skip', 'mae_trt', 'mae_ffd', 'mae_gaze']:
        vals = [r[metric] for r in all_results]
        print(f"  {metric:>10s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}  "
              f"(range: {np.min(vals):.4f} - {np.max(vals):.4f})")

    # ---- Comparison with trained models ----
    print(f"\n  For reference, trained model performance on GECO test:")
    print(f"    LLaMA+EZR v2-delta (trained): r_TRT ~0.466, r_FFD ~0.205, r_Gaze ~0.389")
    print(f"    Faithful (trained):            r_TRT ~0.454, r_FFD ~0.220, r_Gaze ~0.390")
    print(f"\n  If pretrained baseline is close to trained, fine-tuning adds little value")
    print(f"  and the focus should shift to architecture / loss design / LoRA.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--freeze", type=int, default=None)
    args = parser.parse_args()

    if args.freeze is not None:
        freeze_layers = args.freeze
    else:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model)
        n_layers = cfg.num_hidden_layers
        freeze_layers = int(n_layers * 0.75)
        print(f"Auto-freeze: {freeze_layers}/{n_layers} layers")

    main(model_name=args.model, freeze_layers=freeze_layers)
