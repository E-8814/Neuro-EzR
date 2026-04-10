"""
Empirical ceiling curve on Provo: correlate model predictions against
k-participant averages for k = 1, 2, 3, ..., 84.

If the curve is still rising steeply at k=84, the model is data-limited.
If it has plateaued, the model is feature-limited and the plateau is the
practical ceiling for this class of models.

Usage:
    python3 -u src_v2/break_the_ceiling/ceiling_curve_provo.py
    python3 -u src_v2/break_the_ceiling/ceiling_curve_provo.py --checkpoint path/to/best_model.pt
"""

import os
import sys
import random
import argparse
import numpy as np
from collections import defaultdict

import torch
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lm_model'))

from data_loader import load_provo
from model_llama_faithful_sh import NeuralEZReaderLLaMA


def get_model_predictions(model, dataset, device, batch_size=8):
    """
    Run model on all unique sentences in Provo.
    Returns dict: (text_id, sentence_number, word_idx) -> {trt, ffd, gaze, skip}
    """
    # Get unique sentences
    unique = {}
    for sd in dataset:
        key = (sd.text_id, sd.sentence_number)
        if key not in unique:
            unique[key] = sd

    sentences = sorted(unique.values(), key=lambda s: (s.text_id, s.sentence_number))

    predictions = {}
    model.eval()

    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]

            word_lists = [sd.tokens for sd in batch]
            pred_vals = pad_sequence(
                [torch.tensor([w.predictability for w in sd.words], dtype=torch.float32) for sd in batch],
                batch_first=True,
            ).to(device)
            wlens = pad_sequence(
                [torch.tensor([len(t) for t in sd.tokens], dtype=torch.float32) for sd in batch],
                batch_first=True,
            ).to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(word_lists, pred_vals, wlens)

            for b_idx, sd in enumerate(batch):
                for w_idx in range(len(sd.words)):
                    word_key = (sd.text_id, sd.sentence_number, w_idx)
                    predictions[word_key] = {
                        'trt': pred['conditional_trt'][b_idx, w_idx].cpu().item(),
                        'ffd': pred['first_fixation'][b_idx, w_idx].cpu().item(),
                        'gaze': pred['gaze_duration'][b_idx, w_idx].cpu().item(),
                        'skip': pred['skip_prob'][b_idx, w_idx].cpu().item(),
                    }

    return predictions


def build_word_index(dataset):
    """
    Build index: (text_id, sentence_number, word_idx) -> {participant_id: WordData}
    """
    index = defaultdict(dict)
    for sd in dataset:
        for i, w in enumerate(sd.words):
            key = (sd.text_id, sd.sentence_number, i)
            index[key][sd.participant_id] = w
    return index


def correlate_at_k(predictions, word_index, participants, k, n_repeats, rng):
    """
    Sample k participants, average their data, correlate with model predictions.
    Repeat n_repeats times. Return mean correlations.
    """
    metrics = ['trt', 'ffd', 'gaze', 'skip']
    all_corrs = {m: [] for m in metrics}

    for _ in range(n_repeats):
        sampled = rng.sample(participants, k)
        sampled_set = set(sampled)

        model_trt, human_trt = [], []
        model_ffd, human_ffd = [], []
        model_gaze, human_gaze = [], []
        model_skip, human_skip = [], []

        for word_key, pdata in word_index.items():
            if word_key not in predictions:
                continue

            pred = predictions[word_key]

            # Average human data from sampled participants
            trts, ffds, gazes, skips = [], [], [], []
            for pid in sampled_set:
                if pid not in pdata:
                    continue
                w = pdata[pid]
                if w.was_skipped:
                    skips.append(1.0)
                else:
                    skips.append(0.0)
                    if w.total_reading_time > 0:
                        trts.append(w.total_reading_time)
                    if w.first_fixation_duration > 0:
                        ffds.append(w.first_fixation_duration)
                    if w.gaze_duration > 0:
                        gazes.append(w.gaze_duration)

            if trts:
                model_trt.append(pred['trt'])
                human_trt.append(np.mean(trts))
            if ffds:
                model_ffd.append(pred['ffd'])
                human_ffd.append(np.mean(ffds))
            if gazes:
                model_gaze.append(pred['gaze'])
                human_gaze.append(np.mean(gazes))
            if skips:
                model_skip.append(pred['skip'])
                human_skip.append(np.mean(skips))

        def corr(a, b):
            a, b = np.array(a), np.array(b)
            if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0:
                return np.corrcoef(a, b)[0, 1]
            return 0.0

        all_corrs['trt'].append(corr(model_trt, human_trt))
        all_corrs['ffd'].append(corr(model_ffd, human_ffd))
        all_corrs['gaze'].append(corr(model_gaze, human_gaze))
        all_corrs['skip'].append(corr(model_skip, human_skip))

    return {m: (np.mean(all_corrs[m]), np.std(all_corrs[m])) for m in metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "..",
                                             "checkpoints", "faithful_sh",
                                             "geco_TinyLlama_TinyLlama-1.1B-Chat-v1.0",
                                             "best_model.pt"))
    parser.add_argument("--n_repeats", type=int, default=100,
                        help="Random samples per k value")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load Provo ----
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    et_path = os.path.join(data_dir, "Provo_Corpus-Eyetracking_Data.csv")

    print("Loading Provo Corpus...")
    dataset = load_provo(et_path)
    participants = sorted(set(sd.participant_id for sd in dataset))
    print(f"  {len(dataset):,} observations, {len(participants)} participants")

    # ---- Load model ----
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_name = ckpt.get('model_name', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    freeze_layers = ckpt.get('freeze_layers', 12)
    hidden_dim = ckpt.get('hidden_dim', 256)

    model = NeuralEZReaderLLaMA(
        model_name=model_name,
        freeze_layers=freeze_layers,
        hidden_dim=hidden_dim,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Model: {model_name}, delta={model.delta.item():.4f}")

    # ---- Get model predictions (once) ----
    print("\nRunning model on all Provo sentences...")
    predictions = get_model_predictions(model, dataset, device)
    print(f"  Predictions for {len(predictions):,} word positions")

    # ---- Build word index ----
    word_index = build_word_index(dataset)
    print(f"  Word index: {len(word_index):,} positions")

    # ---- Ceiling curve ----
    # k values: 1,2,3,...,10, then 15,20,25,...,84
    k_values = list(range(1, min(11, len(participants) + 1)))
    k_values += list(range(15, len(participants) + 1, 5))
    if len(participants) not in k_values:
        k_values.append(len(participants))
    k_values = sorted(set(k_values))

    print(f"\nComputing ceiling curve for k = {k_values}")
    print(f"  {args.n_repeats} random samples per k\n")

    print(f"{'k':>4s} | {'r_TRT':>10s} {'±':>4s} | {'r_FFD':>10s} {'±':>4s} | "
          f"{'r_Gaze':>10s} {'±':>4s} | {'r_Skip':>10s} {'±':>4s}")
    print("-" * 80)

    results = []
    for k in k_values:
        n_rep = args.n_repeats if k < len(participants) else 1
        corrs = correlate_at_k(predictions, word_index, participants, k, n_rep, rng)
        results.append((k, corrs))

        print(f"{k:4d} | {corrs['trt'][0]:10.4f} {corrs['trt'][1]:4.3f} | "
              f"{corrs['ffd'][0]:10.4f} {corrs['ffd'][1]:4.3f} | "
              f"{corrs['gaze'][0]:10.4f} {corrs['gaze'][1]:4.3f} | "
              f"{corrs['skip'][0]:10.4f} {corrs['skip'][1]:4.3f}")

    # ---- Summary ----
    print("\n" + "=" * 80)
    print("CEILING CURVE SUMMARY")
    print("=" * 80)

    for metric in ['trt', 'ffd', 'gaze', 'skip']:
        r_at_1 = results[0][1][metric][0]
        r_at_max = results[-1][1][metric][0]

        # Find where correlation reaches 95% of max
        threshold = 0.95 * r_at_max
        k_95 = None
        for k, corrs in results:
            if corrs[metric][0] >= threshold:
                k_95 = k
                break

        # Check if still rising: compare last two points
        if len(results) >= 2:
            r_second_last = results[-2][1][metric][0]
            still_rising = r_at_max - r_second_last

        print(f"\n  {metric.upper()}:")
        print(f"    r at k=1:  {r_at_1:.4f}")
        print(f"    r at k={k_values[-1]}: {r_at_max:.4f}")
        print(f"    k for 95% of max: {k_95}")
        if len(results) >= 2:
            print(f"    Still rising at end: {still_rising:+.4f} "
                  f"({'yes' if abs(still_rising) > 0.005 else 'plateaued'})")

    # ---- Also compute split-half ceiling for comparison ----
    print("\n" + "=" * 80)
    print("SPLIT-HALF NOISE CEILING (for comparison)")
    print("=" * 80)

    half = len(participants) // 2
    metrics = ['trt', 'ffd', 'gaze', 'skip']
    half_corrs = {m: [] for m in metrics}

    for _ in range(200):
        shuffled = participants.copy()
        rng.shuffle(shuffled)
        group_a = set(shuffled[:half])
        group_b = set(shuffled[half:])

        a_vals = {m: [] for m in metrics}
        b_vals = {m: [] for m in metrics}

        for word_key, pdata in word_index.items():
            a_trts, b_trts = [], []
            a_ffds, b_ffds = [], []
            a_gazes, b_gazes = [], []
            a_skips, b_skips = [], []

            for pid in group_a:
                if pid not in pdata:
                    continue
                w = pdata[pid]
                if w.was_skipped:
                    a_skips.append(1.0)
                else:
                    a_skips.append(0.0)
                    if w.total_reading_time > 0: a_trts.append(w.total_reading_time)
                    if w.first_fixation_duration > 0: a_ffds.append(w.first_fixation_duration)
                    if w.gaze_duration > 0: a_gazes.append(w.gaze_duration)

            for pid in group_b:
                if pid not in pdata:
                    continue
                w = pdata[pid]
                if w.was_skipped:
                    b_skips.append(1.0)
                else:
                    b_skips.append(0.0)
                    if w.total_reading_time > 0: b_trts.append(w.total_reading_time)
                    if w.first_fixation_duration > 0: b_ffds.append(w.first_fixation_duration)
                    if w.gaze_duration > 0: b_gazes.append(w.gaze_duration)

            if a_trts and b_trts:
                a_vals['trt'].append(np.mean(a_trts)); b_vals['trt'].append(np.mean(b_trts))
            if a_ffds and b_ffds:
                a_vals['ffd'].append(np.mean(a_ffds)); b_vals['ffd'].append(np.mean(b_ffds))
            if a_gazes and b_gazes:
                a_vals['gaze'].append(np.mean(a_gazes)); b_vals['gaze'].append(np.mean(b_gazes))
            if a_skips and b_skips:
                a_vals['skip'].append(np.mean(a_skips)); b_vals['skip'].append(np.mean(b_skips))

        def corr(x, y):
            x, y = np.array(x), np.array(y)
            if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0:
                return np.corrcoef(x, y)[0, 1]
            return 0.0

        for m in metrics:
            r_h = corr(a_vals[m], b_vals[m])
            half_corrs[m].append(r_h)

    print(f"\n{'Metric':<8s} | {'r_half':>8s} | {'r_full (SB)':>12s} | {'sqrt(r_full)':>12s} | "
          f"{'Model r (k=all)':>15s} | {'% of r_full':>11s} | {'% of sqrt':>9s}")
    print("-" * 90)

    for m in metrics:
        r_h = np.mean(half_corrs[m])
        r_f = 2 * r_h / (1 + abs(r_h))
        r_sqrt = np.sqrt(r_f)
        model_r = results[-1][1][m][0]

        pct_rf = 100 * model_r / r_f if r_f > 0 else 0
        pct_sqrt = 100 * model_r / r_sqrt if r_sqrt > 0 else 0

        print(f"{m.upper():<8s} | {r_h:8.4f} | {r_f:12.4f} | {r_sqrt:12.4f} | "
              f"{model_r:15.4f} | {pct_rf:10.1f}% | {pct_sqrt:8.1f}%")


if __name__ == "__main__":
    main()
