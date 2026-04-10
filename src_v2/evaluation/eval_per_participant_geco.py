"""
Per-participant evaluation on GECO.

Loads the aggregated-trained faithful_sh model and evaluates its predictions
against each of the 14 individual GECO participants separately.

This answers: can a model trained on "average reader" predict individual readers?

Usage:
    python3 -u src_v2/evaluation/eval_per_participant_geco.py
    python3 -u src_v2/evaluation/eval_per_participant_geco.py --checkpoint path/to/best_model.pt
"""

import os
import sys
import argparse
import numpy as np
from collections import defaultdict

import torch
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lm_model'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'archive', 'original_ezreader'))

from model_llama_faithful_sh import NeuralEZReaderLLaMA
from geco_loader import load_geco, split_geco


def get_model_predictions(model, raw_dataset, device, batch_size=8):
    """
    Run model on all unique sentences. Returns dict:
    (text_id, sentence_number, word_idx) -> {trt, ffd, gaze, skip}
    """
    # Get unique sentences (one per text_id, sentence_number)
    unique = {}
    for sd in raw_dataset:
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


def evaluate_participant(predictions, participant_data):
    """
    Correlate model predictions with one participant's reading data.
    Only uses words where the participant has valid data.
    """
    model_trt, human_trt = [], []
    model_ffd, human_ffd = [], []
    model_gaze, human_gaze = [], []
    model_skip, human_skip = [], []

    for sd in participant_data:
        for w_idx, w in enumerate(sd.words):
            word_key = (sd.text_id, sd.sentence_number, w_idx)
            if word_key not in predictions:
                continue

            pred = predictions[word_key]

            if w.was_skipped:
                human_skip.append(1.0)
                model_skip.append(pred['skip'])
            else:
                human_skip.append(0.0)
                model_skip.append(pred['skip'])
                if w.total_reading_time > 0:
                    model_trt.append(pred['trt'])
                    human_trt.append(w.total_reading_time)
                if w.first_fixation_duration > 0:
                    model_ffd.append(pred['ffd'])
                    human_ffd.append(w.first_fixation_duration)
                if w.gaze_duration > 0:
                    model_gaze.append(pred['gaze'])
                    human_gaze.append(w.gaze_duration)

    def corr(a, b):
        a, b = np.array(a), np.array(b)
        if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0:
            return np.corrcoef(a, b)[0, 1]
        return 0.0

    return {
        'r_trt': corr(model_trt, human_trt),
        'r_ffd': corr(model_ffd, human_ffd),
        'r_gaze': corr(model_gaze, human_gaze),
        'r_skip': corr(model_skip, human_skip),
        'n_words': len(human_skip),
        'n_fixated': len(human_trt),
        'mean_human_trt': np.mean(human_trt) if human_trt else 0,
        'mean_human_ffd': np.mean(human_ffd) if human_ffd else 0,
        'skip_rate': np.mean(human_skip) if human_skip else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "..",
                                             "checkpoints", "faithful_sh",
                                             "geco_TinyLlama_TinyLlama-1.1B-Chat-v1.0",
                                             "best_model.pt"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load GECO ----
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    reading_path = os.path.join(data_dir, "Geco_MonolingualReadingData.csv")
    material_path = os.path.join(data_dir, "Geco_EnglishMaterial.csv")
    pred_path = os.path.join(data_dir, "geco_predictability.pkl")

    print("Loading GECO Corpus...")
    raw_dataset = load_geco(reading_path, material_path, pred_path)
    print(f"  {len(raw_dataset):,} observations")

    participants = sorted(set(sd.participant_id for sd in raw_dataset))
    print(f"  {len(participants)} participants: {participants}")

    # Split into train/val/test
    train_raw, val_raw, test_raw = split_geco(raw_dataset)
    test_text_ids = set(sd.text_id for sd in test_raw)
    val_text_ids = set(sd.text_id for sd in val_raw)
    held_out_ids = test_text_ids | val_text_ids

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

    # ---- Get predictions (once, on all sentences) ----
    print("\nRunning model on all GECO sentences...")
    predictions = get_model_predictions(model, raw_dataset, device)
    print(f"  Predictions for {len(predictions):,} word positions")

    # ---- Group data by participant ----
    by_participant = defaultdict(list)
    for sd in raw_dataset:
        by_participant[sd.participant_id].append(sd)

    # ---- Evaluate per participant ----
    # On test set sentences only (unseen during training)
    print("\n" + "=" * 100)
    print("PER-PARTICIPANT EVALUATION (test set sentences only)")
    print("=" * 100)

    print(f"\n{'Participant':<14s} | {'r_TRT':>7s} {'r_FFD':>7s} {'r_Gaze':>7s} {'r_Skip':>7s} | "
          f"{'words':>6s} {'fixated':>7s} {'skip%':>6s} {'mean_TRT':>9s} {'mean_FFD':>9s}")
    print("-" * 100)

    all_results = []
    for pid in participants:
        # Filter to test sentences
        test_data = [sd for sd in by_participant[pid] if sd.text_id in held_out_ids]
        if not test_data:
            continue

        result = evaluate_participant(predictions, test_data)
        result['participant'] = pid
        all_results.append(result)

        print(f"{pid:<14s} | {result['r_trt']:7.3f} {result['r_ffd']:7.3f} "
              f"{result['r_gaze']:7.3f} {result['r_skip']:7.3f} | "
              f"{result['n_words']:6d} {result['n_fixated']:7d} "
              f"{result['skip_rate']:5.1%} {result['mean_human_trt']:8.0f}ms "
              f"{result['mean_human_ffd']:8.0f}ms")

    # ---- Summary ----
    print("-" * 100)
    mean_trt = np.mean([r['r_trt'] for r in all_results])
    mean_ffd = np.mean([r['r_ffd'] for r in all_results])
    mean_gaze = np.mean([r['r_gaze'] for r in all_results])
    mean_skip = np.mean([r['r_skip'] for r in all_results])
    std_trt = np.std([r['r_trt'] for r in all_results])
    std_ffd = np.std([r['r_ffd'] for r in all_results])
    std_gaze = np.std([r['r_gaze'] for r in all_results])
    std_skip = np.std([r['r_skip'] for r in all_results])

    print(f"{'MEAN':<14s} | {mean_trt:7.3f} {mean_ffd:7.3f} "
          f"{mean_gaze:7.3f} {mean_skip:7.3f}")
    print(f"{'STD':<14s} | {std_trt:7.3f} {std_ffd:7.3f} "
          f"{std_gaze:7.3f} {std_skip:7.3f}")

    # ---- Also evaluate on ALL sentences (train+val+test) ----
    print("\n" + "=" * 100)
    print("PER-PARTICIPANT EVALUATION (all sentences)")
    print("=" * 100)

    print(f"\n{'Participant':<14s} | {'r_TRT':>7s} {'r_FFD':>7s} {'r_Gaze':>7s} {'r_Skip':>7s} | "
          f"{'words':>6s} {'fixated':>7s} {'skip%':>6s} {'mean_TRT':>9s} {'mean_FFD':>9s}")
    print("-" * 100)

    all_results_full = []
    for pid in participants:
        result = evaluate_participant(predictions, by_participant[pid])
        result['participant'] = pid
        all_results_full.append(result)

        print(f"{pid:<14s} | {result['r_trt']:7.3f} {result['r_ffd']:7.3f} "
              f"{result['r_gaze']:7.3f} {result['r_skip']:7.3f} | "
              f"{result['n_words']:6d} {result['n_fixated']:7d} "
              f"{result['skip_rate']:5.1%} {result['mean_human_trt']:8.0f}ms "
              f"{result['mean_human_ffd']:8.0f}ms")

    print("-" * 100)
    mean_trt = np.mean([r['r_trt'] for r in all_results_full])
    mean_ffd = np.mean([r['r_ffd'] for r in all_results_full])
    mean_gaze = np.mean([r['r_gaze'] for r in all_results_full])
    mean_skip = np.mean([r['r_skip'] for r in all_results_full])
    std_trt = np.std([r['r_trt'] for r in all_results_full])
    std_ffd = np.std([r['r_ffd'] for r in all_results_full])
    std_gaze = np.std([r['r_gaze'] for r in all_results_full])
    std_skip = np.std([r['r_skip'] for r in all_results_full])

    print(f"{'MEAN':<14s} | {mean_trt:7.3f} {mean_ffd:7.3f} "
          f"{mean_gaze:7.3f} {mean_skip:7.3f}")
    print(f"{'STD':<14s} | {std_trt:7.3f} {std_ffd:7.3f} "
          f"{std_gaze:7.3f} {std_skip:7.3f}")

    # ---- Participant variability analysis ----
    print("\n" + "=" * 100)
    print("INDIVIDUAL DIFFERENCES")
    print("=" * 100)

    skip_rates = [r['skip_rate'] for r in all_results_full]
    mean_trts = [r['mean_human_trt'] for r in all_results_full]
    mean_ffds = [r['mean_human_ffd'] for r in all_results_full]

    print(f"\n  Skip rate across participants:  {np.mean(skip_rates):.1%} +/- {np.std(skip_rates):.1%}  "
          f"(range: {np.min(skip_rates):.1%} - {np.max(skip_rates):.1%})")
    print(f"  Mean TRT across participants:  {np.mean(mean_trts):.0f}ms +/- {np.std(mean_trts):.0f}ms  "
          f"(range: {np.min(mean_trts):.0f} - {np.max(mean_trts):.0f}ms)")
    print(f"  Mean FFD across participants:  {np.mean(mean_ffds):.0f}ms +/- {np.std(mean_ffds):.0f}ms  "
          f"(range: {np.min(mean_ffds):.0f} - {np.max(mean_ffds):.0f}ms)")

    # Correlation between individual differences and model performance
    r_trt_vs_skip = np.corrcoef([r['r_trt'] for r in all_results_full], skip_rates)[0, 1]
    r_trt_vs_speed = np.corrcoef([r['r_trt'] for r in all_results_full], mean_trts)[0, 1]

    print(f"\n  Correlation between model r_TRT and participant skip rate: {r_trt_vs_skip:.3f}")
    print(f"  Correlation between model r_TRT and participant mean TRT:  {r_trt_vs_speed:.3f}")
    print(f"  (Positive = model predicts slower/more-fixating readers better)")


if __name__ == "__main__":
    main()
