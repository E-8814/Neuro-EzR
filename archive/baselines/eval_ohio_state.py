"""
Quick evaluation of saved Ohio State model checkpoints.
Loads whatever best_model_*.pth files exist and reports correlations.

Usage:
    python3 -u previous_implementations_of_word_level_predictions/eval_ohio_state.py
"""

import os
import sys
import numpy as np
import torch
from transformers import RobertaModel, RobertaTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cmcl21_st'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src_v2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'original_ezreader'))

from model import RobertaForGazePrediction
from data_loader import load_provo, aggregate_by_sentence
from geco_loader import load_geco, split_geco
from run_ohio_state_on_geco import convert_to_ohio_format, evaluate_model


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "roberta-base"

    # Find checkpoint dir
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints_ohio_state_roberta_base")
    if not os.path.exists(ckpt_dir):
        print(f"No checkpoint dir found at {ckpt_dir}")
        return

    # Find available checkpoints
    metrics = []
    for m in ["ffd", "gaze", "trt", "skip"]:
        path = os.path.join(ckpt_dir, f"best_model_{m}.pth")
        if os.path.exists(path):
            metrics.append(m)

    if not metrics:
        print("No checkpoints found!")
        return

    print(f"Found checkpoints for: {', '.join(m.upper() for m in metrics)}")

    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    # Load GECO test
    print("\nLoading GECO...")
    reading_path = os.path.join(data_dir, "Geco_MonolingualReadingData.csv")
    material_path = os.path.join(data_dir, "Geco_EnglishMaterial.csv")
    pred_path = os.path.join(data_dir, "geco_predictability.pkl")
    geco_raw = load_geco(reading_path, material_path, pred_path)
    _, _, test_raw = split_geco(geco_raw)

    geco_agg = aggregate_by_sentence(geco_raw, min_participants=5)
    test_ids = set(sd.text_id for sd in test_raw)
    test_agg = [a for a in geco_agg if a.text_id in test_ids]

    print("  Converting GECO test...")
    test_data = convert_to_ohio_format(test_agg, tokenizer)
    print(f"  Test: {len(test_data[0])} sentences")

    # Load Provo
    print("\nLoading Provo...")
    et_path = os.path.join(data_dir, "Provo_Corpus-Eyetracking_Data.csv")
    provo_raw = load_provo(et_path)
    provo_agg = aggregate_by_sentence(provo_raw, min_participants=10)
    provo_data = convert_to_ohio_format(provo_agg, tokenizer)
    print(f"  Provo: {len(provo_data[0])} sentences")

    # Evaluate each metric
    input_size = {"roberta-base": 768, "roberta-large": 1024}

    print(f"\n{'=' * 70}")
    print(f"Ohio State RobertaForGazePrediction — Results")
    print(f"{'=' * 70}")
    print(f"\n  {'Metric':<10s} {'GECO test r':>12s} {'GECO MAE':>10s} {'Provo r':>10s} {'Provo MAE':>10s}")
    print(f"  {'-' * 55}")

    for metric in metrics:
        roberta = RobertaModel.from_pretrained(model_name)
        m = RobertaForGazePrediction(
            pretrained=roberta, input_dim=input_size[model_name],
            dropout_1=0.1, hidden_dim=385, activation="relu", dropout_2=0.1,
        ).to(device)
        m.load_state_dict(torch.load(
            os.path.join(ckpt_dir, f"best_model_{metric}.pth"),
            map_location=device, weights_only=False))
        m.eval()

        r_test, mae_test, _, _, _ = evaluate_model(m, test_data, metric, tokenizer, device)
        r_provo, mae_provo, _, _, _ = evaluate_model(m, provo_data, metric, tokenizer, device)

        print(f"  {metric.upper():<10s} {r_test:>12.3f} {mae_test:>10.1f} {r_provo:>10.3f} {mae_provo:>10.1f}")

        del m, roberta
        torch.cuda.empty_cache()

    print(f"\nDone!")


if __name__ == "__main__":
    main()
