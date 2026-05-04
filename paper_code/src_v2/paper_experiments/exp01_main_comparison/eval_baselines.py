"""
Evaluate baseline models (excluding toronto) and emit per-seed JSONs in
the same format as eval_all_models.py, so aggregate_v2.py picks them up.

Coverage:
  - bert_regression × 5 seeds: load each saved best_model.pt, run
    inference on GECO test + Provo aggregated, compute the standard
    12-metric suite. Real per-seed evaluation.
  - ohio_state_roberta × 5 seeds: per seed, load 4 metric-specific
    .pth files (best_model_ffd/gaze/trt/skip.pth), run inference for
    each metric, assemble joint metrics dict. Real per-seed evaluation.
  - linear_regression / lightgbm / gpt2_surprisal: deterministic
    single-run baselines that don't save fitted models. We re-run their
    training scripts in a subprocess and parse the test-set metrics
    block from stdout. Single seed.

Output:
  src_v2/paper_experiments/exp01_main_comparison/results/raw/baselines/
      <name>_seed<N>.json

Usage (from byzantium srun shell, neuro_ezr env):
  python -u src_v2/paper_experiments/exp01_main_comparison/eval_baselines.py
  python -u .../eval_baselines.py --bert_only       # just bert × 5 seeds
  python -u .../eval_baselines.py --ohio_only       # just ohio × 5 seeds
  python -u .../eval_baselines.py --skip_bert       # everything except bert
  python -u .../eval_baselines.py --skip_ohio       # everything except ohio
  python -u .../eval_baselines.py --skip_subprocess # everything except subprocess
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sp_stats
from torch.nn.utils.rnn import pad_sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
SRC_V2 = os.path.join(REPO_ROOT, "src_v2")
ARCHIVE_BASELINES = os.path.join(REPO_ROOT, "archive", "baselines")
ORIG_EZ = os.path.join(REPO_ROOT, "archive", "original_ezreader")

sys.path.insert(0, SRC_V2)
sys.path.insert(0, os.path.join(SRC_V2, "lm_model"))
sys.path.insert(0, ARCHIVE_BASELINES)
sys.path.insert(0, ORIG_EZ)

from data_loader import load_provo, aggregate_by_sentence
from geco_loader import load_geco, split_geco
from bert_regression import BertDirectRegression


RAW_DIR = Path(_HERE) / "results" / "raw" / "baselines"
RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = os.path.join(REPO_ROOT, "data")

BERT_SEEDS = [1, 2, 3, 42, 100]
OHIO_SEEDS = [1, 2, 3, 42, 100]
OHIO_METRICS = ["ffd", "gaze", "trt", "skip"]
DETERMINISTIC_BASELINES = [
    ("linear_regression",   "linear_regression.py"),
    ("lightgbm",            "lightgbm_baseline.py"),
    ("gpt2_surprisal",      "gpt2_surprisal.py"),
]


# --------------------------------------------------------------------------- #
#  Data loading (cached at module level so we only do it once per run)
# --------------------------------------------------------------------------- #

_DATA_CACHE = {}

def load_eval_data():
    if "geco_test" in _DATA_CACHE:
        return _DATA_CACHE["geco_test"], _DATA_CACHE["provo"]

    print("Loading GECO...")
    geco_raw = load_geco(
        os.path.join(DATA_DIR, "Geco_MonolingualReadingData.csv"),
        os.path.join(DATA_DIR, "Geco_EnglishMaterial.csv"),
        os.path.join(DATA_DIR, "geco_predictability.pkl"),
    )
    train_raw, val_raw, _ = split_geco(geco_raw)
    aggregated = aggregate_by_sentence(geco_raw, min_participants=5)
    train_ids = set(sd.text_id for sd in train_raw)
    val_ids = set(sd.text_id for sd in val_raw)
    geco_test = [a for a in aggregated
                 if a.text_id not in train_ids and a.text_id not in val_ids]

    print("Loading Provo...")
    provo_raw = load_provo(os.path.join(DATA_DIR, "Provo_Corpus-Eyetracking_Data.csv"))
    provo_agg = aggregate_by_sentence(provo_raw, min_participants=5)

    print(f"  GECO test: {len(geco_test)} sentences | Provo: {len(provo_agg)} sentences")
    _DATA_CACHE["geco_test"] = geco_test
    _DATA_CACHE["provo"] = provo_agg
    return geco_test, provo_agg


# --------------------------------------------------------------------------- #
#  BERT eval (real Option B)
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
    return word_lists, pred_vals, wlens


def collect_bert_preds(model, agg_data, device, batch_size=8):
    model.eval()
    pred_trt, pred_ffd, pred_gaze, pred_skip = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(agg_data), batch_size):
            batch = agg_data[i:i + batch_size]
            word_lists, pred_vals, wlens = collate_aggregated(batch, device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(word_lists, pred_vals, wlens)
            for b in range(len(batch)):
                sl = len(batch[b].tokens)
                pred_trt.extend(pred["total_reading_time"][b, :sl].cpu().tolist())
                pred_ffd.extend(pred["first_fixation"][b, :sl].cpu().tolist())
                pred_gaze.extend(pred["gaze_duration"][b, :sl].cpu().tolist())
                pred_skip.extend(pred["skip_prob"][b, :sl].cpu().tolist())
    return (np.array(pred_trt), np.array(pred_ffd),
            np.array(pred_gaze), np.array(pred_skip))


def get_human(agg_data):
    trt, ffd, gaze, skip = [], [], [], []
    for a in agg_data:
        trt.extend(a.mean_trt)
        ffd.extend(a.mean_ffd)
        gaze.extend(a.mean_gaze)
        skip.extend(a.skip_rate)
    return (np.array(trt), np.array(ffd), np.array(gaze), np.array(skip))


def _r(pred, human):
    pred, human = np.asarray(pred), np.asarray(human)
    if len(pred) > 2 and np.std(pred) > 0 and np.std(human) > 0:
        return float(sp_stats.pearsonr(pred, human)[0])
    return 0.0


def metrics_summary(pred_trt, pred_ffd, pred_gaze, pred_skip,
                    h_trt, h_ffd, h_gaze, h_skip):
    return {
        "r_trt":   _r(pred_trt, h_trt),
        "r_ffd":   _r(pred_ffd, h_ffd),
        "r_gaze":  _r(pred_gaze, h_gaze),
        "r_skip":  _r(pred_skip, h_skip),
        "mae_trt":  float(np.mean(np.abs(pred_trt - h_trt))),
        "mae_ffd":  float(np.mean(np.abs(pred_ffd - h_ffd))),
        "mae_gaze": float(np.mean(np.abs(pred_gaze - h_gaze))),
        "bias_trt":  float(np.mean(pred_trt) - np.mean(h_trt)),
        "bias_ffd":  float(np.mean(pred_ffd) - np.mean(h_ffd)),
        "bias_gaze": float(np.mean(pred_gaze) - np.mean(h_gaze)),
        "mean_pred_skip":  float(np.mean(pred_skip)),
        "mean_human_skip": float(np.mean(h_skip)),
    }


def eval_bert_seed(seed, device):
    geco_test, provo_agg = load_eval_data()
    ckpt_path = os.path.join(
        ARCHIVE_BASELINES, "checkpoints_bert_regression",
        f"seed{seed}", "best_model.pt",
    )
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    print(f"  loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = BertDirectRegression(
        bert_model_name=ckpt.get("bert_model_name", "bert-base-uncased"),
        freeze_bert_layers=ckpt.get("freeze_bert_layers", 8),
        hidden_dim=ckpt.get("hidden_dim", 256),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("  predicting GECO test...")
    p_trt, p_ffd, p_gaze, p_skip = collect_bert_preds(model, geco_test, device)
    h_trt, h_ffd, h_gaze, h_skip = get_human(geco_test)
    geco_summary = metrics_summary(p_trt, p_ffd, p_gaze, p_skip,
                                    h_trt, h_ffd, h_gaze, h_skip)

    print("  predicting Provo...")
    p_trt, p_ffd, p_gaze, p_skip = collect_bert_preds(model, provo_agg, device)
    h_trt, h_ffd, h_gaze, h_skip = get_human(provo_agg)
    provo_summary = metrics_summary(p_trt, p_ffd, p_gaze, p_skip,
                                     h_trt, h_ffd, h_gaze, h_skip)

    del model
    torch.cuda.empty_cache()

    return {"geco_test": geco_summary, "provo": provo_summary}


# --------------------------------------------------------------------------- #
#  Ohio State RoBERTa eval (4 metric-specific models per seed)
# --------------------------------------------------------------------------- #

def _ohio_predict_one_metric(model, data, metric_name, tokenizer, device):
    """Run a single-metric Ohio model over all sentences; return preds[]."""
    (sent_strings, sent_first_idx, sent_wlen, sent_prop,
     sent_ffd, sent_gaze, sent_trt, sent_skip, sent_words) = data

    model.eval()
    all_preds = []
    with torch.no_grad():
        for j in range(0, len(sent_strings), 20):
            batch = sent_strings[j:j + 20]
            first_idx = sent_first_idx[j:j + 20]
            flat_wlen = [w for sub in sent_wlen[j:j + 20] for w in sub]
            flat_prop = [p for sub in sent_prop[j:j + 20] for p in sub]

            enc = tokenizer(list(batch), padding=True, truncation=True,
                            return_tensors="pt").to(device)
            wlen_t = torch.FloatTensor(flat_wlen).view(-1, 1).to(device)
            prop_t = torch.FloatTensor(flat_prop).view(-1, 1).to(device)

            outputs = model(**enc, first_idx=first_idx,
                            wlen=wlen_t, prop=prop_t)
            all_preds.extend(outputs.cpu().numpy().flatten().tolist())
    return np.array(all_preds)


def _ohio_human_targets(data):
    """Mirror evaluate_model's flattening — return (h_trt, h_ffd, h_gaze, h_skip)."""
    (_, _, _, _, sent_ffd, sent_gaze, sent_trt, sent_skip, _) = data
    h_trt   = np.array([v for sub in sent_trt   for v in sub])
    h_ffd   = np.array([v for sub in sent_ffd   for v in sub])
    h_gaze  = np.array([v for sub in sent_gaze  for v in sub])
    h_skip  = np.array([v for sub in sent_skip  for v in sub])
    return h_trt, h_ffd, h_gaze, h_skip


def _ohio_predict_all_metrics(seed, ohio_data, tokenizer, device, model_short="roberta-base"):
    """Load each metric's checkpoint, run inference, return preds dict."""
    from transformers import RobertaModel
    sys.path.insert(0, ARCHIVE_BASELINES)
    from model import RobertaForGazePrediction

    ckpt_dir = os.path.join(
        ARCHIVE_BASELINES,
        f"checkpoints_ohio_state_roberta",
        f"seed{seed}",
    )

    preds = {}
    for metric in OHIO_METRICS:
        path = os.path.join(ckpt_dir, f"best_model_{metric}.pth")
        if not os.path.isfile(path):
            print(f"    [missing] {path} — using zeros for {metric}")
            preds[metric] = None
            continue
        roberta = RobertaModel.from_pretrained(model_short)
        m = RobertaForGazePrediction(
            pretrained=roberta, input_dim=768,
            dropout_1=0.1, hidden_dim=385, activation="relu", dropout_2=0.1,
        ).to(device)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        m.eval()
        preds[metric] = _ohio_predict_one_metric(m, ohio_data, metric, tokenizer, device)
        del m, roberta
        torch.cuda.empty_cache()
    return preds


def eval_ohio_seed(seed, device):
    """Evaluate one Ohio seed on GECO test + Provo, return summaries dict."""
    from transformers import RobertaTokenizer
    sys.path.insert(0, ARCHIVE_BASELINES)
    from run_ohio_state_on_geco import convert_to_ohio_format

    geco_test, provo_agg = load_eval_data()
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    print("  converting data to ohio format...")
    geco_data = convert_to_ohio_format(geco_test, tokenizer)
    provo_data = convert_to_ohio_format(provo_agg, tokenizer)

    out = {}
    for label, data in (("geco_test", geco_data), ("provo", provo_data)):
        print(f"  predicting {label}...")
        per_metric = _ohio_predict_all_metrics(seed, data, tokenizer, device)
        h_trt, h_ffd, h_gaze, h_skip = _ohio_human_targets(data)

        # Substitute zeros for any metric whose checkpoint was missing,
        # so metrics_summary can still produce a (degraded) entry.
        def safe(arr, h):
            return arr if arr is not None else np.zeros_like(h)
        p_trt  = safe(per_metric["trt"],  h_trt)
        p_ffd  = safe(per_metric["ffd"],  h_ffd)
        p_gaze = safe(per_metric["gaze"], h_gaze)
        p_skip = safe(per_metric["skip"], h_skip)

        out[label] = metrics_summary(p_trt, p_ffd, p_gaze, p_skip,
                                      h_trt, h_ffd, h_gaze, h_skip)
    return out


# --------------------------------------------------------------------------- #
#  Deterministic baselines: subprocess re-run + log parsing
# --------------------------------------------------------------------------- #

_METRIC_LABEL_MAP = {
    "r_FFD":  "r_ffd",   "r_Gaze": "r_gaze",
    "r_TRT":  "r_trt",   "r_Skip": "r_skip",
    "MAE_FFD": "mae_ffd", "MAE_TRT": "mae_trt",
}


def _parse_block(block):
    """Extract any metrics matched by _METRIC_LABEL_MAP from a log block."""
    out = {}
    for label, key in _METRIC_LABEL_MAP.items():
        m = re.search(rf"{re.escape(label)}\s*=\s*(-?\d+(?:\.\d+)?)", block)
        if m:
            out[key] = float(m.group(1))
    return out


def parse_baseline_stdout(log_text: str):
    """Pull out (geco_test, provo) metric dicts from the script's printout."""
    # We expect 'GECO Test (...)' to head the GECO block.
    geco_block_match = re.search(
        r"GECO Test \([^)]*\).*?(?=Provo|Done|$)", log_text, re.DOTALL,
    )
    provo_block_match = re.search(
        r"Provo[^(\n]*\([^)]*\).*?(?=Done|$)", log_text, re.DOTALL,
    )
    geco_summary = _parse_block(geco_block_match.group()) if geco_block_match else {}
    provo_summary = _parse_block(provo_block_match.group()) if provo_block_match else {}
    return {"geco_test": geco_summary, "provo": provo_summary}


def run_subprocess_baseline(script_filename: str):
    """Run a baseline script and return its stdout as a string."""
    cmd = [sys.executable, "-u", os.path.join("archive", "baselines", script_filename)]
    proc = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{script_filename} exited {proc.returncode}\n--- stderr tail ---\n"
            + (proc.stderr or "")[-2000:]
        )
    return proc.stdout + "\n" + (proc.stderr or "")


# --------------------------------------------------------------------------- #
#  Driver
# --------------------------------------------------------------------------- #

def write_json(name, seed, datasets):
    out_path = RAW_DIR / f"{name}_seed{seed}.json"
    payload = {"model": name, "seed": seed, "datasets": datasets}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_only", action="store_true")
    parser.add_argument("--ohio_only", action="store_true")
    parser.add_argument("--skip_bert", action="store_true")
    parser.add_argument("--skip_ohio", action="store_true")
    parser.add_argument("--skip_subprocess", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Output dir: {RAW_DIR}")

    only_one = args.bert_only or args.ohio_only
    do_bert  = not args.skip_bert  and (not only_one or args.bert_only)
    do_ohio  = not args.skip_ohio  and (not only_one or args.ohio_only)
    do_subp  = not args.skip_subprocess and not only_one

    # ---- BERT ---- #
    if do_bert:
        for seed in BERT_SEEDS:
            out_path = RAW_DIR / f"bert_regression_seed{seed}.json"
            if out_path.exists() and not args.overwrite:
                print(f"[skip] {out_path.name} exists (use --overwrite to redo)")
                continue
            print(f"\n>> bert_regression seed={seed}")
            try:
                datasets = eval_bert_seed(seed, device)
            except FileNotFoundError as e:
                print(f"  [missing checkpoint] {e}")
                continue
            written = write_json("bert_regression", seed, datasets)
            print(f"  -> {written}")

    # ---- Ohio State RoBERTa ---- #
    if do_ohio:
        for seed in OHIO_SEEDS:
            out_path = RAW_DIR / f"ohio_state_roberta_seed{seed}.json"
            if out_path.exists() and not args.overwrite:
                print(f"[skip] {out_path.name} exists (use --overwrite to redo)")
                continue
            print(f"\n>> ohio_state_roberta seed={seed}")
            try:
                datasets = eval_ohio_seed(seed, device)
            except Exception as e:
                print(f"  ERROR: {type(e).__name__}: {e}")
                continue
            written = write_json("ohio_state_roberta", seed, datasets)
            print(f"  -> {written}")

    # ---- deterministic baselines ---- #
    if do_subp:
        for name, script in DETERMINISTIC_BASELINES:
            out_path = RAW_DIR / f"{name}_seed1.json"
            if out_path.exists() and not args.overwrite:
                print(f"[skip] {out_path.name} exists")
                continue
            print(f"\n>> {name} (subprocess re-run)")
            try:
                stdout = run_subprocess_baseline(script)
            except RuntimeError as e:
                print(f"  ERROR: {e}")
                continue
            datasets = parse_baseline_stdout(stdout)
            if not datasets["geco_test"] and not datasets["provo"]:
                print(f"  WARN: parsed no metrics from {name}'s stdout — "
                      f"skipping JSON write.")
                continue
            written = write_json(name, 1, datasets)
            print(f"  -> {written}  (geco keys={list(datasets['geco_test'])}, "
                  f"provo keys={list(datasets['provo'])})")

    print("\nDone. Now re-run aggregate_v2.py to fold the baselines into the comparison.")


if __name__ == "__main__":
    main()
