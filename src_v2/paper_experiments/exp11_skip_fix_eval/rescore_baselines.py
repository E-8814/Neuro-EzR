"""
Re-score all baselines with skip computed BOTH on all words (reproduces
the existing table numbers as a sanity check) and on the comparable
population (words 1..L-1, sentence-initial words excluded).

Baselines predict each word's skip at the word's own row, so the
comparable-population selection is simply word_position > 0.

Covered here:
  - linear_regression, gpt2_surprisal, lightgbm  (retrained, deterministic;
    mirrors complete_metrics/03_train_eval_flat_baselines.py)
  - bert_regression, ohio_state_roberta          (existing checkpoints,
    5 seeds; mirrors complete_metrics/02_eval_bert_ohio.py)

NOT covered: ez_reader_classical (Monte-Carlo simulation re-run; the
classical simulator also cannot skip word 1, so its published number
carries the same penalty the cascade did — rerun separately if needed).

Outputs:
    results/raw/baselines/<name>_seed<N>.json

Usage:
    python -u rescore_baselines.py
    python -u rescore_baselines.py --only flat
    python -u rescore_baselines.py --only bert
    python -u rescore_baselines.py --only ohio
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
SRC_V2 = os.path.abspath(os.path.join(_HERE, "..", ".."))
REPO_ROOT = os.path.dirname(SRC_V2)
EXP01_DIR = os.path.join(SRC_V2, "paper_experiments", "exp01_main_comparison")
COMPLETE_METRICS = os.path.join(EXP01_DIR, "complete_metrics")
ARCHIVE_BASELINES = os.path.join(REPO_ROOT, "archive", "baselines")
ORIG_EZ = os.path.join(REPO_ROOT, "archive", "original_ezreader")
DATA_DIR = os.path.join(REPO_ROOT, "data")

for p in (SRC_V2, EXP01_DIR, COMPLETE_METRICS, ARCHIVE_BASELINES, ORIG_EZ, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from paper_experiments import config  # noqa: E402

from skip_metrics import (  # noqa: E402
    same_index_pairs, skip_summary, positions_from_agg, positions_from_sublists,
)


SEEDS = config.SEEDS
OUT_DIR = Path(_HERE) / "results" / "raw" / "baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) > 2 and a.std() > 0 and b.std() > 0:
        return float(np.corrcoef(a, b)[0, 1])
    return 0.0


def block_for(p_trt, p_ffd, p_gaze, p_skip,
              h_trt, h_ffd, h_gaze, h_skip, positions) -> dict:
    """Time metrics on all words + skip on all words AND words 1..L-1."""
    n = len(np.asarray(p_skip))
    assert len(positions) == n, (
        f"position/prediction length mismatch: {len(positions)} vs {n}"
    )
    out = {}
    for m, p, h in (("trt", p_trt, h_trt), ("ffd", p_ffd, h_ffd),
                    ("gaze", p_gaze, h_gaze)):
        p, h = np.asarray(p, float), np.asarray(h, float)
        out[f"r_{m}"] = _corr(p, h)
        out[f"mae_{m}"] = float(np.mean(np.abs(p - h)))
        out[f"bias_{m}"] = float(np.mean(p) - np.mean(h))
    out["n_words_all"] = int(n)
    out["skip_all"] = skip_summary(p_skip, h_skip)
    sp, st = same_index_pairs(p_skip, h_skip, positions)
    out["skip_cmp"] = skip_summary(sp, st)
    return out


# --------------------------------------------------------------------------- #
#  Flat baselines (retrained, mirrors 03_train_eval_flat_baselines.py)
# --------------------------------------------------------------------------- #


def _load_corpora():
    from data_loader import load_provo, aggregate_by_sentence
    from geco_loader import load_geco, split_geco
    print("Loading GECO...")
    geco_raw = load_geco(
        os.path.join(DATA_DIR, "Geco_MonolingualReadingData.csv"),
        os.path.join(DATA_DIR, "Geco_EnglishMaterial.csv"),
        os.path.join(DATA_DIR, "geco_predictability.pkl"),
    )
    train_raw, val_raw, _ = split_geco(geco_raw)
    geco_agg = aggregate_by_sentence(geco_raw, min_participants=5)
    train_ids = set(sd.text_id for sd in train_raw)
    val_ids = set(sd.text_id for sd in val_raw)
    train_agg = [a for a in geco_agg if a.text_id in train_ids]
    test_agg = [a for a in geco_agg
                if a.text_id not in train_ids and a.text_id not in val_ids]
    print("Loading Provo...")
    provo_raw = load_provo(
        os.path.join(DATA_DIR, "Provo_Corpus-Eyetracking_Data.csv"),
    )
    provo_agg = aggregate_by_sentence(provo_raw, min_participants=10)
    return train_agg, test_agg, provo_agg


def _flat_predicts(name, train_agg, test_agg, provo_agg):
    """Returns dict corpus -> (p_trt, p_ffd, p_gaze, p_skip, y_trt, y_ffd, y_gaze, y_skip)."""
    mod_name = {"linear_regression": "linear_regression",
                "gpt2_surprisal": "gpt2_surprisal",
                "lightgbm": "lightgbm_baseline"}[name]
    mod = importlib.import_module(mod_name)
    subtlex = mod.load_subtlexus(os.path.join(DATA_DIR, "SUBTLEXus.txt"))

    if name == "linear_regression":
        ex = lambda agg: mod.extract_features(agg, subtlex)[:5]
    elif name == "gpt2_surprisal":
        surp = mod.GPT2SurprisalComputer(model_name="gpt2")
        ex = lambda agg: mod.extract_features_with_surprisal(agg, subtlex, surp)[:5]
    else:
        surp = mod.GPT2SurprisalComputer(model_name="gpt2")
        ex = lambda agg: mod.extract_rich_features(agg, subtlex, surp)[:5]

    print("  extracting features...")
    X_tr, y_ffd_tr, y_gaze_tr, y_trt_tr, y_skip_tr = ex(train_agg)
    X_te, y_ffd_te, y_gaze_te, y_trt_te, y_skip_te = ex(test_agg)
    X_pr, y_ffd_pr, y_gaze_pr, y_trt_pr, y_skip_pr = ex(provo_agg)

    print("  training...")
    if name == "lightgbm":
        try:
            import lightgbm as lgb
            params = dict(
                objective="regression", metric="mae", n_estimators=500,
                learning_rate=0.05, max_depth=-1, num_leaves=31,
                min_child_samples=20, verbose=-1, random_state=1,
            )
            mk = lambda: lgb.LGBMRegressor(**params)
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            mk = lambda: GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=5, random_state=1)
        ffd_m, gaze_m, trt_m, skip_m = mk(), mk(), mk(), mk()
        clip_skip = True
    else:
        ffd_m = mod.LinearRegressionModel()
        gaze_m = mod.LinearRegressionModel()
        trt_m = mod.LinearRegressionModel()
        skip_m = mod.LogisticRegressionModel(lr=0.1, epochs=1000)
        clip_skip = False
    ffd_m.fit(X_tr, y_ffd_tr); gaze_m.fit(X_tr, y_gaze_tr)
    trt_m.fit(X_tr, y_trt_tr); skip_m.fit(X_tr, y_skip_tr)

    def predict(X):
        ps = np.asarray(skip_m.predict(X))
        if clip_skip:
            ps = ps.clip(0.0, 1.0)
        return (np.asarray(trt_m.predict(X)), np.asarray(ffd_m.predict(X)),
                np.asarray(gaze_m.predict(X)), ps)

    out = {}
    for corpus, X, ys in (("geco_test", X_te, (y_trt_te, y_ffd_te, y_gaze_te, y_skip_te)),
                          ("provo", X_pr, (y_trt_pr, y_ffd_pr, y_gaze_pr, y_skip_pr))):
        p_trt, p_ffd, p_gaze, p_skip = predict(X)
        out[corpus] = (p_trt, p_ffd, p_gaze, p_skip) + tuple(np.asarray(y) for y in ys)
    return out


def run_flat(train_agg, test_agg, provo_agg, force):
    pos = {"geco_test": positions_from_agg(test_agg),
           "provo": positions_from_agg(provo_agg)}
    for name in ("linear_regression", "gpt2_surprisal", "lightgbm"):
        out_path = OUT_DIR / f"{name}_seed1.json"
        if out_path.exists() and not force:
            print(f">> {name}: exists, skipping")
            continue
        print(f"\n========== {name} ==========")
        t0 = time.time()
        preds = _flat_predicts(name, train_agg, test_agg, provo_agg)
        payload = {"model": name, "seed": 1, "datasets": {}}
        for corpus, tup in preds.items():
            p_trt, p_ffd, p_gaze, p_skip, y_trt, y_ffd, y_gaze, y_skip = tup
            payload["datasets"][corpus] = block_for(
                p_trt, p_ffd, p_gaze, p_skip,
                y_trt, y_ffd, y_gaze, y_skip, pos[corpus],
            )
        out_path.write_text(json.dumps(payload, indent=2, default=float))
        print(f">> {name}: wrote {out_path.name} ({time.time()-t0:.1f}s)")


# --------------------------------------------------------------------------- #
#  BERT (existing checkpoints, mirrors 02_eval_bert_ohio.py)
# --------------------------------------------------------------------------- #


def run_bert(device, force):
    from eval_baselines import load_eval_data, collect_bert_preds, get_human
    from bert_regression import BertDirectRegression

    geco_test, provo_agg = load_eval_data()
    pos = {"geco_test": positions_from_agg(geco_test),
           "provo": positions_from_agg(provo_agg)}

    for seed in SEEDS:
        out_path = OUT_DIR / f"bert_regression_seed{seed}.json"
        if out_path.exists() and not force:
            print(f">> bert seed={seed}: exists, skipping")
            continue
        ckpt_path = os.path.join(
            ARCHIVE_BASELINES, "checkpoints_bert_regression",
            f"seed{seed}", "best_model.pt",
        )
        if not os.path.isfile(ckpt_path):
            print(f">> bert seed={seed}: MISSING checkpoint, skipping")
            continue
        print(f"\n========== bert_regression seed={seed} ==========")
        model = BertDirectRegression().to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        payload = {"model": "bert_regression", "seed": seed, "datasets": {}}
        for corpus, data in (("geco_test", geco_test), ("provo", provo_agg)):
            p_trt, p_ffd, p_gaze, p_skip = collect_bert_preds(model, data, device)
            h_trt, h_ffd, h_gaze, h_skip = get_human(data)
            payload["datasets"][corpus] = block_for(
                p_trt, p_ffd, p_gaze, p_skip,
                h_trt, h_ffd, h_gaze, h_skip, pos[corpus],
            )
        out_path.write_text(json.dumps(payload, indent=2, default=float))
        print(f">> wrote {out_path.name}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
#  Ohio State RoBERTa (existing checkpoints)
# --------------------------------------------------------------------------- #


def run_ohio(device, force):
    from transformers import RobertaTokenizer
    from eval_baselines import (
        load_eval_data, _ohio_predict_all_metrics, _ohio_human_targets,
    )
    from run_ohio_state_on_geco import convert_to_ohio_format

    geco_test, provo_agg = load_eval_data()
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    print("  converting data to ohio format...")
    geco_data = convert_to_ohio_format(geco_test, tokenizer)
    provo_data = convert_to_ohio_format(provo_agg, tokenizer)
    # positions from the converted format's own per-sentence sublists
    # (index 6 = sent_trt; see _ohio_human_targets)
    pos = {"geco_test": positions_from_sublists(geco_data[6]),
           "provo": positions_from_sublists(provo_data[6])}

    for seed in SEEDS:
        out_path = OUT_DIR / f"ohio_state_roberta_seed{seed}.json"
        if out_path.exists() and not force:
            print(f">> ohio seed={seed}: exists, skipping")
            continue
        print(f"\n========== ohio_state_roberta seed={seed} ==========")
        payload = {"model": "ohio_state_roberta", "seed": seed, "datasets": {}}
        ok = True
        for corpus, data in (("geco_test", geco_data), ("provo", provo_data)):
            per_metric = _ohio_predict_all_metrics(seed, data, tokenizer, device)
            h_trt, h_ffd, h_gaze, h_skip = _ohio_human_targets(data)
            if any(per_metric[k] is None for k in ("trt", "ffd", "gaze", "skip")):
                print(f">> ohio seed={seed}: missing metric checkpoint, skipping seed")
                ok = False
                break
            payload["datasets"][corpus] = block_for(
                per_metric["trt"], per_metric["ffd"],
                per_metric["gaze"], per_metric["skip"],
                h_trt, h_ffd, h_gaze, h_skip, pos[corpus],
            )
        if ok:
            out_path.write_text(json.dumps(payload, indent=2, default=float))
            print(f">> wrote {out_path.name}")
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["flat", "bert", "ohio"], default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.only in (None, "flat"):
        train_agg, test_agg, provo_agg = _load_corpora()
        run_flat(train_agg, test_agg, provo_agg, args.force)
    if args.only in (None, "bert"):
        run_bert(device, args.force)
    if args.only in (None, "ohio"):
        run_ohio(device, args.force)

    print("\nDone.")


if __name__ == "__main__":
    main()
