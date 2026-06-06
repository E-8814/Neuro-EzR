"""
Train + evaluate the three flat baselines (linear regression, LightGBM,
GPT-2 surprisal) and emit complete-metrics JSONs.

These baselines have no saved checkpoints (their training is fast and
deterministic, so the original pipeline doesn't bother). To produce
augmented metrics (mae_gaze, mae_skip, bias_*, n_words) we retrain
them here using their existing classes/feature-extractors imported
from archive/baselines/, then run inference on GECO test + Provo and
route the predictions through metrics_summary_complete().

Outputs:
    complete_metrics/results/raw/baselines/linear_regression_seed1.json
    complete_metrics/results/raw/baselines/lightgbm_seed1.json
    complete_metrics/results/raw/baselines/gpt2_surprisal_seed1.json

Usage:
    python -u .../03_train_eval_flat_baselines.py
    python -u .../03_train_eval_flat_baselines.py --only linear_regression
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

_HERE = os.path.dirname(os.path.abspath(__file__))
SRC_V2 = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
REPO_ROOT = os.path.dirname(SRC_V2)
ARCHIVE_BASELINES = os.path.join(REPO_ROOT, "archive", "baselines")
ORIG_EZ = os.path.join(REPO_ROOT, "archive", "original_ezreader")
DATA_DIR = os.path.join(REPO_ROOT, "data")

for p in (SRC_V2, ARCHIVE_BASELINES, ORIG_EZ, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_loader import load_provo, aggregate_by_sentence  # noqa: E402
from geco_loader import load_geco, split_geco              # noqa: E402

from metrics import metrics_summary_complete  # local


OUT_DIR = Path(_HERE) / "results" / "raw" / "baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
#  Shared data loading
# --------------------------------------------------------------------------- #


def _load_geco_test_and_provo():
    """Returns (train_agg, val_agg, test_agg, provo_agg)."""
    print("Loading GECO...")
    geco_raw = load_geco(
        os.path.join(DATA_DIR, "Geco_MonolingualReadingData.csv"),
        os.path.join(DATA_DIR, "Geco_EnglishMaterial.csv"),
        os.path.join(DATA_DIR, "geco_predictability.pkl"),
    )
    train_raw, val_raw, test_raw = split_geco(geco_raw)
    geco_agg = aggregate_by_sentence(geco_raw, min_participants=5)
    train_ids = set(sd.text_id for sd in train_raw)
    val_ids   = set(sd.text_id for sd in val_raw)
    train_agg = [a for a in geco_agg if a.text_id in train_ids]
    val_agg   = [a for a in geco_agg if a.text_id in val_ids]
    test_agg  = [a for a in geco_agg
                 if a.text_id not in train_ids and a.text_id not in val_ids]
    print(f"  GECO  train={len(train_agg)}  val={len(val_agg)}  test={len(test_agg)}")

    print("Loading Provo (cross-corpus)...")
    provo_raw = load_provo(
        os.path.join(DATA_DIR, "Provo_Corpus-Eyetracking_Data.csv"),
    )
    provo_agg = aggregate_by_sentence(provo_raw, min_participants=10)
    print(f"  Provo  {len(provo_agg)} sentences")

    return train_agg, val_agg, test_agg, provo_agg


# --------------------------------------------------------------------------- #
#  Linear regression baseline
# --------------------------------------------------------------------------- #


def _eval_linear_regression(train_agg, test_agg, provo_agg):
    print("\n========== linear_regression ==========")
    mod = importlib.import_module("linear_regression")
    subtlex = mod.load_subtlexus(os.path.join(DATA_DIR, "SUBTLEXus.txt"))

    print("  extracting features...")
    X_tr, y_ffd_tr, y_gaze_tr, y_trt_tr, y_skip_tr, _ = mod.extract_features(train_agg, subtlex)
    X_te, y_ffd_te, y_gaze_te, y_trt_te, y_skip_te, _ = mod.extract_features(test_agg, subtlex)
    X_pr, y_ffd_pr, y_gaze_pr, y_trt_pr, y_skip_pr, _ = mod.extract_features(provo_agg, subtlex)

    print("  training...")
    ffd_model  = mod.LinearRegressionModel(); ffd_model.fit(X_tr, y_ffd_tr)
    gaze_model = mod.LinearRegressionModel(); gaze_model.fit(X_tr, y_gaze_tr)
    trt_model  = mod.LinearRegressionModel(); trt_model.fit(X_tr, y_trt_tr)
    skip_model = mod.LogisticRegressionModel(lr=0.1, epochs=1000); skip_model.fit(X_tr, y_skip_tr)

    def predict(X):
        return (
            np.asarray(trt_model.predict(X)),
            np.asarray(ffd_model.predict(X)),
            np.asarray(gaze_model.predict(X)),
            np.asarray(skip_model.predict(X)),
        )

    p_trt, p_ffd, p_gaze, p_skip = predict(X_te)
    geco = metrics_summary_complete(p_trt, p_ffd, p_gaze, p_skip,
                                    y_trt_te, y_ffd_te, y_gaze_te, y_skip_te)
    p_trt, p_ffd, p_gaze, p_skip = predict(X_pr)
    provo = metrics_summary_complete(p_trt, p_ffd, p_gaze, p_skip,
                                     y_trt_pr, y_ffd_pr, y_gaze_pr, y_skip_pr)

    return {"model": "linear_regression", "seed": 1,
            "datasets": {"geco_test": geco, "provo": provo}}


# --------------------------------------------------------------------------- #
#  LightGBM baseline (rich features + GPT-2 surprisal)
# --------------------------------------------------------------------------- #


def _eval_lightgbm(train_agg, test_agg, provo_agg):
    print("\n========== lightgbm ==========")
    mod = importlib.import_module("lightgbm_baseline")
    subtlex = mod.load_subtlexus(os.path.join(DATA_DIR, "SUBTLEXus.txt"))

    print("  loading GPT-2 for surprisal features...")
    surp = mod.GPT2SurprisalComputer(model_name="gpt2")

    print("  extracting features...")
    # extract_rich_features returns a 5-tuple (no tokens_list).
    X_tr, y_ffd_tr, y_gaze_tr, y_trt_tr, y_skip_tr = mod.extract_rich_features(train_agg, subtlex, surp)
    X_te, y_ffd_te, y_gaze_te, y_trt_te, y_skip_te = mod.extract_rich_features(test_agg, subtlex, surp)
    X_pr, y_ffd_pr, y_gaze_pr, y_trt_pr, y_skip_pr = mod.extract_rich_features(provo_agg, subtlex, surp)

    print("  training...")
    try:
        import lightgbm as lgb
        params = dict(
            objective="regression", metric="mae", n_estimators=500,
            learning_rate=0.05, max_depth=-1, num_leaves=31,
            min_child_samples=20, verbose=-1, random_state=1,
        )
        ffd_model  = lgb.LGBMRegressor(**params); ffd_model.fit(X_tr, y_ffd_tr)
        gaze_model = lgb.LGBMRegressor(**params); gaze_model.fit(X_tr, y_gaze_tr)
        trt_model  = lgb.LGBMRegressor(**params); trt_model.fit(X_tr, y_trt_tr)
        # For skip: regression head clamped to [0, 1] (matches existing baseline)
        skip_model = lgb.LGBMRegressor(**params); skip_model.fit(X_tr, y_skip_tr)
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        gbr_kw = dict(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=1)
        ffd_model  = GradientBoostingRegressor(**gbr_kw); ffd_model.fit(X_tr, y_ffd_tr)
        gaze_model = GradientBoostingRegressor(**gbr_kw); gaze_model.fit(X_tr, y_gaze_tr)
        trt_model  = GradientBoostingRegressor(**gbr_kw); trt_model.fit(X_tr, y_trt_tr)
        skip_model = GradientBoostingRegressor(**gbr_kw); skip_model.fit(X_tr, y_skip_tr)

    def predict(X):
        return (
            np.asarray(trt_model.predict(X)),
            np.asarray(ffd_model.predict(X)),
            np.asarray(gaze_model.predict(X)),
            np.asarray(skip_model.predict(X)).clip(0.0, 1.0),
        )

    p_trt, p_ffd, p_gaze, p_skip = predict(X_te)
    geco = metrics_summary_complete(p_trt, p_ffd, p_gaze, p_skip,
                                    y_trt_te, y_ffd_te, y_gaze_te, y_skip_te)
    p_trt, p_ffd, p_gaze, p_skip = predict(X_pr)
    provo = metrics_summary_complete(p_trt, p_ffd, p_gaze, p_skip,
                                     y_trt_pr, y_ffd_pr, y_gaze_pr, y_skip_pr)

    return {"model": "lightgbm", "seed": 1,
            "datasets": {"geco_test": geco, "provo": provo}}


# --------------------------------------------------------------------------- #
#  GPT-2 surprisal baseline
# --------------------------------------------------------------------------- #


def _eval_gpt2_surprisal(train_agg, test_agg, provo_agg):
    print("\n========== gpt2_surprisal ==========")
    mod = importlib.import_module("gpt2_surprisal")
    subtlex = mod.load_subtlexus(os.path.join(DATA_DIR, "SUBTLEXus.txt"))

    print("  loading GPT-2 surprisal computer...")
    surp = mod.GPT2SurprisalComputer(model_name="gpt2")

    print("  extracting features...")
    X_tr, y_ffd_tr, y_gaze_tr, y_trt_tr, y_skip_tr, _ = mod.extract_features_with_surprisal(train_agg, subtlex, surp)
    X_te, y_ffd_te, y_gaze_te, y_trt_te, y_skip_te, _ = mod.extract_features_with_surprisal(test_agg, subtlex, surp)
    X_pr, y_ffd_pr, y_gaze_pr, y_trt_pr, y_skip_pr, _ = mod.extract_features_with_surprisal(provo_agg, subtlex, surp)

    print("  training...")
    ffd_model  = mod.LinearRegressionModel(); ffd_model.fit(X_tr, y_ffd_tr)
    gaze_model = mod.LinearRegressionModel(); gaze_model.fit(X_tr, y_gaze_tr)
    trt_model  = mod.LinearRegressionModel(); trt_model.fit(X_tr, y_trt_tr)
    skip_model = mod.LogisticRegressionModel(lr=0.1, epochs=1000); skip_model.fit(X_tr, y_skip_tr)

    def predict(X):
        return (
            np.asarray(trt_model.predict(X)),
            np.asarray(ffd_model.predict(X)),
            np.asarray(gaze_model.predict(X)),
            np.asarray(skip_model.predict(X)),
        )

    p_trt, p_ffd, p_gaze, p_skip = predict(X_te)
    geco = metrics_summary_complete(p_trt, p_ffd, p_gaze, p_skip,
                                    y_trt_te, y_ffd_te, y_gaze_te, y_skip_te)
    p_trt, p_ffd, p_gaze, p_skip = predict(X_pr)
    provo = metrics_summary_complete(p_trt, p_ffd, p_gaze, p_skip,
                                     y_trt_pr, y_ffd_pr, y_gaze_pr, y_skip_pr)

    return {"model": "gpt2_surprisal", "seed": 1,
            "datasets": {"geco_test": geco, "provo": provo}}


# --------------------------------------------------------------------------- #
#  Driver
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["linear_regression", "lightgbm",
                                           "gpt2_surprisal"],
                        default=None,
                        help="Run just one baseline (default: all three).")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    train_agg, _, test_agg, provo_agg = _load_geco_test_and_provo()

    runners = {
        "linear_regression": _eval_linear_regression,
        "lightgbm":          _eval_lightgbm,
        "gpt2_surprisal":    _eval_gpt2_surprisal,
    }
    if args.only:
        runners = {args.only: runners[args.only]}

    for name, fn in runners.items():
        out_path = OUT_DIR / f"{name}_seed1.json"
        if out_path.exists() and not args.force:
            print(f">> {name}: {out_path.name} exists, skipping (use --force to redo).")
            continue
        t0 = time.time()
        try:
            payload = fn(train_agg, test_agg, provo_agg)
        except Exception as exc:
            print(f">> {name}: FAILED -> {exc!r}")
            continue
        out_path.write_text(json.dumps(payload, indent=2, default=float))
        print(f">> {name}: wrote {out_path.name}  ({time.time()-t0:.1f}s)")

    print("\nDone.")


if __name__ == "__main__":
    main()
