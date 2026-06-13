"""
Re-run the classical E-Z Reader simulator eval (Reichle 2003, fitted
params, 200 MC runs/sentence) and score it BOTH ways:

    skip_all : all words            (sanity check vs the existing
               ez_reader_classical_seed1.json)
    skip_cmp : words 1..L-1         (the comparable population used by
               every other row in the v3 tables; classical predicts each
               word's skip at the word's own row -> same-index selection)

This time the per-word predictions are SAVED, so any future re-scoring
needs no simulation re-run.

Reuses _run_corpus() and the fitted-parameter loading from
exp01_main_comparison/complete_metrics/04_eval_ez_classical.py
(imported via importlib; not modified).

Outputs:
    results/raw/ez_reader_classical_v3_seed1.json   (model: ez_reader_classical)
    results/perword/ez_reader_classical_{geco_test,provo}.csv

Usage:
    python -u eval_ez_classical_v3.py --workers 16
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import sys
import time
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
SRC_V2 = os.path.abspath(os.path.join(_HERE, "..", ".."))
REPO_ROOT = os.path.dirname(SRC_V2)
CM_DIR = os.path.join(SRC_V2, "paper_experiments", "exp01_main_comparison",
                      "complete_metrics")
ARCHIVE_BASELINES = os.path.join(REPO_ROOT, "archive", "baselines")
ORIG_EZ = os.path.join(REPO_ROOT, "archive", "original_ezreader")

for p in (SRC_V2, CM_DIR, ARCHIVE_BASELINES, ORIG_EZ, _HERE,
          os.path.join(CM_DIR, "ez_classical")):
    if p not in sys.path:
        sys.path.insert(0, p)

from paper_experiments.utils.load_data import (  # noqa: E402
    load_geco_aggregated, load_provo_aggregated, load_subtlex,
)

from skip_metrics import (  # noqa: E402
    same_index_pairs, skip_summary, positions_from_agg,
)

# import 04_eval_ez_classical (module name starts with a digit)
_spec = importlib.util.spec_from_file_location(
    "eval_ez_classical_v2", os.path.join(CM_DIR, "04_eval_ez_classical.py"))
_m04 = importlib.util.module_from_spec(_spec)
# register under its alias BEFORE exec so multiprocessing (fork) children can
# unpickle the worker function by module name
sys.modules["eval_ez_classical_v2"] = _m04
_spec.loader.exec_module(_m04)

RAW_DIR = Path(_HERE) / "results" / "raw"
PW_DIR = Path(_HERE) / "results" / "perword"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PW_DIR.mkdir(parents=True, exist_ok=True)


def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) > 2 and a.std() > 0 and b.std() > 0:
        return float(np.corrcoef(a, b)[0, 1])
    return 0.0


def block_for(p_trt, p_ffd, p_gaze, p_skip,
              h_trt, h_ffd, h_gaze, h_skip, positions) -> dict:
    out = {}
    for m, p, h in (("trt", p_trt, h_trt), ("ffd", p_ffd, h_ffd),
                    ("gaze", p_gaze, h_gaze)):
        out[f"r_{m}"] = _corr(p, h)
        out[f"mae_{m}"] = float(np.mean(np.abs(np.asarray(p) - np.asarray(h))))
        out[f"bias_{m}"] = float(np.mean(p) - np.mean(h))
    out["n_words_all"] = int(len(p_trt))
    out["skip_all"] = skip_summary(p_skip, h_skip)
    sp, st = same_index_pairs(p_skip, h_skip, positions)
    out["skip_cmp"] = skip_summary(sp, st)
    return out


def dump_perword(agg_list, arrays, out_path):
    p_trt, p_ffd, p_gaze, p_skip = arrays[:4]
    words = [t for a in agg_list for t in a.tokens]
    sent = [i for i, a in enumerate(agg_list) for _ in a.tokens]
    pos = [j for a in agg_list for j in range(len(a.tokens))]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sentence_idx", "word_position", "word",
                    "pred_trt", "pred_ffd", "pred_gaze", "pred_skip"])
        for i in range(len(words)):
            w.writerow([sent[i], pos[i], words[i],
                        p_trt[i], p_ffd[i], p_gaze[i], p_skip[i]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", type=int, default=200)
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_path = RAW_DIR / "ez_reader_classical_v3_seed1.json"
    if out_path.exists() and not args.force:
        print(f">> {out_path.name} exists, skipping.")
        return

    # fitted params, same loading logic as the v2 script
    fitted_path = Path(CM_DIR) / "ez_classical" / "fitted_params.json"
    model_params = None
    params_source = "Reichle 2003 defaults"
    if fitted_path.exists():
        payload = json.loads(fitted_path.read_text())
        model_params = payload.get("fitted_params", None)
        if model_params:
            params_source = f"FITTED params from {fitted_path.name}"
    print(f"Classical-engine parameters: {params_source}")

    random.seed(1)
    np.random.seed(1)

    subtlex = load_subtlex()
    geco_test = load_geco_aggregated("test")
    provo = load_provo_aggregated()
    if args.limit:
        geco_test, provo = geco_test[:args.limit], provo[:args.limit]
    print(f"GECO test {len(geco_test)} | Provo {len(provo)} sentences")

    payload = {
        "model": "ez_reader_classical",
        "seed": 1,
        "num_mc_runs": args.num_runs,
        "is_classical": True,
        "params_source": params_source,
        "model_params": model_params,
        "skip_population": "skip_cmp = words 1..L-1; skip_all = all words",
        "datasets": {},
    }
    t0 = time.time()
    for label, data in (("geco_test", geco_test), ("provo", provo)):
        print(f"\n========== {label} ==========")
        arrays = _m04._run_corpus(
            data, subtlex, args.num_runs, args.workers, label,
            model_params=model_params,
        )
        positions = positions_from_agg(data)
        assert len(positions) == len(arrays[0]), "position/pred length mismatch"
        payload["datasets"][label] = block_for(*arrays, positions)
        dump_perword(data, arrays, PW_DIR / f"ez_reader_classical_{label}.csv")
        b = payload["datasets"][label]
        print(f"  sanity (all-words r_skip): {b['skip_all']['r_skip']:+.3f}")
        print(f"  comparable (1..L-1) r_skip: {b['skip_cmp']['r_skip']:+.3f}  "
              f"AUC={b['skip_cmp']['skip_auc']:.3f}")

    out_path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nWrote {out_path}  (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
