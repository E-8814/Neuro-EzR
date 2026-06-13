"""
Aggregate the v3 exact permutation null.

With all 1,716 balanced splits completed, the observed fast/slow split is
one of them, and the EXACT one-sided p-value is

    p = #{splits with T >= T_observed} / 1716

(the observed split counts itself, so p >= 1/1716 ≈ 0.00058).

Also reports, for every cognitive parameter, the observed |%shift|
(fast vs slow, signed too) against the null distribution of |%shift|
across all splits — the v3 analogue of Figure 1.

Outputs:
    results/perm_summary_v3.json
    results/perm_distribution_v3.csv
    (printed markdown report)

Usage:
    python -u aggregate_v3.py
"""

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
EXP09 = os.path.abspath(os.path.join(_HERE, ".."))
PERM_V2 = os.path.join(EXP09, "permutation_null")
SRC_V2 = os.path.abspath(os.path.join(EXP09, "..", ".."))

for p in (SRC_V2, EXP09, PERM_V2, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from _cog_fit import enumerate_balanced_splits, split_index  # noqa: E402
from fit_per_group import FAST_READERS, SLOW_READERS  # noqa: E402


PERMS_DIR = Path(_HERE) / "results" / "perms"
OUT_JSON = Path(_HERE) / "results" / "perm_summary_v3.json"
OUT_CSV = Path(_HERE) / "results" / "perm_distribution_v3.csv"

PARAMS = ["l1_base_offset", "l1_freq_coef", "alpha1_reichle", "alpha2_reichle",
          "delta", "epsilon", "M1", "M2_eq_I", "lambda_refix", "refix_pivot",
          "skip_temperature"]


def pct_shift(fast_val: float, slow_val: float) -> float:
    denom = abs(fast_val) if abs(fast_val) > 1e-9 else 1.0
    return 100.0 * (slow_val - fast_val) / denom


def main():
    records = []
    for path in sorted(PERMS_DIR.glob("perm_*.json")):
        if path.name.endswith(".error.json"):
            continue
        records.append(json.loads(path.read_text()))
    if not records:
        print("No completed perms found.")
        sys.exit(1)
    by_index = {r["index"]: r for r in records}
    print(f"Loaded {len(records)} completed splits.")

    pids = sorted(set(records[0]["group_a"]) | set(records[0]["group_b"]))
    splits = enumerate_balanced_splits(pids)
    n_total = len(splits)

    obs_idx = split_index(splits, sorted(FAST_READERS), sorted(SLOW_READERS))
    print(f"Observed fast/slow split is canonical index {obs_idx}.")
    if obs_idx not in by_index:
        print("ERROR: observed split not completed yet — run perm_v3.py to completion.")
        sys.exit(2)

    obs = by_index[obs_idx]
    # Orient observed groups: which side is FAST?
    if frozenset(obs["group_a"]) == frozenset(FAST_READERS):
        fast_cog, slow_cog = obs["cog_a"], obs["cog_b"]
    else:
        fast_cog, slow_cog = obs["cog_b"], obs["cog_a"]

    T_obs = obs["T"]
    Ts = np.array([by_index[i]["T"] for i in sorted(by_index)])
    n_done = len(Ts)
    exact_p = float(np.sum(Ts >= T_obs)) / n_done

    print(f"\nT_observed = {T_obs:+.3f}")
    print(f"Null over {n_done}/{n_total} splits: max={Ts.max():+.3f} "
          f"mean={Ts.mean():+.3f} p95={np.percentile(Ts, 95):+.3f}")
    print(f"EXACT one-sided p = {np.sum(Ts >= T_obs)}/{n_done} = {exact_p:.5f}")

    # Per-parameter observed shift vs null |shift| distribution
    print(f"\n| param | obs %shift (fast→slow) | null |%shift| p95 | exceedance "
          f"(#null ≥ |obs|) |")
    print("|---|---|---|---|")
    param_rows = []
    for prm in PARAMS:
        if prm not in fast_cog:
            continue
        obs_shift = pct_shift(fast_cog[prm], slow_cog[prm])
        null_abs = []
        for i, r in by_index.items():
            if i == obs_idx:
                continue
            null_abs.append(abs(pct_shift(r["cog_a"][prm], r["cog_b"][prm])))
        null_abs = np.array(null_abs)
        exceed = int(np.sum(null_abs >= abs(obs_shift)))
        p95 = float(np.percentile(null_abs, 95))
        print(f"| {prm} | {obs_shift:+.2f}% | {p95:.2f}% | {exceed}/{len(null_abs)} |")
        param_rows.append({
            "param": prm, "obs_pct_shift": obs_shift,
            "null_abs_p95": p95, "null_exceed": exceed,
            "n_null": len(null_abs),
        })

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "T"])
        for i in sorted(by_index):
            w.writerow([i, by_index[i]["T"]])

    OUT_JSON.write_text(json.dumps({
        "n_completed": n_done,
        "n_total": n_total,
        "observed_index": obs_idx,
        "T_observed": T_obs,
        "exact_p_one_sided": exact_p,
        "null_T_max": float(Ts.max()),
        "null_T_mean": float(Ts.mean()),
        "fast_cog": fast_cog,
        "slow_cog": slow_cog,
        "per_param": param_rows,
        "model_recipe": "v4c_v3_dualctx_next",
    }, indent=2, default=float))
    print(f"\nWrote {OUT_JSON}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
