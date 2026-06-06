# Permutation null for the H3 group fit

Addresses §Weakness 1 of the paper:

> "The per-group dissociation rests on n=7 readers per group with no
> bootstrap CIs; the headline finding therefore needs uncertainty
> quantification before it can be treated as established, not merely
> suggestive."

We test whether the observed lexical-vs-motor dissociation could arise
from random groupings of the 14 GECO readers. With n=14 there are
exactly **C(14,7)/2 = 1,716 unique balanced 7/7 splits**, so we can
*enumerate* the null instead of Monte-Carloing it.

## Test statistic

```
T = mean( |%Δ_lexical| )  −  mean( |%Δ_motor| )
```

with `lexical = {δ, λ_refix, ε}` and `motor = {M₁, M₂, τ}`,
`%Δ = 100·(slow − fast)/|fast|`. (T is symmetric under fast↔slow
because of the absolute values.)

The observed split (FAST_READERS vs SLOW_READERS in
`fit_per_group.py`) gives T_obs ≈ +16 percentage points. We compute T
for all 1,716 random groupings and report
`p = (1 + #{T_perm ≥ T_obs}) / (1 + n_perm)`.

## Files

```
permutation_null/
├── _cog_fit.py              shared helpers: cog-only forward, fit loop, T
├── 01_cache_features.py     cache frozen-neural outputs per participant
├── 02_sanity_check.py       cached fast/slow fit ≈ published exp09 numbers
├── 03a_perm_cached.py       enumerate 1,716 splits (used if sanity passes)
├── 03b_perm_live.py         300 random splits on live model (fallback)
├── 04_aggregate.py          collate JSONs → CSV / summary / histogram PDF
├── run_slurm.sh             sbatch wrapper (orchestrates 01 → 02 → 3a/3b → 04)
└── results/
    ├── cache/               one .pt per participant (~few MB each)
    ├── sanity_check.json    pass/fail + diagnostics
    ├── perms/               cached path: one .json per completed split
    ├── perms_live/          live fallback: one .json per completed split
    ├── perm_distribution.csv  long-form null distribution
    ├── perm_summary.json    {n, T_obs, p_value, ...}
    └── perm_histogram.pdf   small panel for the paper
```

## Running it

### One-shot

```bash
sbatch src_v2/paper_experiments/exp09_per_participant_cog_fits/permutation_null/run_slurm.sh
```

The script orchestrates everything: caches features (idempotent), runs
the sanity check, branches to the cached enumeration if sanity passes
or to the live fallback if not, and finally aggregates whatever is
complete. Re-submit to resume — every split's result is its own JSON,
and missing indices are picked at random within each run.

### Multiple GPUs / multiple SLURM jobs

Just submit the script multiple times. The 03a/03b loops detect which
indices are already done by listing JSONs in `results/perms[/_live]/`,
and pick uniformly at random from what's missing. Race-safety is
provided by atomic JSON writes (`.tmp` then `os.replace`); collisions
at worst recompute one split.

### Monitoring

```bash
ls src_v2/paper_experiments/exp09_per_participant_cog_fits/permutation_null/results/perms/ | wc -l
# expect this number to climb toward 1716

cat src_v2/paper_experiments/exp09_per_participant_cog_fits/permutation_null/results/sanity_check.json
# expect "passed": true
```

To see partial results without waiting for completion:

```bash
python src_v2/paper_experiments/exp09_per_participant_cog_fits/permutation_null/04_aggregate.py
```

## Compute estimates

Cached path (preferred):
- One per-group cog-scalar fit on cached features: a few seconds on a
  modern GPU (no LM forward; just SGD on a tiny tensor).
- 1,716 splits × 2 groups = 3,432 fits.
- Wall-clock total: roughly **1–4 GPU-hours** depending on per-fit
  speed.
- One 8-hour SLURM submission almost always finishes the whole thing.

Live fallback path:
- One fit involves a full TinyLlama forward per batch — ~2–5 minutes
  per group, ~5–10 minutes per split.
- 300 splits → ~25–50 GPU-hours total.
- Submit several jobs back-to-back. Each picks up where the last
  stopped.

## Why the sanity check matters

Caching trades correctness-by-construction for speed. The risk is that
the cached features come out subtly different from what the live model
produces (wrong tokenization, dropout/eval mode mismatch, autocast
precision, etc.) and the entire 1,716-permutation result is silently
wrong.

The sanity check guards against that: we run the *actual* fast/slow
split through the cached path and require its fitted cog scalars to
match the published values to within a relative tolerance (default
10%). If they don't, the cached pipeline is not trustworthy, and
`run_slurm.sh` automatically routes to the live fallback.

## Output

After enough perms have completed, `04_aggregate.py` produces:

- `perm_summary.json` — primary result for the paper
  (`p_value_one_sided_geq` is the headline number).
- `perm_histogram.pdf` — small panel showing the null distribution
  with a vertical line at T_obs.
- `perm_distribution.csv` — long-form data behind the histogram.

Cite the result inline near Figure 2:

> "Across all 1,716 balanced 7/7 splits of the 14 readers, the observed
> dissociation T = … exceeds N% of permuted values (p = …)."
