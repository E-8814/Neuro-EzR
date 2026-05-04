"""
Data loaders for the paper-experiments pipeline.

Wraps the existing GECO and Provo loaders from
`archive/original_ezreader/` with caching and a uniform interface.

Provides:
    - load_geco_aggregated()         GECO test set, aggregated by sentence
    - load_geco_per_participant()    GECO raw, grouped by participant
    - load_provo_aggregated()        Provo, aggregated by sentence
    - load_subtlex()                 SUBTLEX frequency dict
"""

import os
import sys
from functools import lru_cache

import numpy as np

# Allow importing the existing data loaders from archive/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "archive", "original_ezreader"))

# These imports come from archive/original_ezreader/.
from data_loader import aggregate_by_sentence, load_provo  # noqa: E402
from geco_loader import load_geco, split_geco              # noqa: E402

from .. import config  # noqa: E402  (intentionally relative)


# --------------------------------------------------------------------------- #
#  GECO loading
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def _load_geco_raw():
    """Load GECO raw per-participant observations once and cache."""
    return load_geco(
        str(config.GECO_READING_FILE),
        str(config.GECO_MATERIAL_FILE),
        str(config.GECO_PRED_FILE),
    )


@lru_cache(maxsize=1)
def _split_geco_cached():
    """Train/val/test split."""
    raw = _load_geco_raw()
    return split_geco(raw)


def load_geco_aggregated(split: str = "test", min_participants: int = 5):
    """
    Returns sentence-aggregated GECO data for the requested split.

    Args:
        split: one of {"train", "val", "test"}.
        min_participants: drop sentences with fewer participants than this.

    Returns:
        list of AggregatedSentence objects (with .tokens, .mean_trt,
        .mean_ffd, .mean_gaze, .skip_rate, .text_id, etc.)
    """
    raw = _load_geco_raw()
    train_raw, val_raw, test_raw = _split_geco_cached()
    aggregated = aggregate_by_sentence(raw, min_participants=min_participants)

    train_text_ids = set(sd.text_id for sd in train_raw)
    val_text_ids = set(sd.text_id for sd in val_raw)

    if split == "train":
        return [a for a in aggregated if a.text_id in train_text_ids]
    if split == "val":
        return [a for a in aggregated if a.text_id in val_text_ids]
    if split == "test":
        return [
            a for a in aggregated
            if a.text_id not in train_text_ids and a.text_id not in val_text_ids
        ]
    raise ValueError(f"Unknown split: {split!r}")


def load_geco_per_participant(split: str = None):
    """
    Returns GECO raw observations grouped by participant_id.

    Args:
        split: optional split filter. If "test", only sentences in the
            test split are kept (per-participant evaluation typically
            wants this).

    Returns:
        Dict[participant_id (str)] -> List[SentenceData]
    """
    raw = _load_geco_raw()

    if split is not None:
        train_raw, val_raw, test_raw = _split_geco_cached()
        if split == "train":
            target = train_raw
        elif split == "val":
            target = val_raw
        elif split == "test":
            target = test_raw
        else:
            raise ValueError(f"Unknown split: {split!r}")
        # Filter raw by being in the target set's (text_id, sentence_number, participant) tuples.
        # Cheaper: target IS the same SentenceData list we want to return.
        raw = target

    by_participant = {}
    for sd in raw:
        pid = sd.participant_id
        by_participant.setdefault(pid, []).append(sd)
    return by_participant


def list_geco_participants():
    """Return sorted list of participant IDs in GECO."""
    by_p = load_geco_per_participant()
    return sorted(by_p.keys())


# --------------------------------------------------------------------------- #
#  Provo loading
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def _load_provo_raw():
    """Load Provo raw per-participant observations once."""
    return load_provo(str(config.PROVO_FILE))


def load_provo_aggregated(min_participants: int = 5):
    """
    Returns sentence-aggregated Provo data.

    Provo is a single corpus (no train/val/test split — used as held-out
    cross-corpus test).
    """
    raw = _load_provo_raw()
    return aggregate_by_sentence(raw, min_participants=min_participants)


# --------------------------------------------------------------------------- #
#  SUBTLEX
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def load_subtlex():
    """Load SUBTLEX-US frequency dict (lowercased -> raw count)."""
    import csv
    freq = {}
    with open(str(config.SUBTLEX_FILE), "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            freq[row["Word"].lower()] = int(row["FREQcount"])
    return freq


def word_frequency(token: str, subtlex: dict) -> int:
    """Look up frequency for a token, with a few cleanup heuristics."""
    w = token.lower().strip(".,;:!?\"'()[]{}").replace("’", "'")
    if w in subtlex:
        return max(1, subtlex[w])
    for variant in (w.replace("'", ""), w.split("'")[0], w.split("-")[0]):
        if variant in subtlex:
            return max(1, subtlex[variant])
    length = len(w)
    if length <= 3:
        return 50000
    if length <= 5:
        return 10000
    if length <= 7:
        return 2000
    return 500


# --------------------------------------------------------------------------- #
#  Standalone smoke test
# --------------------------------------------------------------------------- #


if __name__ == "__main__":
    print("Loading GECO test...")
    test_agg = load_geco_aggregated("test")
    print(f"  {len(test_agg)} test sentences")

    print("Loading per-participant GECO test...")
    by_p = load_geco_per_participant(split="test")
    print(f"  {len(by_p)} participants: {sorted(by_p.keys())[:5]}...")

    print("Loading Provo...")
    provo = load_provo_aggregated()
    print(f"  {len(provo)} sentences")

    print("Loading SUBTLEX...")
    sx = load_subtlex()
    print(f"  {len(sx):,} entries")
    print("OK.")
