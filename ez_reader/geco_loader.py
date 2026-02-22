"""
Data loader for the GECO Corpus (Monolingual English).

Parses GECO CSVs into the same SentenceData / AggregatedSentence structures
used by data_loader.py for the Provo Corpus.

GECO data:
  - 14 monolingual English participants
  - ~5,284 sentences from an Agatha Christie novel
  - ~54,362 unique words
  - 774,015 word-level observations

Key column mappings (GECO → our format):
    WORD                          → word
    WORD_LENGTH (from material)   → word_length
    WORD_FIRST_FIXATION_DURATION  → first_fixation_duration  ('.' = 0)
    WORD_GAZE_DURATION            → gaze_duration            ('.' = 0)
    WORD_TOTAL_READING_TIME       → total_reading_time       ('.' = 0)
    WORD_SKIP                     → was_skipped              (0/1)
    PP_NR                         → participant_id
    SENTENCE_ID (from material)   → text_id + sentence_number
    GPT-2 predictability (cached) → predictability

Usage:
    python geco_loader.py
"""

import os
import sys
import csv
import pickle
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# Reuse data structures from data_loader (same directory)
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import WordData, SentenceData, AggregatedSentence, aggregate_by_sentence


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _safe_float(val, default=0.0):
    """Parse numeric, returning default for '.', NA, empty."""
    if val is None:
        return default
    val = str(val).strip()
    if val in (".", "NA", "", "nan"):
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _safe_int(val, default=0):
    if val is None:
        return default
    val = str(val).strip()
    if val in (".", "NA", "", "nan"):
        return default
    try:
        return int(float(val))
    except ValueError:
        return default


# --------------------------------------------------------------------------- #
#  Build WORD_ID → SENTENCE_ID mapping from material file
# --------------------------------------------------------------------------- #

def load_material(material_path):
    """
    Load GECO EnglishMaterial CSV.

    Returns:
        word_to_sentence: dict mapping WORD_ID → SENTENCE_ID
        sentence_words:   dict mapping SENTENCE_ID → ordered list of (WORD_ID, word_text)
    """
    word_to_sentence = {}
    sentence_words = defaultdict(list)

    with open(material_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word_id = row["WORD_ID"]
            sent_id = row["SENTENCE_ID"]
            word = row["WORD"]
            word_to_sentence[word_id] = sent_id
            sentence_words[sent_id].append((word_id, word))

    return word_to_sentence, dict(sentence_words)


# --------------------------------------------------------------------------- #
#  Encode SENTENCE_ID as numeric text_id + sentence_number
# --------------------------------------------------------------------------- #

def encode_sentence_ids(sentence_ids):
    """
    Convert GECO SENTENCE_IDs (e.g. '1-1', '1-2', '2-5') to
    (text_id, sentence_number) pairs.

    GECO format: 'PART-SENTENCE_NUM'
    Each unique sentence gets its own text_id (starting at 1001) so that
    split_geco() can do a proper 80/10/10 split at sentence granularity.
    Provo uses text_ids 1-55, so offset >= 1000 avoids collision.
    """
    mapping = {}
    sorted_ids = sorted(sentence_ids, key=lambda s: (
        int(str(s).split("-")[0]) if "-" in str(s) else 9999,
        int(str(s).split("-")[1]) if "-" in str(s) else 0,
    ))
    for i, sent_id in enumerate(sorted_ids):
        parts = str(sent_id).split("-")
        if len(parts) == 2:
            sent_num = int(parts[1])
        else:
            sent_num = 0
        text_id = 1001 + i  # unique per sentence, offset from Provo
        mapping[sent_id] = (text_id, sent_num)
    return mapping


# --------------------------------------------------------------------------- #
#  Load predictability
# --------------------------------------------------------------------------- #

def load_predictability(pred_path):
    """
    Load cached GPT-2 predictability from pickle file.

    Returns:
        dict: SENTENCE_ID → {'words': [...], 'predictability': [...]}
    """
    if not os.path.exists(pred_path):
        return None

    with open(pred_path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- #
#  Main loader
# --------------------------------------------------------------------------- #

def load_geco(
    reading_data_path: str,
    material_path: str,
    predictability_path: Optional[str] = None,
    min_sentence_length: int = 2,
    max_sentence_length: int = 50,
) -> List[SentenceData]:
    """
    Load the GECO Corpus and return a list of SentenceData objects
    (one per participant × sentence), matching the data_loader.py interface.

    Parameters
    ----------
    reading_data_path : str
        Path to Geco_MonolingualReadingData.csv
    material_path : str
        Path to Geco_EnglishMaterial.csv
    predictability_path : str or None
        Path to geco_predictability.pkl (from compute_predictability.py).
        If None or missing, predictability defaults to 0.0 for all words.
    min_sentence_length : int
    max_sentence_length : int

    Returns
    -------
    list[SentenceData]
    """
    # 1. Load material to get WORD_ID → SENTENCE_ID mapping
    print("  Loading GECO material...")
    word_to_sentence, sentence_words = load_material(material_path)
    sent_id_mapping = encode_sentence_ids(sentence_words.keys())
    print(f"    {len(sentence_words)} sentences, {len(word_to_sentence)} words")

    # 2. Load predictability (if available)
    pred_data = None
    if predictability_path:
        pred_data = load_predictability(predictability_path)
        if pred_data:
            print(f"    Loaded predictability for {len(pred_data)} sentences")
        else:
            print("    Predictability file not found, using 0.0 for all words")

    # 3. Build per-word predictability lookup: WORD_ID → predictability
    word_predictability = {}
    if pred_data:
        for sent_id, words_list in sentence_words.items():
            if sent_id in pred_data:
                preds = pred_data[sent_id]['predictability']
                for i, (word_id, _) in enumerate(words_list):
                    if i < len(preds):
                        word_predictability[word_id] = preds[i]

    # 4. Read the eye-tracking data and bucket by (participant, sentence)
    print("  Loading GECO reading data...")
    # Key = (participant_id, sentence_id)
    buckets: Dict[Tuple[str, str], List[dict]] = defaultdict(list)

    with open(reading_data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word_id = row["WORD_ID"]
            if word_id not in word_to_sentence:
                continue
            sent_id = word_to_sentence[word_id]
            pid = row["PP_NR"]
            buckets[(pid, sent_id)].append(row)

    print(f"    {len(buckets)} participant × sentence combinations")

    # 5. Convert each bucket to a SentenceData
    dataset: List[SentenceData] = []

    for (pid, sent_id), rows in buckets.items():
        # Sort by word position within trial
        rows.sort(key=lambda r: _safe_int(r.get("WORD_ID_WITHIN_TRIAL", 0)))

        # Get the canonical word order from material
        canonical_words = sentence_words.get(sent_id, [])
        canonical_word_ids = [wid for wid, _ in canonical_words]

        # Build a lookup from WORD_ID → row
        row_by_word_id = {}
        for r in rows:
            row_by_word_id[r["WORD_ID"]] = r

        words: List[WordData] = []
        for word_pos, (word_id, word_text) in enumerate(canonical_words):
            r = row_by_word_id.get(word_id)
            if r is None:
                continue

            ffd = _safe_float(r.get("WORD_FIRST_FIXATION_DURATION"), 0.0)
            gaze = _safe_float(r.get("WORD_GAZE_DURATION"), 0.0)
            total = _safe_float(r.get("WORD_TOTAL_READING_TIME"), 0.0)
            skipped = _safe_int(r.get("WORD_SKIP"), 0) == 1
            pred = word_predictability.get(word_id, 0.0)

            wd = WordData(
                word=word_text.strip(),
                word_length=len(word_text.strip()),
                predictability=pred,
                first_fixation_duration=ffd,
                gaze_duration=gaze,
                total_reading_time=total,
                was_skipped=skipped,
                regression_in=False,  # GECO doesn't have a simple regression_in flag
                word_number=word_pos + 1,
                sentence_number=sent_id_mapping.get(sent_id, (9999, 0))[1],
                word_in_sentence=word_pos + 1,
                text_id=sent_id_mapping.get(sent_id, (9999, 0))[0],
                participant_id=pid,
            )
            words.append(wd)

        # Filter by sentence length
        if len(words) < min_sentence_length or len(words) > max_sentence_length:
            continue

        text_id, sent_num = sent_id_mapping.get(sent_id, (9999, 0))

        sd = SentenceData(
            text_id=text_id,
            sentence_number=sent_num,
            participant_id=pid,
            words=words,
        )
        dataset.append(sd)

    # Sort for reproducibility
    dataset.sort(key=lambda s: (s.text_id, s.sentence_number, s.participant_id))

    return dataset


# --------------------------------------------------------------------------- #
#  Split by text_id (same logic as data_loader.split_dataset)
# --------------------------------------------------------------------------- #

def split_geco(
    dataset: List[SentenceData],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """Split GECO by text_id to avoid data leakage."""
    import random
    rng = random.Random(seed)

    text_ids = sorted(set(sd.text_id for sd in dataset))
    rng.shuffle(text_ids)

    n_train = int(len(text_ids) * train_ratio)
    n_val = int(len(text_ids) * val_ratio)

    train_texts = set(text_ids[:n_train])
    val_texts = set(text_ids[n_train:n_train + n_val])

    train = [sd for sd in dataset if sd.text_id in train_texts]
    val = [sd for sd in dataset if sd.text_id in val_texts]
    test = [sd for sd in dataset if sd.text_id not in train_texts and sd.text_id not in val_texts]

    return train, val, test


# --------------------------------------------------------------------------- #
#  CLI: quick stats
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    reading_path = os.path.join(data_dir, "Geco_MonolingualReadingData.csv")
    material_path = os.path.join(data_dir, "Geco_EnglishMaterial.csv")
    pred_path = os.path.join(data_dir, "geco_predictability.pkl")

    print("Loading GECO Corpus...")
    dataset = load_geco(reading_path, material_path, pred_path)

    print(f"\n=== GECO Corpus Stats ===")
    print(f"Total sentence-observations: {len(dataset):,}")

    participants = set(sd.participant_id for sd in dataset)
    print(f"Unique participants: {len(participants)}")

    unique_sents = set((sd.text_id, sd.sentence_number) for sd in dataset)
    print(f"Unique sentences: {len(unique_sents)}")

    all_words = [w for sd in dataset for w in sd.words]
    print(f"Total word-observations: {len(all_words):,}")

    # Reading time stats
    ffds = [w.first_fixation_duration for w in all_words if not w.was_skipped and w.first_fixation_duration > 0]
    totals = [w.total_reading_time for w in all_words if not w.was_skipped and w.total_reading_time > 0]
    skips = [w.was_skipped for w in all_words]

    if ffds:
        print(f"\nReading time stats (non-skipped):")
        print(f"  FFD:  mean={sum(ffds)/len(ffds):.1f}ms, n={len(ffds):,}")
        print(f"  TRT:  mean={sum(totals)/len(totals):.1f}ms, n={len(totals):,}")
        print(f"  Skip rate: {sum(skips)/len(skips)*100:.1f}%")

    # Sentence lengths
    lengths = [len(sd) for sd in dataset]
    print(f"\nSentence lengths: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.1f}")

    # Aggregation test
    print("\nAggregating...")
    agg = aggregate_by_sentence(dataset, min_participants=5)
    print(f"Aggregated sentences (min 5 participants): {len(agg)}")

    if agg:
        print(f"\nSample aggregated sentence:")
        s = agg[0]
        print(f"  {s}")
        for i, tok in enumerate(s.tokens[:8]):
            print(f"    {tok:12s} TRT={s.mean_trt[i]:6.1f}ms  FFD={s.mean_ffd[i]:6.1f}ms  skip={s.skip_rate[i]:.2f}")

    # Split test
    train, val, test = split_geco(dataset)
    print(f"\nSplit: train={len(train):,}  val={len(val):,}  test={len(test):,}")
