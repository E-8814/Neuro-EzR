"""
Scanpath loader for Provo and GECO corpora.

Extracts per-participant fixation sequences (scanpaths) from raw eye-tracking
CSV data. Each scanpath is an ordered sequence of fixations: which word was
fixated, for how long, and in what order.

Fixation order is reconstructed from IA_FIRST_FIXATION_INDEX (Provo) or
WORD_FIRST_FIXATION_INDEX (GECO), which record the ordinal fixation number
within the trial.
"""

import csv
import os
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


# --------------------------------------------------------------------------- #
#  Data structures
# --------------------------------------------------------------------------- #

@dataclass
class Fixation:
    """Single fixation event in a scanpath."""
    word_index: int      # 0-based position within the sentence
    duration: float      # first fixation duration on this word (ms)
    timestamp: float     # for ordering (IA_FIRST_FIXATION_TIME or index)


@dataclass
class ScanpathData:
    """Ordered fixation sequence for one participant reading one sentence."""
    text_id: int
    sentence_number: int
    participant_id: str
    tokens: List[str]
    word_lengths: List[int]
    predictabilities: List[float]
    fixations: List[Fixation]       # ordered by time
    # Per-word ground truth (for evaluation)
    word_ffd: List[float]           # first fixation duration per word (0 if skipped)
    word_gaze: List[float]          # gaze duration per word
    word_trt: List[float]           # total reading time per word
    word_skipped: List[bool]        # skip flag per word


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _safe_float(val, default=0.0):
    if val is None:
        return default
    val = str(val).strip()
    if val in (".", "NA", "", "nan", "NaN"):
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _safe_int(val, default=0):
    if val is None:
        return default
    val = str(val).strip()
    if val in (".", "NA", "", "nan", "NaN"):
        return default
    try:
        return int(float(val))
    except ValueError:
        return default


# --------------------------------------------------------------------------- #
#  Provo scanpath loader
# --------------------------------------------------------------------------- #

def load_provo_scanpaths(
    eyetracking_path: str,
    min_sentence_length: int = 3,
    max_sentence_length: int = 50,
    min_fixations: int = 2,
) -> List[ScanpathData]:
    """
    Load scanpaths from the Provo Corpus.

    For each (participant, sentence), extracts the ordered sequence of
    first-pass fixations by sorting non-skipped words by their
    IA_FIRST_FIXATION_INDEX.

    Returns list of ScanpathData sorted by (text_id, sentence_number, participant_id).
    """
    # Bucket rows by (participant, text, sentence)
    buckets: Dict[Tuple[str, int, int], List[dict]] = defaultdict(list)

    with open(eyetracking_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["Participant_ID"]
            text_id = _safe_int(row["Text_ID"])
            sent_num = _safe_int(row["Sentence_Number"])
            buckets[(pid, text_id, sent_num)].append(row)

    dataset: List[ScanpathData] = []

    for (pid, text_id, sent_num), rows in buckets.items():
        # Sort by word position in sentence
        rows.sort(key=lambda r: _safe_int(r["Word_In_Sentence_Number"]))

        n_words = len(rows)
        if n_words < min_sentence_length or n_words > max_sentence_length:
            continue

        # Extract per-word data
        tokens = []
        word_lengths = []
        predictabilities = []
        word_ffd = []
        word_gaze = []
        word_trt = []
        word_skipped = []
        fixations = []

        for word_pos, r in enumerate(rows):
            word_text = (r.get("Word_Cleaned") or r["Word"]).strip()
            tokens.append(word_text)
            word_lengths.append(len(word_text))
            predictabilities.append(_safe_float(r.get("OrthographicMatch"), 0.0))

            ffd = _safe_float(r.get("IA_FIRST_FIXATION_DURATION"), 0.0)
            gaze = _safe_float(r.get("IA_FIRST_RUN_DWELL_TIME"), 0.0)
            trt = _safe_float(r.get("IA_DWELL_TIME"), 0.0)
            skipped = _safe_int(r.get("IA_SKIP"), 0) == 1

            word_ffd.append(ffd)
            word_gaze.append(gaze)
            word_trt.append(trt)
            word_skipped.append(skipped)

            # Build fixation if word was not skipped and has valid data
            if not skipped and ffd > 0:
                fix_index = _safe_float(r.get("IA_FIRST_FIXATION_INDEX"), -1)
                fix_time = _safe_float(r.get("IA_FIRST_FIXATION_TIME"), -1)
                # Use fixation time if available, otherwise index
                timestamp = fix_time if fix_time > 0 else fix_index
                if timestamp > 0:
                    fixations.append(Fixation(
                        word_index=word_pos,
                        duration=ffd,
                        timestamp=timestamp,
                    ))

        # Sort fixations by timestamp to get actual reading order
        fixations.sort(key=lambda f: f.timestamp)

        if len(fixations) < min_fixations:
            continue

        dataset.append(ScanpathData(
            text_id=text_id,
            sentence_number=sent_num,
            participant_id=pid,
            tokens=tokens,
            word_lengths=word_lengths,
            predictabilities=predictabilities,
            fixations=fixations,
            word_ffd=word_ffd,
            word_gaze=word_gaze,
            word_trt=word_trt,
            word_skipped=word_skipped,
        ))

    dataset.sort(key=lambda s: (s.text_id, s.sentence_number, s.participant_id))
    return dataset


# --------------------------------------------------------------------------- #
#  GECO scanpath loader
# --------------------------------------------------------------------------- #

def load_geco_scanpaths(
    reading_data_path: str,
    material_path: str,
    predictability_path: Optional[str] = None,
    min_sentence_length: int = 3,
    max_sentence_length: int = 50,
    min_fixations: int = 2,
) -> List[ScanpathData]:
    """
    Load scanpaths from the GECO Corpus.

    GECO has WORD_FIRST_FIXATION_INDEX and WORD_FIRST_FIXATION_TIME columns
    that allow reconstruction of fixation order.
    """
    # 1. Load material for sentence structure
    from collections import defaultdict as dd
    word_to_sentence = {}
    sentence_words = dd(list)

    with open(material_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word_id = row["WORD_ID"]
            sent_id = row["SENTENCE_ID"]
            word_to_sentence[word_id] = sent_id
            sentence_words[sent_id].append((word_id, row["WORD"]))

    # 2. Encode sentence IDs
    sorted_sent_ids = sorted(sentence_words.keys(), key=lambda s: (
        int(str(s).split("-")[0]) if "-" in str(s) else 9999,
        int(str(s).split("-")[1]) if "-" in str(s) else 0,
    ))
    sent_id_to_numeric = {}
    for i, sid in enumerate(sorted_sent_ids):
        parts = str(sid).split("-")
        sent_num = int(parts[1]) if len(parts) == 2 else 0
        sent_id_to_numeric[sid] = (1001 + i, sent_num)

    # 3. Load predictability
    word_predictability = {}
    if predictability_path and os.path.exists(predictability_path):
        with open(predictability_path, "rb") as f:
            pred_data = pickle.load(f)
        for sent_id, words_list in sentence_words.items():
            if sent_id in pred_data:
                preds = pred_data[sent_id]['predictability']
                for i, (word_id, _) in enumerate(words_list):
                    if i < len(preds):
                        word_predictability[word_id] = preds[i]

    # 4. Read eye-tracking data, bucket by (participant, sentence)
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

    # 5. Build scanpaths
    dataset: List[ScanpathData] = []

    for (pid, sent_id), rows in buckets.items():
        canonical = sentence_words.get(sent_id, [])
        canonical_ids = [wid for wid, _ in canonical]

        row_by_id = {r["WORD_ID"]: r for r in rows}

        n_words = len(canonical)
        if n_words < min_sentence_length or n_words > max_sentence_length:
            continue

        tokens = []
        word_lengths_list = []
        predictabilities = []
        word_ffd = []
        word_gaze = []
        word_trt = []
        word_skipped = []
        fixations = []

        for word_pos, (word_id, word_text) in enumerate(canonical):
            word_text = word_text.strip()
            tokens.append(word_text)
            word_lengths_list.append(len(word_text))
            predictabilities.append(word_predictability.get(word_id, 0.0))

            r = row_by_id.get(word_id)
            if r is None:
                word_ffd.append(0.0)
                word_gaze.append(0.0)
                word_trt.append(0.0)
                word_skipped.append(True)
                continue

            ffd = _safe_float(r.get("WORD_FIRST_FIXATION_DURATION"), 0.0)
            gaze = _safe_float(r.get("WORD_GAZE_DURATION"), 0.0)
            trt = _safe_float(r.get("WORD_TOTAL_READING_TIME"), 0.0)
            skipped = _safe_int(r.get("WORD_SKIP"), 0) == 1

            word_ffd.append(ffd)
            word_gaze.append(gaze)
            word_trt.append(trt)
            word_skipped.append(skipped)

            if not skipped and ffd > 0:
                fix_index = _safe_float(r.get("WORD_FIRST_FIXATION_INDEX"), -1)
                fix_time = _safe_float(r.get("WORD_FIRST_FIXATION_TIME"), -1)
                timestamp = fix_time if fix_time > 0 else fix_index
                if timestamp > 0:
                    fixations.append(Fixation(
                        word_index=word_pos,
                        duration=ffd,
                        timestamp=timestamp,
                    ))

        fixations.sort(key=lambda f: f.timestamp)

        if len(fixations) < min_fixations:
            continue

        text_id, sent_num = sent_id_to_numeric.get(sent_id, (9999, 0))

        dataset.append(ScanpathData(
            text_id=text_id,
            sentence_number=sent_num,
            participant_id=pid,
            tokens=tokens,
            word_lengths=word_lengths_list,
            predictabilities=predictabilities,
            fixations=fixations,
            word_ffd=word_ffd,
            word_gaze=word_gaze,
            word_trt=word_trt,
            word_skipped=word_skipped,
        ))

    dataset.sort(key=lambda s: (s.text_id, s.sentence_number, s.participant_id))
    return dataset


# --------------------------------------------------------------------------- #
#  Split by text_id
# --------------------------------------------------------------------------- #

def split_scanpaths(
    dataset: List[ScanpathData],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[ScanpathData], List[ScanpathData], List[ScanpathData]]:
    """Split scanpaths by text_id to avoid data leakage."""
    rng = random.Random(seed)
    text_ids = sorted(set(sp.text_id for sp in dataset))
    rng.shuffle(text_ids)

    n_train = int(len(text_ids) * train_ratio)
    n_val = int(len(text_ids) * val_ratio)

    train_texts = set(text_ids[:n_train])
    val_texts = set(text_ids[n_train:n_train + n_val])

    train = [sp for sp in dataset if sp.text_id in train_texts]
    val = [sp for sp in dataset if sp.text_id in val_texts]
    test = [sp for sp in dataset if sp.text_id not in train_texts and sp.text_id not in val_texts]

    return train, val, test


# --------------------------------------------------------------------------- #
#  Aggregate scanpaths to per-word means (for evaluation comparison with v2)
# --------------------------------------------------------------------------- #

def aggregate_scanpaths(dataset: List[ScanpathData], min_participants: int = 5):
    """
    Aggregate per-participant scanpaths into per-sentence word-level means.
    Returns list of dicts with mean_ffd, mean_gaze, mean_trt, skip_rate per word.
    """
    groups = defaultdict(list)
    for sp in dataset:
        groups[(sp.text_id, sp.sentence_number)].append(sp)

    aggregated = []
    for (text_id, sent_num), scanpaths in groups.items():
        if len(scanpaths) < min_participants:
            continue

        ref = scanpaths[0]
        n_words = len(ref.tokens)

        # Check consistency
        scanpaths = [sp for sp in scanpaths if len(sp.tokens) == n_words]
        if len(scanpaths) < min_participants:
            continue

        ffd_sums = [0.0] * n_words
        gaze_sums = [0.0] * n_words
        trt_sums = [0.0] * n_words
        ffd_counts = [0] * n_words
        gaze_counts = [0] * n_words
        trt_counts = [0] * n_words
        skip_counts = [0] * n_words

        for sp in scanpaths:
            for i in range(n_words):
                if sp.word_skipped[i]:
                    skip_counts[i] += 1
                else:
                    if sp.word_ffd[i] > 0:
                        ffd_sums[i] += sp.word_ffd[i]
                        ffd_counts[i] += 1
                    if sp.word_gaze[i] > 0:
                        gaze_sums[i] += sp.word_gaze[i]
                        gaze_counts[i] += 1
                    if sp.word_trt[i] > 0:
                        trt_sums[i] += sp.word_trt[i]
                        trt_counts[i] += 1

        n_p = len(scanpaths)
        aggregated.append({
            'text_id': text_id,
            'sentence_number': sent_num,
            'tokens': ref.tokens,
            'word_lengths': ref.word_lengths,
            'predictabilities': ref.predictabilities,
            'mean_ffd': [ffd_sums[i] / ffd_counts[i] if ffd_counts[i] > 0 else 0.0 for i in range(n_words)],
            'mean_gaze': [gaze_sums[i] / gaze_counts[i] if gaze_counts[i] > 0 else 0.0 for i in range(n_words)],
            'mean_trt': [trt_sums[i] / trt_counts[i] if trt_counts[i] > 0 else 0.0 for i in range(n_words)],
            'skip_rate': [skip_counts[i] / n_p for i in range(n_words)],
            'n_participants': n_p,
        })

    aggregated.sort(key=lambda a: (a['text_id'], a['sentence_number']))
    return aggregated


# --------------------------------------------------------------------------- #
#  CLI: quick stats
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import sys

    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    ET_PATH = os.path.join(DATA_DIR, "Provo_Corpus-Eyetracking_Data.csv")

    if not os.path.exists(ET_PATH):
        print(f"ERROR: {ET_PATH} not found.")
        sys.exit(1)

    print("Loading Provo scanpaths...")
    dataset = load_provo_scanpaths(ET_PATH)

    print(f"\n=== Provo Scanpath Stats ===")
    print(f"Total scanpaths: {len(dataset):,}")

    participants = set(sp.participant_id for sp in dataset)
    print(f"Unique participants: {len(participants)}")

    unique_sents = set((sp.text_id, sp.sentence_number) for sp in dataset)
    print(f"Unique sentences: {len(unique_sents)}")

    # Fixation stats
    fix_counts = [len(sp.fixations) for sp in dataset]
    print(f"\nFixations per scanpath: min={min(fix_counts)}, max={max(fix_counts)}, "
          f"mean={sum(fix_counts)/len(fix_counts):.1f}")

    # Sentence lengths
    sent_lens = [len(sp.tokens) for sp in dataset]
    print(f"Words per sentence: min={min(sent_lens)}, max={max(sent_lens)}, "
          f"mean={sum(sent_lens)/len(sent_lens):.1f}")

    # Fixation durations
    all_durs = [f.duration for sp in dataset for f in sp.fixations]
    print(f"Fixation duration: mean={sum(all_durs)/len(all_durs):.1f}ms, n={len(all_durs):,}")

    # Skip rate
    all_skips = [s for sp in dataset for s in sp.word_skipped]
    print(f"Skip rate: {sum(all_skips)/len(all_skips)*100:.1f}%")

    # Saccade length distribution
    saccade_lens = []
    for sp in dataset:
        for i in range(len(sp.fixations) - 1):
            sac = sp.fixations[i + 1].word_index - sp.fixations[i].word_index
            saccade_lens.append(sac)

    if saccade_lens:
        forward = [s for s in saccade_lens if s > 0]
        backward = [s for s in saccade_lens if s < 0]
        print(f"\nSaccade stats:")
        print(f"  Forward:    {len(forward):,} ({len(forward)/len(saccade_lens)*100:.1f}%), "
              f"mean length={sum(forward)/len(forward):.1f}" if forward else "  Forward: 0")
        print(f"  Backward:   {len(backward):,} ({len(backward)/len(saccade_lens)*100:.1f}%), "
              f"mean length={sum(abs(s) for s in backward)/len(backward):.1f}" if backward else "  Backward: 0")
        print(f"  Refixation: {sum(1 for s in saccade_lens if s == 0)}")

    # Split
    train, val, test = split_scanpaths(dataset)
    print(f"\nSplit: train={len(train):,}  val={len(val):,}  test={len(test):,}")

    # Sample scanpath
    print(f"\n=== Sample Scanpath ===")
    sp = dataset[0]
    print(f"  Participant: {sp.participant_id}")
    print(f"  Sentence: {' '.join(sp.tokens[:10])}{'...' if len(sp.tokens) > 10 else ''}")
    print(f"  Fixations ({len(sp.fixations)}):")
    for i, fix in enumerate(sp.fixations[:10]):
        word = sp.tokens[fix.word_index] if fix.word_index < len(sp.tokens) else "???"
        print(f"    {i+1}. word[{fix.word_index}] '{word}' -> {fix.duration:.0f}ms")
