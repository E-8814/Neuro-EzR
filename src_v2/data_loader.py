"""
Data loader for the Provo Corpus (v2).

Only change from v1: split ratios are 70/15/15 instead of 80/10/10,
giving ~23 val sentences (~400 words) vs the previous 13 (249 words).
"""

import csv
import os
import math
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


# --------------------------------------------------------------------------- #
#  Core data structures
# --------------------------------------------------------------------------- #

class WordData:
    """Eye-tracking data for a single word from a single participant."""

    __slots__ = (
        "word", "word_length", "predictability",
        "first_fixation_duration", "gaze_duration", "total_reading_time",
        "was_skipped", "regression_in",
        "word_number", "sentence_number", "word_in_sentence",
        "text_id", "participant_id",
    )

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return (
            f"WordData('{self.word}', FFD={self.first_fixation_duration}, "
            f"gaze={self.gaze_duration}, total={self.total_reading_time}, "
            f"skip={self.was_skipped})"
        )


class SentenceData:
    """All word-level observations for one sentence from one participant."""

    def __init__(self, text_id: int, sentence_number: int,
                 participant_id: str, words: List[WordData]):
        self.text_id = text_id
        self.sentence_number = sentence_number
        self.participant_id = participant_id
        self.words = words  # ordered by word position

    @property
    def tokens(self) -> List[str]:
        return [w.word for w in self.words]

    @property
    def total_reading_times(self) -> List[float]:
        return [w.total_reading_time for w in self.words]

    @property
    def first_fixation_durations(self) -> List[float]:
        return [w.first_fixation_duration for w in self.words]

    @property
    def gaze_durations(self) -> List[float]:
        return [w.gaze_duration for w in self.words]

    @property
    def skip_flags(self) -> List[bool]:
        return [w.was_skipped for w in self.words]

    def __repr__(self):
        text = " ".join(self.tokens)
        if len(text) > 60:
            text = text[:60] + "..."
        return f"SentenceData(text={self.text_id}, sent={self.sentence_number}, " \
               f"subj={self.participant_id}, n_words={len(self.words)}, '{text}')"

    def __len__(self):
        return len(self.words)


# --------------------------------------------------------------------------- #
#  Helper: safe numeric parsing
# --------------------------------------------------------------------------- #

def _safe_float(val: str, default: float = 0.0) -> float:
    """Parse a numeric string, returning *default* for NA / '.' / empty."""
    if not val or val.strip() in ("NA", ".", ""):
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _safe_int(val: str, default: int = 0) -> int:
    if not val or val.strip() in ("NA", ".", ""):
        return default
    try:
        return int(val)
    except ValueError:
        return default


# --------------------------------------------------------------------------- #
#  Main loader
# --------------------------------------------------------------------------- #

def load_provo(
    eyetracking_path: str,
    min_sentence_length: int = 2,
    max_sentence_length: int = 50,
    participants: Optional[List[str]] = None,
) -> List[SentenceData]:
    """
    Load the Provo Corpus eye-tracking CSV and return a list of
    SentenceData objects (one per participant x sentence combination).
    """

    # ---- First pass: bucket rows by (participant, text, sentence) ----
    buckets: Dict[Tuple[str, int, int], List[dict]] = defaultdict(list)

    with open(eyetracking_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["Participant_ID"]
            if participants and pid not in participants:
                continue

            text_id = _safe_int(row["Text_ID"])
            sent_num = _safe_int(row["Sentence_Number"])

            buckets[(pid, text_id, sent_num)].append(row)

    # ---- Second pass: convert each bucket into a SentenceData ----
    dataset: List[SentenceData] = []

    for (pid, text_id, sent_num), rows in buckets.items():
        # Sort by word position within sentence
        rows.sort(key=lambda r: _safe_int(r["Word_In_Sentence_Number"]))

        words: List[WordData] = []
        for r in rows:
            ffd = _safe_float(r["IA_FIRST_FIXATION_DURATION"], default=0.0)
            gaze = _safe_float(r["IA_FIRST_RUN_DWELL_TIME"], default=0.0)
            total = _safe_float(r["IA_DWELL_TIME"], default=0.0)
            skipped = _safe_int(r["IA_SKIP"], default=0) == 1
            reg_in = _safe_int(r["IA_REGRESSION_IN"], default=0) == 1
            pred = _safe_float(r["OrthographicMatch"], default=0.0)
            word_text = r["Word_Cleaned"] if r.get("Word_Cleaned") else r["Word"]

            wd = WordData(
                word=word_text.strip(),
                word_length=len(word_text.strip()),
                predictability=pred,
                first_fixation_duration=ffd,
                gaze_duration=gaze,
                total_reading_time=total,
                was_skipped=skipped,
                regression_in=reg_in,
                word_number=_safe_int(r["Word_Number"]),
                sentence_number=sent_num,
                word_in_sentence=_safe_int(r["Word_In_Sentence_Number"]),
                text_id=text_id,
                participant_id=pid,
            )
            words.append(wd)

        # Filter by sentence length
        if len(words) < min_sentence_length or len(words) > max_sentence_length:
            continue

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


def get_unique_sentences(dataset: List[SentenceData]) -> Dict[Tuple[int, int], List[str]]:
    """
    Return a dict mapping (text_id, sentence_number) -> list of word tokens.
    """
    sentences = {}
    for sd in dataset:
        key = (sd.text_id, sd.sentence_number)
        if key not in sentences:
            sentences[key] = sd.tokens
    return sentences


def compute_word_frequency_from_corpus(dataset: List[SentenceData]) -> Dict[str, int]:
    """
    Compute a simple word frequency table from the corpus tokens.
    """
    freq = defaultdict(int)
    seen = set()
    for sd in dataset:
        key = (sd.text_id, sd.sentence_number)
        if key in seen:
            continue
        seen.add(key)
        for w in sd.words:
            freq[w.word.lower()] += 1
    return dict(freq)


# --------------------------------------------------------------------------- #
#  Aggregated sentence data (averaged across participants)
# --------------------------------------------------------------------------- #

class AggregatedSentence:
    """
    Per-sentence data averaged across ALL participants.
    """

    def __init__(self, text_id: int, sentence_number: int,
                 tokens: List[str],
                 predictabilities: List[float],
                 mean_trt: List[float],
                 mean_ffd: List[float],
                 mean_gaze: List[float],
                 skip_rate: List[float],
                 n_participants: int):
        self.text_id = text_id
        self.sentence_number = sentence_number
        self.tokens = tokens
        self.predictabilities = predictabilities
        self.mean_trt = mean_trt
        self.mean_ffd = mean_ffd
        self.mean_gaze = mean_gaze
        self.skip_rate = skip_rate
        self.n_participants = n_participants

    def __repr__(self):
        text = " ".join(self.tokens)
        if len(text) > 60:
            text = text[:60] + "..."
        return (f"AggregatedSentence(text={self.text_id}, sent={self.sentence_number}, "
                f"n_words={len(self.tokens)}, n_subj={self.n_participants}, '{text}')")

    def __len__(self):
        return len(self.tokens)


def aggregate_by_sentence(
    dataset: List[SentenceData],
    min_participants: int = 10,
) -> List[AggregatedSentence]:
    """
    Aggregate per-participant SentenceData into per-sentence averages.
    """
    # Group by (text_id, sentence_number)
    groups: Dict[Tuple[int, int], List[SentenceData]] = defaultdict(list)
    for sd in dataset:
        groups[(sd.text_id, sd.sentence_number)].append(sd)

    aggregated: List[AggregatedSentence] = []

    for (text_id, sent_num), observations in groups.items():
        if len(observations) < min_participants:
            continue

        ref = observations[0]
        n_words = len(ref.words)

        observations = [obs for obs in observations if len(obs.words) == n_words]
        if len(observations) < min_participants:
            continue

        tokens = ref.tokens
        predictabilities = [w.predictability for w in ref.words]

        trt_sums = [0.0] * n_words
        ffd_sums = [0.0] * n_words
        gaze_sums = [0.0] * n_words
        skip_counts = [0] * n_words
        trt_counts = [0] * n_words
        ffd_counts = [0] * n_words
        gaze_counts = [0] * n_words

        for obs in observations:
            for i, w in enumerate(obs.words):
                if w.was_skipped:
                    skip_counts[i] += 1
                else:
                    if w.total_reading_time > 0:
                        trt_sums[i] += w.total_reading_time
                        trt_counts[i] += 1
                    if w.first_fixation_duration > 0:
                        ffd_sums[i] += w.first_fixation_duration
                        ffd_counts[i] += 1
                    if w.gaze_duration > 0:
                        gaze_sums[i] += w.gaze_duration
                        gaze_counts[i] += 1

        n_participants = len(observations)

        mean_trt = [trt_sums[i] / trt_counts[i] if trt_counts[i] > 0 else 0.0
                    for i in range(n_words)]
        mean_ffd = [ffd_sums[i] / ffd_counts[i] if ffd_counts[i] > 0 else 0.0
                    for i in range(n_words)]
        mean_gaze = [gaze_sums[i] / gaze_counts[i] if gaze_counts[i] > 0 else 0.0
                     for i in range(n_words)]
        skip_rate = [skip_counts[i] / n_participants for i in range(n_words)]

        agg = AggregatedSentence(
            text_id=text_id,
            sentence_number=sent_num,
            tokens=tokens,
            predictabilities=predictabilities,
            mean_trt=mean_trt,
            mean_ffd=mean_ffd,
            mean_gaze=mean_gaze,
            skip_rate=skip_rate,
            n_participants=n_participants,
        )
        aggregated.append(agg)

    aggregated.sort(key=lambda a: (a.text_id, a.sentence_number))
    return aggregated


def split_aggregated(
    aggregated: List[AggregatedSentence],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[AggregatedSentence], List[AggregatedSentence], List[AggregatedSentence]]:
    """Split aggregated sentences by TEXT to avoid data leakage. v2: 70/15/15."""
    import random
    rng = random.Random(seed)
    text_ids = sorted(set(a.text_id for a in aggregated))
    rng.shuffle(text_ids)
    n_train = int(len(text_ids) * train_ratio)
    n_val = int(len(text_ids) * val_ratio)
    train_texts = set(text_ids[:n_train])
    val_texts = set(text_ids[n_train:n_train + n_val])
    train = [a for a in aggregated if a.text_id in train_texts]
    val = [a for a in aggregated if a.text_id in val_texts]
    test = [a for a in aggregated if a.text_id not in train_texts and a.text_id not in val_texts]
    return train, val, test


def split_dataset(
    dataset: List[SentenceData],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[SentenceData], List[SentenceData], List[SentenceData]]:
    """
    Split by TEXT (not by sentence or participant) to avoid data leakage.
    v2: 70/15/15 instead of 80/10/10.
    """
    import random
    rng = random.Random(seed)

    text_ids = sorted(set(sd.text_id for sd in dataset))
    rng.shuffle(text_ids)

    n_train = int(len(text_ids) * train_ratio)
    n_val = int(len(text_ids) * val_ratio)

    train_texts = set(text_ids[:n_train])
    val_texts = set(text_ids[n_train:n_train + n_val])
    test_texts = set(text_ids[n_train + n_val:])

    train = [sd for sd in dataset if sd.text_id in train_texts]
    val = [sd for sd in dataset if sd.text_id in val_texts]
    test = [sd for sd in dataset if sd.text_id in test_texts]

    return train, val, test


# --------------------------------------------------------------------------- #
#  CLI: quick stats when run directly
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import sys

    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    ET_PATH = os.path.join(DATA_DIR, "Provo_Corpus-Eyetracking_Data.csv")

    if not os.path.exists(ET_PATH):
        print(f"ERROR: {ET_PATH} not found. Download the Provo Corpus first.")
        sys.exit(1)

    print("Loading Provo Corpus...")
    dataset = load_provo(ET_PATH)

    print(f"\n=== Provo Corpus Stats ===")
    print(f"Total sentence-observations: {len(dataset):,}")

    participants = set(sd.participant_id for sd in dataset)
    print(f"Unique participants: {len(participants)}")

    unique_sents = get_unique_sentences(dataset)
    print(f"Unique sentences: {len(unique_sents)}")

    all_words = [w for sd in dataset for w in sd.words]
    print(f"Total word-observations: {len(all_words):,}")

    # Reading time stats (exclude skipped words)
    ffds = [w.first_fixation_duration for w in all_words if not w.was_skipped and w.first_fixation_duration > 0]
    totals = [w.total_reading_time for w in all_words if not w.was_skipped and w.total_reading_time > 0]
    skips = [w.was_skipped for w in all_words]

    print(f"\nReading time stats (non-skipped words):")
    print(f"  First Fixation Duration: mean={sum(ffds)/len(ffds):.1f}ms, n={len(ffds):,}")
    print(f"  Total Reading Time:      mean={sum(totals)/len(totals):.1f}ms, n={len(totals):,}")
    print(f"  Skip rate:               {sum(skips)/len(skips)*100:.1f}%")

    # Sentence length distribution
    lengths = [len(sd) for sd in dataset]
    print(f"\nSentence lengths: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.1f}")

    # Train/val/test split (v2: 70/15/15)
    train, val, test = split_dataset(dataset)
    print(f"\nSplits (by text, 70/15/15): train={len(train):,}, val={len(val):,}, test={len(test):,}")

    # Show a few example sentences
    print(f"\n=== Sample Sentences ===")
    for sd in dataset[:3]:
        print(sd)
        for w in sd.words[:5]:
            print(f"  {w}")
        if len(sd.words) > 5:
            print(f"  ... ({len(sd.words) - 5} more words)")
        print()
