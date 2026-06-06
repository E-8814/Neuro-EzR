"""
Loader for the Provo cloze-predictability norms.

The Provo predictability CSV has one row per (Text_ID, Word_Number,
participant response). To compute per-word predictability:

  predictability(word) = sum of Response_Proportion across rows where
                         Response (lowercased, stripped) matches Word.

This corresponds to the standard cloze-probability definition: the
fraction of participants whose guess matched the actual word.

Words with no matching response receive predictability = 0.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple


def _norm(s: str) -> str:
    """Normalize a word/response: lowercase, strip whitespace + outer punctuation."""
    if s is None:
        return ""
    s = s.lower().strip()
    return s.strip(".,;:!?\"'()[]{}").replace("’", "'")


def load_provo_predictability(csv_path: str | Path) -> Dict[Tuple[int, int], float]:
    """
    Returns a dict keyed by (Text_ID, Word_Number) -> cloze probability of
    the actual word given the preceding context.

    Both keys are int (matching the CSV columns).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Provo predictability CSV not found: {csv_path}")

    out: Dict[Tuple[int, int], float] = defaultdict(float)
    word_for_key: Dict[Tuple[int, int], str] = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                text_id = int(row["Text_ID"])
                word_num = int(row["Word_Number"])
            except (KeyError, ValueError):
                continue
            actual = _norm(row.get("Word", ""))
            response = _norm(row.get("Response", ""))
            try:
                rp = float(row.get("Response_Proportion", "0") or "0")
            except ValueError:
                rp = 0.0
            key = (text_id, word_num)
            word_for_key[key] = actual
            if actual and response and actual == response:
                out[key] += rp

    return dict(out)


def predictability_for_token(
    pred_dict: Dict[Tuple[int, int], float],
    text_id: int,
    word_number: int,
    default: float = 0.05,
) -> float:
    """Look up predictability for one (text_id, word_number); fallback to default."""
    return float(pred_dict.get((text_id, word_number), default))


# --------------------------------------------------------------------------- #
#  CLI sanity check
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/u384661/Neuro_EZR/data/Provo_Corpus-Predictability_Norms.csv"
    d = load_provo_predictability(path)
    print(f"Loaded {len(d)} (Text_ID, Word_Number) keys.")
    print("Distribution of predictability values:")
    vals = list(d.values())
    if vals:
        import statistics as _s
        print(f"  min={min(vals):.3f}  max={max(vals):.3f}  "
              f"mean={_s.mean(vals):.3f}  median={_s.median(vals):.3f}")
        # Show 5 examples
        print("Examples:")
        for k, v in list(d.items())[:5]:
            print(f"  text={k[0]:>3d} word_num={k[1]:>3d}  pred={v:.3f}")
