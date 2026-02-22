"""
Build real word features for a fair EZ Reader comparison.

1. SUBTLEXus word frequencies (74K words from movie subtitles)
2. Cloze predictability from Provo norms (proportion who guessed the exact word)

Outputs a lookup dict: (text_id, word_number) -> {frequency, predictability}
"""

import os
import sys
import csv
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ez_reader'))
from data_loader import load_provo, aggregate_by_sentence


def load_subtlexus(path):
    """Load SUBTLEXus into a dict: word_lower -> frequency_count."""
    freq = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            word = row['Word'].lower()
            count = int(row['FREQcount'])
            freq[word] = count
    return freq


def load_provo_predictability(path):
    """
    Load cloze predictability from Provo norms.

    The norms file has multiple responses per word. For each word,
    predictability = proportion of respondents who guessed the EXACT word.

    Returns: dict of (text_id, word_number) -> cloze_predictability (0-1)
    """
    # The eye-tracking data already has OrthographicMatch which IS the cloze predictability
    # But let's compute it directly from the norms as a cross-check
    #
    # Each row in the norms file is one response.
    # We need: for each word position, what proportion guessed correctly?
    # The "correct" response is the actual word in the text.

    # First, find the actual word at each position
    word_at_position = {}  # (text_id, word_number) -> actual_word
    all_responses = defaultdict(list)  # (text_id, word_number) -> [(response, count, total)]

    with open(path, 'r', encoding='latin-1') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text_id = int(row['Text_ID'])
            word_num = int(row['Word_Number'])
            word = row['Word'].strip()
            response = row['Response'].strip()
            count = int(row['Response_Count'])
            total = int(row['Total_Response_Count'])
            proportion = float(row['Response_Proportion'])

            key = (text_id, word_num)
            word_at_position[key] = word
            all_responses[key].append((response, count, total, proportion))

    # For each word position, find the proportion who guessed the actual word
    predictabilities = {}
    for key, word in word_at_position.items():
        responses = all_responses[key]
        total = responses[0][2] if responses else 40  # typically 40 respondents

        # Find if anyone guessed the exact word
        correct_count = 0
        for resp, count, _, prop in responses:
            if resp.lower() == word.lower():
                correct_count = count
                break

        predictabilities[key] = correct_count / total

    return predictabilities


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    # Load SUBTLEXus
    subtlex_path = os.path.join(data_dir, 'SUBTLEXus.txt')
    print("Loading SUBTLEXus...")
    subtlex = load_subtlexus(subtlex_path)
    print(f"  {len(subtlex):,} words loaded")
    print(f"  'the' = {subtlex.get('the', 'N/A'):,}")
    print(f"  'rumblings' = {subtlex.get('rumblings', 'N/A')}")
    print(f"  'microphone' = {subtlex.get('microphone', 'N/A')}")

    # Load Provo predictability
    norms_path = os.path.join(data_dir, 'Provo_Corpus-Predictability_Norms.csv')
    print("\nLoading Provo cloze predictability...")
    pred = load_provo_predictability(norms_path)
    print(f"  {len(pred):,} word positions loaded")

    # Show some examples
    print("\n  Sample predictabilities:")
    et_path = os.path.join(data_dir, 'Provo_Corpus-Eyetracking_Data.csv')
    raw = load_provo(et_path)
    agg = aggregate_by_sentence(raw, min_participants=10)

    for s in agg[:2]:
        print(f"\n  Sentence: \"{' '.join(s.tokens[:8])}...\"")
        for i, tok in enumerate(s.tokens[:8]):
            # word_number in the data is 1-indexed and refers to position in the text
            # We need to map from our sentence to the norms
            # The eye-tracking data has Word_Number which maps to norms
            word_key = (s.text_id, i + 2)  # approximate - word_number offset
            cloze = pred.get(word_key, -1)

            # Get frequency
            freq = subtlex.get(tok.lower(), 0)
            old_pred = s.predictabilities[i]

            print(f"    {tok:<14s} freq={freq:>8,}  "
                  f"cloze_pred={cloze:.3f}  ortho_match={old_pred:.3f}")

    # Check coverage
    print("\n\nCoverage check:")
    all_words = set()
    missing = set()
    for s in agg:
        for tok in s.tokens:
            w = tok.lower()
            all_words.add(w)
            if w not in subtlex:
                missing.add(w)

    print(f"  Unique words in corpus: {len(all_words)}")
    print(f"  Found in SUBTLEXus: {len(all_words) - len(missing)}")
    print(f"  Missing: {len(missing)}")
    if missing:
        print(f"  Examples missing: {list(missing)[:10]}")


if __name__ == "__main__":
    main()
