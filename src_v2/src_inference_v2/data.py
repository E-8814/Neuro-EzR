"""
Data loading for GECO and Provo corpora.

Thin wrapper around existing loaders + collation utilities for batching.
"""

import os
import sys
import csv

import torch
from torch.nn.utils.rnn import pad_sequence

# --- Path setup: import existing data loaders ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_V2_DIR = os.path.join(_THIS_DIR, '..')
_ROOT_DIR = os.path.join(_THIS_DIR, '..', '..')
_EZR_DIR = os.path.join(_ROOT_DIR, 'archive', 'original_ezreader')

# src_v2/ first so its data_loader.py (v2, 70/15/15 split) takes priority
sys.path.insert(0, _EZR_DIR)
sys.path.insert(0, _SRC_V2_DIR)

from data_loader import (
    WordData, SentenceData, AggregatedSentence,
    load_provo, aggregate_by_sentence, split_aggregated, split_dataset,
)
from geco_loader import load_geco, split_geco


def get_data_dir():
    """Resolve the data/ directory path."""
    return os.path.join(_ROOT_DIR, 'data')


# --------------------------------------------------------------------------- #
#  Collation functions for mini-batching
# --------------------------------------------------------------------------- #

def collate_sentences(batch, device):
    """Pad a list of SentenceData (per-participant) into batched tensors."""
    word_lists = [sd.tokens for sd in batch]
    pred_vals = pad_sequence(
        [torch.tensor([w.predictability for w in sd.words], dtype=torch.float32)
         for sd in batch],
        batch_first=True,
    ).to(device)
    wlens = pad_sequence(
        [torch.tensor([len(t) for t in sd.tokens], dtype=torch.float32)
         for sd in batch],
        batch_first=True,
    ).to(device)
    h_trt = pad_sequence(
        [torch.tensor(sd.total_reading_times, dtype=torch.float32)
         for sd in batch],
        batch_first=True,
    ).to(device)
    h_ffd = pad_sequence(
        [torch.tensor(sd.first_fixation_durations, dtype=torch.float32)
         for sd in batch],
        batch_first=True,
    ).to(device)
    h_gaze = pad_sequence(
        [torch.tensor(sd.gaze_durations, dtype=torch.float32)
         for sd in batch],
        batch_first=True,
    ).to(device)
    h_skip = pad_sequence(
        [torch.tensor([1.0 if s else 0.0 for s in sd.skip_flags],
                       dtype=torch.float32)
         for sd in batch],
        batch_first=True,
    ).to(device)
    return word_lists, pred_vals, wlens, h_trt, h_ffd, h_gaze, h_skip


def collate_aggregated(batch, device):
    """Pad a list of AggregatedSentence (averaged) into batched tensors."""
    word_lists = [a.tokens for a in batch]
    pred_vals = pad_sequence(
        [torch.tensor(a.predictabilities, dtype=torch.float32) for a in batch],
        batch_first=True,
    ).to(device)
    wlens = pad_sequence(
        [torch.tensor([len(t) for t in a.tokens], dtype=torch.float32)
         for a in batch],
        batch_first=True,
    ).to(device)
    h_trt = pad_sequence(
        [torch.tensor(a.mean_trt, dtype=torch.float32) for a in batch],
        batch_first=True,
    ).to(device)
    h_ffd = pad_sequence(
        [torch.tensor(a.mean_ffd, dtype=torch.float32) for a in batch],
        batch_first=True,
    ).to(device)
    h_gaze = pad_sequence(
        [torch.tensor(a.mean_gaze, dtype=torch.float32) for a in batch],
        batch_first=True,
    ).to(device)
    h_skip = pad_sequence(
        [torch.tensor(a.skip_rate, dtype=torch.float32) for a in batch],
        batch_first=True,
    ).to(device)
    return word_lists, pred_vals, wlens, h_trt, h_ffd, h_gaze, h_skip


# --------------------------------------------------------------------------- #
#  SUBTLEXus frequency (for original EZ Reader comparison)
# --------------------------------------------------------------------------- #

def load_subtlexus(path=None):
    """Load SUBTLEXus frequency database."""
    if path is None:
        path = os.path.join(get_data_dir(), 'SUBTLEXus.txt')
    freq = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            freq[row['Word'].lower()] = int(row['FREQcount'])
    return freq


def get_frequency(word, subtlex):
    """Look up a word's frequency in SUBTLEXus."""
    w = word.lower().strip(".,;:!?\"'()[]{}").replace("\u2019", "'")
    return subtlex.get(w, 1)
