"""
Baseline 4: Fine-tuned BERT + Direct Regression Heads.

This is the most important baseline — the standard approach in NLP for
word-level prediction tasks (Hollenstein et al. 2021, Wiechmann & Qiao 2022,
CMCL shared tasks). Directly fine-tunes BERT to predict reading times from
contextual word representations, WITHOUT any cognitive model (no E-Z Reader).

Architecture:
    word tokens -> BERT (subword) -> first-subword pooling -> word representations
    -> 4 independent regression heads -> (FFD, Gaze, TRT, Skip)

This is a direct comparison to the Neural EZ Reader (model_bert.py), which
uses the same BERT backbone but routes predictions through the DifferentiableEZReader.

Trained on GECO, evaluated on GECO test + full Provo (cross-corpus).

Usage:
    # Default: bert-base-uncased (110M params)
    python3 -u previous_implementations_of_word_level_predictions/bert_regression.py

    # Smaller models:
    python3 -u previous_implementations_of_word_level_predictions/bert_regression.py --bert prajjwal1/bert-mini
    python3 -u previous_implementations_of_word_level_predictions/bert_regression.py --bert prajjwal1/bert-small
"""

import os
import sys
import time
import random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn.utils.rnn import pad_sequence

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src_v2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'original_ezreader'))

from transformers import BertModel, BertTokenizerFast
from data_loader import load_provo, aggregate_by_sentence
from geco_loader import load_geco, split_geco


# --------------------------------------------------------------------------- #
#  Model: BERT + Direct Regression Heads
# --------------------------------------------------------------------------- #

class BertDirectRegression(nn.Module):
    """
    Fine-tuned BERT with direct regression heads for reading time prediction.
    No cognitive model architecture — pure neural regression.

    Architecture mirrors model_bert.py's NeuralEZReaderBERT but replaces the
    DifferentiableEZReader with direct regression heads.
    """

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        freeze_bert_layers: int = 8,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # --- BERT encoder ---
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)
        bert_dim = self.bert.config.hidden_size

        # Freeze lower BERT layers
        if freeze_bert_layers > 0:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            for layer_idx in range(min(freeze_bert_layers, len(self.bert.encoder.layer))):
                for param in self.bert.encoder.layer[layer_idx].parameters():
                    param.requires_grad = False

        # --- Projection ---
        self.projection = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # --- Direct regression heads (same architecture as model_llama_direct.py) ---
        self.trt_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )
        self.ffd_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )
        self.gaze_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )
        self.skip_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Learnable scales
        self.trt_scale = nn.Parameter(torch.tensor(100.0))
        self.ffd_scale = nn.Parameter(torch.tensor(100.0))
        self.gaze_scale = nn.Parameter(torch.tensor(100.0))

    def _tokenize_and_align(self, word_lists, device):
        encodings = self.tokenizer(
            word_lists,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        batch_word_maps = []
        max_words = 0
        for batch_idx in range(len(word_lists)):
            word_ids = encodings.word_ids(batch_index=batch_idx)
            word_map = {}
            for subword_idx, word_idx in enumerate(word_ids):
                if word_idx is None:
                    continue
                if word_idx not in word_map:
                    word_map[word_idx] = [subword_idx, subword_idx + 1]
                else:
                    word_map[word_idx][1] = subword_idx + 1

            n_words = len(word_lists[batch_idx])
            spans = []
            for w_idx in range(n_words):
                if w_idx in word_map:
                    spans.append(tuple(word_map[w_idx]))
                else:
                    spans.append((1, 2))  # fallback to [CLS] +1
            batch_word_maps.append(spans)
            max_words = max(max_words, n_words)

        return input_ids, attention_mask, batch_word_maps, max_words

    def _pool_subwords_to_words(self, bert_output, batch_word_maps, max_words, device):
        """First-subword pooling (same as model_bert.py)."""
        batch_size = bert_output.size(0)
        bert_dim = bert_output.size(2)
        idx = torch.zeros(batch_size, max_words, dtype=torch.long)
        for b in range(batch_size):
            for w_idx, (start, end) in enumerate(batch_word_maps[b]):
                idx[b, w_idx] = start
        idx = idx.to(device)
        word_repr = torch.gather(
            bert_output, 1, idx.unsqueeze(-1).expand(-1, -1, bert_dim)
        )
        return word_repr

    def forward(self, word_lists, predictability, word_lengths):
        """
        Args:
            word_lists:     list of list of str
            predictability: (batch, seq_len) — accepted but NOT used
            word_lengths:   (batch, seq_len) — accepted but NOT used
        """
        device = predictability.device
        seq_len = predictability.size(1)

        input_ids, attention_mask, word_maps, max_words = self._tokenize_and_align(
            word_lists, device
        )
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

        word_repr = self._pool_subwords_to_words(
            bert_out, word_maps, max_words, device
        )
        projected = self.projection(word_repr)

        # Direct predictions
        trt = self.trt_head(projected).squeeze(-1) * self.trt_scale
        ffd = self.ffd_head(projected).squeeze(-1) * self.ffd_scale
        gaze = self.gaze_head(projected).squeeze(-1) * self.gaze_scale
        skip = self.skip_head(projected).squeeze(-1)

        trt = trt[:, :seq_len].clamp(min=1.0, max=1500.0)
        ffd = ffd[:, :seq_len].clamp(min=1.0, max=1000.0)
        gaze = gaze[:, :seq_len].clamp(min=1.0, max=1500.0)
        skip = skip[:, :seq_len]

        return {
            'total_reading_time': trt,
            'first_fixation': ffd,
            'gaze_duration': gaze,
            'skip_prob': skip,
        }


# --------------------------------------------------------------------------- #
#  Collate functions
# --------------------------------------------------------------------------- #

def collate_sentences(batch, device):
    word_lists = [sd.tokens for sd in batch]
    pred_vals = pad_sequence(
        [torch.tensor([w.predictability for w in sd.words], dtype=torch.float32) for sd in batch],
        batch_first=True).to(device)
    wlens = pad_sequence(
        [torch.tensor([len(t) for t in sd.tokens], dtype=torch.float32) for sd in batch],
        batch_first=True).to(device)
    h_trt = pad_sequence(
        [torch.tensor(sd.total_reading_times, dtype=torch.float32) for sd in batch],
        batch_first=True).to(device)
    h_ffd = pad_sequence(
        [torch.tensor(sd.first_fixation_durations, dtype=torch.float32) for sd in batch],
        batch_first=True).to(device)
    h_gaze = pad_sequence(
        [torch.tensor(sd.gaze_durations, dtype=torch.float32) for sd in batch],
        batch_first=True).to(device)
    h_skip = pad_sequence(
        [torch.tensor([1.0 if s else 0.0 for s in sd.skip_flags], dtype=torch.float32) for sd in batch],
        batch_first=True).to(device)
    return word_lists, pred_vals, wlens, h_trt, h_ffd, h_gaze, h_skip


def collate_aggregated(batch, device):
    word_lists = [a.tokens for a in batch]
    pred_vals = pad_sequence(
        [torch.tensor(a.predictabilities, dtype=torch.float32) for a in batch],
        batch_first=True).to(device)
    wlens = pad_sequence(
        [torch.tensor([len(t) for t in a.tokens], dtype=torch.float32) for a in batch],
        batch_first=True).to(device)
    h_trt = pad_sequence(
        [torch.tensor(a.mean_trt, dtype=torch.float32) for a in batch],
        batch_first=True).to(device)
    h_ffd = pad_sequence(
        [torch.tensor(a.mean_ffd, dtype=torch.float32) for a in batch],
        batch_first=True).to(device)
    h_gaze = pad_sequence(
        [torch.tensor(a.mean_gaze, dtype=torch.float32) for a in batch],
        batch_first=True).to(device)
    h_skip = pad_sequence(
        [torch.tensor(a.skip_rate, dtype=torch.float32) for a in batch],
        batch_first=True).to(device)
    return word_lists, pred_vals, wlens, h_trt, h_ffd, h_gaze, h_skip


# --------------------------------------------------------------------------- #
#  Logger
# --------------------------------------------------------------------------- #

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()


# --------------------------------------------------------------------------- #
#  Loss function
# --------------------------------------------------------------------------- #

def compute_loss(pred, human_trt, human_ffd, human_gaze, human_skip):
    """
    Combined loss: MSE on reading times + BCE on skip.
    Weights: 0.25*TRT + 0.25*FFD + 0.25*Gaze + 0.25*Skip
    """
    trt_loss = nn.functional.mse_loss(pred['total_reading_time'].float(), human_trt)
    ffd_loss = nn.functional.mse_loss(pred['first_fixation'].float(), human_ffd)
    gaze_loss = nn.functional.mse_loss(pred['gaze_duration'].float(), human_gaze)

    skip_pred = pred['skip_prob'].float().clamp(1e-6, 1 - 1e-6)
    skip_loss = nn.functional.binary_cross_entropy(skip_pred, human_skip)

    total = 0.25 * trt_loss + 0.25 * ffd_loss + 0.25 * gaze_loss + 0.25 * skip_loss

    return total, {
        'trt': trt_loss.item(),
        'ffd': ffd_loss.item(),
        'gaze': gaze_loss.item(),
        'skip': skip_loss.item(),
        'total': total.item(),
    }


# --------------------------------------------------------------------------- #
#  Evaluation
# --------------------------------------------------------------------------- #

def evaluate_detailed(model, agg_data, device, batch_size=16):
    model.eval()
    all_pred_trt, all_human_trt = [], []
    all_pred_ffd, all_human_ffd = [], []
    all_pred_gaze, all_human_gaze = [], []
    all_pred_skip, all_human_skip = [], []
    total_loss = 0.0
    n = 0

    with torch.no_grad():
        for i in range(0, len(agg_data), batch_size):
            batch = agg_data[i:i + batch_size]
            word_lists, pred_vals, wlens, h_trt, h_ffd, h_gaze, h_skip = collate_aggregated(batch, device)

            pred = model(word_lists, pred_vals, wlens)
            loss, _ = compute_loss(pred, h_trt, h_ffd, h_gaze, h_skip)
            total_loss += loss.item() * len(batch)
            n += len(batch)

            for b in range(len(batch)):
                seq_len = len(batch[b].tokens)
                all_pred_trt.extend(pred['total_reading_time'][b, :seq_len].cpu().tolist())
                all_human_trt.extend(batch[b].mean_trt)
                all_pred_ffd.extend(pred['first_fixation'][b, :seq_len].cpu().tolist())
                all_human_ffd.extend(batch[b].mean_ffd)
                all_pred_gaze.extend(pred['gaze_duration'][b, :seq_len].cpu().tolist())
                all_human_gaze.extend(batch[b].mean_gaze)
                all_pred_skip.extend(pred['skip_prob'][b, :seq_len].cpu().tolist())
                all_human_skip.extend(batch[b].skip_rate)

    avg_loss = total_loss / max(n, 1)

    def corr(a, b):
        a, b = np.array(a), np.array(b)
        if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0:
            return np.corrcoef(a, b)[0, 1]
        return 0.0

    return {
        'loss': avg_loss,
        'r_trt': corr(all_pred_trt, all_human_trt),
        'r_ffd': corr(all_pred_ffd, all_human_ffd),
        'r_gaze': corr(all_pred_gaze, all_human_gaze),
        'r_skip': corr(all_pred_skip, all_human_skip),
        'mae_trt': np.mean(np.abs(np.array(all_pred_trt) - np.array(all_human_trt))),
        'mae_ffd': np.mean(np.abs(np.array(all_pred_ffd) - np.array(all_human_ffd))),
        'mean_pred_trt': np.mean(all_pred_trt),
        'mean_human_trt': np.mean(all_human_trt),
    }


def print_sample_predictions(model, agg_data, device, n_sentences=3, n_words=8):
    model.eval()
    with torch.no_grad():
        for s_idx in range(min(n_sentences, len(agg_data))):
            s = agg_data[s_idx]
            word_list = [s.tokens]
            pv = torch.tensor(s.predictabilities, dtype=torch.float32).unsqueeze(0).to(device)
            wl = torch.tensor([len(t) for t in s.tokens], dtype=torch.float32).unsqueeze(0).to(device)
            p = model(word_list, pv, wl)

            title = ' '.join(s.tokens[:6]) + ('...' if len(s.tokens) > 6 else '')
            print(f"  Sentence {s_idx+1}: \"{title}\"")
            print(f"  {'word':<14s} {'pTRT':>6s} {'hTRT':>6s} {'err':>6s} | "
                  f"{'pFFD':>6s} {'hFFD':>6s} | {'pSkip':>6s} {'hSkip':>6s}")
            print(f"  {'-'*70}")

            for i in range(min(n_words, len(s.tokens))):
                pt = p['total_reading_time'][0, i].item()
                pf = p['first_fixation'][0, i].item()
                ps = p['skip_prob'][0, i].item()
                ht = s.mean_trt[i]
                hf = s.mean_ffd[i]
                hs = s.skip_rate[i]
                err = pt - ht
                print(f"  {s.tokens[i]:<14s} {pt:6.0f} {ht:6.0f} {err:+6.0f} | "
                      f"{pf:6.0f} {hf:6.0f} | {ps:6.2f} {hs:6.2f}")
            print()


# --------------------------------------------------------------------------- #
#  Training
# --------------------------------------------------------------------------- #

def train(
    data_dir="../data",
    num_epochs=50,
    bert_lr=2e-5,
    head_lr=5e-4,
    batch_size=16,
    accumulation_steps=4,
    save_dir="checkpoints_bert_regression",
    seed=42,
    bert_model_name="bert-base-uncased",
    freeze_bert_layers=8,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load GECO ----
    reading_path = os.path.join(data_dir, "Geco_MonolingualReadingData.csv")
    material_path = os.path.join(data_dir, "Geco_EnglishMaterial.csv")
    pred_path = os.path.join(data_dir, "geco_predictability.pkl")

    print("Loading GECO Corpus...")
    raw_dataset = load_geco(reading_path, material_path, pred_path)
    train_raw, val_raw, test_raw = split_geco(raw_dataset)
    print(f"  Train: {len(train_raw):,} | Val: {len(val_raw):,} | Test: {len(test_raw):,}")

    aggregated = aggregate_by_sentence(raw_dataset, min_participants=5)
    train_ids = set(sd.text_id for sd in train_raw)
    val_ids = set(sd.text_id for sd in val_raw)
    train_agg = [a for a in aggregated if a.text_id in train_ids]
    val_agg = [a for a in aggregated if a.text_id in val_ids]
    test_agg = [a for a in aggregated if a.text_id not in train_ids and a.text_id not in val_ids]
    print(f"  Aggregated: {len(train_agg)} train | {len(val_agg)} val | {len(test_agg)} test")

    # ---- Model ----
    print(f"\nBuilding BERT Direct Regression model: {bert_model_name}")
    print(f"  Freezing first {freeze_bert_layers} BERT layers")
    model = BertDirectRegression(
        bert_model_name=bert_model_name,
        freeze_bert_layers=freeze_bert_layers,
    ).to(device)

    # Differential learning rates
    bert_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("bert."):
            bert_params.append(param)
        else:
            head_params.append(param)

    n_bert = sum(p.numel() for p in bert_params)
    n_head = sum(p.numel() for p in head_params)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"  Total parameters:    {total:,}")
    print(f"  Frozen (BERT):       {n_frozen:,}")
    print(f"  Trainable (BERT):    {n_bert:,} (lr={bert_lr})")
    print(f"  Trainable (heads):   {n_head:,} (lr={head_lr})")

    optimizer = optim.AdamW([
        {"params": bert_params, "lr": bert_lr, "weight_decay": 0.01},
        {"params": head_params, "lr": head_lr, "weight_decay": 0.0},
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-7)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(save_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(save_dir, "training_log.txt"))

    best_val_corr = -1.0

    print("\n" + "=" * 90)
    print("Training: BERT + Direct Regression Heads (NO E-Z Reader)")
    print(f"  Batch size: {batch_size} | Grad accum: {accumulation_steps}")
    print(f"  Effective batch size: {batch_size * accumulation_steps}")
    print(f"  AMP: {use_amp}")
    print(f"  Loss: 0.25*TRT + 0.25*FFD + 0.25*Gaze + 0.25*Skip")
    print("=" * 90)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()

        epoch_data = train_raw.copy()
        random.shuffle(epoch_data)

        epoch_loss = 0.0
        epoch_trt = 0.0
        epoch_ffd = 0.0
        epoch_gaze = 0.0
        epoch_skip = 0.0
        n_samples = 0

        optimizer.zero_grad()
        n_batches = (len(epoch_data) + batch_size - 1) // batch_size

        for step in range(n_batches):
            batch = epoch_data[step * batch_size : (step + 1) * batch_size]
            word_lists, pred_vals, wlens, h_trt, h_ffd, h_gaze, h_skip = collate_sentences(batch, device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(word_lists, pred_vals, wlens)
            loss, parts = compute_loss(pred, h_trt, h_ffd, h_gaze, h_skip)

            loss = loss / accumulation_steps
            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == n_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += parts['total']
            epoch_trt += parts['trt']
            epoch_ffd += parts['ffd']
            epoch_gaze += parts['gaze']
            epoch_skip += parts['skip']
            n_samples += len(batch)

        epoch_loss /= n_samples
        epoch_trt /= n_samples
        epoch_ffd /= n_samples
        epoch_gaze /= n_samples
        epoch_skip /= n_samples
        elapsed = time.time() - t0

        # Validation
        val_metrics = evaluate_detailed(model, val_agg, device)
        scheduler.step(val_metrics['loss'])
        bert_lr_now = optimizer.param_groups[0]['lr']
        head_lr_now = optimizer.param_groups[1]['lr']

        is_best = val_metrics['r_trt'] > best_val_corr

        print(f"\n[Epoch {epoch:3d}/{num_epochs}] {elapsed:.1f}s | "
              f"bert_lr={bert_lr_now:.2e} head_lr={head_lr_now:.2e}")
        print(f"  Train: loss={epoch_loss:.1f} "
              f"(trt={epoch_trt:.0f} ffd={epoch_ffd:.0f} gaze={epoch_gaze:.0f} skip={epoch_skip:.3f})")
        print(f"  Val:   loss={val_metrics['loss']:.1f} | "
              f"r_TRT={val_metrics['r_trt']:.3f}  "
              f"r_FFD={val_metrics['r_ffd']:.3f}  "
              f"r_Gaze={val_metrics['r_gaze']:.3f}  "
              f"r_skip={val_metrics['r_skip']:.3f}")
        print(f"  Val:   MAE_TRT={val_metrics['mae_trt']:.1f}ms  "
              f"MAE_FFD={val_metrics['mae_ffd']:.1f}ms")
        print(f"  Pred:  mean_TRT={val_metrics['mean_pred_trt']:.0f}ms "
              f"(human={val_metrics['mean_human_trt']:.0f}ms)")

        if epoch % 5 == 1 or is_best:
            print_sample_predictions(model, train_agg, device, n_sentences=2, n_words=6)

        if is_best:
            best_val_corr = val_metrics['r_trt']
            print(f"  ** NEW BEST (r_TRT={val_metrics['r_trt']:.3f}) **")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'bert_model_name': bert_model_name,
                'freeze_bert_layers': freeze_bert_layers,
                'val_metrics': val_metrics,
            }, os.path.join(save_dir, "best_model.pt"))

    # ---- Final evaluation ----
    print("\n" + "=" * 90)
    print(f"Training complete! Best val r_TRT = {best_val_corr:.3f}")
    print("=" * 90)

    # Reload best model
    best_ckpt = torch.load(os.path.join(save_dir, "best_model.pt"), map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])
    model.eval()
    print(f"  Loaded best model from epoch {best_ckpt['epoch']}")

    # GECO test
    test_metrics = evaluate_detailed(model, test_agg, device)
    print(f"\nGECO Test Set:")
    print(f"  r_TRT={test_metrics['r_trt']:.3f}  "
          f"r_FFD={test_metrics['r_ffd']:.3f}  "
          f"r_Gaze={test_metrics['r_gaze']:.3f}  "
          f"r_skip={test_metrics['r_skip']:.3f}")
    print(f"  MAE_TRT={test_metrics['mae_trt']:.1f}ms  MAE_FFD={test_metrics['mae_ffd']:.1f}ms")

    print("\nSample test predictions:")
    print_sample_predictions(model, test_agg, device, n_sentences=3, n_words=8)

    # Cross-corpus: Provo
    print("\nLoading Provo for cross-corpus evaluation...")
    et_path = os.path.join(data_dir, "Provo_Corpus-Eyetracking_Data.csv")
    provo_raw = load_provo(et_path)
    provo_agg = aggregate_by_sentence(provo_raw, min_participants=10)
    print(f"  Provo: {len(provo_agg)} sentences, {sum(len(a.tokens) for a in provo_agg)} words")

    provo_metrics = evaluate_detailed(model, provo_agg, device)
    print(f"\nProvo (cross-corpus):")
    print(f"  r_TRT={provo_metrics['r_trt']:.3f}  "
          f"r_FFD={provo_metrics['r_ffd']:.3f}  "
          f"r_Gaze={provo_metrics['r_gaze']:.3f}  "
          f"r_skip={provo_metrics['r_skip']:.3f}")
    print(f"  MAE_TRT={provo_metrics['mae_trt']:.1f}ms  MAE_FFD={provo_metrics['mae_ffd']:.1f}ms")

    print("\nSample Provo predictions:")
    print_sample_predictions(model, provo_agg, device, n_sentences=3, n_words=8)

    # ---- Summary comparison ----
    print("\n" + "=" * 90)
    print("SUMMARY: BERT Direct Regression (no E-Z Reader)")
    print("=" * 90)
    print(f"  {'Metric':<20s} {'GECO Test':>12s} {'Provo (cross)':>14s}")
    print(f"  {'-'*50}")
    print(f"  {'r_TRT':<20s} {test_metrics['r_trt']:>12.3f} {provo_metrics['r_trt']:>14.3f}")
    print(f"  {'r_FFD':<20s} {test_metrics['r_ffd']:>12.3f} {provo_metrics['r_ffd']:>14.3f}")
    print(f"  {'r_Gaze':<20s} {test_metrics['r_gaze']:>12.3f} {provo_metrics['r_gaze']:>14.3f}")
    print(f"  {'r_Skip':<20s} {test_metrics['r_skip']:>12.3f} {provo_metrics['r_skip']:>14.3f}")
    print(f"  {'MAE_TRT (ms)':<20s} {test_metrics['mae_trt']:>12.1f} {provo_metrics['mae_trt']:>14.1f}")
    print(f"  {'MAE_FFD (ms)':<20s} {test_metrics['mae_ffd']:>12.1f} {provo_metrics['mae_ffd']:>14.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert", type=str, default="bert-base-uncased",
                        help="BERT model name")
    parser.add_argument("--freeze", type=int, default=None,
                        help="Layers to freeze (default: half)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--bert_lr", type=float, default=2e-5)
    parser.add_argument("--head_lr", type=float, default=5e-4)
    args = parser.parse_args()

    if args.freeze is not None:
        freeze_layers = args.freeze
    else:
        from transformers import BertConfig
        cfg = BertConfig.from_pretrained(args.bert)
        freeze_layers = cfg.num_hidden_layers // 2

    model_short = args.bert.replace("/", "_")
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    save_dir = os.path.join(os.path.dirname(__file__), f"checkpoints_bert_direct_{model_short}")

    train(
        data_dir=data_dir,
        num_epochs=args.epochs,
        bert_lr=args.bert_lr,
        head_lr=args.head_lr,
        batch_size=args.batch_size,
        accumulation_steps=args.accum,
        save_dir=save_dir,
        bert_model_name=args.bert,
        freeze_bert_layers=freeze_layers,
    )
