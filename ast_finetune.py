"""
Option B: Fine-tune Audio Spectrogram Transformer (AST) on FMA-small.

Uses MIT/ast-finetuned-audioset-10-10-0.4593 from HuggingFace — an AST model
pre-trained on AudioSet (~2M clips). Only the classification head is replaced
for 8 genres. Differential learning rates are used: 1e-5 for the backbone,
1e-4 for the new head.

At inference, three fixed 10-second crops (start / middle / end) are averaged
for track-level prediction (test-time augmentation).

Requirements:
    pip install transformers

Reads:  data/fma_small/         (original 29-second MP3s)
        features/mel_specs.npz  (for track IDs, labels, and the shared split)
Saves:  results/B AST Fine-tune/
"""

import time
import warnings
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

try:
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
except ImportError:
    raise SystemExit("Install the transformers library first:  pip install transformers")

from cnn_training_utils import (
    DEVICE,
    clone_state_dict,
    count_parameters,
    finalize_experiment,
    load_mel_cache,
    make_split,
    seed_everything,
)
from reporting_utils import ROOT, experiment_dir, print_section

warnings.filterwarnings("ignore")

OUT_DIR    = experiment_dir("B AST Fine-tune")
FMA_AUDIO  = ROOT / "data" / "fma_small"
MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
TARGET_SR  = 16_000
CLIP_SEC   = 10.0         # AST was pre-trained on 10-second clips
N_CROPS    = 3            # crops averaged at eval time (start / middle / end)
EPOCHS     = 30
PATIENCE   = 8


# ── Helpers ───────────────────────────────────────────────────────────────────

def audio_path(track_id):
    tid = f"{int(track_id):06d}"
    return FMA_AUDIO / tid[:3] / f"{tid}.mp3"


def load_waveform(track_id):
    path = audio_path(track_id)
    y, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    return y


def fixed_crop(y, crop_idx, n_crops, clip_samples):
    """Return one of n_crops evenly-spaced fixed crops of exactly clip_samples."""
    total = len(y)
    if total <= clip_samples:
        return np.pad(y, (0, clip_samples - total))
    max_start = total - clip_samples
    start = int(round(max_start * crop_idx / max(n_crops - 1, 1)))
    return y[start : start + clip_samples]


def random_crop(y, clip_samples):
    total = len(y)
    if total <= clip_samples:
        return np.pad(y, (0, clip_samples - total))
    start = np.random.randint(0, total - clip_samples + 1)
    return y[start : start + clip_samples]


# ── Dataset ───────────────────────────────────────────────────────────────────

class FMADataset(Dataset):
    """
    train=True  → one random 10-second crop per track, returns (1024, 128).
    train=False → N_CROPS fixed crops per track, returns (N_CROPS, 1024, 128).
    """

    def __init__(self, track_ids, labels, feature_extractor, train):
        self.track_ids = track_ids
        self.labels    = labels
        self.fe        = feature_extractor
        self.train     = train
        self.clip_samples = int(CLIP_SEC * TARGET_SR)

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        y     = load_waveform(self.track_ids[idx])
        label = int(self.labels[idx])

        if self.train:
            crops = [random_crop(y, self.clip_samples)]
        else:
            crops = [
                fixed_crop(y, i, N_CROPS, self.clip_samples)
                for i in range(N_CROPS)
            ]

        inputs = self.fe(
            crops,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding="max_length",
        )
        features = inputs["input_values"]   # (n_crops, time, freq)

        if self.train:
            features = features.squeeze(0)  # (time, freq)

        return features, torch.tensor(label, dtype=torch.long)


# ── Training / evaluation ─────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, use_amp, scaler):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for features, labels in loader:
        features = features.to(DEVICE, non_blocking=True)   # (B, time, freq)
        labels   = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(input_values=features).logits
            loss   = criterion(logits, labels)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        correct    += (logits.argmax(1) == labels).sum().item()
        total_loss += loss.item() * len(labels)
        n          += len(labels)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, use_amp):
    model.eval()
    total_loss, n = 0.0, 0
    all_true, all_pred, all_proba = [], [], []
    for features, labels in loader:
        # features: (B, N_CROPS, time, freq)
        B, NC = features.shape[0], features.shape[1]
        flat     = features.view(B * NC, *features.shape[2:]).to(DEVICE, non_blocking=True)
        labels_d = labels.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits_flat = model(input_values=flat).logits          # (B*NC, classes)
        logits = logits_flat.view(B, NC, -1).mean(dim=1)           # average crops
        loss   = criterion(logits, labels_d)
        proba  = torch.softmax(logits, dim=1).cpu().numpy()
        all_true.extend(labels.numpy())
        all_pred.extend(proba.argmax(1))
        all_proba.extend(proba)
        total_loss += loss.item() * B
        n          += B
    y_true  = np.array(all_true)
    y_pred  = np.array(all_pred)
    y_proba = np.array(all_proba)
    return (
        total_loss / n,
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average="macro"),
        y_true, y_pred, y_proba,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    seed_everything()
    print(f"Using device: {DEVICE}", flush=True)

    # Use the same track set and split as every other CNN experiment.
    _mels, labels, track_ids = load_mel_cache()
    if track_ids is None:
        feat = np.load(ROOT / "features" / "features.npz", allow_pickle=True)
        track_ids = feat["track_ids"]

    le, y, idx_train, idx_val, idx_test = make_split(labels)

    missing = [tid for tid in track_ids if not audio_path(tid).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} audio files not found under {FMA_AUDIO}. "
            "This script must run where the original FMA-small MP3s are present."
        )

    print_section("B AST Fine-tune")
    print(f"Loading feature extractor from {MODEL_NAME}")
    fe = ASTFeatureExtractor.from_pretrained(MODEL_NAME)

    loader_kw = dict(num_workers=2, pin_memory=DEVICE.type == "cuda")
    train_loader = DataLoader(
        FMADataset(track_ids[idx_train], y[idx_train], fe, train=True),
        batch_size=16, shuffle=True, **loader_kw,
    )
    val_loader = DataLoader(
        FMADataset(track_ids[idx_val], y[idx_val], fe, train=False),
        batch_size=8, shuffle=False, **loader_kw,
    )
    test_loader = DataLoader(
        FMADataset(track_ids[idx_test], y[idx_test], fe, train=False),
        batch_size=8, shuffle=False, **loader_kw,
    )

    print(f"Building model (num_labels={len(le.classes_)})")
    model = ASTForAudioClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(le.classes_),
        ignore_mismatched_sizes=True,
    ).to(DEVICE)
    param_count, trainable_count = count_parameters(model)
    print(f"Parameters: {param_count:,} (trainable: {trainable_count:,})")

    # Differential LR: backbone gets 10× smaller LR than the new head.
    backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
    head_params     = [p for n, p in model.named_parameters() if "classifier"     in n]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": 1e-5},
            {"params": head_params,     "lr": 1e-4},
        ],
        weight_decay=1e-2,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-7
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    use_amp = DEVICE.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_state, best_epoch, best_val_f1 = None, 0, -1.0
    best_val_true = best_val_pred = best_val_proba = None
    no_improve = 0
    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "lr")}
    start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, use_amp, scaler)
        val_loss, val_acc, val_f1, val_true, val_pred, val_proba = evaluate(
            model, val_loader, criterion, use_amp
        )
        scheduler.step(val_f1)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if val_f1 > best_val_f1:
            best_state = clone_state_dict(model)
            best_epoch = epoch
            best_val_f1 = val_f1
            best_val_true, best_val_pred, best_val_proba = val_true, val_pred, val_proba
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"  Epoch {epoch:2d}/{EPOCHS}  loss={tr_loss:.4f}  "
            f"train={tr_acc:.4f}  val_acc={val_acc:.4f}  "
            f"val_f1={val_f1:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}",
            flush=True,
        )
        if no_improve >= PATIENCE:
            print(f"  Early stop at epoch {epoch} (best epoch {best_epoch})", flush=True)
            break

    training_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)
    _, test_acc, test_f1, test_true, test_pred, test_proba = evaluate(
        model, test_loader, criterion, use_amp
    )
    epochs_run = len(history["train_loss"])
    print(
        f"\nAST fine-tune: best_epoch={best_epoch}  val_f1={best_val_f1:.4f}  "
        f"test_acc={test_acc:.4f}  test_f1={test_f1:.4f}  runtime={training_seconds:.1f}s",
        flush=True,
    )

    result = {
        "label":                  "AST fine-tune",
        "val_true":               best_val_true,
        "val_pred":               best_val_pred,
        "val_proba":              best_val_proba,
        "test_true":              test_true,
        "test_pred":              test_pred,
        "test_proba":             test_proba,
        "test_indices":           idx_test,
        "best_epoch":             best_epoch,
        "best_val_f1":            best_val_f1,
        "param_count":            param_count,
        "trainable_param_count":  trainable_count,
        "training_seconds":       training_seconds,
        "epochs_run":             epochs_run,
        "history":                history,
    }
    finalize_experiment([result], OUT_DIR, le.classes_, "B AST Fine-tune", track_ids=track_ids)


if __name__ == "__main__":
    main()
