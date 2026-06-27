"""
Option A: CNN backbone + Transformer encoder across track segments.

Instead of averaging segment probabilities independently (Part 2.6), a small
Transformer encoder learns cross-segment context before classifying.

Architecture:
    [seg_0, seg_1, seg_2, seg_3]
        ↓ shared CNN backbone (RegularisedCNN without classifier head)
    [emb_0, emb_1, emb_2, emb_3]   (128-dim each)
        ↓ prepend learnable CLS token
    TransformerEncoder (2 layers, 4 heads, dim=128)
        ↓ CLS output
    Dropout → Linear → 8 genres

Reads:  features/mel_segments.npz  (run extract_mel_segments.py first)
Saves:  results/A CNN Transformer/
"""

import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

from cnn_training_utils import (
    DEVICE,
    AugmentConfig,
    RegularisedCNN,
    TrainConfig,
    clone_state_dict,
    count_parameters,
    current_lr,
    finalize_experiment,
    load_mel_cache,
    make_optimizer,
    make_scheduler,
    make_split,
    seed_everything,
    spec_augment,
)
from reporting_utils import ROOT, experiment_dir, print_section

OUT_DIR = experiment_dir("A CNN Transformer")
SEGMENT_CACHE = ROOT / "features" / "mel_segments.npz"


# ── Data ─────────────────────────────────────────────────────────────────────

def load_segments():
    assert SEGMENT_CACHE.exists(), (
        "Run extract_mel_segments.py first to build features/mel_segments.npz."
    )
    data = np.load(SEGMENT_CACHE, allow_pickle=True)
    return (
        data["segments"].astype(np.float32),  # (n_tracks, n_segs, H, W)
        data["labels"],
        data["track_ids"],
    )


class TrackDataset(Dataset):
    """Returns all segments for a track as a single (n_segs, 1, H, W) tensor."""

    def __init__(self, segments, labels, track_indices, augment_cfg=None):
        self.segments = segments
        self.labels = labels
        self.track_indices = np.asarray(track_indices)
        self.augment_cfg = augment_cfg or AugmentConfig()

    def __len__(self):
        return len(self.track_indices)

    def __getitem__(self, i):
        idx = self.track_indices[i]
        segs = torch.tensor(self.segments[idx], dtype=torch.float32)  # (n_segs, H, W)
        segs = segs.unsqueeze(1)                                       # (n_segs, 1, H, W)
        if self.augment_cfg.specaugment:
            segs = torch.stack([spec_augment(s, self.augment_cfg) for s in segs])
        label = int(self.labels[idx])
        return segs, torch.tensor(label, dtype=torch.long)


# ── Model ─────────────────────────────────────────────────────────────────────

class CNNBackbone(nn.Module):
    """Moderately regularised CNN up to the global average pool — no classifier."""

    EMBED_DIM = 128

    def __init__(self):
        super().__init__()
        cnn = RegularisedCNN(
            n_classes=8,
            block_dropouts=(0.05, 0.10, 0.15),
            fc_dropouts=(0.5, 0.3),
        )
        self.block1 = cnn.block1
        self.drop1  = cnn.drop1
        self.block2 = cnn.block2
        self.drop2  = cnn.drop2
        self.block3 = cnn.block3
        self.drop3  = cnn.drop3
        self.pool   = cnn.pool

    def forward(self, x):          # x: (B, 1, H, W)
        x = self.drop1(self.block1(x))
        x = self.drop2(self.block2(x))
        x = self.drop3(self.block3(x))
        return self.pool(x).flatten(1)   # (B, 128)


class CNNSegmentTransformer(nn.Module):
    """CNN per segment + Transformer encoder across segments."""

    def __init__(self, n_classes=8, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.backbone = CNNBackbone()
        d = CNNBackbone.EMBED_DIM

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=d * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(d, n_classes),
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, segs):       # segs: (B, N, 1, H, W)
        B, N, C, H, W = segs.shape
        embeds = self.backbone(segs.view(B * N, C, H, W)).view(B, N, -1)  # (B, N, d)
        cls    = self.cls_token.expand(B, -1, -1)                          # (B, 1, d)
        tokens = torch.cat([cls, embeds], dim=1)                           # (B, N+1, d)
        out    = self.transformer(tokens)                                   # (B, N+1, d)
        return self.classifier(out[:, 0])                                  # CLS token


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, augment_cfg, use_amp, scaler):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for segs, labels in loader:
        segs   = segs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            out  = model(segs)
            loss = criterion(out, labels)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        correct    += (out.argmax(1) == labels).sum().item()
        total_loss += loss.item() * len(labels)
        n          += len(labels)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, use_amp):
    model.eval()
    total_loss, n = 0.0, 0
    all_true, all_pred, all_proba = [], [], []
    for segs, labels in loader:
        segs      = segs.to(DEVICE, non_blocking=True)
        labels_d  = labels.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            out  = model(segs)
            loss = criterion(out, labels_d)
        proba = torch.softmax(out, dim=1).cpu().numpy()
        all_true.extend(labels.numpy())
        all_pred.extend(proba.argmax(1))
        all_proba.extend(proba)
        total_loss += loss.item() * len(labels)
        n          += len(labels)
    y_true  = np.array(all_true)
    y_pred  = np.array(all_pred)
    y_proba = np.array(all_proba)
    return (
        total_loss / n,
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average="macro"),
        y_true, y_pred, y_proba,
    )


def run(segments, y, le, idx_train, idx_val, idx_test):
    cfg        = TrainConfig(epochs=300, lr=5e-4, weight_decay=1e-4, patience=48, batch_size=64)
    augment_cfg = AugmentConfig(specaugment=False, mixup=False)

    seed_everything()
    model = CNNSegmentTransformer(n_classes=len(le.classes_)).to(DEVICE)
    param_count, trainable_count = count_parameters(model)
    print(f"\nCNN + Transformer: parameters={param_count:,} (trainable={trainable_count:,})")

    use_amp   = cfg.amp and DEVICE.type == "cuda"
    scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)
    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    def make_loader(indices, train):
        aug = augment_cfg if train else AugmentConfig()
        return DataLoader(
            TrackDataset(segments, y, indices, augment_cfg=aug),
            batch_size=cfg.batch_size,
            shuffle=train,
            num_workers=cfg.num_workers,
            pin_memory=DEVICE.type == "cuda",
            persistent_workers=cfg.num_workers > 0,
        )

    train_loader = make_loader(idx_train, train=True)
    val_loader   = make_loader(idx_val,   train=False)
    test_loader  = make_loader(idx_test,  train=False)

    best_state, best_epoch, best_val_f1 = None, 0, -1.0
    best_val_true = best_val_pred = best_val_proba = None
    no_improve = 0
    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "lr")}
    start = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, criterion, augment_cfg, use_amp, scaler
        )
        val_loss, val_acc, val_f1, val_true, val_pred, val_proba = evaluate(
            model, val_loader, criterion, use_amp
        )
        scheduler.step(val_f1)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr(optimizer))

        if val_f1 > best_val_f1:
            best_state = clone_state_dict(model)
            best_epoch = epoch
            best_val_f1 = val_f1
            best_val_true, best_val_pred, best_val_proba = val_true, val_pred, val_proba
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{cfg.epochs}  loss={tr_loss:.4f}  "
                f"train={tr_acc:.4f}  val_acc={val_acc:.4f}  "
                f"val_f1={val_f1:.4f}  lr={current_lr(optimizer):.2e}",
                flush=True,
            )
        if no_improve >= cfg.patience:
            print(f"  Early stop at epoch {epoch} (best epoch {best_epoch})", flush=True)
            break

    training_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)
    _, test_acc, test_f1, test_true, test_pred, test_proba = evaluate(
        model, test_loader, criterion, use_amp
    )
    epochs_run = len(history["train_loss"])
    print(
        f"\nCNN + Transformer: best_epoch={best_epoch}  val_f1={best_val_f1:.4f}  "
        f"test_acc={test_acc:.4f}  test_f1={test_f1:.4f}  runtime={training_seconds:.1f}s",
        flush=True,
    )
    return {
        "label": "CNN + Transformer",
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


def main():
    seed_everything()
    print(f"Using device: {DEVICE}", flush=True)

    _mels, mel_labels, track_ids = load_mel_cache()
    le, y, idx_train, idx_val, idx_test = make_split(mel_labels)
    segments, seg_labels, _ = load_segments()

    assert np.array_equal(mel_labels, seg_labels), (
        "Segment cache label mismatch — re-run extract_mel_segments.py."
    )

    print_section("A CNN + Transformer")
    print(f"Segments: {segments.shape}  split: train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}")

    result = run(segments, y, le, idx_train, idx_val, idx_test)
    finalize_experiment([result], OUT_DIR, le.classes_, "A CNN Transformer", track_ids=track_ids)


if __name__ == "__main__":
    main()
