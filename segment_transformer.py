"""Part 2.6: CNN backbone + Transformer encoder across track segments.

Instead of averaging segment probabilities independently, a small Transformer
encoder learns cross-segment context before classifying the full track.

Architecture:
    [seg_0, seg_1, seg_2, seg_3]
        ↓ shared Multi-shape CNN backbone without classifier head
    [emb_0, emb_1, emb_2, emb_3]   (128-dim each)
        ↓ prepend learnable CLS token + positional embeddings
    TransformerEncoder (2 layers, 4 heads, dim=128)
        ↓ CLS output
    Dropout → Linear → 8 genres

Reads:  features/mel_segments.npz  (run extract_mel_segments.py first)
Saves:  results/2.6 Segment Transformer/
"""

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

from cnn_training_utils import (
    DEVICE,
    AugmentConfig,
    MultiShapeCNN,
    TrainConfig,
    clone_state_dict,
    count_parameters,
    current_lr,
    finalize_experiment,
    make_optimizer,
    make_scheduler,
    make_split,
    mixup_batch,
    mixup_loss,
    seed_everything,
    spec_augment,
)
from reporting_utils import RESULTS_ROOT, ROOT, experiment_dir, print_section

OUT_DIR = experiment_dir("2.6 Segment Transformer")
SEGMENT_CACHE = ROOT / "features" / "mel_segments.npz"
SEGMENT_AVERAGING_SELECTION = RESULTS_ROOT / "2.5 Segment Averaging" / "selected_model.txt"


def strip_source(label):
    return str(label).split(" - no augmentation")[0]


def selected_base_and_augment(label):
    base = strip_source(label)
    specaugment = "SpecAugment" in base
    mixup = "Mixup" in base
    for suffix in [" - SpecAugment + Mixup", " - SpecAugment", " - Mixup"]:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    if base != "Multi-shape CNN":
        raise ValueError(
            "Segment Transformer currently uses the Multi-shape CNN backbone. "
            f"Selected CNN branch {label!r} is not supported."
        )
    return base, AugmentConfig(
        specaugment=specaugment,
        mixup=mixup,
        spec_t=30,
        spec_f=20,
        mixup_alpha=0.3,
    )


def selected_from_segment_averaging():
    if not SEGMENT_AVERAGING_SELECTION.exists():
        return None
    for line in SEGMENT_AVERAGING_SELECTION.read_text(encoding="utf-8").splitlines():
        if line.startswith("Selected model:"):
            return line.split(":", 1)[1].strip()
    return None


def fallback_selected_cnn():
    candidate_paths = [
        RESULTS_ROOT / "2.4 Augmentation ablation" / "augmentation_comparison.csv",
        RESULTS_ROOT / "2.3 Multi-shape CNN" / "metrics.csv",
    ]
    rows = []
    for path in candidate_paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        val_df = df[df["split"] == "val"].copy()
        val_df["source"] = path.parent.name
        rows.append(val_df)
    if not rows:
        raise FileNotFoundError(
            "Run segment_averaging.py or the Multi-shape/Augmentation experiments "
            "before segment_transformer.py."
        )
    candidates = pd.concat(rows, ignore_index=True)
    supported = []
    for label in candidates["model"]:
        try:
            selected_base_and_augment(label)
            supported.append(True)
        except ValueError:
            supported.append(False)
    candidates = candidates[supported].sort_values("f1_macro", ascending=False)
    if candidates.empty:
        raise ValueError("No supported Multi-shape CNN candidate found for Segment Transformer.")
    return candidates.iloc[0]["model"]


def select_cnn_branch():
    selected = selected_from_segment_averaging() or fallback_selected_cnn()
    _base, augment_cfg = selected_base_and_augment(selected)
    train_cfg = TrainConfig(
        epochs=300,
        lr=1e-3,
        weight_decay=1e-4,
        patience=48,
        batch_size=128,
        label_smoothing=0.1,
    )
    (OUT_DIR / "selected_model.txt").write_text(
        "\n".join(
            [
                "CNN branch selected for Segment Transformer",
                "",
                f"Selected model: {selected}",
                f"SpecAugment: {augment_cfg.specaugment}",
                f"Mixup: {augment_cfg.mixup}",
            ]
        ),
        encoding="utf-8",
    )
    return selected, train_cfg, augment_cfg


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

class MultiShapeBackbone(nn.Module):
    """Multi-shape CNN up to the global average pool, without classifier."""

    EMBED_DIM = 128

    def __init__(self):
        super().__init__()
        cnn = MultiShapeCNN(
            n_classes=8,
            branch_width=8,
            block_dropouts=(0.05, 0.10, 0.15),
            classifier_dropout=0.45,
        )
        self.local_branch = cnn.local_branch
        self.timbre_branch = cnn.timbre_branch
        self.temporal_branch = cnn.temporal_branch
        self.block1 = cnn.block1
        self.drop1 = cnn.drop1
        self.block2 = cnn.block2
        self.drop2 = cnn.drop2
        self.block3 = cnn.block3
        self.drop3 = cnn.drop3
        self.pool = cnn.pool

    def forward(self, x):          # x: (B, 1, H, W)
        height, width = x.shape[-2:]
        x = torch.cat(
            [
                self.local_branch(x)[..., :height, :width],
                self.timbre_branch(x)[..., :height, :width],
                self.temporal_branch(x)[..., :height, :width],
            ],
            dim=1,
        )
        x = self.drop1(self.block1(x))
        x = self.drop2(self.block2(x))
        x = self.drop3(self.block3(x))
        return self.pool(x).flatten(1)   # (B, 128)


class SegmentTransformer(nn.Module):
    """Multi-shape CNN per segment + Transformer encoder across segments."""

    def __init__(self, n_classes=8, n_segments=4, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.backbone = MultiShapeBackbone()
        d = MultiShapeBackbone.EMBED_DIM

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_segments + 1, d))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=d * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.classifier = nn.Sequential(
            nn.Dropout(0.45),
            nn.Linear(d, n_classes),
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, segs):       # segs: (B, N, 1, H, W)
        B, N, C, H, W = segs.shape
        embeds = self.backbone(segs.reshape(B * N, C, H, W)).reshape(B, N, -1)  # (B, N, d)
        cls = self.cls_token.expand(B, -1, -1)                              # (B, 1, d)
        tokens = torch.cat([cls, embeds], dim=1)                           # (B, N+1, d)
        tokens = tokens + self.pos_embed[:, : N + 1]
        out = self.transformer(tokens)                                      # (B, N+1, d)
        return self.classifier(out[:, 0])                                  # CLS token


CNNSegmentTransformer = SegmentTransformer


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, augment_cfg, use_amp, scaler):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for segs, labels in loader:
        segs   = segs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            if augment_cfg.mixup:
                segs, labels_a, labels_b, lam = mixup_batch(segs, labels, augment_cfg.mixup_alpha)
                out = model(segs)
                loss = mixup_loss(criterion, out, labels_a, labels_b, lam)
                correct += (out.argmax(1) == labels_a).sum().item()
            else:
                out = model(segs)
                loss = criterion(out, labels)
                correct += (out.argmax(1) == labels).sum().item()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
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


def run(segments, y, le, idx_train, idx_val, idx_test, selected_cnn, cfg, augment_cfg):
    label = f"{selected_cnn} - Segment Transformer"
    seed_everything()
    model = SegmentTransformer(n_classes=len(le.classes_), n_segments=segments.shape[1]).to(DEVICE)
    param_count, trainable_count = count_parameters(model)
    print(f"\n{label}: parameters={param_count:,} (trainable={trainable_count:,})")

    use_amp   = cfg.amp and DEVICE.type == "cuda"
    scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)
    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

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
        f"\n{label}: best_epoch={best_epoch}  val_f1={best_val_f1:.4f}  "
        f"test_acc={test_acc:.4f}  test_f1={test_f1:.4f}  runtime={training_seconds:.1f}s",
        flush=True,
    )
    return {
        "label": label,
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

    segments, seg_labels, track_ids = load_segments()
    le, y, idx_train, idx_val, idx_test = make_split(seg_labels)
    selected_cnn, train_cfg, augment_cfg = select_cnn_branch()

    print_section("2.6 Segment Transformer")
    print(f"Selected CNN branch: {selected_cnn}")
    print(f"Augmentation: specaugment={augment_cfg.specaugment} mixup={augment_cfg.mixup}")
    print(f"Segments: {segments.shape}  split: train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}")

    result = run(segments, y, le, idx_train, idx_val, idx_test, selected_cnn, train_cfg, augment_cfg)
    finalize_experiment([result], OUT_DIR, le.classes_, "2.6 Segment Transformer", track_ids=track_ids)


if __name__ == "__main__":
    main()
