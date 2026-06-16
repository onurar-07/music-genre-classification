"""
CNN music genre classifier on mel spectrograms.
Run AFTER extract_features.py (needs features/features.npz for track IDs + labels).

Phases:
  1. Extract mel spectrograms and cache to features/mel_specs.npz  (once, ~30 min)
  2. Train CNN — no augmentation
  3. Train CNN — with SpecAugment (time/frequency masking)
  4. Full comparison plot: RF/MLP baseline vs CNN vs CNN+SpecAugment
"""

import random
import warnings
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix,
)
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import zoom

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent
FEATURES_DIR = ROOT / "features"
RESULTS_DIR  = ROOT / "results"
MEL_CACHE    = FEATURES_DIR / "mel_specs.npz"
FMA_AUDIO    = ROOT / "data" / "fma_small"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
SR           = 22050
DURATION     = 29.0
N_MELS       = 128
HOP_LEN      = 512
MEL_W        = 128    # resize time axis → (128 × 128) input to CNN

BATCH_SIZE   = 32
EPOCHS       = 40
LR           = 1e-3
WEIGHT_DECAY = 1e-4
RANDOM_STATE = 42
TEST_SIZE    = 0.20


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except AttributeError:
        pass
    return torch.device("cpu")


DEVICE = get_device()
print(f"Using device: {DEVICE}")


# ── Mel spectrogram extraction ────────────────────────────────────────────────
def audio_path(track_id: int) -> Path:
    tid = f"{track_id:06d}"
    return FMA_AUDIO / tid[:3] / f"{tid}.mp3"


def extract_mel(y: np.ndarray, sr: int) -> np.ndarray:
    """Log-mel spectrogram, per-sample normalised, resized to (N_MELS, MEL_W)."""
    mel    = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LEN, fmax=sr // 2
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Resize time axis to fixed width
    if mel_db.shape[1] != MEL_W:
        scale  = (N_MELS / mel_db.shape[0], MEL_W / mel_db.shape[1])
        mel_db = zoom(mel_db, scale, order=1)
    # Per-sample normalisation
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    return mel_db.astype(np.float32)


def build_mel_cache():
    """Extract mel spectrograms for all tracks in features.npz and cache them."""
    data      = np.load(FEATURES_DIR / "features.npz", allow_pickle=True)
    track_ids = data["track_ids"]
    labels    = data["labels"]

    print(f"Extracting mel spectrograms for {len(track_ids)} tracks…")
    mels, valid_idx = [], []

    for i, tid in enumerate(tqdm(track_ids, desc="Mel extraction")):
        path = audio_path(int(tid))
        if not path.exists():
            continue
        try:
            y, sr = librosa.load(path, sr=SR, duration=DURATION, mono=True)
            mels.append(extract_mel(y, sr))
            valid_idx.append(i)
        except Exception as exc:
            tqdm.write(f"  skip {tid}: {exc}")

    mels   = np.stack(mels)
    labels = labels[valid_idx]

    np.savez_compressed(
        MEL_CACHE,
        mels   = mels.astype(np.float16),   # float16 keeps file ~250 MB
        labels = labels,
    )
    print(f"Saved {len(mels)} mel specs → features/mel_specs.npz")
    return mels.astype(np.float32), labels


def load_mel_cache():
    data = np.load(MEL_CACHE, allow_pickle=True)
    return data["mels"].astype(np.float32), data["labels"]


# ── Dataset ───────────────────────────────────────────────────────────────────
class MelDataset(Dataset):
    def __init__(self, mels: np.ndarray, labels: np.ndarray, augment: bool = False):
        # Add channel dimension: (N, 1, H, W)
        self.mels    = torch.tensor(mels[:, None, :, :], dtype=torch.float32)
        self.labels  = torch.tensor(labels, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mel = self.mels[idx]
        if self.augment:
            mel = spec_augment(mel)
        return mel, self.labels[idx]


def spec_augment(mel: torch.Tensor,
                 num_time_masks: int = 2, T: int = 25,
                 num_freq_masks: int = 2, F: int = 15) -> torch.Tensor:
    """SpecAugment: randomly zero-out time and frequency strips."""
    mel = mel.clone()
    _, n_mels, n_frames = mel.shape
    for _ in range(num_time_masks):
        t  = random.randint(0, min(T, n_frames))
        t0 = random.randint(0, max(0, n_frames - t))
        mel[:, :, t0:t0 + t] = 0.0
    for _ in range(num_freq_masks):
        f  = random.randint(0, min(F, n_mels))
        f0 = random.randint(0, max(0, n_mels - f))
        mel[:, f0:f0 + f, :] = 0.0
    return mel


# ── CNN model ─────────────────────────────────────────────────────────────────
class GenreCNN(nn.Module):
    """
    Four convolutional blocks with batch norm, each followed by 2×2 max-pool.
    Global average pooling collapses spatial dims before the classifier head.
    Input: (B, 1, 128, 128)  →  Output: (B, n_classes)
    """
    def __init__(self, n_classes: int = 8):
        super().__init__()
        self.features = nn.Sequential(
            self._block(1,   32),   # → (B,  32, 64, 64)
            self._block(32,  64),   # → (B,  64, 32, 32)
            self._block(64,  128),  # → (B, 128, 16, 16)
            self._block(128, 256),  # → (B, 256,  8,  8)
        )
        self.pool       = nn.AdaptiveAvgPool2d(1)   # → (B, 256, 1, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, n_classes),
        )

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


# ── Training helpers ──────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out  = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (out.argmax(1) == y).sum().item()
        n          += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def predict(model, loader):
    model.eval()
    all_pred, all_true = [], []
    for X, y in loader:
        pred = model(X.to(DEVICE)).argmax(1).cpu().numpy()
        all_pred.extend(pred)
        all_true.extend(y.numpy())
    return np.array(all_true), np.array(all_pred)


def run_cnn(train_ds, test_ds, label: str, le) -> dict:
    model     = GenreCNN(n_classes=len(le.classes_)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    best_acc, best_pred, best_true = 0.0, None, None
    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        y_true, y_pred  = predict(model, test_loader)
        val_acc         = accuracy_score(y_true, y_pred)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        scheduler.step()

        if val_acc > best_acc:
            best_acc, best_pred, best_true = val_acc, y_pred, y_true

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  "
                  f"loss={tr_loss:.4f}  train={tr_acc:.4f}  val={val_acc:.4f}")

    f1 = f1_score(best_true, best_pred, average="macro")
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"  Best accuracy : {best_acc:.4f}   F1-macro : {f1:.4f}")
    print(classification_report(best_true, best_pred,
                                 target_names=le.classes_, zero_division=0))
    return {
        "label":   label,
        "acc":     best_acc,
        "f1":      f1,
        "y_te":    best_true,
        "y_pred":  best_pred,
        "history": history,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_training_history(results_list: list):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for r in results_list:
        h = r["history"]
        axes[0].plot(h["train_loss"], label=r["label"])
        axes[1].plot(h["val_acc"],    label=r["label"])
    for ax, title, ylabel in zip(
        axes,
        ["Training loss", "Validation accuracy"],
        ["Cross-entropy loss", "Accuracy"],
    ):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "cnn_training_history.png", dpi=150)
    plt.close()


def plot_confusion_matrix(y_te, y_pred, class_names, title: str, path: Path):
    cm      = confusion_matrix(y_te, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.4, ax=ax, cbar=True)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_full_comparison(cnn_results: list):
    """Merge CNN results with the RF/MLP baseline from train_evaluate.py."""
    baseline_csv = RESULTS_DIR / "results_summary.csv"
    if baseline_csv.exists():
        base = pd.read_csv(baseline_csv)
        # Pick the single best RF and single best MLP row
        best_rf  = (base[base["Configuration"].str.startswith("RF")]
                    .sort_values("F1-macro", ascending=False).head(1))
        best_mlp = (base[base["Configuration"].str.startswith("MLP")]
                    .sort_values("F1-macro", ascending=False).head(1))
        prior = (pd.concat([best_rf, best_mlp])
                   .rename(columns={"Configuration": "label",
                                    "Accuracy": "acc", "F1-macro": "f1"})
                   [["label", "acc", "f1"]])
    else:
        prior = pd.DataFrame(columns=["label", "acc", "f1"])

    cnn_df = pd.DataFrame([{"label": r["label"], "acc": r["acc"], "f1": r["f1"]}
                            for r in cnn_results])
    all_df = pd.concat([prior, cnn_df], ignore_index=True)

    n_base = len(prior)
    colors = (["#5B8DB8"] * n_base +
              ["#E07B54",          # CNN no aug
               "#2E9E6B"])         # CNN + SpecAugment

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col in zip(axes, ["acc", "f1"]):
        ylabel = "Accuracy" if col == "acc" else "F1-macro"
        bars = ax.bar(all_df["label"], all_df[col],
                      color=colors[:len(all_df)], edgecolor="white")
        ax.set_ylim(0, 1.0)
        ax.set_title(ylabel, fontsize=13)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)
        for bar, val in zip(bars, all_df[col]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + 0.01, f"{val:.3f}", ha="center", fontsize=9)

    fig.suptitle("Full comparison: Baseline  →  CNN  →  CNN + SpecAugment",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "full_comparison.png", dpi=150)
    plt.close()

    all_df.to_csv(RESULTS_DIR / "full_comparison.csv", index=False)
    print("\n=== Full comparison ===")
    print(all_df.to_string(index=False))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # 1. Load or build mel spectrogram cache
    if MEL_CACHE.exists():
        print("Loading cached mel spectrograms…")
        mels, labels = load_mel_cache()
    else:
        mels, labels = build_mel_cache()

    print(f"Mel specs shape : {mels.shape}   dtype : {mels.dtype}")

    le = LabelEncoder()
    y  = le.fit_transform(labels)

    idx            = np.arange(len(y))
    idx_tr, idx_te = train_test_split(idx, test_size=TEST_SIZE,
                                      random_state=RANDOM_STATE, stratify=y)
    mels_tr, mels_te = mels[idx_tr], mels[idx_te]
    y_tr,    y_te    = y[idx_tr],    y[idx_te]

    test_ds = MelDataset(mels_te, y_te, augment=False)

    # Phase 1 — CNN, no augmentation
    print("\n" + "=" * 60)
    print("  Phase 1 — CNN on mel spectrograms  (no augmentation)")
    print("=" * 60)
    res_cnn = run_cnn(
        MelDataset(mels_tr, y_tr, augment=False),
        test_ds, "CNN – Mel (no aug)", le,
    )

    # Phase 2 — CNN + SpecAugment
    print("\n" + "=" * 60)
    print("  Phase 2 — CNN on mel spectrograms  (+ SpecAugment)")
    print("=" * 60)
    res_aug = run_cnn(
        MelDataset(mels_tr, y_tr, augment=True),
        test_ds, "CNN – Mel + SpecAugment", le,
    )

    cnn_results = [res_cnn, res_aug]

    # Training curves
    plot_training_history(cnn_results)

    # Confusion matrix for best CNN model
    best = max(cnn_results, key=lambda r: r["f1"])
    plot_confusion_matrix(
        best["y_te"], best["y_pred"], le.classes_,
        f"Confusion matrix — {best['label']}",
        RESULTS_DIR / "confusion_matrix_cnn.png",
    )

    # Full comparison vs RF/MLP baseline
    plot_full_comparison(cnn_results)

    print("\nSaved to results/:")
    print("  cnn_training_history.png")
    print("  confusion_matrix_cnn.png")
    print("  full_comparison.png")
    print("  full_comparison.csv")


if __name__ == "__main__":
    main()
