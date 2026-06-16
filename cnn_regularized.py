"""
Anti-overfitting CNN for music genre classification.
Addresses the overfitting observed in cnn_classifier.py (low train loss, low val acc).

Techniques applied:
  1. Smaller model        — fewer parameters → less memorisation
  2. Spatial dropout      — Dropout2d after every conv block
  3. Strong SpecAugment   — larger time/freq masks
  4. Mixup augmentation   — blends two samples + their labels
  5. Label smoothing      — prevents over-confident predictions
  6. Early stopping       — stops when val accuracy plateaus

Loads mel specs from features/mel_specs.npz (already built by cnn_classifier.py).
Results saved to results/ for comparison with previous runs.
"""

import random
import warnings
import numpy as np
import pandas as pd
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

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent
FEATURES_DIR = ROOT / "features"
RESULTS_DIR  = ROOT / "results"
MEL_CACHE    = FEATURES_DIR / "mel_specs.npz"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
BATCH_SIZE   = 32
EPOCHS       = 60
LR           = 5e-4
WEIGHT_DECAY = 1e-3     # stronger than before (was 1e-4)
PATIENCE     = 12       # early stopping patience
RANDOM_STATE = 42
TEST_SIZE    = 0.20

# SpecAugment — larger masks than cnn_classifier.py
SPEC_T       = 40       # max time mask width  (was 25)
SPEC_F       = 25       # max freq mask width  (was 15)
N_TIME_MASKS = 2
N_FREQ_MASKS = 2

# Mixup
MIXUP_ALPHA  = 0.3


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


# ── Data loading ──────────────────────────────────────────────────────────────
def load_mel_cache():
    assert MEL_CACHE.exists(), (
        "features/mel_specs.npz not found — run cnn_classifier.py first."
    )
    data = np.load(MEL_CACHE, allow_pickle=True)
    return data["mels"].astype(np.float32), data["labels"]


# ── Augmentation ──────────────────────────────────────────────────────────────
def spec_augment(mel: torch.Tensor) -> torch.Tensor:
    mel = mel.clone()
    _, n_mels, n_frames = mel.shape
    for _ in range(N_TIME_MASKS):
        t  = random.randint(0, min(SPEC_T, n_frames))
        t0 = random.randint(0, max(0, n_frames - t))
        mel[:, :, t0:t0 + t] = 0.0
    for _ in range(N_FREQ_MASKS):
        f  = random.randint(0, min(SPEC_F, n_mels))
        f0 = random.randint(0, max(0, n_mels - f))
        mel[:, f0:f0 + f, :] = 0.0
    return mel


def mixup_batch(X: torch.Tensor, y: torch.Tensor, alpha: float = MIXUP_ALPHA):
    """Mix two random samples: X_mix = λ·X + (1-λ)·X_shuffled."""
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(X.size(0), device=X.device)
    X_mix = lam * X + (1 - lam) * X[idx]
    return X_mix, y, y[idx], lam


def mixup_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ── Dataset ───────────────────────────────────────────────────────────────────
class MelDataset(Dataset):
    def __init__(self, mels: np.ndarray, labels: np.ndarray, augment: bool = False):
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


# ── Smaller, regularised CNN ──────────────────────────────────────────────────
class RegularisedCNN(nn.Module):
    """
    Lighter than GenreCNN (3 blocks instead of 4, fewer channels).
    Dropout2d after each block applies spatial dropout — drops entire
    feature maps rather than individual neurons, which is more effective
    for convolutional layers.
    Input: (B, 1, 128, 128) → Output: (B, n_classes)
    """
    def __init__(self, n_classes: int = 8):
        super().__init__()
        self.block1 = self._conv_block(1,   32)
        self.drop1  = nn.Dropout2d(0.1)

        self.block2 = self._conv_block(32,  64)
        self.drop2  = nn.Dropout2d(0.2)

        self.block3 = self._conv_block(64,  128)
        self.drop3  = nn.Dropout2d(0.3)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.6),                # strong dropout before FC
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, n_classes),
        )

    @staticmethod
    def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.block1(x))
        x = self.drop2(self.block2(x))
        x = self.drop3(self.block3(x))
        x = self.pool(x).flatten(1)
        return self.classifier(x)


# ── Training loop ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, use_mixup: bool):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        if use_mixup:
            X, y_a, y_b, lam = mixup_batch(X, y)
            optimizer.zero_grad()
            out  = model(X)
            loss = mixup_loss(criterion, out, y_a, y_b, lam)
            # Accuracy tracked on original labels
            correct += (out.argmax(1) == y_a).sum().item()
        else:
            optimizer.zero_grad()
            out  = model(X)
            loss = criterion(out, y)
            correct += (out.argmax(1) == y).sum().item()

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
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


def run(train_ds, test_ds, label: str, le, use_mixup: bool) -> dict:
    model     = RegularisedCNN(n_classes=len(le.classes_)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    # Label smoothing reduces over-confidence on training set
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    best_acc = 0.0
    best_pred, best_true = None, None
    epochs_no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, criterion, use_mixup
        )
        y_true, y_pred = predict(model, test_loader)
        val_acc        = accuracy_score(y_true, y_pred)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        scheduler.step()

        if val_acc > best_acc:
            best_acc   = val_acc
            best_pred  = y_pred
            best_true  = y_true
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  "
                  f"loss={tr_loss:.4f}  train={tr_acc:.4f}  val={val_acc:.4f}"
                  + ("  ← best" if epochs_no_improve == 0 else ""))

        if epochs_no_improve >= PATIENCE:
            print(f"  Early stop at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

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
def plot_overfitting_comparison(results: list):
    """Side-by-side: train acc vs val acc — shows how much overfitting shrank."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for r in results:
        h = r["history"]
        axes[0].plot(h["train_loss"], label=r["label"])
        axes[1].plot(h["val_acc"],    label=r["label"])
        # Show train acc as dashed to highlight train/val gap
        axes[1].plot(h["train_acc"], linestyle="--",
                     alpha=0.4, color=axes[1].get_lines()[-1].get_color())

    axes[0].set_title("Training loss  (lower = memorising more)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].set_title("Val accuracy (solid) vs Train accuracy (dashed)\n"
                      "Smaller gap = less overfitting")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "regularised_training.png", dpi=150)
    plt.close()


def plot_confusion_matrix(y_te, y_pred, class_names, title: str, path: Path):
    cm      = confusion_matrix(y_te, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.4, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_full_comparison(reg_results: list):
    """Load all previous results and add the regularised models."""
    rows = []

    # Previous CNN results
    cnn_csv = RESULTS_DIR / "full_comparison.csv"
    if cnn_csv.exists():
        prev = pd.read_csv(cnn_csv)
        rows.append(prev)

    # New regularised results
    rows.append(pd.DataFrame([
        {"label": r["label"], "acc": r["acc"], "f1": r["f1"]}
        for r in reg_results
    ]))

    all_df = pd.concat(rows, ignore_index=True).drop_duplicates("label")

    n_prev   = len(all_df) - len(reg_results)
    colors   = (["#5B8DB8"] * n_prev +
                ["#9B59B6", "#E74C3C"])   # purple, red for new models

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, col in zip(axes, ["acc", "f1"]):
        ylabel = "Accuracy" if col == "acc" else "F1-macro"
        bars   = ax.bar(all_df["label"], all_df[col],
                        color=colors[:len(all_df)], edgecolor="white")
        ax.set_ylim(0, 1.0)
        ax.set_title(ylabel, fontsize=13)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)
        for bar, val in zip(bars, all_df[col]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + 0.01, f"{val:.3f}", ha="center", fontsize=8)

    fig.suptitle(
        "All models — RF  →  CNN  →  CNN+SpecAug  →  Regularised CNN",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "all_models_comparison.png", dpi=150)
    plt.close()

    all_df.to_csv(RESULTS_DIR / "all_models_comparison.csv", index=False)
    print("\n=== All models comparison ===")
    print(all_df.to_string(index=False))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    mels, labels = load_mel_cache()
    print(f"Loaded mel specs: {mels.shape}")

    le = LabelEncoder()
    y  = le.fit_transform(labels)

    idx            = np.arange(len(y))
    idx_tr, idx_te = train_test_split(idx, test_size=TEST_SIZE,
                                      random_state=RANDOM_STATE, stratify=y)
    mels_tr, mels_te = mels[idx_tr], mels[idx_te]
    y_tr,    y_te    = y[idx_tr],    y[idx_te]

    test_ds = MelDataset(mels_te, y_te, augment=False)

    # Run 1: smaller model + dropout + SpecAugment + label smoothing
    print("\n" + "=" * 60)
    print("  Run 1 — Regularised CNN  (no Mixup)")
    print("  Techniques: smaller model, spatial dropout, strong SpecAugment,")
    print("              label smoothing, early stopping")
    print("=" * 60)
    res_reg = run(
        MelDataset(mels_tr, y_tr, augment=True),
        test_ds,
        "Reg. CNN (no Mixup)",
        le,
        use_mixup=False,
    )

    # Run 2: same + Mixup
    print("\n" + "=" * 60)
    print("  Run 2 — Regularised CNN  (+ Mixup)")
    print("  Same as Run 1 + Mixup data augmentation")
    print("=" * 60)
    res_mix = run(
        MelDataset(mels_tr, y_tr, augment=True),
        test_ds,
        "Reg. CNN + Mixup",
        le,
        use_mixup=True,
    )

    reg_results = [res_reg, res_mix]

    plot_overfitting_comparison(reg_results)

    best = max(reg_results, key=lambda r: r["f1"])
    plot_confusion_matrix(
        best["y_te"], best["y_pred"], le.classes_,
        f"Confusion matrix — {best['label']}",
        RESULTS_DIR / "confusion_matrix_regularised.png",
    )

    plot_full_comparison(reg_results)

    print("\nSaved to results/:")
    print("  regularised_training.png     (train loss + val/train gap)")
    print("  confusion_matrix_regularised.png")
    print("  all_models_comparison.png    (all 5 models side by side)")
    print("  all_models_comparison.csv")


if __name__ == "__main__":
    main()
