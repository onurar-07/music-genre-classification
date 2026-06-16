"""
Balanced regularisation CNN — middle ground between cnn_classifier.py (overfits)
and cnn_regularized.py (too aggressive, train acc < val acc).

Changes vs cnn_regularized.py:
  - LR back to 1e-3       (was 5e-4 — too slow to learn)
  - Weight decay 1e-4     (was 1e-3 — too strong)
  - Dropout2d 0.05/0.1/0.15  (was 0.1/0.2/0.3 — too aggressive)
  - Final dropout 0.5/0.3    (was 0.6/0.4)
  - SpecAugment T=30, F=20   (was T=40, F=25 — slightly toned down)

Goal: train acc ≈ val acc (small gap), both increasing over epochs.
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
LR           = 1e-3     # original LR — learns fast enough
WEIGHT_DECAY = 1e-4     # mild weight decay
PATIENCE     = 12
RANDOM_STATE = 42
TEST_SIZE    = 0.20

SPEC_T       = 30       # slightly toned-down SpecAugment
SPEC_F       = 20
N_TIME_MASKS = 2
N_FREQ_MASKS = 2

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


# ── Data ──────────────────────────────────────────────────────────────────────
def load_mel_cache():
    assert MEL_CACHE.exists(), "Run cnn_classifier.py first to build mel cache."
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


def mixup_batch(X, y, alpha=MIXUP_ALPHA):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(X.size(0), device=X.device)
    return lam * X + (1 - lam) * X[idx], y, y[idx], lam


def mixup_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ── Dataset ───────────────────────────────────────────────────────────────────
class MelDataset(Dataset):
    def __init__(self, mels, labels, augment=False):
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


# ── Balanced CNN model ────────────────────────────────────────────────────────
class BalancedCNN(nn.Module):
    """
    Same 3-block architecture as RegularisedCNN but with lighter dropout:
      Dropout2d: 0.05 / 0.10 / 0.15  (was 0.1 / 0.2 / 0.3)
      Final FC dropout: 0.5 / 0.3    (was 0.6 / 0.4)
    Gives the model enough capacity to learn while still preventing overfitting.
    """
    def __init__(self, n_classes=8):
        super().__init__()
        self.block1 = self._conv_block(1,   32)
        self.drop1  = nn.Dropout2d(0.05)

        self.block2 = self._conv_block(32,  64)
        self.drop2  = nn.Dropout2d(0.10)

        self.block3 = self._conv_block(64,  128)
        self.drop3  = nn.Dropout2d(0.15)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    @staticmethod
    def _conv_block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        x = self.drop1(self.block1(x))
        x = self.drop2(self.block2(x))
        x = self.drop3(self.block3(x))
        x = self.pool(x).flatten(1)
        return self.classifier(x)


# ── Training ──────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, use_mixup):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        if use_mixup:
            X, y_a, y_b, lam = mixup_batch(X, y)
            optimizer.zero_grad()
            out  = model(X)
            loss = mixup_loss(criterion, out, y_a, y_b, lam)
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


def run(train_ds, test_ds, label, le, use_mixup):
    model     = BalancedCNN(n_classes=len(le.classes_)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    best_acc, best_pred, best_true = 0.0, None, None
    no_improve = 0
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
            best_acc, best_pred, best_true = val_acc, y_pred, y_true
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 5 == 0:
            gap = tr_acc - val_acc
            print(f"  Epoch {epoch:3d}/{EPOCHS}  "
                  f"loss={tr_loss:.4f}  train={tr_acc:.4f}  val={val_acc:.4f}  "
                  f"gap={gap:+.4f}")

        if no_improve >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
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
def plot_training(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for r in results:
        h = r["history"]
        c = axes[1].plot(h["val_acc"], label=r["label"])[0].get_color()
        axes[0].plot(h["train_loss"], label=r["label"], color=c)
        axes[1].plot(h["train_acc"], linestyle="--", alpha=0.4, color=c)

    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].set_title("Val accuracy (solid) vs Train accuracy (dashed)\n"
                      "Smaller gap = better balance")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "balanced_training.png", dpi=150)
    plt.close()


def plot_confusion_matrix(y_te, y_pred, class_names, title, path):
    cm      = confusion_matrix(y_te, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.4, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_all_models(balanced_results):
    """Combine all previous results + balanced results into one final chart."""
    rows = []
    for csv in ["full_comparison.csv", "all_models_comparison.csv"]:
        p = RESULTS_DIR / csv
        if p.exists():
            rows.append(pd.read_csv(p))

    rows.append(pd.DataFrame([
        {"label": r["label"], "acc": r["acc"], "f1": r["f1"]}
        for r in balanced_results
    ]))

    all_df = pd.concat(rows, ignore_index=True).drop_duplicates("label")

    n_prev = len(all_df) - len(balanced_results)
    colors = ["#5B8DB8"] * n_prev + ["#F39C12", "#27AE60"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
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

    fig.suptitle("All models compared", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "all_models_final.png", dpi=150)
    plt.close()

    all_df.to_csv(RESULTS_DIR / "all_models_final.csv", index=False)
    print("\n=== All models ===")
    print(all_df.to_string(index=False))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    mels, labels = load_mel_cache()
    print(f"Loaded: {mels.shape}")

    le = LabelEncoder()
    y  = le.fit_transform(labels)

    idx            = np.arange(len(y))
    idx_tr, idx_te = train_test_split(idx, test_size=TEST_SIZE,
                                      random_state=RANDOM_STATE, stratify=y)
    mels_tr, mels_te = mels[idx_tr], mels[idx_te]
    y_tr,    y_te    = y[idx_tr],    y[idx_te]
    test_ds = MelDataset(mels_te, y_te, augment=False)

    print("\n" + "=" * 60)
    print("  Run 1 — Balanced CNN  (SpecAugment only)")
    print("=" * 60)
    res1 = run(MelDataset(mels_tr, y_tr, augment=True),
               test_ds, "Balanced CNN", le, use_mixup=False)

    print("\n" + "=" * 60)
    print("  Run 2 — Balanced CNN  (SpecAugment + Mixup)")
    print("=" * 60)
    res2 = run(MelDataset(mels_tr, y_tr, augment=True),
               test_ds, "Balanced CNN + Mixup", le, use_mixup=True)

    results = [res1, res2]

    plot_training(results)

    best = max(results, key=lambda r: r["f1"])
    plot_confusion_matrix(
        best["y_te"], best["y_pred"], le.classes_,
        f"Confusion matrix — {best['label']}",
        RESULTS_DIR / "confusion_matrix_balanced.png",
    )

    plot_all_models(results)

    print("\nSaved to results/:")
    print("  balanced_training.png")
    print("  confusion_matrix_balanced.png")
    print("  all_models_final.png      ← all models side by side")
    print("  all_models_final.csv")


if __name__ == "__main__":
    main()
