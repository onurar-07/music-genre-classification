"""
Balanced regularisation CNN experiments for FMA-small genre classification.

Uses the shared train/validation/test split. Validation F1-macro drives early
stopping and best-epoch selection; final metrics are reported on test.
"""

import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

from experiment_utils import (
    ROOT,
    compute_scores,
    experiment_dir,
    plot_confusion_matrix,
    plot_metrics,
    plot_training_history,
    save_history,
    save_metrics,
    split_indices,
    update_global_comparison,
    write_classification_report,
)

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

MEL_CACHE = ROOT / "features" / "mel_specs.npz"
OUT_DIR = experiment_dir("Balanced CNN")

BATCH_SIZE = 32
EPOCHS = 60
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 12

SPEC_T = 30
SPEC_F = 20
N_TIME_MASKS = 2
N_FREQ_MASKS = 2
MIXUP_ALPHA = 0.3


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
print(f"Using device: {DEVICE}", flush=True)


def load_mel_cache():
    assert MEL_CACHE.exists(), "Run extract_mel_specs.py first to build features/mel_specs.npz."
    data = np.load(MEL_CACHE, allow_pickle=True)
    return data["mels"].astype(np.float32), data["labels"]


def spec_augment(mel):
    mel = mel.clone()
    _, n_mels, n_frames = mel.shape
    for _ in range(N_TIME_MASKS):
        t = random.randint(0, min(SPEC_T, n_frames))
        t0 = random.randint(0, max(0, n_frames - t))
        mel[:, :, t0:t0 + t] = 0.0
    for _ in range(N_FREQ_MASKS):
        f = random.randint(0, min(SPEC_F, n_mels))
        f0 = random.randint(0, max(0, n_mels - f))
        mel[:, f0:f0 + f, :] = 0.0
    return mel


def mixup_batch(X, y, alpha=MIXUP_ALPHA):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(X.size(0), device=X.device)
    return lam * X + (1 - lam) * X[idx], y, y[idx], lam


def mixup_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class MelDataset(Dataset):
    def __init__(self, mels, labels, augment=False):
        self.mels = torch.tensor(mels[:, None, :, :], dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mel = self.mels[idx]
        if self.augment:
            mel = spec_augment(mel)
        return mel, self.labels[idx]


class BalancedCNN(nn.Module):
    def __init__(self, n_classes=8):
        super().__init__()
        self.block1 = self._conv_block(1, 32)
        self.drop1 = nn.Dropout2d(0.05)
        self.block2 = self._conv_block(32, 64)
        self.drop2 = nn.Dropout2d(0.10)
        self.block3 = self._conv_block(64, 128)
        self.drop3 = nn.Dropout2d(0.15)
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


def train_epoch(model, loader, optimizer, criterion, use_mixup):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        if use_mixup:
            X, y_a, y_b, lam = mixup_batch(X, y)
            out = model(X)
            loss = mixup_loss(criterion, out, y_a, y_b, lam)
            correct += (out.argmax(1) == y_a).sum().item()
        else:
            out = model(X)
            loss = criterion(out, y)
            correct += (out.argmax(1) == y).sum().item()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, n = 0.0, 0
    all_true, all_pred = [], []
    for X, y in loader:
        X, y_dev = X.to(DEVICE), y.to(DEVICE)
        out = model(X)
        loss = criterion(out, y_dev)
        pred = out.argmax(1).cpu().numpy()
        all_true.extend(y.numpy())
        all_pred.extend(pred)
        total_loss += loss.item() * len(y)
        n += len(y)
    scores = compute_scores(np.array(all_true), np.array(all_pred))
    return total_loss / n, scores["accuracy"], scores["f1_macro"], np.array(all_true), np.array(all_pred)


def clone_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def run(train_ds, val_ds, test_ds, label, le, use_mixup):
    model = BalancedCNN(n_classes=len(le.classes_)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    best_state, best_epoch, best_val_f1 = None, 0, -1.0
    best_val_true, best_val_pred = None, None
    no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, use_mixup)
        val_loss, val_acc, val_f1, val_true, val_pred = evaluate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if val_f1 > best_val_f1:
            best_state = clone_state_dict(model)
            best_epoch = epoch
            best_val_f1 = val_f1
            best_val_true, best_val_pred = val_true, val_pred
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{EPOCHS}  loss={tr_loss:.4f}  "
                f"train={tr_acc:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}",
                flush=True,
            )
        if no_improve >= PATIENCE:
            print(f"  Early stop at epoch {epoch} (best epoch {best_epoch})", flush=True)
            break

    model.load_state_dict(best_state)
    _, test_acc, test_f1, test_true, test_pred = evaluate(model, test_loader, criterion)
    print(f"\n{label}: best_epoch={best_epoch}  val_f1={best_val_f1:.4f}  test_acc={test_acc:.4f}  test_f1={test_f1:.4f}")
    return {
        "label": label,
        "val_true": best_val_true,
        "val_pred": best_val_pred,
        "test_true": test_true,
        "test_pred": test_pred,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "history": history,
    }


def main():
    mels, labels = load_mel_cache()
    print(f"Loaded mel specs: {mels.shape}")
    le = LabelEncoder()
    y = le.fit_transform(labels)
    idx_train, idx_val, idx_test = split_indices(y)
    print(f"Split sizes: train={len(idx_train)}  val={len(idx_val)}  test={len(idx_test)}")

    val_ds = MelDataset(mels[idx_val], y[idx_val], augment=False)
    test_ds = MelDataset(mels[idx_test], y[idx_test], augment=False)

    print("\n" + "=" * 60)
    print("  Run 1 - Balanced CNN (SpecAugment only)")
    print("=" * 60)
    res_plain = run(
        MelDataset(mels[idx_train], y[idx_train], augment=True),
        val_ds,
        test_ds,
        "Balanced CNN",
        le,
        use_mixup=False,
    )

    print("\n" + "=" * 60)
    print("  Run 2 - Balanced CNN (SpecAugment + Mixup)")
    print("=" * 60)
    res_mix = run(
        MelDataset(mels[idx_train], y[idx_train], augment=True),
        val_ds,
        test_ds,
        "Balanced CNN + Mixup",
        le,
        use_mixup=True,
    )

    results = [res_plain, res_mix]
    metrics_df = save_metrics(results, OUT_DIR)
    save_history(results, OUT_DIR)
    plot_training_history(results, OUT_DIR)
    plot_metrics(metrics_df, OUT_DIR, "Balanced CNN models")
    best = max(results, key=lambda r: r["best_val_f1"])
    plot_confusion_matrix(best["test_true"], best["test_pred"], le.classes_, OUT_DIR, f"Confusion matrix - {best['label']}")
    write_classification_report(best, le.classes_, OUT_DIR)
    update_global_comparison()

    print("\n=== Metrics ===")
    print(metrics_df.to_string(index=False))
    print("\nSaved to results/Balanced CNN/:")
    print("  metrics.csv")
    print("  metrics.png")
    print("  training_history.csv")
    print("  training_history.png")
    print("  classification_report.txt")
    print("  confusion_matrix.png")


if __name__ == "__main__":
    main()
