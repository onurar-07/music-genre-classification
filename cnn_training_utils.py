"""Shared CNN models, augmentations, and training loop for experiments."""

import random
import time
import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

from reporting_utils import (
    ROOT,
    compute_scores,
    plot_confusion_matrix,
    plot_metrics,
    plot_training_history,
    print_metrics_summary,
    print_saved_outputs,
    save_branch_probabilities,
    save_history,
    save_metrics,
    save_predictions,
    split_indices,
    update_global_comparison,
    write_classification_report,
)

warnings.filterwarnings("ignore")

MEL_CACHE = ROOT / "features" / "mel_specs.npz"
RANDOM_SEED = 42


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 12
    label_smoothing: float = 0.0
    optimizer: str = "adam"
    num_workers: int = 2
    amp: bool = True


@dataclass
class AugmentConfig:
    specaugment: bool = False
    mixup: bool = False
    spec_t: int = 25
    spec_f: int = 15
    time_masks: int = 2
    freq_masks: int = 2
    mixup_alpha: float = 0.3


def seed_everything(seed=RANDOM_SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


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
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass


def load_mel_cache():
    assert MEL_CACHE.exists(), "Run extract_mel_specs.py first to build features/mel_specs.npz."
    data = np.load(MEL_CACHE, allow_pickle=True)
    track_ids = data["track_ids"] if "track_ids" in data.files else None
    return data["mels"].astype(np.float32), data["labels"], track_ids


def make_split(labels):
    le = LabelEncoder()
    y = le.fit_transform(labels)
    idx_train, idx_val, idx_test = split_indices(y)
    return le, y, idx_train, idx_val, idx_test


def spec_augment(mel, cfg: AugmentConfig):
    mel = mel.clone()
    _, n_mels, n_frames = mel.shape
    for _ in range(cfg.time_masks):
        t = random.randint(0, min(cfg.spec_t, n_frames))
        t0 = random.randint(0, max(0, n_frames - t))
        mel[:, :, t0:t0 + t] = 0.0
    for _ in range(cfg.freq_masks):
        f = random.randint(0, min(cfg.spec_f, n_mels))
        f0 = random.randint(0, max(0, n_mels - f))
        mel[:, f0:f0 + f, :] = 0.0
    return mel


class MelDataset(Dataset):
    def __init__(self, mels, labels, augment_cfg=None):
        self.mels = torch.tensor(mels[:, None, :, :], dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.augment_cfg = augment_cfg or AugmentConfig()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mel = self.mels[idx]
        if self.augment_cfg.specaugment:
            mel = spec_augment(mel, self.augment_cfg)
        return mel, self.labels[idx]


class PlainCNN(nn.Module):
    def __init__(self, n_classes=8):
        super().__init__()
        self.features = nn.Sequential(
            self._block(1, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(256, n_classes))

    @staticmethod
    def _block(in_ch, out_ch):
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
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


class RegularisedCNN(nn.Module):
    def __init__(
        self,
        n_classes=8,
        block_dropouts=(0.1, 0.2, 0.3),
        fc_dropouts=(0.6, 0.4),
    ):
        super().__init__()
        self.block1 = self._conv_block(1, 32)
        self.drop1 = nn.Dropout2d(block_dropouts[0])
        self.block2 = self._conv_block(32, 64)
        self.drop2 = nn.Dropout2d(block_dropouts[1])
        self.block3 = self._conv_block(64, 128)
        self.drop3 = nn.Dropout2d(block_dropouts[2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(fc_dropouts[0]),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(fc_dropouts[1]),
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


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNetGenreCNN(nn.Module):
    def __init__(
        self,
        n_classes=8,
        stage_dropouts=(0.08, 0.10, 0.15, 0.20),
        classifier_dropout=0.45,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(32, 32, blocks=2, stride=1, dropout=stage_dropouts[0])
        self.stage2 = self._make_stage(32, 64, blocks=2, stride=2, dropout=stage_dropouts[1])
        self.stage3 = self._make_stage(64, 128, blocks=2, stride=2, dropout=stage_dropouts[2])
        self.stage4 = self._make_stage(128, 256, blocks=1, stride=2, dropout=stage_dropouts[3])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(classifier_dropout), nn.Linear(256, n_classes))

    @staticmethod
    def _make_stage(in_ch, out_ch, blocks, stride, dropout):
        layers = [ResidualBlock(in_ch, out_ch, stride=stride, dropout=dropout)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_ch, out_ch, dropout=dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def mixup_batch(X, y, alpha):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(X.size(0), device=X.device)
    return lam * X + (1 - lam) * X[idx], y, y[idx], lam


def mixup_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_epoch(model, loader, optimizer, criterion, augment_cfg, scaler=None, use_amp=False):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for X, y in loader:
        X = X.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            if augment_cfg.mixup:
                X, y_a, y_b, lam = mixup_batch(X, y, augment_cfg.mixup_alpha)
                out = model(X)
                loss = mixup_loss(criterion, out, y_a, y_b, lam)
                correct += (out.argmax(1) == y_a).sum().item()
            else:
                out = model(X)
                loss = criterion(out, y)
                correct += (out.argmax(1) == y).sum().item()
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, use_amp=False):
    model.eval()
    total_loss, n = 0.0, 0
    all_true, all_pred, all_proba = [], [], []
    for X, y in loader:
        X = X.to(DEVICE, non_blocking=True)
        y_dev = y.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(X)
            loss = criterion(out, y_dev)
        proba = torch.softmax(out, dim=1).cpu().numpy()
        pred = proba.argmax(1)
        all_true.extend(y.numpy())
        all_pred.extend(pred)
        all_proba.extend(proba)
        total_loss += loss.item() * len(y)
        n += len(y)
    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    y_proba = np.array(all_proba)
    scores = compute_scores(y_true, y_pred)
    return total_loss / n, scores["accuracy"], scores["f1_macro"], y_true, y_pred, y_proba


def loader_kwargs(cfg: TrainConfig, shuffle):
    kwargs = {
        "batch_size": cfg.batch_size,
        "shuffle": shuffle,
        "num_workers": cfg.num_workers,
        "pin_memory": DEVICE.type == "cuda",
    }
    if cfg.num_workers > 0:
        kwargs["persistent_workers"] = True
    return kwargs


def clone_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def make_optimizer(model, cfg: TrainConfig):
    if cfg.optimizer == "adamw":
        return optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    return optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def run_model(
    model_factory,
    label,
    mels,
    y,
    le,
    idx_train,
    idx_val,
    idx_test,
    train_cfg=None,
    augment_cfg=None,
):
    train_cfg = train_cfg or TrainConfig()
    augment_cfg = augment_cfg or AugmentConfig()
    model = model_factory(len(le.classes_)).to(DEVICE)
    param_count, trainable_param_count = count_parameters(model)
    print(
        f"\n{label}: parameters={param_count:,} "
        f"(trainable={trainable_param_count:,})",
        flush=True,
    )
    use_amp = train_cfg.amp and DEVICE.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    optimizer = make_optimizer(model, train_cfg)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg.label_smoothing)

    train_loader = DataLoader(
        MelDataset(mels[idx_train], y[idx_train], augment_cfg=augment_cfg),
        **loader_kwargs(train_cfg, shuffle=True),
    )
    val_loader = DataLoader(
        MelDataset(mels[idx_val], y[idx_val], augment_cfg=AugmentConfig()),
        **loader_kwargs(train_cfg, shuffle=False),
    )
    test_loader = DataLoader(
        MelDataset(mels[idx_test], y[idx_test], augment_cfg=AugmentConfig()),
        **loader_kwargs(train_cfg, shuffle=False),
    )

    best_state, best_epoch, best_val_f1 = None, 0, -1.0
    best_val_true, best_val_pred, best_val_proba = None, None, None
    no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    start_time = time.perf_counter()

    for epoch in range(1, train_cfg.epochs + 1):
        tr_loss, tr_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            augment_cfg,
            scaler=scaler,
            use_amp=use_amp,
        )
        val_loss, val_acc, val_f1, val_true, val_pred, val_proba = evaluate(
            model,
            val_loader,
            criterion,
            use_amp=use_amp,
        )
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
            best_val_proba = val_proba
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{train_cfg.epochs}  loss={tr_loss:.4f}  "
                f"train={tr_acc:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}",
                flush=True,
            )
        if no_improve >= train_cfg.patience:
            print(f"  Early stop at epoch {epoch} (best epoch {best_epoch})", flush=True)
            break

    training_seconds = time.perf_counter() - start_time
    model.load_state_dict(best_state)
    _, test_acc, test_f1, test_true, test_pred, test_proba = evaluate(
        model,
        test_loader,
        criterion,
        use_amp=use_amp,
    )
    epochs_run = len(history["train_loss"])
    print(
        f"\n{label}: best_epoch={best_epoch}  val_f1={best_val_f1:.4f}  "
        f"test_acc={test_acc:.4f}  test_f1={test_f1:.4f}  "
        f"time={training_seconds:.1f}s",
        flush=True,
    )

    return {
        "label": label,
        "val_true": best_val_true,
        "val_pred": best_val_pred,
        "val_proba": best_val_proba,
        "test_true": test_true,
        "test_pred": test_pred,
        "test_proba": test_proba,
        "test_indices": idx_test,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "param_count": param_count,
        "trainable_param_count": trainable_param_count,
        "training_seconds": training_seconds,
        "epochs_run": epochs_run,
        "history": history,
    }


def prepare_data():
    seed_everything()
    print(f"Using device: {DEVICE}", flush=True)
    mels, labels, track_ids = load_mel_cache()
    le, y, idx_train, idx_val, idx_test = make_split(labels)
    print(f"Loaded mel specs: {mels.shape}")
    print(f"Split sizes: train={len(idx_train)}  val={len(idx_val)}  test={len(idx_test)}")
    return mels, labels, track_ids, le, y, idx_train, idx_val, idx_test


def finalize_experiment(results, out_dir, class_names, title, track_ids=None):
    metrics_df = save_metrics(results, out_dir)
    proba_saved = save_branch_probabilities(results, out_dir)
    save_history(results, out_dir)
    history_plot_saved = plot_training_history(results, out_dir)
    metrics_plot_saved = plot_metrics(metrics_df, out_dir, title)
    best = max(results, key=lambda r: r["best_val_f1"])
    plot_confusion_matrix(best["test_true"], best["test_pred"], class_names, out_dir, f"Confusion matrix - {best['label']}")
    write_classification_report(best, class_names, out_dir)
    save_predictions(best, class_names, out_dir, test_indices=best.get("test_indices"), track_ids=track_ids)
    update_global_comparison()

    print_metrics_summary(metrics_df)
    print_saved_outputs(
        out_dir,
        [
            "metrics.csv",
            "metrics.png" if metrics_plot_saved else "",
            "training_history.csv",
            "training_history.png" if history_plot_saved else "",
            "classification_report.txt",
            "confusion_matrix.png",
            "predictions.csv",
            "high_confidence_errors.csv",
            "branch_probabilities.npz" if proba_saved else "",
            "branch_probability_index.csv" if proba_saved else "",
        ],
    )
    return metrics_df
