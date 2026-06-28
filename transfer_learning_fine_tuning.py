"""Part 5.4: fine-tune the better audio-pretrained transfer model.

The script first compares the validation F1-macro scores from Part 5.2
PANNs-CNN14 and Part 5.3 AST frozen-embedding experiments. It then fine-tunes
only the stronger family, using the same shared split and reporting format as
the other experiments.
"""

import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from cnn_training_utils import (
    DEVICE,
    clone_state_dict,
    count_parameters,
    current_lr,
    finalize_experiment,
    load_mel_cache,
    make_split,
    seed_everything,
)
from reporting_utils import (
    RESULTS_ROOT,
    ROOT,
    compute_scores,
    experiment_dir,
    fma_audio_path,
    print_section,
)

warnings.filterwarnings("ignore")

OUT_DIR = experiment_dir("5.4 Fine Tuning")
SELECTION_PATH = OUT_DIR / "selection_candidates.csv"

SAMPLE_RATE = 16000
CLIP_DURATION = 10.0
FULL_DURATION = 29.0
SEGMENTS_PER_TRACK = 4


@dataclass
class FineTuneConfig:
    epochs: int = 30
    full_batch_size: int = 8
    segment_batch_size: int = 2
    lr: float = 2e-5
    head_lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 8
    num_workers: int = 2
    amp: bool = True


def check_audio_files(track_ids):
    missing = [track_id for track_id in track_ids if not fma_audio_path(track_id).exists()]
    if missing:
        examples = ", ".join(str(int(track_id)) for track_id in missing[:5])
        raise FileNotFoundError(
            f"Fine-tuning requires raw FMA audio under data/fma_small. "
            f"Missing {len(missing)} files. Examples: {examples}"
        )


def load_waveform(track_id, sample_rate=SAMPLE_RATE):
    y, _sr = librosa.load(
        fma_audio_path(track_id),
        sr=sample_rate,
        mono=True,
        duration=FULL_DURATION,
    )
    target = int(FULL_DURATION * sample_rate)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    return y[:target].astype(np.float32)


def center_crop(waveform, sample_rate=SAMPLE_RATE):
    clip_samples = int(CLIP_DURATION * sample_rate)
    if len(waveform) < clip_samples:
        waveform = np.pad(waveform, (0, clip_samples - len(waveform)))
    start = max(0, (len(waveform) - clip_samples) // 2)
    return waveform[start:start + clip_samples].astype(np.float32)


def waveform_segments(waveform, sample_rate=SAMPLE_RATE):
    clip_samples = int(CLIP_DURATION * sample_rate)
    if len(waveform) < clip_samples:
        waveform = np.pad(waveform, (0, clip_samples - len(waveform)))
    max_start = len(waveform) - clip_samples
    starts = np.linspace(0, max_start, SEGMENTS_PER_TRACK).round().astype(int)
    return np.stack(
        [waveform[start:start + clip_samples] for start in starts],
    ).astype(np.float32)


class TrackClipDataset(Dataset):
    def __init__(self, track_ids, labels, indices, sample_rate=SAMPLE_RATE):
        self.track_ids = np.asarray(track_ids)
        self.labels = np.asarray(labels)
        self.indices = np.asarray(indices)
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        idx = int(self.indices[item])
        waveform = load_waveform(self.track_ids[idx], sample_rate=self.sample_rate)
        clip = center_crop(waveform, sample_rate=self.sample_rate)
        return torch.tensor(clip, dtype=torch.float32), torch.tensor(int(self.labels[idx]))


class TrackSegmentsDataset(Dataset):
    def __init__(self, track_ids, labels, indices, sample_rate=SAMPLE_RATE):
        self.track_ids = np.asarray(track_ids)
        self.labels = np.asarray(labels)
        self.indices = np.asarray(indices)
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        idx = int(self.indices[item])
        waveform = load_waveform(self.track_ids[idx], sample_rate=self.sample_rate)
        clips = waveform_segments(waveform, sample_rate=self.sample_rate)
        return torch.tensor(clips, dtype=torch.float32), torch.tensor(int(self.labels[idx]))


def loader_kwargs(batch_size, shuffle, num_workers):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": DEVICE.type == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return kwargs


def read_candidate_metrics():
    specs = [
        ("panns", "5.2 PANNs-CNN14", RESULTS_ROOT / "5.2 PANNs-CNN14" / "metrics.csv"),
        ("ast", "5.3 AST", RESULTS_ROOT / "5.3 AST" / "metrics.csv"),
    ]
    rows = []
    missing = []
    for family, experiment, path in specs:
        if not path.exists():
            missing.append(f"{experiment}: {path.relative_to(ROOT)}")
            continue
        df = pd.read_csv(path)
        val_df = df[df["split"] == "val"].copy()
        val_df["family"] = family
        val_df["experiment"] = experiment
        val_df["source_path"] = str(path.relative_to(ROOT))
        rows.append(val_df)
    if missing:
        raise FileNotFoundError(
            "Run both Part 5.2 and Part 5.3 before 5.4 fine-tuning. Missing:\n"
            + "\n".join(missing)
        )
    candidates = pd.concat(rows, ignore_index=True)
    candidates = candidates.sort_values("f1_macro", ascending=False)
    candidates.to_csv(SELECTION_PATH, index=False)
    return candidates


def select_pretrained_family():
    candidates = read_candidate_metrics()
    selected = candidates.iloc[0]
    use_segments = "Segment Averaging" in selected["model"]
    lines = [
        "Selected audio-pretrained model for fine-tuning",
        "",
        candidates[["experiment", "model", "f1_macro", "accuracy", "best_epoch"]].to_string(index=False),
        "",
        f"Selected experiment: {selected['experiment']}",
        f"Selected model: {selected['model']}",
        f"Selected validation F1-macro: {selected['f1_macro']:.4f}",
        f"Fine-tuning family: {selected['family']}",
        f"Use segment averaging: {use_segments}",
    ]
    (OUT_DIR / "selected_pretrained_model.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print_section("5.4 Fine Tuning selection")
    print(candidates[["experiment", "model", "f1_macro", "accuracy"]].to_string(index=False))
    print(f"Selected: {selected['model']} from {selected['experiment']}")
    return selected["family"], use_segments, selected


def configure_ast_model(n_classes):
    try:
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
    except ImportError as exc:
        raise ImportError(
            "transfer_learning_fine_tuning.py selected AST but transformers is missing. "
            "Install dependencies with: pip3 install -r requirements.txt"
        ) from exc

    from transfer_learning_ast import MODEL_NAME

    feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_NAME)
    model = ASTForAudioClassification.from_pretrained(
        MODEL_NAME,
        num_labels=n_classes,
        ignore_mismatched_sizes=True,
    )
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if (
            "classifier" in name
            or "layer.11" in name
            or "layernorm" in name.lower()
            or "layer_norm" in name.lower()
        ):
            param.requires_grad = True
    model.to(DEVICE)
    return model, feature_extractor


def configure_panns_model(n_classes):
    try:
        from transfer_learning_panns_cnn14 import Cnn14, download_checkpoint
    except ImportError as exc:
        raise ImportError(
            "transfer_learning_fine_tuning.py selected PANNs-CNN14 but torchlibrosa is missing. "
            "Install dependencies with: pip3 install -r requirements.txt"
        ) from exc

    checkpoint_path = download_checkpoint()
    model = Cnn14(classes_num=527)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.fc_audioset = nn.Linear(2048, n_classes)

    for param in model.parameters():
        param.requires_grad = False
    for module in [model.conv_block6, model.fc1, model.fc_audioset]:
        for param in module.parameters():
            param.requires_grad = True
    model.to(DEVICE)
    return model, None


def make_optimizer(model, cfg):
    head_keywords = ["classifier", "fc_audioset"]
    head_params = []
    body_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(keyword in name for keyword in head_keywords):
            head_params.append(param)
        else:
            body_params.append(param)
    groups = []
    if body_params:
        groups.append({"params": body_params, "lr": cfg.lr})
    if head_params:
        groups.append({"params": head_params, "lr": cfg.head_lr})
    return optim.AdamW(groups, weight_decay=cfg.weight_decay)


def ast_logits(model, feature_extractor, clips):
    clips_np = [clip.detach().cpu().numpy().astype(np.float32) for clip in clips]
    inputs = feature_extractor(
        clips_np,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs["input_values"].to(DEVICE, non_blocking=True)
    return model(input_values=input_values).logits


def panns_logits(model, clips):
    output = model(clips.to(DEVICE, non_blocking=True))
    return output["logits"]


def forward_logits(family, model, feature_extractor, clips):
    if family == "ast":
        return ast_logits(model, feature_extractor, clips)
    return panns_logits(model, clips)


def keep_frozen_batchnorm_eval(model):
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            params = list(module.parameters())
            if params and not any(param.requires_grad for param in params):
                module.eval()


def train_epoch(family, model, feature_extractor, loader, optimizer, criterion, scaler, use_amp):
    model.train()
    if family == "panns":
        keep_frozen_batchnorm_eval(model)
    total_loss, correct, n = 0.0, 0, 0
    for clips, y in tqdm(loader, desc="train", leave=False):
        y = y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = forward_logits(family, model, feature_extractor, clips)
            loss = criterion(logits, y)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        correct += (logits.argmax(1) == y).sum().item()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / n, correct / n


def train_segment_epoch(family, model, feature_extractor, loader, optimizer, criterion, scaler, use_amp):
    model.train()
    if family == "panns":
        keep_frozen_batchnorm_eval(model)
    total_loss, correct, n, n_loss = 0.0, 0, 0, 0
    for clips, y in tqdm(loader, desc="train", leave=False):
        batch_size, n_segments, n_samples = clips.shape
        flat_clips = clips.reshape(batch_size * n_segments, n_samples)
        y = y.to(DEVICE, non_blocking=True)
        y_rep = y.repeat_interleave(n_segments)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = forward_logits(family, model, feature_extractor, flat_clips)
            loss = criterion(logits, y_rep)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        track_logits = logits.reshape(batch_size, n_segments, -1).mean(dim=1)
        correct += (track_logits.argmax(1) == y).sum().item()
        total_loss += loss.item() * len(y_rep)
        n_loss += len(y_rep)
        n += len(y)
    return total_loss / max(1, n_loss), correct / n


@torch.no_grad()
def evaluate(family, model, feature_extractor, loader, criterion):
    model.eval()
    total_loss, n = 0.0, 0
    all_true, all_pred, all_proba = [], [], []
    for clips, y in tqdm(loader, desc="eval", leave=False):
        y_dev = y.to(DEVICE, non_blocking=True)
        logits = forward_logits(family, model, feature_extractor, clips)
        loss = criterion(logits, y_dev)
        proba = torch.softmax(logits, dim=1).detach().cpu().numpy()
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


@torch.no_grad()
def evaluate_segments(family, model, feature_extractor, loader, criterion):
    model.eval()
    total_loss, n, n_loss = 0.0, 0, 0
    all_true, all_pred, all_proba = [], [], []
    for clips, y in tqdm(loader, desc="eval", leave=False):
        batch_size, n_segments, n_samples = clips.shape
        flat_clips = clips.reshape(batch_size * n_segments, n_samples)
        y_dev = y.to(DEVICE, non_blocking=True)
        y_rep = y_dev.repeat_interleave(n_segments)
        logits = forward_logits(family, model, feature_extractor, flat_clips)
        loss = criterion(logits, y_rep)
        proba = torch.softmax(logits, dim=1).reshape(batch_size, n_segments, -1)
        avg_proba = proba.mean(dim=1).detach().cpu().numpy()
        pred = avg_proba.argmax(1)
        all_true.extend(y.numpy())
        all_pred.extend(pred)
        all_proba.extend(avg_proba)
        total_loss += loss.item() * len(y_rep)
        n_loss += len(y_rep)
        n += len(y)
    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    y_proba = np.array(all_proba)
    scores = compute_scores(y_true, y_pred)
    return (
        total_loss / max(1, n_loss),
        scores["accuracy"],
        scores["f1_macro"],
        y_true,
        y_pred,
        y_proba,
    )


def run_fine_tuning(family, use_segments, track_ids, y, le, idx_train, idx_val, idx_test):
    cfg = FineTuneConfig()
    if family == "panns":
        from transfer_learning_panns_cnn14 import SAMPLE_RATE as sample_rate

        model, feature_extractor = configure_panns_model(len(le.classes_))
        cfg.lr = 5e-5
        cfg.head_lr = 2e-4
        cfg.full_batch_size = 6
        cfg.segment_batch_size = 2
    else:
        sample_rate = SAMPLE_RATE
        model, feature_extractor = configure_ast_model(len(le.classes_))

    param_count, trainable_param_count = count_parameters(model)
    label_family = "AST" if family == "ast" else "PANNs-CNN14"
    label = f"Fine-tuned {label_family}"
    if use_segments:
        label += " - Segment Averaging"

    print_section(label)
    print(
        f"parameters={param_count:,} trainable={trainable_param_count:,} "
        f"sample_rate={sample_rate} clip={CLIP_DURATION:.1f}s segments={use_segments}"
    )

    dataset_cls = TrackSegmentsDataset if use_segments else TrackClipDataset
    batch_size = cfg.segment_batch_size if use_segments else cfg.full_batch_size
    train_loader = DataLoader(
        dataset_cls(track_ids, y, idx_train, sample_rate=sample_rate),
        **loader_kwargs(batch_size, shuffle=True, num_workers=cfg.num_workers),
    )
    val_loader = DataLoader(
        dataset_cls(track_ids, y, idx_val, sample_rate=sample_rate),
        **loader_kwargs(batch_size, shuffle=False, num_workers=cfg.num_workers),
    )
    test_loader = DataLoader(
        dataset_cls(track_ids, y, idx_test, sample_rate=sample_rate),
        **loader_kwargs(batch_size, shuffle=False, num_workers=cfg.num_workers),
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = make_optimizer(model, cfg)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
    )
    use_amp = cfg.amp and DEVICE.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": [], "lr": []}

    best_state, best_epoch, best_val_f1 = None, 0, -1.0
    best_val_true, best_val_pred, best_val_proba = None, None, None
    no_improve = 0
    start_time = time.perf_counter()
    train_fn = train_segment_epoch if use_segments else train_epoch
    eval_fn = evaluate_segments if use_segments else evaluate

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_fn(
            family,
            model,
            feature_extractor,
            train_loader,
            optimizer,
            criterion,
            scaler,
            use_amp,
        )
        val_loss, val_acc, val_f1, val_true, val_pred, val_proba = eval_fn(
            family,
            model,
            feature_extractor,
            val_loader,
            criterion,
        )
        scheduler.step(val_f1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
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

        print(
            f"  Epoch {epoch:3d}/{cfg.epochs} loss={train_loss:.4f} "
            f"train={train_acc:.4f} val_acc={val_acc:.4f} "
            f"val_f1={val_f1:.4f} lr={current_lr(optimizer):.2e}",
            flush=True,
        )
        if no_improve >= cfg.patience:
            print(f"  Early stop at epoch {epoch} (best epoch {best_epoch})")
            break

    training_seconds = time.perf_counter() - start_time
    model.load_state_dict(best_state)
    _, test_acc, test_f1, test_true, test_pred, test_proba = eval_fn(
        family,
        model,
        feature_extractor,
        test_loader,
        criterion,
    )
    print(
        f"\n{label}: best_epoch={best_epoch} val_f1={best_val_f1:.4f} "
        f"test_acc={test_acc:.4f} test_f1={test_f1:.4f} "
        f"runtime={training_seconds:.1f}s",
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
        "epochs_run": len(history["train_loss"]),
        "history": history,
    }


def main():
    seed_everything()
    family, use_segments, _selected = select_pretrained_family()
    _mels, labels, track_ids = load_mel_cache()
    check_audio_files(track_ids)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    idx_train, idx_val, idx_test = make_split(labels)[2:]
    result = run_fine_tuning(
        family,
        use_segments,
        track_ids,
        y,
        le,
        idx_train,
        idx_val,
        idx_test,
    )
    finalize_experiment(
        [result],
        OUT_DIR,
        le.classes_,
        "5.4 Fine Tuning",
        track_ids=track_ids,
    )


if __name__ == "__main__":
    main()
