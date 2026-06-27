"""Part 2.5: segment-based training with track-level probability averaging."""

import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from cnn_training_utils import (
    DEVICE,
    AugmentConfig,
    MultiShapeCNN,
    PlainCNN,
    PlainCNNRegularisation,
    ResNetGenreCNN,
    TrainConfig,
    clone_state_dict,
    count_parameters,
    finalize_experiment,
    load_mel_cache,
    loader_kwargs,
    current_lr,
    make_scheduler,
    make_optimizer,
    make_split,
    mixup_batch,
    mixup_loss,
    seed_everything,
    spec_augment,
)
from reporting_utils import (
    RESULTS_ROOT,
    ROOT,
    experiment_dir,
    plot_metrics,
    print_metrics_summary,
    print_saved_outputs,
    print_section,
)

OUT_DIR = experiment_dir("2.5 Segment Averaging")
SEGMENT_CACHE = ROOT / "features" / "mel_segments.npz"
FMA_AUDIO = ROOT / "data" / "fma_small"

SR = 22050
DURATION = 29.0
N_MELS = 128
HOP_LEN = 512
SEGMENT_FRAMES = 128
SEGMENTS_PER_TRACK = 4


def audio_path(track_id):
    tid = f"{int(track_id):06d}"
    return FMA_AUDIO / tid[:3] / f"{tid}.mp3"


def plain_factory(n_classes):
    return PlainCNN(n_classes=n_classes)


def regularised_factory(n_classes):
    return PlainCNNRegularisation(n_classes=n_classes)


def resnet_factory(n_classes):
    return ResNetGenreCNN(n_classes=n_classes)


def multi_shape_factory(n_classes):
    return MultiShapeCNN(n_classes=n_classes)


MODEL_CONFIGS = {
    "Plain CNN": {
        "factory": plain_factory,
        "train_cfg": TrainConfig(epochs=300, lr=1e-3, weight_decay=0.0, patience=48),
        "augment_cfg": AugmentConfig(specaugment=False, mixup=False),
        "spec_t": 25,
        "spec_f": 15,
    },
    "Plain CNN - Regularisation": {
        "factory": regularised_factory,
        "train_cfg": TrainConfig(
            epochs=300,
            lr=1e-3,
            weight_decay=1e-4,
            patience=48,
            label_smoothing=0.1,
        ),
        "augment_cfg": AugmentConfig(specaugment=False, mixup=False),
        "spec_t": 30,
        "spec_f": 20,
    },
    "ResNet CNN": {
        "factory": resnet_factory,
        "train_cfg": TrainConfig(
            epochs=300,
            lr=1e-3,
            weight_decay=5e-4,
            patience=48,
            label_smoothing=0.10,
            optimizer="adamw",
        ),
        "augment_cfg": AugmentConfig(specaugment=False, mixup=False),
        "spec_t": 30,
        "spec_f": 20,
    },
    "Multi-shape CNN": {
        "factory": multi_shape_factory,
        "train_cfg": TrainConfig(
            epochs=300,
            lr=1e-3,
            weight_decay=1e-4,
            patience=48,
            label_smoothing=0.1,
        ),
        "augment_cfg": AugmentConfig(specaugment=False, mixup=False),
        "spec_t": 30,
        "spec_f": 20,
    },
}


def strip_source(label):
    return label.split(" - no augmentation")[0]


def parse_selected_label(label):
    base = strip_source(label)
    specaugment = "SpecAugment" in label
    mixup = "Mixup" in label
    for suffix in [" - SpecAugment + Mixup", " - SpecAugment", " - Mixup"]:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    if base not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported selected model for segment training: {label}")

    cfg = MODEL_CONFIGS[base].copy()
    cfg["base_name"] = base
    cfg["augment_cfg"] = AugmentConfig(
        specaugment=specaugment,
        mixup=mixup,
        spec_t=cfg["spec_t"],
        spec_f=cfg["spec_f"],
        mixup_alpha=0.3,
    )
    return cfg


def load_candidate_metrics():
    preferred = RESULTS_ROOT / "2.4 Augmentation ablation" / "augmentation_comparison.csv"
    fallback_paths = [
        RESULTS_ROOT / "2.3 Multi-shape CNN" / "metrics.csv",
        RESULTS_ROOT / "2.2 ResNet CNN" / "metrics.csv",
        RESULTS_ROOT / "2.1 Plain CNN" / "metrics.csv",
    ]
    paths = [preferred] if preferred.exists() else [p for p in fallback_paths if p.exists()]
    if not paths:
        raise FileNotFoundError("Run Part 2.1-2.4 before segment_averaging.py.")

    rows = []
    for path in paths:
        df = pd.read_csv(path)
        val_df = df[df["split"] == "val"].copy()
        val_df["source"] = path.parent.name
        val_df["source_path"] = str(path)
        rows.append(val_df)
    candidates = pd.concat(rows, ignore_index=True)
    supported = []
    for label in candidates["model"]:
        try:
            parse_selected_label(label)
            supported.append(True)
        except ValueError:
            supported.append(False)
    candidates = candidates[supported].sort_values("f1_macro", ascending=False)
    if candidates.empty:
        raise ValueError("No supported CNN candidates found for segment training.")
    return candidates


def select_model():
    candidates = load_candidate_metrics()
    candidates.to_csv(OUT_DIR / "selection_candidates.csv", index=False)
    selected = candidates.iloc[0]
    selected_model = selected["model"]

    baseline_path = Path(selected["source_path"])
    baseline = pd.read_csv(baseline_path)
    baseline = baseline[baseline["model"] == selected_model].copy()
    baseline["model"] = f"{selected_model} - previous best"
    baseline["probability_source_dir"] = selected.get("probability_source_dir", selected["source"])
    baseline["probability_model"] = selected.get("probability_model", selected_model)
    baseline.to_csv(OUT_DIR / "previous_best_baseline.csv", index=False)

    lines = [
        "Model selected for segment-based training",
        "",
        candidates[["source", "model", "accuracy", "f1_macro", "best_epoch"]].to_string(index=False),
        "",
        f"Selected model: {selected_model}",
        f"Selected validation F1-macro: {selected['f1_macro']:.4f}",
    ]
    (OUT_DIR / "selected_model.txt").write_text("\n".join(lines), encoding="utf-8")

    print_section("Model selection")
    print(candidates[["source", "model", "f1_macro", "best_epoch"]].to_string(index=False))
    print(f"Selected model: {selected_model} (val_f1={selected['f1_macro']:.4f})")
    return selected_model, parse_selected_label(selected_model), baseline


def extract_track_segments(track_id):
    path = audio_path(track_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing audio file: {path}")
    y, sr = librosa.load(path, sr=SR, duration=DURATION, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        hop_length=HOP_LEN,
        fmax=sr // 2,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)

    if mel_db.shape[1] < SEGMENT_FRAMES:
        pad = SEGMENT_FRAMES - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode="edge")

    max_start = mel_db.shape[1] - SEGMENT_FRAMES
    starts = np.linspace(0, max_start, SEGMENTS_PER_TRACK).round().astype(int)
    return np.stack([mel_db[:, start:start + SEGMENT_FRAMES] for start in starts]).astype(np.float32)


def build_segment_cache(track_ids, labels):
    print_section("Build segment cache")
    missing = [track_id for track_id in track_ids if not audio_path(track_id).exists()]
    if missing:
        examples = ", ".join(str(int(track_id)) for track_id in missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} FMA-small audio files under {FMA_AUDIO}. "
            f"Examples: {examples}. Run extract_mel_segments.py locally where the audio exists, "
            f"then copy {SEGMENT_CACHE.relative_to(ROOT)} to this environment."
        )
    print(f"Tracks: {len(track_ids)}  segments_per_track={SEGMENTS_PER_TRACK}")
    segments = []
    for track_id in tqdm(track_ids, desc="Segment extraction"):
        segments.append(extract_track_segments(track_id))
    segments = np.stack(segments).astype(np.float16)
    np.savez_compressed(
        SEGMENT_CACHE,
        segments=segments,
        labels=labels,
        track_ids=track_ids,
        segments_per_track=np.array(SEGMENTS_PER_TRACK),
        segment_frames=np.array(SEGMENT_FRAMES),
    )
    print(f"Saved: {SEGMENT_CACHE.relative_to(ROOT)}")
    return segments, labels, track_ids


def load_segment_cache(expected_track_ids, expected_labels):
    if SEGMENT_CACHE.exists():
        data = np.load(SEGMENT_CACHE, allow_pickle=True)
        required = {"segments", "labels", "track_ids", "segments_per_track", "segment_frames"}
        labels_ok = (
            "labels" in data.files
            and len(data["labels"]) == len(expected_labels)
            and np.array_equal(data["labels"], expected_labels)
        )
        track_ids_ok = (
            "track_ids" in data.files
            and (
                expected_track_ids is None
                or np.array_equal(data["track_ids"], expected_track_ids)
            )
        )
        ok = (
            required.issubset(data.files)
            and int(data["segments_per_track"]) == SEGMENTS_PER_TRACK
            and int(data["segment_frames"]) == SEGMENT_FRAMES
            and labels_ok
            and track_ids_ok
        )
        if ok:
            return data["segments"], data["labels"], data["track_ids"]
        print_section("Segment cache mismatch")
        print(f"Ignoring stale cache: {SEGMENT_CACHE.relative_to(ROOT)}")
    if expected_track_ids is None:
        raise ValueError(
            "Missing track_ids for segment extraction and no compatible "
            "features/mel_segments.npz cache was found. Run extract_mel_segments.py "
            "locally where the original audio exists, then copy the cache to this environment."
        )
    return build_segment_cache(expected_track_ids, expected_labels)


class SegmentDataset(Dataset):
    def __init__(self, segments, labels, track_indices, augment_cfg=None):
        self.segments = segments
        self.labels = labels
        self.track_indices = np.asarray(track_indices)
        self.augment_cfg = augment_cfg or AugmentConfig()
        self.items = [
            (track_pos, seg_idx)
            for track_pos in range(len(self.track_indices))
            for seg_idx in range(self.segments.shape[1])
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        track_pos, seg_idx = self.items[idx]
        global_idx = self.track_indices[track_pos]
        mel = torch.tensor(self.segments[global_idx, seg_idx][None, :, :], dtype=torch.float32)
        if self.augment_cfg.specaugment:
            mel = spec_augment(mel, self.augment_cfg)
        label = int(self.labels[global_idx])
        return mel, torch.tensor(label, dtype=torch.long), torch.tensor(track_pos, dtype=torch.long)


def train_segment_epoch(
    model,
    loader,
    optimizer,
    criterion,
    augment_cfg,
    scaler=None,
    use_amp=False,
):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for X, y, _track_pos in loader:
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
def evaluate_track_level(model, loader, criterion, track_indices, labels, n_classes, use_amp=False):
    model.eval()
    total_loss, n = 0.0, 0
    proba_sum = np.zeros((len(track_indices), n_classes), dtype=np.float64)
    counts = np.zeros(len(track_indices), dtype=np.float64)
    for X, y, track_pos in loader:
        X = X.to(DEVICE, non_blocking=True)
        y_dev = y.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(X)
            loss = criterion(out, y_dev)
        proba = torch.softmax(out, dim=1).cpu().numpy()
        for row, pos in enumerate(track_pos.numpy()):
            proba_sum[pos] += proba[row]
            counts[pos] += 1
        total_loss += loss.item() * len(y)
        n += len(y)

    avg_proba = proba_sum / counts[:, None]
    y_true = labels[track_indices]
    y_pred = avg_proba.argmax(1)
    return (
        total_loss / n,
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average="macro"),
        y_true,
        y_pred,
        avg_proba,
    )


def run_segment_model(model_cfg, label, segments, y, le, idx_train, idx_val, idx_test):
    train_cfg = model_cfg["train_cfg"]
    augment_cfg = model_cfg["augment_cfg"]
    seed_everything()
    model = model_cfg["factory"](len(le.classes_)).to(DEVICE)
    param_count, trainable_param_count = count_parameters(model)
    print(f"\n{label}: parameters={param_count:,} (trainable={trainable_param_count:,})")

    use_amp = train_cfg.amp and DEVICE.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    optimizer = make_optimizer(model, train_cfg)
    scheduler = make_scheduler(optimizer, train_cfg)
    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg.label_smoothing)

    train_loader = DataLoader(
        SegmentDataset(segments, y, idx_train, augment_cfg=augment_cfg),
        **loader_kwargs(train_cfg, shuffle=True),
    )
    val_loader = DataLoader(
        SegmentDataset(segments, y, idx_val, augment_cfg=AugmentConfig()),
        **loader_kwargs(train_cfg, shuffle=False),
    )
    test_loader = DataLoader(
        SegmentDataset(segments, y, idx_test, augment_cfg=AugmentConfig()),
        **loader_kwargs(train_cfg, shuffle=False),
    )

    best_state, best_epoch, best_val_f1 = None, 0, -1.0
    best_val_true, best_val_pred, best_val_proba = None, None, None
    no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": [], "lr": []}
    start_time = time.perf_counter()

    for epoch in range(1, train_cfg.epochs + 1):
        tr_loss, tr_acc = train_segment_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            augment_cfg,
            scaler=scaler,
            use_amp=use_amp,
        )
        val_loss, val_acc, val_f1, val_true, val_pred, val_proba = evaluate_track_level(
            model,
            val_loader,
            criterion,
            idx_val,
            y,
            len(le.classes_),
            use_amp=use_amp,
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
                f"  Epoch {epoch:3d}/{train_cfg.epochs}  loss={tr_loss:.4f}  "
                f"train={tr_acc:.4f}  val_acc={val_acc:.4f}  "
                f"val_f1={val_f1:.4f}  lr={current_lr(optimizer):.2e}",
                flush=True,
            )
        if no_improve >= train_cfg.patience:
            print(f"  Early stop at epoch {epoch} (best epoch {best_epoch})", flush=True)
            break

    training_seconds = time.perf_counter() - start_time
    model.load_state_dict(best_state)
    _, test_acc, test_f1, test_true, test_pred, test_proba = evaluate_track_level(
        model,
        test_loader,
        criterion,
        idx_test,
        y,
        len(le.classes_),
        use_amp=use_amp,
    )
    epochs_run = len(history["train_loss"])
    print(
        f"\n{label}: best_epoch={best_epoch}  val_f1={best_val_f1:.4f}  "
        f"test_acc={test_acc:.4f}  test_f1={test_f1:.4f}  runtime={training_seconds:.1f}s",
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


def main():
    seed_everything()
    selected_model, model_cfg, baseline_metrics = select_model()
    _mels, labels, track_ids = load_mel_cache()
    le, y, idx_train, idx_val, idx_test = make_split(labels)
    segments, _segment_labels, segment_track_ids = load_segment_cache(track_ids, labels)

    print_section(f"2.5 Segment Averaging - {selected_model}")
    print(f"Segments: {segments.shape}  split: train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}")
    result = run_segment_model(
        model_cfg,
        f"{selected_model} - Segment Averaging",
        segments,
        y,
        le,
        idx_train,
        idx_val,
        idx_test,
    )

    metrics_df = finalize_experiment(
        [result],
        OUT_DIR,
        le.classes_,
        "2.5 Segment Averaging",
        track_ids=segment_track_ids,
    )

    comparison = pd.concat([baseline_metrics, metrics_df], ignore_index=True, sort=False)
    comparison.loc[comparison["probability_source_dir"].isna(), "probability_source_dir"] = OUT_DIR.name
    comparison.loc[comparison["probability_model"].isna(), "probability_model"] = comparison["model"]
    comparison.to_csv(OUT_DIR / "segment_comparison.csv", index=False)
    plot_metrics(comparison, OUT_DIR, "2.5 Segment Averaging comparison")
    print_metrics_summary(comparison, "Segment comparison")
    print_saved_outputs(
        OUT_DIR,
        [
            "selection_candidates.csv",
            "selected_model.txt",
            "previous_best_baseline.csv",
            "segment_comparison.csv",
        ],
    )


if __name__ == "__main__":
    main()
