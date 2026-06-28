"""Part 5.2: PANNs-CNN14 transfer learning with frozen AudioSet embeddings.

PANNs-CNN14 is an audio-pretrained model, so this experiment reads the original
FMA MP3 files instead of reusing the project mel-spectrogram cache as input.
It extracts 2048-dimensional CNN14 embeddings, caches them, then trains compact
MLP classifiers on full-track and segment-averaged embeddings.
"""

import time
import urllib.request
import warnings
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

try:
    from torchlibrosa.stft import LogmelFilterBank, Spectrogram
except ImportError as exc:
    raise ImportError(
        "transfer_learning_panns_cnn14.py requires torchlibrosa. "
        "Install project dependencies with: pip3 install -r requirements.txt"
    ) from exc

from cnn_training_utils import (
    DEVICE,
    finalize_experiment,
    load_mel_cache,
    make_split,
    seed_everything,
)
from reporting_utils import (
    ROOT,
    experiment_dir,
    fma_audio_path,
    print_section,
)

warnings.filterwarnings("ignore")

OUT_DIR = experiment_dir("5.2 PANNs-CNN14")
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
CHECKPOINT_PATH = MODELS_DIR / "Cnn14_mAP=0.431.pth"
CHECKPOINT_URL = (
    "https://zenodo.org/record/3987831/files/"
    "Cnn14_mAP%3D0.431.pth?download=1"
)
EMBEDDING_CACHE = ROOT / "features" / "panns_cnn14_embeddings.npz"

SAMPLE_RATE = 32000
WINDOW_SIZE = 1024
HOP_SIZE = 320
MEL_BINS = 64
FMIN = 50
FMAX = 14000
FULL_DURATION = 29.0
SEGMENT_DURATION = 10.0
SEGMENTS_PER_TRACK = 4
FULL_BATCH_SIZE = 8
SEGMENT_BATCH_SIZE = 16


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, pool_size=(2, 2), pool_type="avg"):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x = F.avg_pool2d(x, kernel_size=pool_size) + F.max_pool2d(
                x,
                kernel_size=pool_size,
            )
        else:
            raise ValueError(f"Unsupported pool_type: {pool_type}")
        return x


class Cnn14(nn.Module):
    def __init__(self, classes_num=527):
        super().__init__()
        self.spectrogram_extractor = Spectrogram(
            n_fft=WINDOW_SIZE,
            hop_length=HOP_SIZE,
            win_length=WINDOW_SIZE,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )
        self.logmel_extractor = LogmelFilterBank(
            sr=SAMPLE_RATE,
            n_fft=WINDOW_SIZE,
            n_mels=MEL_BINS,
            fmin=FMIN,
            fmax=FMAX,
            ref=1.0,
            amin=1e-10,
            top_db=None,
            freeze_parameters=True,
        )

        self.bn0 = nn.BatchNorm2d(MEL_BINS)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

    def forward(self, input):
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type="avg")
        x = F.dropout(x, p=0.5, training=self.training)

        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        logits = self.fc_audioset(embedding)
        clipwise_output = torch.sigmoid(logits)
        return {"clipwise_output": clipwise_output, "embedding": embedding, "logits": logits}


def download_checkpoint():
    if CHECKPOINT_PATH.exists():
        return CHECKPOINT_PATH
    print_section("Download PANNs-CNN14 checkpoint")
    print(f"Saving to: {CHECKPOINT_PATH.relative_to(ROOT)}")
    urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT_PATH)
    return CHECKPOINT_PATH


def load_panns_model():
    checkpoint_path = download_checkpoint()
    model = Cnn14(classes_num=527)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"Checkpoint load: missing={len(missing)} unexpected={len(unexpected)}")
    model.to(DEVICE)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def check_audio_files(track_ids):
    missing = [track_id for track_id in track_ids if not fma_audio_path(track_id).exists()]
    if missing:
        examples = ", ".join(str(int(track_id)) for track_id in missing[:5])
        raise FileNotFoundError(
            f"PANNs-CNN14 requires raw FMA audio under data/fma_small. "
            f"Missing {len(missing)} files. Examples: {examples}"
        )


def load_waveform(track_id):
    path = fma_audio_path(track_id)
    y, _sr = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=FULL_DURATION)
    target = int(FULL_DURATION * SAMPLE_RATE)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    return y[:target].astype(np.float32)


def waveform_segments(waveform):
    segment_samples = int(SEGMENT_DURATION * SAMPLE_RATE)
    if len(waveform) < segment_samples:
        waveform = np.pad(waveform, (0, segment_samples - len(waveform)))
    max_start = len(waveform) - segment_samples
    starts = np.linspace(0, max_start, SEGMENTS_PER_TRACK).round().astype(int)
    return np.stack(
        [waveform[start:start + segment_samples] for start in starts],
    ).astype(np.float32)


@torch.no_grad()
def embed_waveforms(model, waveforms, batch_size):
    embeddings = []
    for start in range(0, len(waveforms), batch_size):
        batch = torch.tensor(waveforms[start:start + batch_size], dtype=torch.float32)
        batch = batch.to(DEVICE, non_blocking=True)
        output = model(batch)
        embeddings.append(output["embedding"].detach().cpu().numpy())
    return np.concatenate(embeddings, axis=0).astype(np.float32)


def build_embedding_cache(track_ids, labels):
    check_audio_files(track_ids)
    model = load_panns_model()
    full_embeddings = []
    segment_embeddings = []

    print_section("Extract PANNs-CNN14 embeddings")
    print(
        f"Tracks: {len(track_ids)}  full_duration={FULL_DURATION:.1f}s  "
        f"segments={SEGMENTS_PER_TRACK}x{SEGMENT_DURATION:.1f}s"
    )
    for start in tqdm(range(0, len(track_ids), FULL_BATCH_SIZE), desc="PANNs batches"):
        batch_ids = track_ids[start:start + FULL_BATCH_SIZE]
        waveforms = np.stack([load_waveform(track_id) for track_id in batch_ids])
        full_embeddings.append(embed_waveforms(model, waveforms, FULL_BATCH_SIZE))
        segments = np.concatenate([waveform_segments(waveform) for waveform in waveforms])
        embedded_segments = embed_waveforms(model, segments, SEGMENT_BATCH_SIZE)
        segment_embeddings.append(
            embedded_segments.reshape(len(batch_ids), SEGMENTS_PER_TRACK, -1)
        )

    full_embeddings = np.concatenate(full_embeddings, axis=0)
    segment_embeddings = np.concatenate(segment_embeddings, axis=0)
    np.savez_compressed(
        EMBEDDING_CACHE,
        track_ids=track_ids,
        labels=labels,
        full_embeddings=full_embeddings.astype(np.float16),
        segment_embeddings=segment_embeddings.astype(np.float16),
        sample_rate=np.array(SAMPLE_RATE),
        full_duration=np.array(FULL_DURATION),
        segment_duration=np.array(SEGMENT_DURATION),
        segments_per_track=np.array(SEGMENTS_PER_TRACK),
    )
    print(f"Saved: {EMBEDDING_CACHE.relative_to(ROOT)}")
    return full_embeddings, segment_embeddings


def load_or_build_embeddings(track_ids, labels):
    if EMBEDDING_CACHE.exists():
        data = np.load(EMBEDDING_CACHE, allow_pickle=True)
        required = {"track_ids", "labels", "full_embeddings", "segment_embeddings"}
        cache_ok = (
            required.issubset(data.files)
            and np.array_equal(data["track_ids"], track_ids)
            and np.array_equal(data["labels"], labels)
            and (
                "segments_per_track" not in data.files
                or int(data["segments_per_track"]) == SEGMENTS_PER_TRACK
            )
        )
        if cache_ok:
            print_section("Load cached PANNs-CNN14 embeddings")
            print(f"Cache: {EMBEDDING_CACHE.relative_to(ROOT)}")
            return (
                data["full_embeddings"].astype(np.float32),
                data["segment_embeddings"].astype(np.float32),
            )
        print_section("PANNs embedding cache mismatch")
        print(f"Ignoring stale cache: {EMBEDDING_CACHE.relative_to(ROOT)}")
    return build_embedding_cache(track_ids, labels)


def make_mlp():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(512, 128),
                    activation="relu",
                    alpha=1e-4,
                    learning_rate_init=1e-3,
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=42,
                ),
            ),
        ]
    )


def mlp_param_count(model):
    clf = model.named_steps["clf"]
    return sum(w.size for w in clf.coefs_) + sum(b.size for b in clf.intercepts_)


def train_full_embedding_model(X, y, idx_train, idx_val, idx_test):
    label = "PANNs-CNN14 embeddings - MLP"
    start_time = time.perf_counter()
    model = make_mlp()
    model.fit(X[idx_train], y[idx_train])
    training_seconds = time.perf_counter() - start_time
    param_count = mlp_param_count(model)
    val_proba = model.predict_proba(X[idx_val])
    test_proba = model.predict_proba(X[idx_test])
    return {
        "label": label,
        "val_true": y[idx_val],
        "val_pred": val_proba.argmax(1),
        "val_proba": val_proba,
        "test_true": y[idx_test],
        "test_pred": test_proba.argmax(1),
        "test_proba": test_proba,
        "test_indices": idx_test,
        "best_epoch": getattr(model.named_steps["clf"], "n_iter_", ""),
        "best_val_f1": "",
        "param_count": param_count,
        "trainable_param_count": param_count,
        "training_seconds": training_seconds,
        "epochs_run": getattr(model.named_steps["clf"], "n_iter_", ""),
    }


def train_segment_embedding_model(segment_X, y, idx_train, idx_val, idx_test):
    label = "PANNs-CNN14 embeddings - MLP - Segment Averaging"
    n_segments = segment_X.shape[1]
    X_train = segment_X[idx_train].reshape(-1, segment_X.shape[-1])
    y_train = np.repeat(y[idx_train], n_segments)
    start_time = time.perf_counter()
    model = make_mlp()
    model.fit(X_train, y_train)
    training_seconds = time.perf_counter() - start_time
    param_count = mlp_param_count(model)

    def averaged_proba(indices):
        flat = segment_X[indices].reshape(-1, segment_X.shape[-1])
        proba = model.predict_proba(flat).reshape(len(indices), n_segments, -1)
        return proba.mean(axis=1)

    val_proba = averaged_proba(idx_val)
    test_proba = averaged_proba(idx_test)
    return {
        "label": label,
        "val_true": y[idx_val],
        "val_pred": val_proba.argmax(1),
        "val_proba": val_proba,
        "test_true": y[idx_test],
        "test_pred": test_proba.argmax(1),
        "test_proba": test_proba,
        "test_indices": idx_test,
        "best_epoch": getattr(model.named_steps["clf"], "n_iter_", ""),
        "best_val_f1": "",
        "param_count": param_count,
        "trainable_param_count": param_count,
        "training_seconds": training_seconds,
        "epochs_run": getattr(model.named_steps["clf"], "n_iter_", ""),
    }


def fill_best_val_f1(results):
    from reporting_utils import compute_scores

    for result in results:
        result["best_val_f1"] = compute_scores(
            result["val_true"],
            result["val_pred"],
        )["f1_macro"]


def main():
    seed_everything()
    _mels, labels, track_ids = load_mel_cache()
    le = LabelEncoder()
    y = le.fit_transform(labels)
    idx_train, idx_val, idx_test = make_split(labels)[2:]
    full_embeddings, segment_embeddings = load_or_build_embeddings(track_ids, labels)

    print_section("5.2 PANNs-CNN14")
    print(
        f"Embeddings: full={full_embeddings.shape}  "
        f"segments={segment_embeddings.shape}  device={DEVICE}"
    )
    results = [
        train_full_embedding_model(full_embeddings, y, idx_train, idx_val, idx_test),
        train_segment_embedding_model(segment_embeddings, y, idx_train, idx_val, idx_test),
    ]
    fill_best_val_f1(results)
    finalize_experiment(
        results,
        OUT_DIR,
        le.classes_,
        "5.2 PANNs-CNN14",
        track_ids=track_ids,
    )


if __name__ == "__main__":
    main()
