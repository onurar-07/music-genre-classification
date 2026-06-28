"""Part 5.3: AST transfer learning with frozen AudioSet embeddings.

This experiment uses an AudioSet-pretrained Audio Spectrogram Transformer (AST)
as a frozen embedding extractor. AST is trained on 10-second audio clips, so the
full-track branch uses a center 10-second crop and the segment branch averages
four 10-second crops across each track.
"""

import time
import warnings

import librosa
import numpy as np
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

try:
    from transformers import ASTFeatureExtractor, ASTModel
except ImportError as exc:
    raise ImportError(
        "transfer_learning_ast.py requires transformers. "
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

OUT_DIR = experiment_dir("5.3 AST")
EMBEDDING_CACHE = ROOT / "features" / "ast_embeddings.npz"
MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

SAMPLE_RATE = 16000
FULL_DURATION = 29.0
CROP_DURATION = 10.0
SEGMENTS_PER_TRACK = 4
BATCH_SIZE = 8


def check_audio_files(track_ids):
    missing = [track_id for track_id in track_ids if not fma_audio_path(track_id).exists()]
    if missing:
        examples = ", ".join(str(int(track_id)) for track_id in missing[:5])
        raise FileNotFoundError(
            f"AST requires raw FMA audio under data/fma_small. "
            f"Missing {len(missing)} files. Examples: {examples}"
        )


def load_waveform(track_id):
    path = fma_audio_path(track_id)
    y, _sr = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=FULL_DURATION)
    target = int(FULL_DURATION * SAMPLE_RATE)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    return y[:target].astype(np.float32)


def center_crop(waveform):
    crop_samples = int(CROP_DURATION * SAMPLE_RATE)
    if len(waveform) < crop_samples:
        waveform = np.pad(waveform, (0, crop_samples - len(waveform)))
    start = max(0, (len(waveform) - crop_samples) // 2)
    return waveform[start:start + crop_samples].astype(np.float32)


def waveform_segments(waveform):
    crop_samples = int(CROP_DURATION * SAMPLE_RATE)
    if len(waveform) < crop_samples:
        waveform = np.pad(waveform, (0, crop_samples - len(waveform)))
    max_start = len(waveform) - crop_samples
    starts = np.linspace(0, max_start, SEGMENTS_PER_TRACK).round().astype(int)
    return np.stack(
        [waveform[start:start + crop_samples] for start in starts],
    ).astype(np.float32)


def load_ast_model():
    print_section("Load AudioSet-pretrained AST")
    print(f"Model: {MODEL_NAME}")
    feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_NAME)
    model = ASTModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return feature_extractor, model


@torch.no_grad()
def embed_clips(feature_extractor, model, clips):
    embeddings = []
    for start in range(0, len(clips), BATCH_SIZE):
        batch_clips = [clip.astype(np.float32) for clip in clips[start:start + BATCH_SIZE]]
        inputs = feature_extractor(
            batch_clips,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs["input_values"].to(DEVICE, non_blocking=True)
        output = model(input_values=input_values)
        if getattr(output, "pooler_output", None) is not None:
            embedding = output.pooler_output
        else:
            embedding = output.last_hidden_state[:, 0]
        embeddings.append(embedding.detach().cpu().numpy())
    return np.concatenate(embeddings, axis=0).astype(np.float32)


def build_embedding_cache(track_ids, labels):
    check_audio_files(track_ids)
    feature_extractor, model = load_ast_model()
    full_embeddings = []
    segment_embeddings = []

    print_section("Extract AST embeddings")
    print(
        f"Tracks: {len(track_ids)}  crop_duration={CROP_DURATION:.1f}s  "
        f"segments={SEGMENTS_PER_TRACK}x{CROP_DURATION:.1f}s"
    )
    for start in tqdm(range(0, len(track_ids), BATCH_SIZE), desc="AST batches"):
        batch_ids = track_ids[start:start + BATCH_SIZE]
        waveforms = [load_waveform(track_id) for track_id in batch_ids]
        full_clips = np.stack([center_crop(waveform) for waveform in waveforms])
        full_embeddings.append(embed_clips(feature_extractor, model, full_clips))

        segments = np.concatenate([waveform_segments(waveform) for waveform in waveforms])
        embedded_segments = embed_clips(feature_extractor, model, segments)
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
        model_name=np.array(MODEL_NAME),
        sample_rate=np.array(SAMPLE_RATE),
        crop_duration=np.array(CROP_DURATION),
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
            print_section("Load cached AST embeddings")
            print(f"Cache: {EMBEDDING_CACHE.relative_to(ROOT)}")
            return (
                data["full_embeddings"].astype(np.float32),
                data["segment_embeddings"].astype(np.float32),
            )
        print_section("AST embedding cache mismatch")
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
    label = "AST embeddings - MLP"
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
    label = "AST embeddings - MLP - Segment Averaging"
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

    print_section("5.3 AST")
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
        "5.3 AST",
        track_ids=track_ids,
    )


if __name__ == "__main__":
    main()
