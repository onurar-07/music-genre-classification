"""Part 5.3: AST transfer learning with frozen AudioSet embeddings.

This experiment uses an AudioSet-pretrained Audio Spectrogram Transformer (AST)
as a frozen embedding extractor. AST is trained on 10-second audio clips, so the
full-track branch uses a center 10-second crop and the segment branch averages
four 10-second crops across each track.
"""

import time
import warnings
import zipfile

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from cnn_training_utils import (
    DEVICE,
    clone_state_dict,
    finalize_experiment,
    load_mel_cache,
    make_split,
    seed_everything,
)
from reporting_utils import (
    ROOT,
    compute_scores,
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
DEFAULT_MIXUP_ALPHA = 0.2
BEST_SEGMENT_MIXUP_ALPHA = 0.5
BEST_SEGMENT_AGGREGATION = "logit"
ENSEMBLE_SEEDS = (42, 43, 44)
TORCH_MLP_BATCH_SIZE = 512
TORCH_MLP_EPOCHS = 120
TORCH_MLP_PATIENCE = 8
TORCH_MLP_LR = 1e-3
TORCH_MLP_WEIGHT_DECAY = 1e-3


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
    try:
        from transformers import ASTFeatureExtractor, ASTModel
    except ImportError as exc:
        raise ImportError(
            "Building AST embeddings requires transformers. "
            "Install project dependencies with: pip3 install -r requirements.txt"
        ) from exc

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


def load_labels_and_track_ids():
    if EMBEDDING_CACHE.exists():
        data = np.load(EMBEDDING_CACHE, allow_pickle=True)
        if {"labels", "track_ids"}.issubset(data.files):
            print_section("Load labels from cached AST embeddings")
            print(f"Cache: {EMBEDDING_CACHE.relative_to(ROOT)}")
            return data["labels"], data["track_ids"]

    try:
        _mels, labels, track_ids = load_mel_cache()
        return labels, track_ids
    except zipfile.BadZipFile as exc:
        raise zipfile.BadZipFile(
            "features/mel_specs.npz is corrupted, and labels/track_ids could not "
            "be recovered from features/ast_embeddings.npz."
        ) from exc


def make_mlp_classifier():
    return MLPClassifier(
        hidden_layer_sizes=(256,),
        activation="relu",
        alpha=3e-3,
        learning_rate_init=1e-3,
        max_iter=1,
        random_state=42,
    )


def mlp_param_count(model):
    clf = model.named_steps["clf"]
    return sum(w.size for w in clf.coefs_) + sum(b.size for b in clf.intercepts_)


class TorchEmbeddingMLP(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_sizes=(256, 64), dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def torch_param_count(model):
    return sum(param.numel() for param in model.parameters())


def soft_cross_entropy(logits, soft_targets):
    return -(soft_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def mixup_embeddings(X, y_soft, alpha=DEFAULT_MIXUP_ALPHA):
    if alpha <= 0:
        return X, y_soft
    lam = np.random.beta(alpha, alpha)
    indices = torch.randperm(X.size(0), device=X.device)
    mixed_X = lam * X + (1.0 - lam) * X[indices]
    mixed_y = lam * y_soft + (1.0 - lam) * y_soft[indices]
    return mixed_X, mixed_y


@torch.no_grad()
def predict_logits_torch(model, scaler, X, batch_size=2048):
    model.eval()
    X_scaled = scaler.transform(X).astype(np.float32)
    logits_list = []
    for start in range(0, len(X_scaled), batch_size):
        batch = torch.tensor(X_scaled[start:start + batch_size], dtype=torch.float32)
        logits = model(batch.to(DEVICE, non_blocking=True))
        logits_list.append(logits.cpu().numpy())
    return np.concatenate(logits_list, axis=0)


def softmax_np(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def predict_proba_torch(model, scaler, X, batch_size=2048):
    return softmax_np(predict_logits_torch(model, scaler, X, batch_size=batch_size))


def aggregate_segment_predictions(model, scaler, segment_X, indices, aggregation):
    n_segments = segment_X.shape[1]
    flat = segment_X[indices].reshape(-1, segment_X.shape[-1])
    flat_logits = predict_logits_torch(model, scaler, flat)
    logits = flat_logits.reshape(len(indices), n_segments, -1)
    if aggregation == "probability":
        probabilities = softmax_np(flat_logits).reshape(len(indices), n_segments, -1)
        return probabilities.mean(axis=1)
    if aggregation == "logit":
        return softmax_np(logits.mean(axis=1))
    raise ValueError(f"Unknown segment aggregation: {aggregation}")


def train_torch_mlp_with_mixup(
    X_train,
    y_train,
    n_classes,
    predict_train,
    predict_val,
    mixup_alpha=DEFAULT_MIXUP_ALPHA,
    seed=42,
    max_epochs=TORCH_MLP_EPOCHS,
    patience=TORCH_MLP_PATIENCE,
    tol=1e-4,
):
    seed_everything(seed)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train).astype(np.float32)
    train_dataset = TensorDataset(
        torch.tensor(X_scaled, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=TORCH_MLP_BATCH_SIZE,
        shuffle=True,
        pin_memory=DEVICE.type == "cuda",
    )
    model = TorchEmbeddingMLP(X_train.shape[1], n_classes).to(DEVICE)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=TORCH_MLP_LR,
        weight_decay=TORCH_MLP_WEIGHT_DECAY,
    )
    history = {"train_loss": [], "train_acc": [], "val_acc": [], "val_f1": [], "lr": []}
    best_state, best_val_f1, best_epoch, no_improve = None, -1.0, 0, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss, seen = 0.0, 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE, non_blocking=True)
            y_batch = y_batch.to(DEVICE, non_blocking=True)
            y_soft = F.one_hot(y_batch, num_classes=n_classes).float()
            X_mix, y_mix = mixup_embeddings(X_batch, y_soft, alpha=mixup_alpha)

            optimizer.zero_grad(set_to_none=True)
            logits = model(X_mix)
            loss = soft_cross_entropy(logits, y_mix)
            loss.backward()
            optimizer.step()

            batch_size = X_batch.size(0)
            running_loss += float(loss.item()) * batch_size
            seen += batch_size

        train_pred = predict_train(model, scaler)
        val_true, val_pred = predict_val(model, scaler)
        val_scores = compute_scores(val_true, val_pred)
        train_acc = float((train_pred[0] == train_pred[1]).mean())
        val_acc = float(val_scores["accuracy"])
        val_f1 = float(val_scores["f1_macro"])
        history["train_loss"].append(running_loss / max(1, seen))
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if val_f1 > best_val_f1 + tol:
            best_state = clone_state_dict(model)
            best_val_f1 = val_f1
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler, history, best_epoch


def fit_mlp_with_history(
    X_train,
    y_train,
    score_train,
    score_val,
    max_epochs=500,
    patience=5,
    tol=1e-4,
):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    clf = make_mlp_classifier()
    classes = np.unique(y_train)
    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    best_state, best_val, best_epoch, no_improve = None, -1.0, 0, 0

    for epoch in range(1, max_epochs + 1):
        if epoch == 1:
            clf.partial_fit(X_scaled, y_train, classes=classes)
        else:
            clf.partial_fit(X_scaled, y_train)

        model = Pipeline([("scaler", scaler), ("clf", clf)])
        train_acc = score_train(model)
        val_acc = score_val(model)
        history["train_loss"].append(float(clf.loss_))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))

        if val_acc > best_val + tol:
            best_state = (
                [coef.copy() for coef in clf.coefs_],
                [intercept.copy() for intercept in clf.intercepts_],
            )
            best_val = float(val_acc)
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        clf.coefs_, clf.intercepts_ = best_state
    clf.n_iter_ = len(history["train_loss"])
    clf.loss_curve_ = history["train_loss"]
    clf.validation_scores_ = history["val_acc"]
    model = Pipeline([("scaler", scaler), ("clf", clf)])
    return model, history, best_epoch


def train_full_embedding_model(X, y, idx_train, idx_val, idx_test):
    label = "AST embeddings - MLP"
    start_time = time.perf_counter()
    model, history, best_epoch = fit_mlp_with_history(
        X[idx_train],
        y[idx_train],
        score_train=lambda model: model.score(X[idx_train], y[idx_train]),
        score_val=lambda model: model.score(X[idx_val], y[idx_val]),
    )
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
        "best_epoch": best_epoch,
        "best_val_f1": "",
        "param_count": param_count,
        "trainable_param_count": param_count,
        "training_seconds": training_seconds,
        "epochs_run": getattr(model.named_steps["clf"], "n_iter_", ""),
        "history": history,
    }


def train_segment_embedding_model(segment_X, y, idx_train, idx_val, idx_test):
    label = "AST embeddings - MLP - Segment Averaging"
    n_segments = segment_X.shape[1]
    X_train = segment_X[idx_train].reshape(-1, segment_X.shape[-1])
    y_train = np.repeat(y[idx_train], n_segments)

    def averaged_proba(model, indices):
        flat = segment_X[indices].reshape(-1, segment_X.shape[-1])
        proba = model.predict_proba(flat).reshape(len(indices), n_segments, -1)
        return proba.mean(axis=1)

    start_time = time.perf_counter()
    model, history, best_epoch = fit_mlp_with_history(
        X_train,
        y_train,
        score_train=lambda model: (averaged_proba(model, idx_train).argmax(1) == y[idx_train]).mean(),
        score_val=lambda model: (averaged_proba(model, idx_val).argmax(1) == y[idx_val]).mean(),
    )
    training_seconds = time.perf_counter() - start_time
    param_count = mlp_param_count(model)

    val_proba = averaged_proba(model, idx_val)
    test_proba = averaged_proba(model, idx_test)
    return {
        "label": label,
        "val_true": y[idx_val],
        "val_pred": val_proba.argmax(1),
        "val_proba": val_proba,
        "test_true": y[idx_test],
        "test_pred": test_proba.argmax(1),
        "test_proba": test_proba,
        "test_indices": idx_test,
        "best_epoch": best_epoch,
        "best_val_f1": "",
        "param_count": param_count,
        "trainable_param_count": param_count,
        "training_seconds": training_seconds,
        "epochs_run": getattr(model.named_steps["clf"], "n_iter_", ""),
        "history": history,
    }


def train_segment_embedding_mixup_model(
    segment_X,
    y,
    idx_train,
    idx_val,
    idx_test,
    mixup_alpha=DEFAULT_MIXUP_ALPHA,
    aggregation="probability",
    seed=42,
):
    aggregation_label = (
        "Segment Averaging"
        if aggregation == "probability"
        else "Segment Logit Averaging"
    )
    label = f"AST embeddings - Mixup - {aggregation_label}"
    n_classes = int(np.max(y)) + 1
    n_segments = segment_X.shape[1]
    X_train = segment_X[idx_train].reshape(-1, segment_X.shape[-1])
    y_train = np.repeat(y[idx_train], n_segments)

    def predict_indices(model, scaler, indices):
        proba = aggregate_segment_predictions(model, scaler, segment_X, indices, aggregation)
        return y[indices], proba.argmax(1), proba

    start_time = time.perf_counter()
    model, scaler, history, best_epoch = train_torch_mlp_with_mixup(
        X_train,
        y_train,
        n_classes,
        mixup_alpha=mixup_alpha,
        seed=seed,
        predict_train=lambda model, scaler: (
            predict_indices(model, scaler, idx_train)[1],
            y[idx_train],
        ),
        predict_val=lambda model, scaler: predict_indices(model, scaler, idx_val)[:2],
    )
    training_seconds = time.perf_counter() - start_time
    param_count = torch_param_count(model)

    val_true, val_pred, val_proba = predict_indices(model, scaler, idx_val)
    test_true, test_pred, test_proba = predict_indices(model, scaler, idx_test)
    return {
        "label": label,
        "val_true": val_true,
        "val_pred": val_pred,
        "val_proba": val_proba,
        "test_true": test_true,
        "test_pred": test_pred,
        "test_proba": test_proba,
        "test_indices": idx_test,
        "best_epoch": best_epoch,
        "best_val_f1": "",
        "param_count": param_count,
        "trainable_param_count": param_count,
        "training_seconds": training_seconds,
        "epochs_run": len(history["train_loss"]),
        "history": history,
        "seed": seed,
    }


def train_segment_embedding_mixup_ensemble_model(
    segment_X,
    y,
    idx_train,
    idx_val,
    idx_test,
    member_results=None,
    seeds=ENSEMBLE_SEEDS,
):
    label = "AST embeddings - Mixup Ensemble - Segment Logit Averaging"
    member_by_seed = {
        result.get("seed"): result
        for result in (member_results or [])
        if result.get("seed") in seeds
    }
    members = []
    for seed in seeds:
        if seed not in member_by_seed:
            member_by_seed[seed] = train_segment_embedding_mixup_model(
                segment_X,
                y,
                idx_train,
                idx_val,
                idx_test,
                mixup_alpha=BEST_SEGMENT_MIXUP_ALPHA,
                aggregation=BEST_SEGMENT_AGGREGATION,
                seed=seed,
            )
        members.append(member_by_seed[seed])

    val_proba = np.mean([member["val_proba"] for member in members], axis=0)
    test_proba = np.mean([member["test_proba"] for member in members], axis=0)
    return {
        "label": label,
        "val_true": members[0]["val_true"],
        "val_pred": val_proba.argmax(1),
        "val_proba": val_proba,
        "test_true": members[0]["test_true"],
        "test_pred": test_proba.argmax(1),
        "test_proba": test_proba,
        "test_indices": idx_test,
        "best_epoch": "/".join(str(member["best_epoch"]) for member in members),
        "best_val_f1": "",
        "param_count": sum(member["param_count"] for member in members),
        "trainable_param_count": sum(member["trainable_param_count"] for member in members),
        "training_seconds": sum(member["training_seconds"] for member in members),
        "epochs_run": "/".join(str(member["epochs_run"]) for member in members),
    }


def fill_best_val_f1(results):

    for result in results:
        result["best_val_f1"] = compute_scores(
            result["val_true"],
            result["val_pred"],
        )["f1_macro"]


def main():
    seed_everything()
    labels, track_ids = load_labels_and_track_ids()
    le = LabelEncoder()
    y = le.fit_transform(labels)
    idx_train, idx_val, idx_test = make_split(labels)[2:]
    full_embeddings, segment_embeddings = load_or_build_embeddings(track_ids, labels)

    print_section("5.3 AST")
    print(
        f"Embeddings: full={full_embeddings.shape}  "
        f"segments={segment_embeddings.shape}  device={DEVICE}"
    )
    mixup_result = train_segment_embedding_mixup_model(
        segment_embeddings,
        y,
        idx_train,
        idx_val,
        idx_test,
        mixup_alpha=BEST_SEGMENT_MIXUP_ALPHA,
        aggregation=BEST_SEGMENT_AGGREGATION,
        seed=ENSEMBLE_SEEDS[0],
    )
    results = [
        train_full_embedding_model(full_embeddings, y, idx_train, idx_val, idx_test),
        train_segment_embedding_model(segment_embeddings, y, idx_train, idx_val, idx_test),
        mixup_result,
        train_segment_embedding_mixup_ensemble_model(
            segment_embeddings,
            y,
            idx_train,
            idx_val,
            idx_test,
            member_results=[mixup_result],
        ),
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
