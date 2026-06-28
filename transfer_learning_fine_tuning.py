"""Part 5.4: fine-tune the AudioSet-pretrained AST model.

The transfer-learning comparison showed AST as the strongest pretrained audio
backbone, so this script is intentionally AST-only. It fine-tunes a small
trainable subset of AST on raw FMA audio and keeps the same reporting format as
the other experiments. AST feature-extractor inputs are cached in
features/ast_inputs.npz to avoid repeated MP3 decoding and CPU preprocessing
across epochs.
"""

import time
import warnings
from dataclasses import dataclass

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
except ImportError as exc:
    raise ImportError(
        "transfer_learning_fine_tuning.py requires transformers. "
        "Install project dependencies with: pip3 install -r requirements.txt"
    ) from exc

from cnn_training_utils import (
    DEVICE,
    clone_state_dict,
    count_parameters,
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

OUT_DIR = experiment_dir("5.4 Fine Tuning")
AST_INPUT_CACHE = ROOT / "features" / "ast_inputs.npz"
MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

SAMPLE_RATE = 16000
FULL_DURATION = 29.0
CLIP_DURATION = 10.0
SEGMENTS_PER_TRACK = 4
USE_SEGMENT_AVERAGING = True


@dataclass
class FineTuneConfig:
    epochs: int = 30
    batch_size: int = 48
    cache_batch_size: int = 24
    unfrozen_encoder_blocks: int = 1
    lr: float = 2e-5
    head_lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 8
    num_workers: int = 2
    amp: bool = True
    label_smoothing: float = 0.05


def check_audio_files(track_ids):
    missing = [track_id for track_id in track_ids if not fma_audio_path(track_id).exists()]
    if missing:
        examples = ", ".join(str(int(track_id)) for track_id in missing[:5])
        raise FileNotFoundError(
            "AST fine-tuning requires raw FMA audio under data/fma_small. "
            f"Missing {len(missing)} files. Examples: {examples}"
        )


def load_waveform(track_id):
    y, _sr = librosa.load(
        fma_audio_path(track_id),
        sr=SAMPLE_RATE,
        mono=True,
        duration=FULL_DURATION,
    )
    target = int(FULL_DURATION * SAMPLE_RATE)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    return y[:target].astype(np.float32)


def center_crop(waveform):
    clip_samples = int(CLIP_DURATION * SAMPLE_RATE)
    if len(waveform) < clip_samples:
        waveform = np.pad(waveform, (0, clip_samples - len(waveform)))
    start = max(0, (len(waveform) - clip_samples) // 2)
    return waveform[start:start + clip_samples].astype(np.float32)


def waveform_segments(waveform):
    clip_samples = int(CLIP_DURATION * SAMPLE_RATE)
    if len(waveform) < clip_samples:
        waveform = np.pad(waveform, (0, clip_samples - len(waveform)))
    max_start = len(waveform) - clip_samples
    starts = np.linspace(0, max_start, SEGMENTS_PER_TRACK).round().astype(int)
    return np.stack(
        [waveform[start:start + clip_samples] for start in starts],
    ).astype(np.float32)


class ASTInputDataset(Dataset):
    def __init__(
        self,
        full_inputs,
        labels,
        indices,
        use_segments,
        segment_inputs=None,
        segment_indices=None,
    ):
        self.full_inputs = full_inputs
        self.labels = np.asarray(labels)
        self.indices = np.asarray(indices)
        self.use_segments = use_segments
        self.segment_inputs = segment_inputs
        if segment_indices is None:
            self.segment_lookup = {}
        else:
            self.segment_lookup = {
                int(index): pos for pos, index in enumerate(np.asarray(segment_indices))
            }

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        idx = int(self.indices[item])
        if self.use_segments:
            pos = self.segment_lookup[idx]
            inputs = self.segment_inputs[pos]
        else:
            inputs = self.full_inputs[idx]
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(int(self.labels[idx]))


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


def load_feature_extractor():
    return ASTFeatureExtractor.from_pretrained(MODEL_NAME)


def ast_input_values(feature_extractor, clips):
    batch_clips = [clip.astype(np.float32) for clip in clips]
    inputs = feature_extractor(
        batch_clips,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    return inputs["input_values"].numpy().astype(np.float16)


def build_ast_input_cache(track_ids, labels, eval_indices, feature_extractor, cfg):
    check_audio_files(track_ids)
    AST_INPUT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    eval_indices = np.asarray(sorted(int(index) for index in eval_indices))
    full_inputs = None
    segment_inputs = None

    print_section("Build cached AST inputs")
    print(f"Cache: {AST_INPUT_CACHE.relative_to(ROOT)}")

    for start in tqdm(range(0, len(track_ids), cfg.cache_batch_size), desc="center crops"):
        batch_ids = track_ids[start:start + cfg.cache_batch_size]
        waveforms = [load_waveform(track_id) for track_id in batch_ids]
        clips = [center_crop(waveform) for waveform in waveforms]
        batch_inputs = ast_input_values(feature_extractor, clips)
        if full_inputs is None:
            full_inputs = np.empty(
                (len(track_ids),) + batch_inputs.shape[1:],
                dtype=np.float16,
            )
        full_inputs[start:start + len(batch_ids)] = batch_inputs

    for start in tqdm(range(0, len(eval_indices), cfg.cache_batch_size), desc="eval segments"):
        batch_indices = eval_indices[start:start + cfg.cache_batch_size]
        waveforms = [load_waveform(track_ids[index]) for index in batch_indices]
        clips = np.concatenate([waveform_segments(waveform) for waveform in waveforms])
        batch_inputs = ast_input_values(feature_extractor, clips)
        batch_inputs = batch_inputs.reshape(len(batch_indices), SEGMENTS_PER_TRACK, *batch_inputs.shape[1:])
        if segment_inputs is None:
            segment_inputs = np.empty(
                (len(eval_indices), SEGMENTS_PER_TRACK) + batch_inputs.shape[2:],
                dtype=np.float16,
            )
        segment_inputs[start:start + len(batch_indices)] = batch_inputs

    np.savez(
        AST_INPUT_CACHE,
        track_ids=track_ids,
        labels=labels,
        eval_indices=eval_indices,
        full_inputs=full_inputs,
        segment_inputs=segment_inputs,
        model_name=np.array(MODEL_NAME),
        sample_rate=np.array(SAMPLE_RATE),
        clip_duration=np.array(CLIP_DURATION),
        segments_per_track=np.array(SEGMENTS_PER_TRACK),
    )
    print(f"Saved: {AST_INPUT_CACHE.relative_to(ROOT)}")
    return full_inputs, eval_indices, segment_inputs


def load_or_build_ast_inputs(track_ids, labels, eval_indices, feature_extractor, cfg):
    eval_indices = np.asarray(sorted(int(index) for index in eval_indices))
    if AST_INPUT_CACHE.exists():
        data = np.load(AST_INPUT_CACHE, allow_pickle=True)
        required = {"track_ids", "labels", "eval_indices", "full_inputs", "segment_inputs"}
        cache_ok = (
            required.issubset(data.files)
            and np.array_equal(data["track_ids"], track_ids)
            and np.array_equal(data["labels"], labels)
            and np.array_equal(data["eval_indices"], eval_indices)
            and (
                "segments_per_track" not in data.files
                or int(data["segments_per_track"]) == SEGMENTS_PER_TRACK
            )
        )
        if cache_ok:
            print_section("Load cached AST inputs")
            print(f"Cache: {AST_INPUT_CACHE.relative_to(ROOT)}")
            return (
                data["full_inputs"],
                data["eval_indices"],
                data["segment_inputs"],
            )
        print_section("AST input cache mismatch")
        print(f"Ignoring stale cache: {AST_INPUT_CACHE.relative_to(ROOT)}")

    return build_ast_input_cache(track_ids, labels, eval_indices, feature_extractor, cfg)


def set_trainable(module, trainable=True):
    for param in module.parameters():
        param.requires_grad = trainable


def trainable_parameter_count(module):
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def module_parameter_count(module):
    return sum(param.numel() for param in module.parameters())


def is_module_stack(value):
    if isinstance(value, (nn.ModuleList, nn.Sequential)):
        return len(value) > 0 and all(module_parameter_count(module) > 0 for module in value)
    if isinstance(value, (list, tuple)):
        return (
            len(value) > 0
            and all(isinstance(module, nn.Module) for module in value)
            and all(module_parameter_count(module) > 0 for module in value)
        )
    return False


def get_encoder_layers(model):
    candidates = []
    for module_name, module in model.named_modules():
        if "classifier" in module_name:
            continue
        for attr in ("layer", "layers"):
            layers = getattr(module, attr, None)
            if is_module_stack(layers):
                layer_name = f"{module_name}.{attr}" if module_name else attr
                total_params = sum(module_parameter_count(layer) for layer in layers)
                candidates.append((len(layers), total_params, layer_name, layers))

    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        _n_layers, _total_params, layer_name, layers = candidates[0]
        return layers, layer_name

    module_examples = ", ".join(name for name, _module in list(model.named_modules())[:30])
    raise AttributeError(
        "Could not find an AST encoder layer stack. "
        f"First model modules: {module_examples or '(none)'}"
    )


def get_final_layernorm(model):
    for name, module in model.named_modules():
        lower_name = name.lower()
        if "classifier" in lower_name or "encoder" in lower_name:
            continue
        if isinstance(module, nn.LayerNorm) and (
            lower_name.endswith("layernorm") or lower_name.endswith("layer_norm")
        ):
            return module, name
    return None, "final_layernorm"


def configure_trainable_ast_layers(model, cfg):
    set_trainable(model, False)
    encoder_layers, encoder_layer_name = get_encoder_layers(model)
    n_layers = len(encoder_layers)
    n_unfrozen = min(cfg.unfrozen_encoder_blocks, n_layers)
    if n_unfrozen <= 0:
        raise ValueError("unfrozen_encoder_blocks must be positive.")

    for block in encoder_layers[-n_unfrozen:]:
        set_trainable(block, True)

    final_layernorm, final_layernorm_name = get_final_layernorm(model)
    if final_layernorm is not None:
        set_trainable(final_layernorm, True)

    set_trainable(model.classifier, True)

    return {
        "encoder_blocks": [
            {
                "name": f"{encoder_layer_name}.{idx}",
                "params": module_parameter_count(encoder_layers[idx]),
                "trainable": trainable_parameter_count(encoder_layers[idx]),
            }
            for idx in range(n_layers - n_unfrozen, n_layers)
        ],
        "final_layernorm": {
            "name": final_layernorm_name,
            "params": module_parameter_count(final_layernorm) if final_layernorm is not None else 0,
            "trainable": trainable_parameter_count(final_layernorm) if final_layernorm is not None else 0,
        },
        "classifier": {
            "name": "classifier",
            "params": module_parameter_count(model.classifier),
            "trainable": trainable_parameter_count(model.classifier),
        },
    }


def print_trainable_summary(summary):
    rows = []
    rows.extend(summary["encoder_blocks"])
    rows.append(summary["final_layernorm"])
    rows.append(summary["classifier"])
    print_section("Trainable AST modules")
    for row in rows:
        print(f"{row['name']}: {row['trainable']:,}/{row['params']:,} trainable")


def configure_ast_model(n_classes, cfg):
    model = ASTForAudioClassification.from_pretrained(
        MODEL_NAME,
        num_labels=n_classes,
        ignore_mismatched_sizes=True,
    )
    trainable_summary = configure_trainable_ast_layers(model, cfg)

    model.to(DEVICE)
    return model, trainable_summary


def make_optimizer(model, cfg):
    head_params = []
    body_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name:
            head_params.append(param)
        else:
            body_params.append(param)

    groups = []
    if body_params:
        groups.append({"params": body_params, "lr": cfg.lr, "name": "body"})
    if head_params:
        groups.append({"params": head_params, "lr": cfg.head_lr, "name": "head"})
    return optim.AdamW(groups, weight_decay=cfg.weight_decay)


def optimizer_lrs(optimizer):
    values = {group.get("name", f"group{idx}"): group["lr"] for idx, group in enumerate(optimizer.param_groups)}
    return {
        "body": values.get("body", optimizer.param_groups[0]["lr"]),
        "head": values.get("head", values.get("body", optimizer.param_groups[0]["lr"])),
    }


def format_optimizer_lrs(optimizer):
    lrs = optimizer_lrs(optimizer)
    return f"{lrs['body']:.2e}/{lrs['head']:.2e}"


def ast_logits(model, input_values):
    input_values = input_values.to(DEVICE, dtype=torch.float32, non_blocking=True)
    return model(input_values=input_values).logits


def flatten_segment_batch(inputs, y):
    batch_size, n_segments = inputs.shape[:2]
    flat_inputs = inputs.reshape(batch_size * n_segments, *inputs.shape[2:])
    y_rep = y.repeat_interleave(n_segments)
    return flat_inputs, y_rep, batch_size, n_segments


def train_epoch(model, loader, optimizer, criterion, scaler, use_amp, use_segments):
    model.train()
    total_loss, correct, n, n_loss = 0.0, 0, 0, 0
    for inputs, y in tqdm(loader, desc="train", leave=False):
        y = y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_segments:
            forward_inputs, loss_y, batch_size, n_segments = flatten_segment_batch(inputs, y)
        else:
            forward_inputs, loss_y = inputs, y
            batch_size, n_segments = len(y), 1

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = ast_logits(model, forward_inputs)
            loss = criterion(logits, loss_y)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        track_logits = logits.reshape(batch_size, n_segments, -1).mean(dim=1)
        correct += (track_logits.argmax(1) == y).sum().item()
        total_loss += loss.item() * len(loss_y)
        n_loss += len(loss_y)
        n += len(y)

    return total_loss / max(1, n_loss), correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, use_segments):
    model.eval()
    total_loss, n, n_loss = 0.0, 0, 0
    all_true, all_pred, all_proba = [], [], []

    for inputs, y in tqdm(loader, desc="eval", leave=False):
        y_dev = y.to(DEVICE, non_blocking=True)
        if use_segments:
            forward_inputs, loss_y, batch_size, n_segments = flatten_segment_batch(inputs, y_dev)
        else:
            forward_inputs, loss_y = inputs, y_dev
            batch_size, n_segments = len(y), 1

        logits = ast_logits(model, forward_inputs)
        loss = criterion(logits, loss_y)
        proba = torch.softmax(logits, dim=1).reshape(batch_size, n_segments, -1)
        avg_proba = proba.mean(dim=1).detach().cpu().numpy()
        pred = avg_proba.argmax(1)

        all_true.extend(y.numpy())
        all_pred.extend(pred)
        all_proba.extend(avg_proba)
        total_loss += loss.item() * len(loss_y)
        n_loss += len(loss_y)
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


def run_ast_fine_tuning(track_ids, y, le, idx_train, idx_val, idx_test):
    cfg = FineTuneConfig()
    feature_extractor = load_feature_extractor()
    eval_indices = np.concatenate([idx_val, idx_test])
    full_inputs, segment_indices, segment_inputs = load_or_build_ast_inputs(
        track_ids,
        y,
        eval_indices,
        feature_extractor,
        cfg,
    )
    model, trainable_summary = configure_ast_model(len(le.classes_), cfg)
    param_count, trainable_param_count = count_parameters(model)
    label = "Fine-tuned AST"
    if USE_SEGMENT_AVERAGING:
        label += " - Segment Averaging"

    print_section(label)
    print(
        f"model={MODEL_NAME} parameters={param_count:,} "
        f"trainable={trainable_param_count:,} sample_rate={SAMPLE_RATE} "
        f"clip={CLIP_DURATION:.1f}s segments={SEGMENTS_PER_TRACK if USE_SEGMENT_AVERAGING else 1}"
    )
    print_trainable_summary(trainable_summary)

    train_loader = DataLoader(
        ASTInputDataset(full_inputs, y, idx_train, use_segments=False),
        **loader_kwargs(cfg.batch_size, shuffle=True, num_workers=cfg.num_workers),
    )
    val_loader = DataLoader(
        ASTInputDataset(
            full_inputs,
            y,
            idx_val,
            use_segments=USE_SEGMENT_AVERAGING,
            segment_inputs=segment_inputs,
            segment_indices=segment_indices,
        ),
        **loader_kwargs(cfg.batch_size, shuffle=False, num_workers=cfg.num_workers),
    )
    test_loader = DataLoader(
        ASTInputDataset(
            full_inputs,
            y,
            idx_test,
            use_segments=USE_SEGMENT_AVERAGING,
            segment_inputs=segment_inputs,
            segment_indices=segment_indices,
        ),
        **loader_kwargs(cfg.batch_size, shuffle=False, num_workers=cfg.num_workers),
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
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
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "lr": [],
        "head_lr": [],
    }

    best_state, best_epoch, best_val_f1 = None, 0, -1.0
    best_val_true, best_val_pred, best_val_proba = None, None, None
    no_improve = 0
    start_time = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            use_amp,
            False,
        )
        val_loss, val_acc, val_f1, val_true, val_pred, val_proba = evaluate(
            model,
            val_loader,
            criterion,
            USE_SEGMENT_AVERAGING,
        )
        scheduler.step(val_f1)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        lrs = optimizer_lrs(optimizer)
        history["lr"].append(lrs["body"])
        history["head_lr"].append(lrs["head"])

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
            f"val_f1={val_f1:.4f} lr={format_optimizer_lrs(optimizer)}",
            flush=True,
        )
        if no_improve >= cfg.patience:
            print(f"  Early stop at epoch {epoch} (best epoch {best_epoch})", flush=True)
            break

    training_seconds = time.perf_counter() - start_time
    model.load_state_dict(best_state)
    _, test_acc, test_f1, test_true, test_pred, test_proba = evaluate(
        model,
        test_loader,
        criterion,
        USE_SEGMENT_AVERAGING,
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
    _mels, labels, track_ids = load_mel_cache()
    le, y, idx_train, idx_val, idx_test = make_split(labels)
    result = run_ast_fine_tuning(track_ids, y, le, idx_train, idx_val, idx_test)
    finalize_experiment(
        [result],
        OUT_DIR,
        le.classes_,
        "5.4 Fine Tuning",
        track_ids=track_ids,
    )


if __name__ == "__main__":
    main()
