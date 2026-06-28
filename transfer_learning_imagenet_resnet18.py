"""Part 5.1: transfer learning with an ImageNet-pretrained ResNet18.

This experiment reuses the cached log-mel spectrograms as spectrogram images.
It evaluates a frozen-backbone classifier, a partially fine-tuned backbone, and
a segment-averaged partially fine-tuned model using the same project split and
reporting format as the earlier CNN experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import ResNet18_Weights, resnet18
except ImportError as exc:
    raise ImportError(
        "transfer_learning_imagenet_resnet18.py requires torchvision. "
        "Install project dependencies with: pip3 install -r requirements.txt"
    ) from exc

from cnn_training_utils import (
    AugmentConfig,
    TrainConfig,
    finalize_experiment,
    load_mel_cache,
    make_split,
    prepare_data,
    run_model,
    seed_everything,
)
from reporting_utils import experiment_dir, print_section
from segment_averaging import load_segment_cache, run_segment_model

OUT_DIR = experiment_dir("5.1 ImageNet ResNet18")
RESIZE_SIZE = 224


class TransferResNet18(nn.Module):
    def __init__(
        self,
        n_classes=8,
        fine_tune="layer4",
        dropout=0.35,
        resize_size=RESIZE_SIZE,
    ):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.backbone = resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, n_classes),
        )
        self.resize_size = resize_size
        self.fine_tune = fine_tune
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.configure_trainable(fine_tune)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            trainable_children = {
                "classifier": {"fc"},
                "layer4": {"layer4", "fc"},
                "layer3": {"layer3", "layer4", "fc"},
                "all": {"conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"},
            }[self.fine_tune]
            for name, module in self.backbone.named_children():
                if name not in trainable_children:
                    module.eval()
        return self

    def configure_trainable(self, fine_tune):
        for param in self.backbone.parameters():
            param.requires_grad = False

        trainable_prefixes = {
            "classifier": ["fc"],
            "layer4": ["layer4", "fc"],
            "layer3": ["layer3", "layer4", "fc"],
            "all": [""],
        }
        if fine_tune not in trainable_prefixes:
            raise ValueError(
                "fine_tune must be one of: classifier, layer4, layer3, all"
            )
        prefixes = trainable_prefixes[fine_tune]
        for name, param in self.backbone.named_parameters():
            if any(name.startswith(prefix) for prefix in prefixes):
                param.requires_grad = True

    def spectrogram_to_image(self, x):
        # Convert z-scored log-mel inputs into ImageNet-style 3-channel images.
        x_min = x.amin(dim=(-2, -1), keepdim=True)
        x_max = x.amax(dim=(-2, -1), keepdim=True)
        x = (x - x_min) / (x_max - x_min).clamp_min(1e-6)
        x = x.repeat(1, 3, 1, 1)
        if x.shape[-1] != self.resize_size or x.shape[-2] != self.resize_size:
            x = F.interpolate(
                x,
                size=(self.resize_size, self.resize_size),
                mode="bilinear",
                align_corners=False,
            )
        return (x - self.imagenet_mean) / self.imagenet_std

    def forward(self, x):
        return self.backbone(self.spectrogram_to_image(x))


def frozen_factory(n_classes):
    return TransferResNet18(n_classes=n_classes, fine_tune="classifier")


def layer4_factory(n_classes):
    return TransferResNet18(n_classes=n_classes, fine_tune="layer4")


FROZEN_CFG = TrainConfig(
    epochs=120,
    batch_size=128,
    lr=3e-4,
    weight_decay=1e-4,
    patience=24,
    label_smoothing=0.05,
    optimizer="adamw",
    lr_patience=6,
)

LAYER4_CFG = TrainConfig(
    epochs=160,
    batch_size=96,
    lr=1e-4,
    weight_decay=1e-4,
    patience=32,
    label_smoothing=0.05,
    optimizer="adamw",
    lr_patience=8,
)

SEGMENT_LAYER4_CFG = TrainConfig(
    epochs=120,
    batch_size=96,
    lr=1e-4,
    weight_decay=1e-4,
    patience=24,
    label_smoothing=0.05,
    optimizer="adamw",
    lr_patience=6,
)

MIXUP_AUGMENT = AugmentConfig(specaugment=False, mixup=True, mixup_alpha=0.3)


def main():
    seed_everything()
    mels, labels, track_ids, le, y, idx_train, idx_val, idx_test = prepare_data()

    print_section("5.1 ImageNet ResNet18 - full-track mel")
    results = [
        run_model(
            frozen_factory,
            "ImageNet ResNet18 - frozen classifier",
            mels,
            y,
            le,
            idx_train,
            idx_val,
            idx_test,
            train_cfg=FROZEN_CFG,
            augment_cfg=MIXUP_AUGMENT,
        ),
        run_model(
            layer4_factory,
            "ImageNet ResNet18 - layer4 fine-tune",
            mels,
            y,
            le,
            idx_train,
            idx_val,
            idx_test,
            train_cfg=LAYER4_CFG,
            augment_cfg=MIXUP_AUGMENT,
        ),
    ]

    print_section("5.1 ImageNet ResNet18 - segment averaging")
    _mels, labels_for_segments, segment_track_ids = load_mel_cache()
    segment_le, segment_y, seg_train, seg_val, seg_test = make_split(labels_for_segments)
    if list(segment_le.classes_) != list(le.classes_):
        raise ValueError("Segment labels do not match mel-spectrogram labels.")
    segments, _segment_labels, segment_track_ids = load_segment_cache(
        segment_track_ids,
        labels_for_segments,
    )
    segment_cfg = {
        "factory": layer4_factory,
        "train_cfg": SEGMENT_LAYER4_CFG,
        "augment_cfg": MIXUP_AUGMENT,
    }
    results.append(
        run_segment_model(
            segment_cfg,
            "ImageNet ResNet18 - layer4 fine-tune - Segment Averaging",
            segments,
            segment_y,
            segment_le,
            seg_train,
            seg_val,
            seg_test,
        )
    )

    finalize_experiment(
        results,
        OUT_DIR,
        le.classes_,
        "5.1 ImageNet ResNet18",
        track_ids=track_ids,
    )


if __name__ == "__main__":
    main()
