"""Step 3: Architecture improvement with a ResNet-style CNN."""

from reporting_utils import experiment_dir, print_section
from cnn_training_utils import (
    AugmentConfig,
    ResNetGenreCNN,
    TrainConfig,
    finalize_experiment,
    prepare_data,
    run_model,
)

OUT_DIR = experiment_dir("2.3 ResNet CNN")


def resnet_factory(n_classes):
    return ResNetGenreCNN(n_classes=n_classes)


def main():
    mels, _labels, track_ids, le, y, idx_train, idx_val, idx_test = prepare_data()

    print_section("2.3 ResNet CNN")

    result = run_model(
        resnet_factory,
        "ResNet CNN",
        mels,
        y,
        le,
        idx_train,
        idx_val,
        idx_test,
        train_cfg=TrainConfig(
            epochs=300,
            lr=1e-3,
            weight_decay=5e-4,
            patience=48,
            label_smoothing=0.10,
            optimizer="adamw",
        ),
        augment_cfg=AugmentConfig(specaugment=False, mixup=False),
    )

    finalize_experiment([result], OUT_DIR, le.classes_, "2.3 ResNet CNN", track_ids=track_ids)


if __name__ == "__main__":
    main()
