"""Part 2.1: Plain CNN baseline and regularised variant."""

from reporting_utils import experiment_dir, print_section
from cnn_training_utils import (
    AugmentConfig,
    PlainCNN,
    PlainCNNRegularisation,
    TrainConfig,
    finalize_experiment,
    prepare_data,
    run_model,
)

OUT_DIR = experiment_dir("2.1 Plain CNN")


def plain_factory(n_classes):
    return PlainCNN(n_classes=n_classes)


def regularised_factory(n_classes):
    return PlainCNNRegularisation(n_classes=n_classes)


def main():
    mels, _labels, track_ids, le, y, idx_train, idx_val, idx_test = prepare_data()
    baseline_cfg = TrainConfig(epochs=300, lr=1e-3, weight_decay=0.0, patience=48)
    regularised_cfg = TrainConfig(
        epochs=300,
        lr=1e-3,
        weight_decay=1e-4,
        patience=48,
        label_smoothing=0.1,
    )

    print_section("2.1 Plain CNN")

    results = [
        run_model(
            plain_factory,
            "Plain CNN",
            mels,
            y,
            le,
            idx_train,
            idx_val,
            idx_test,
            train_cfg=baseline_cfg,
            augment_cfg=AugmentConfig(specaugment=False, mixup=False),
        ),
        run_model(
            regularised_factory,
            "Plain CNN - Regularisation",
            mels,
            y,
            le,
            idx_train,
            idx_val,
            idx_test,
            train_cfg=regularised_cfg,
            augment_cfg=AugmentConfig(specaugment=False, mixup=False),
        ),
    ]

    finalize_experiment(results, OUT_DIR, le.classes_, "2.1 Plain CNN", track_ids=track_ids)


if __name__ == "__main__":
    main()
