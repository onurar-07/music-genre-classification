"""Step 2: Regularisation ablation."""

from reporting_utils import experiment_dir, print_section
from cnn_training_utils import (
    AugmentConfig,
    RegularisedCNN,
    TrainConfig,
    finalize_experiment,
    prepare_data,
    run_model,
)

OUT_DIR = experiment_dir("2.2 Regularisation ablation")


def heavy_factory(n_classes):
    return RegularisedCNN(
        n_classes=n_classes,
        block_dropouts=(0.1, 0.2, 0.3),
        fc_dropouts=(0.6, 0.4),
    )


def moderate_factory(n_classes):
    return RegularisedCNN(
        n_classes=n_classes,
        block_dropouts=(0.05, 0.10, 0.15),
        fc_dropouts=(0.5, 0.3),
    )


def main():
    mels, _labels, track_ids, le, y, idx_train, idx_val, idx_test = prepare_data()

    print_section("2.2 Regularisation ablation")

    results = [
        run_model(
            heavy_factory,
            "Heavily Regularised CNN",
            mels,
            y,
            le,
            idx_train,
            idx_val,
            idx_test,
            train_cfg=TrainConfig(
                epochs=60,
                lr=5e-4,
                weight_decay=1e-3,
                patience=12,
                label_smoothing=0.1,
            ),
            augment_cfg=AugmentConfig(specaugment=False, mixup=False),
        ),
        run_model(
            moderate_factory,
            "Moderately Regularised CNN",
            mels,
            y,
            le,
            idx_train,
            idx_val,
            idx_test,
            train_cfg=TrainConfig(
                epochs=60,
                lr=1e-3,
                weight_decay=1e-4,
                patience=12,
                label_smoothing=0.1,
            ),
            augment_cfg=AugmentConfig(specaugment=False, mixup=False),
        ),
    ]

    finalize_experiment(results, OUT_DIR, le.classes_, "2.2 Regularisation ablation", track_ids=track_ids)


if __name__ == "__main__":
    main()
