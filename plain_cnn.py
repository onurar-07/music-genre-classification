"""Step 1: Plain CNN baseline."""

from reporting_utils import experiment_dir, print_section
from cnn_training_utils import AugmentConfig, PlainCNN, TrainConfig, finalize_experiment, prepare_data, run_model

OUT_DIR = experiment_dir("2.1 Plain CNN")


def plain_factory(n_classes):
    return PlainCNN(n_classes=n_classes)


def main():
    mels, _labels, track_ids, le, y, idx_train, idx_val, idx_test = prepare_data()
    cfg = TrainConfig(epochs=60, lr=1e-3, weight_decay=1e-4, patience=12)

    print_section("2.1 Plain CNN")

    results = [
        run_model(
            plain_factory,
            "Plain CNN - no augmentation",
            mels,
            y,
            le,
            idx_train,
            idx_val,
            idx_test,
            train_cfg=cfg,
            augment_cfg=AugmentConfig(specaugment=False, mixup=False),
        )
    ]

    finalize_experiment(results, OUT_DIR, le.classes_, "2.1 Plain CNN", track_ids=track_ids)


if __name__ == "__main__":
    main()
