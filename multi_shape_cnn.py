"""Part 2.3: multi-shape CNN filters for musical time-frequency structure."""

from cnn_training_utils import MultiShapeCNN, TrainConfig, finalize_experiment, prepare_data, run_model
from reporting_utils import experiment_dir, print_section

OUT_DIR = experiment_dir("2.3 Multi-shape CNN")


def model_factory(n_classes):
    return MultiShapeCNN(n_classes=n_classes)


def main():
    mels, _labels, track_ids, le, y, idx_train, idx_val, idx_test = prepare_data()
    print_section("2.3 Multi-shape CNN")
    result = run_model(
        model_factory,
        "Multi-shape CNN",
        mels,
        y,
        le,
        idx_train,
        idx_val,
        idx_test,
        train_cfg=TrainConfig(
            epochs=300,
            lr=1e-3,
            weight_decay=1e-4,
            patience=48,
            label_smoothing=0.1,
        ),
    )
    finalize_experiment([result], OUT_DIR, le.classes_, "2.3 Multi-shape CNN", track_ids=track_ids)


if __name__ == "__main__":
    main()
