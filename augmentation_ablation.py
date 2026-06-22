"""Part 2.4: augmentation ablation on the best selected CNN architecture.

This script first compares the validation results from:
  - Part 2.1: Plain CNN
  - Part 2.2: Regularisation ablation
  - Part 2.3: ResNet CNN

It then selects the best architecture by validation F1-macro and runs the
augmentation ablation on that selected model:
  - Uses the already-trained selected model as the no-augmentation baseline
  - Trains SpecAugment
  - Trains Mixup
  - Trains SpecAugment + Mixup
"""

import pandas as pd

from reporting_utils import (
    RESULTS_ROOT,
    experiment_dir,
    plot_metrics,
    print_metrics_summary,
    print_saved_outputs,
    print_section,
)
from cnn_training_utils import (
    AugmentConfig,
    PlainCNN,
    RegularisedCNN,
    ResNetGenreCNN,
    TrainConfig,
    finalize_experiment,
    prepare_data,
    run_model,
)

OUT_DIR = experiment_dir("2.4 Augmentation ablation")


def heavy_factory(n_classes):
    return RegularisedCNN(
        n_classes=n_classes,
        block_dropouts=(0.1, 0.2, 0.3),
        fc_dropouts=(0.6, 0.4),
    )


def plain_factory(n_classes):
    return PlainCNN(n_classes=n_classes)


def moderate_factory(n_classes):
    return RegularisedCNN(
        n_classes=n_classes,
        block_dropouts=(0.05, 0.10, 0.15),
        fc_dropouts=(0.5, 0.3),
    )


def resnet_factory(n_classes):
    return ResNetGenreCNN(n_classes=n_classes)


MODEL_CONFIGS = {
    "Plain CNN": {
        "factory": plain_factory,
        "train_cfg": TrainConfig(
            epochs=300,
            lr=1e-3,
            weight_decay=1e-4,
            patience=48,
        ),
        "spec_t": 25,
        "spec_f": 15,
    },
    "Heavily Regularised CNN": {
        "factory": heavy_factory,
        "train_cfg": TrainConfig(
            epochs=300,
            lr=1e-3,
            weight_decay=1e-3,
            patience=48,
            label_smoothing=0.1,
        ),
        "spec_t": 40,
        "spec_f": 25,
    },
    "Moderately Regularised CNN": {
        "factory": moderate_factory,
        "train_cfg": TrainConfig(
            epochs=300,
            lr=1e-3,
            weight_decay=1e-4,
            patience=48,
            label_smoothing=0.1,
        ),
        "spec_t": 30,
        "spec_f": 20,
    },
    "ResNet CNN": {
        "factory": resnet_factory,
        "train_cfg": TrainConfig(
            epochs=300,
            lr=1e-3,
            weight_decay=5e-4,
            patience=48,
            label_smoothing=0.10,
            optimizer="adamw",
        ),
        "spec_t": 30,
        "spec_f": 20,
    },
}


def load_candidate_metrics():
    paths = [
        RESULTS_ROOT / "2.1 Plain CNN" / "metrics.csv",
        RESULTS_ROOT / "2.2 Regularisation ablation" / "metrics.csv",
        RESULTS_ROOT / "2.3 ResNet CNN" / "metrics.csv",
    ]
    missing = [p for p in paths if not p.exists()]
    if missing:
        missing_text = "\n".join(f"  {p}" for p in missing)
        raise FileNotFoundError(
            "Run plain_cnn.py, regularisation_ablation.py, and resnet_cnn.py before augmentation_ablation.py.\n"
            f"Missing:\n{missing_text}"
        )

    rows = []
    for path in paths:
        df = pd.read_csv(path)
        val_df = df[df["split"] == "val"].copy()
        val_df["source"] = path.parent.name
        rows.append(val_df)
    candidates = pd.concat(rows, ignore_index=True)
    candidates = candidates[candidates["model"].isin(MODEL_CONFIGS)]
    if candidates.empty:
        raise ValueError("No known candidate models found in 2.1/2.2/2.3 metrics.")
    return candidates.sort_values("f1_macro", ascending=False)


def select_model():
    candidates = load_candidate_metrics()
    candidates.to_csv(OUT_DIR / "selection_candidates.csv", index=False)
    selected = candidates.iloc[0]
    selected_model = selected["model"]

    lines = [
        "Architecture/model selection before augmentation ablation",
        "",
        "Candidates ranked by validation F1-macro:",
        candidates[["source", "model", "accuracy", "f1_macro", "best_epoch"]].to_string(index=False),
        "",
        f"Selected model: {selected_model}",
        f"Selected validation F1-macro: {selected['f1_macro']:.4f}",
    ]
    (OUT_DIR / "selected_model.txt").write_text("\n".join(lines), encoding="utf-8")
    print_section("Model selection")
    print(candidates[["source", "model", "f1_macro", "best_epoch"]].to_string(index=False))
    print(f"Selected model: {selected_model} (val_f1={selected['f1_macro']:.4f})")
    source = selected["source"]
    baseline_path = RESULTS_ROOT / source / "metrics.csv"
    baseline = pd.read_csv(baseline_path)
    baseline = baseline[baseline["model"] == selected_model].copy()
    baseline["model"] = selected_model
    baseline["probability_source_dir"] = source
    baseline["probability_model"] = selected_model
    baseline.to_csv(OUT_DIR / "no_augmentation_baseline.csv", index=False)
    return selected_model, MODEL_CONFIGS[selected_model], baseline


def augmentation_runs(base_name, spec_t, spec_f):
    return [
        (f"{base_name} - SpecAugment", AugmentConfig(specaugment=True, mixup=False, spec_t=spec_t, spec_f=spec_f)),
        (f"{base_name} - Mixup", AugmentConfig(specaugment=False, mixup=True, mixup_alpha=0.3)),
        (
            f"{base_name} - SpecAugment + Mixup",
            AugmentConfig(specaugment=True, mixup=True, spec_t=spec_t, spec_f=spec_f, mixup_alpha=0.3),
        ),
    ]


def main():
    selected_model, selected_cfg, baseline_metrics = select_model()
    mels, _labels, track_ids, le, y, idx_train, idx_val, idx_test = prepare_data()

    print_section(f"2.4 Augmentation ablation - {selected_model}")

    results = [
        run_model(
            selected_cfg["factory"],
            label,
            mels,
            y,
            le,
            idx_train,
            idx_val,
            idx_test,
            train_cfg=selected_cfg["train_cfg"],
            augment_cfg=augment_cfg,
        )
        for label, augment_cfg in augmentation_runs(
            selected_model,
            selected_cfg["spec_t"],
            selected_cfg["spec_f"],
        )
    ]

    trained_metrics = finalize_experiment(results, OUT_DIR, le.classes_, "2.4 Augmentation ablation", track_ids=track_ids)

    comparison = pd.concat([baseline_metrics, trained_metrics], ignore_index=True, sort=False)
    comparison.loc[comparison["probability_source_dir"].isna(), "probability_source_dir"] = OUT_DIR.name
    comparison.loc[comparison["probability_model"].isna(), "probability_model"] = comparison["model"]
    comparison.to_csv(OUT_DIR / "augmentation_comparison.csv", index=False)
    plot_metrics(comparison, OUT_DIR, "2.4 Augmentation comparison with reused no-augmentation baseline")
    print_metrics_summary(comparison, "Augmentation comparison")
    print_saved_outputs(
        OUT_DIR,
        [
            "selection_candidates.csv",
            "selected_model.txt",
            "no_augmentation_baseline.csv",
            "augmentation_comparison.csv",
        ],
    )


if __name__ == "__main__":
    main()
