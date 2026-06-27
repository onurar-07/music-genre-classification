"""Summarise model parameter counts and recorded training runtimes."""

import pandas as pd

from cnn_training_utils import (
    MultiShapeCNN,
    PlainCNN,
    PlainCNNRegularisation,
    ResNetGenreCNN,
    count_parameters,
)
from reporting_utils import RESULTS_ROOT, print_saved_outputs, print_section
from segment_transformer import SegmentTransformer


def architecture_rows():
    models = [
        ("Plain CNN", PlainCNN(n_classes=8)),
        ("Plain CNN - Regularisation", PlainCNNRegularisation(n_classes=8)),
        ("ResNet CNN", ResNetGenreCNN(n_classes=8)),
        ("Multi-shape CNN", MultiShapeCNN(n_classes=8)),
        ("Segment Transformer", SegmentTransformer(n_classes=8, n_segments=4)),
    ]
    rows = []
    for name, model in models:
        total, trainable = count_parameters(model)
        rows.append(
            {
                "model": name,
                "param_count": total,
                "trainable_param_count": trainable,
            }
        )
    return pd.DataFrame(rows)


def recorded_training_rows():
    rows = []
    for path in sorted(RESULTS_ROOT.glob("*/metrics.csv")):
        df = pd.read_csv(path)
        if "training_seconds" not in df.columns:
            continue
        test_df = df[df["split"] == "test"].copy()
        test_df["experiment"] = path.parent.name
        rows.append(test_df)
    if not rows:
        return pd.DataFrame()
    cols = [
        "experiment",
        "model",
        "accuracy",
        "f1_macro",
        "param_count",
        "trainable_param_count",
        "training_seconds",
        "epochs_run",
        "best_epoch",
    ]
    df = pd.concat(rows, ignore_index=True)
    return df[[col for col in cols if col in df.columns]]


def main():
    arch_df = architecture_rows()
    arch_path = RESULTS_ROOT / "model_parameter_counts.csv"
    arch_df.to_csv(arch_path, index=False)

    print_section("Architecture parameter counts")
    print(arch_df.to_string(index=False))

    training_df = recorded_training_rows()
    if training_df.empty:
        print_section("Recorded training runtimes")
        print("No metrics.csv files with training runtime found yet.")
        print_saved_outputs(RESULTS_ROOT, ["model_parameter_counts.csv"])
        return

    train_path = RESULTS_ROOT / "model_training_runtimes.csv"
    training_df.to_csv(train_path, index=False)
    print_section("Recorded training runtimes")
    print(training_df.to_string(index=False))
    print_saved_outputs(RESULTS_ROOT, ["model_parameter_counts.csv", "model_training_runtimes.csv"])


if __name__ == "__main__":
    main()
