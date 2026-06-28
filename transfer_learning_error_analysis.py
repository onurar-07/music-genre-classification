"""Part 5.5: error analysis for the best transfer-learning model."""

import pandas as pd

from error_analysis import analyze_predictions
from reporting_utils import RESULTS_ROOT, experiment_dir, print_section

OUT_DIR = experiment_dir("5.5 Error Analysis")
TRANSFER_EXPERIMENTS = [
    "5.4 Fine Tuning",
    "5.3 AST",
    "5.2 PANNs-CNN14",
    "5.1 ImageNet ResNet18",
]


def best_val_row(metrics_path):
    df = pd.read_csv(metrics_path)
    val_df = df[df["split"] == "val"].copy()
    if val_df.empty:
        return None
    return val_df.sort_values("f1_macro", ascending=False).iloc[0]


def find_best_transfer_predictions():
    candidates = []
    for experiment in TRANSFER_EXPERIMENTS:
        exp_dir = RESULTS_ROOT / experiment
        metrics_path = exp_dir / "metrics.csv"
        predictions_path = exp_dir / "predictions.csv"
        if not metrics_path.exists() or not predictions_path.exists():
            continue
        row = best_val_row(metrics_path)
        if row is None:
            continue
        candidates.append(
            {
                "experiment": experiment,
                "model": row["model"],
                "val_accuracy": row["accuracy"],
                "val_f1_macro": row["f1_macro"],
                "best_epoch": row.get("best_epoch", ""),
                "metrics_path": str(metrics_path.relative_to(RESULTS_ROOT.parent)),
                "predictions_path": str(predictions_path.relative_to(RESULTS_ROOT.parent)),
            }
        )
    if not candidates:
        raise FileNotFoundError(
            "No Part 5 predictions.csv found. Run one of 5.1-5.4 before 5.5 error analysis."
        )
    candidates_df = pd.DataFrame(candidates).sort_values("val_f1_macro", ascending=False)
    candidates_df.to_csv(OUT_DIR / "selection_candidates.csv", index=False)
    selected = candidates_df.iloc[0]
    return selected, candidates_df


def main():
    selected, candidates = find_best_transfer_predictions()
    pred_path = RESULTS_ROOT.parent / selected["predictions_path"]
    print_section("5.5 Error Analysis selection")
    print(
        candidates[
            ["experiment", "model", "val_f1_macro", "val_accuracy", "best_epoch"]
        ].to_string(index=False)
    )

    extra_lines = [
        f"Selected experiment: {selected['experiment']}",
        f"Selected model: {selected['model']}",
        f"Selected validation F1-macro: {selected['val_f1_macro']:.4f}",
    ]
    suggestions = [
        "- Inspect Pop and Experimental first; they are consistently difficult genre labels.",
        "- Compare this confusion matrix with 5.2/5.3 to see whether fine-tuning changes the weak classes.",
        "- If errors concentrate in short ambiguous excerpts, try multi-crop or segment-level fusion.",
        "- If high-confidence errors are stylistically reasonable, discuss label ambiguity in the report.",
    ]
    analyze_predictions(
        pred_path,
        out_dir=OUT_DIR,
        title="5.5 Transfer-learning error analysis",
        suggestions=suggestions,
        extra_lines=extra_lines,
    )


if __name__ == "__main__":
    main()
