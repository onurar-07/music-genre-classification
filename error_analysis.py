"""Error analysis for the selected model."""

import pandas as pd
from sklearn.metrics import confusion_matrix

from reporting_utils import RESULTS_ROOT, experiment_dir, print_saved_outputs

import matplotlib.pyplot as plt
import seaborn as sns

OUT_DIR = experiment_dir("4 Error analysis")
DEFAULT_SUGGESTIONS = [
    "- Segment-based training to expose more local temporal detail.",
    "- Multi-crop inference to average predictions over several song excerpts.",
    "- Inspect Pop and Experimental confusions first; they are usually the weakest labels.",
    "- Inspect whether the hybrid branch improves or worsens the weakest CNN classes.",
]
PREFERRED_PREDICTIONS = [
    RESULTS_ROOT / "3 Hybrid Modal" / "predictions.csv",
    RESULTS_ROOT / "2.6 Segment Transformer" / "predictions.csv",
    RESULTS_ROOT / "2.5 Segment Averaging" / "predictions.csv",
    RESULTS_ROOT / "2.4 Augmentation ablation" / "predictions.csv",
    RESULTS_ROOT / "2.3 Multi-shape CNN" / "predictions.csv",
    RESULTS_ROOT / "2.2 ResNet CNN" / "predictions.csv",
    RESULTS_ROOT / "2.1 Plain CNN" / "predictions.csv",
]


def find_predictions():
    for path in PREFERRED_PREDICTIONS:
        if path.exists():
            return path
    matches = sorted(RESULTS_ROOT.glob("*/predictions.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        "No predictions.csv found. Run resnet_cnn.py or augmentation_ablation.py first."
    )


def plot_per_class(per_class, out_dir=OUT_DIR, title="Per-class recall"):
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(per_class["true_label"], per_class["recall"], color="#4C78A8", edgecolor="white")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Recall")
    ax.set_title(title)
    for bar, val in zip(bars, per_class["recall"]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "per_class_recall.png", dpi=150)
    plt.close()


def plot_confusion(df, out_dir=OUT_DIR, title="Error analysis confusion matrix"):
    labels = sorted(df["true_label"].unique())
    cm = confusion_matrix(df["true_label"], df["pred_label"], labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=cm,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.4,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()


def make_data_cleaning_candidates(errors):
    candidates = errors.copy()
    sort_cols = ["confidence"]
    if "confidence_margin" in candidates.columns:
        sort_cols.append("confidence_margin")
    candidates = candidates.sort_values(sort_cols, ascending=False).reset_index(drop=True)
    candidates.insert(0, "review_rank", candidates.index + 1)

    if "confidence_margin" in candidates.columns:
        high_priority = (candidates["confidence"] >= 0.90) & (
            candidates["confidence_margin"] >= 0.70
        )
        medium_priority = (candidates["confidence"] >= 0.75) & (
            candidates["confidence_margin"] >= 0.40
        )
    else:
        high_priority = candidates["confidence"] >= 0.90
        medium_priority = candidates["confidence"] >= 0.75

    candidates["review_priority"] = "Low"
    candidates.loc[medium_priority, "review_priority"] = "Medium"
    candidates.loc[high_priority, "review_priority"] = "High"
    candidates["suggested_action"] = "manual_label_review"

    front_cols = [
        "review_rank",
        "review_priority",
        "suggested_action",
        "track_id",
        "true_label",
        "pred_label",
        "confidence",
        "true_confidence",
        "confidence_margin",
        "mp3_path",
    ]
    front_cols = [col for col in front_cols if col in candidates.columns]
    remaining_cols = [col for col in candidates.columns if col not in front_cols]
    return candidates[front_cols + remaining_cols]


def analyze_predictions(
    pred_path,
    out_dir=OUT_DIR,
    title="Error analysis",
    suggestions=None,
    extra_lines=None,
):
    suggestions = suggestions or DEFAULT_SUGGESTIONS
    extra_lines = extra_lines or []
    df = pd.read_csv(pred_path)

    per_class = (
        df.groupby("true_label")
        .agg(
            support=("correct", "size"),
            correct=("correct", "sum"),
        )
        .reset_index()
    )
    per_class["errors"] = per_class["support"] - per_class["correct"]
    per_class["recall"] = per_class["correct"] / per_class["support"]
    per_class = per_class.sort_values("recall")

    errors = df[~df["correct"]].copy()
    confusion_pairs = (
        errors.groupby(["true_label", "pred_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    high_conf_errors = None
    if "confidence" in errors.columns:
        sort_cols = ["confidence"]
        if "confidence_margin" in errors.columns:
            sort_cols.append("confidence_margin")
        high_conf_errors = errors.sort_values(sort_cols, ascending=False)

    overall_acc = df["correct"].mean()
    per_class.to_csv(out_dir / "per_class_errors.csv", index=False)
    confusion_pairs.to_csv(out_dir / "top_confusions.csv", index=False)
    if high_conf_errors is not None:
        high_conf_errors.to_csv(out_dir / "high_confidence_errors.csv", index=False)
        cleaning_candidates = make_data_cleaning_candidates(high_conf_errors)
        cleaning_candidates.to_csv(out_dir / "data_cleaning_candidates.csv", index=False)
    plot_per_class(per_class, out_dir=out_dir)
    plot_confusion(df, out_dir=out_dir)

    lines = [
        title,
        f"Source predictions: {pred_path.relative_to(RESULTS_ROOT.parent)}",
        f"Overall accuracy: {overall_acc:.4f}",
        *extra_lines,
        "",
        "Weakest classes by recall:",
        per_class.head(5).to_string(index=False),
        "",
        "Most common confusion pairs:",
        confusion_pairs.head(10).to_string(index=False),
    ]
    if high_conf_errors is not None:
        display_cols = [
            col for col in [
                "track_id",
                "true_label",
                "pred_label",
                "confidence",
                "confidence_margin",
                "mp3_path",
            ]
            if col in high_conf_errors.columns
        ]
        lines.extend([
            "",
            "Highest-confidence errors:",
            high_conf_errors[display_cols].head(10).to_string(index=False),
        ])
    lines.extend([
        "",
        "Suggested next improvements:",
        *suggestions,
    ])
    (out_dir / "error_analysis_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print_saved_outputs(
        out_dir,
        [
            "per_class_errors.csv",
            "top_confusions.csv",
            "high_confidence_errors.csv",
            "data_cleaning_candidates.csv",
            "per_class_recall.png",
            "confusion_matrix.png",
            "error_analysis_summary.txt",
        ],
    )
    return {
        "overall_acc": overall_acc,
        "per_class": per_class,
        "confusion_pairs": confusion_pairs,
        "high_conf_errors": high_conf_errors,
    }


def main():
    pred_path = find_predictions()
    analyze_predictions(pred_path)


if __name__ == "__main__":
    main()
