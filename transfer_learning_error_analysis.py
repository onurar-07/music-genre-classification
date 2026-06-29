"""Part 5.5: error analysis for the best transfer-learning model."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from error_analysis import analyze_predictions
from reporting_utils import RESULTS_ROOT, experiment_dir, print_section
import matplotlib.pyplot as plt
import seaborn as sns

OUT_DIR = experiment_dir("5.5 Error Analysis")
TRANSFER_EXPERIMENTS = [
    "5.4 Fine Tuning",
    "5.3 AST",
    "5.2 PANNs-CNN14",
    "5.1 ImageNet ResNet18",
]
CLASS_NAMES = [
    "Electronic",
    "Experimental",
    "Folk",
    "Hip-Hop",
    "Instrumental",
    "International",
    "Pop",
    "Rock",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
CLEANING_CANDIDATES = OUT_DIR / "data_cleaning_candidates.csv"


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


def normalize_subjective_label(label):
    if pd.isna(label):
        return "", "missing"
    text = str(label).strip()
    if not text or text == "-":
        return "", "unresolved"
    parts = [part.strip() for part in text.split("/") if part.strip()]
    in_domain = [part for part in parts if part in CLASS_TO_ID]
    if in_domain:
        return in_domain[0], "in_domain"
    return text, "out_of_taxonomy"


def plot_cleaned_confusion(df, out_dir):
    cm = confusion_matrix(
        df["cleaned_true_label"],
        df["pred_label"],
        labels=CLASS_NAMES,
    )
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=cm,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=0.4,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Cleaned true label")
    ax.set_title("Cleaned-label test confusion matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "cleaned_test_confusion_matrix.png", dpi=150)
    plt.close()


def per_class_scores(df, true_col):
    precision, recall, f1, support = precision_recall_fscore_support(
        df[true_col],
        df["pred_label"],
        labels=CLASS_NAMES,
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "class": CLASS_NAMES,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )


def plot_per_class_comparison(comparison, out_dir):
    metrics = ["precision", "recall", "f1"]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(11, 12), sharex=True)
    x = np.arange(len(CLASS_NAMES))
    width = 0.36
    for ax, metric in zip(axes, metrics):
        ax.bar(
            x - width / 2,
            comparison[f"original_{metric}"],
            width,
            label="Original labels",
            color="#4C78A8",
        )
        ax.bar(
            x + width / 2,
            comparison[f"cleaned_{metric}"],
            width,
            label="Cleaned labels",
            color="#E15759",
        )
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(metric.title())
        ax.grid(axis="y", alpha=0.25)
        for idx, row in comparison.iterrows():
            ax.text(
                idx - width / 2,
                row[f"original_{metric}"] + 0.015,
                f"{row[f'original_{metric}']:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
            ax.text(
                idx + width / 2,
                row[f"cleaned_{metric}"] + 0.015,
                f"{row[f'cleaned_{metric}']:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    axes[0].legend(loc="lower right")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    fig.suptitle("Original vs cleaned test labels: per-class scores", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "cleaned_vs_original_per_class.png", dpi=150)
    plt.close()


def save_label_noise_impact(cleaned, strict_df, metrics, out_dir):
    original = metrics[metrics["evaluation"] == "original_test_labels"].iloc[0]
    cleaned_metric = metrics[
        metrics["evaluation"] == "cleaned_test_labels_in_domain_only"
    ].iloc[0]
    reviewed = cleaned[cleaned["label_review_status"] != "not_reviewed"]
    in_domain = reviewed[reviewed["label_review_status"] == "in_domain"]
    out_of_taxonomy = reviewed[reviewed["label_review_status"] == "out_of_taxonomy"]
    unresolved = reviewed[reviewed["label_review_status"] == "unresolved"]
    changed = in_domain[
        in_domain["original_true_label"] != in_domain["cleaned_true_label"]
    ]

    impact = pd.DataFrame(
        [
            {
                "original_n": int(original["n_samples"]),
                "cleaned_strict_n": int(cleaned_metric["n_samples"]),
                "reviewed_error_candidates": int(len(reviewed)),
                "in_domain_relabelled": int(len(in_domain)),
                "changed_in_domain_labels": int(len(changed)),
                "out_of_taxonomy_labels": int(len(out_of_taxonomy)),
                "unresolved_labels": int(len(unresolved)),
                "original_accuracy": float(original["accuracy"]),
                "cleaned_accuracy": float(cleaned_metric["accuracy"]),
                "accuracy_delta": float(cleaned_metric["accuracy"] - original["accuracy"]),
                "original_macro_f1": float(original["f1_macro"]),
                "cleaned_macro_f1": float(cleaned_metric["f1_macro"]),
                "macro_f1_delta": float(cleaned_metric["f1_macro"] - original["f1_macro"]),
            }
        ]
    )
    impact.to_csv(out_dir / "label_noise_impact_summary.csv", index=False)

    lines = [
        "5.5 Label Noise Impact Summary",
        "",
        f"Reviewed high-confidence error candidates: {len(reviewed)}",
        f"In-domain relabelled samples: {len(in_domain)}",
        f"Changed in-domain labels: {len(changed)}",
        f"Out-of-taxonomy samples excluded from strict metric: {len(out_of_taxonomy)}",
        f"Unresolved samples excluded from strict metric: {len(unresolved)}",
        "",
        "Original test labels:",
        f"- Samples: {int(original['n_samples'])}",
        f"- Accuracy: {original['accuracy']:.4f}",
        f"- Macro-F1: {original['f1_macro']:.4f}",
        "",
        "Strict cleaned test labels:",
        f"- Samples: {int(cleaned_metric['n_samples'])}",
        f"- Accuracy: {cleaned_metric['accuracy']:.4f}",
        f"- Macro-F1: {cleaned_metric['f1_macro']:.4f}",
        "",
        "Impact:",
        f"- Accuracy delta: +{cleaned_metric['accuracy'] - original['accuracy']:.4f}",
        f"- Macro-F1 delta: +{cleaned_metric['f1_macro'] - original['f1_macro']:.4f}",
        "",
        "Interpretation:",
        "The large gap between original-label and cleaned-label performance indicates that label noise and out-of-taxonomy ambiguity account for a substantial part of the apparent test error.",
    ]
    (out_dir / "label_noise_impact_summary.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return impact


def evaluate_cleaned_test_labels(pred_path):
    pred_path = pred_path if pred_path.is_absolute() else RESULTS_ROOT.parent / pred_path
    if not CLEANING_CANDIDATES.exists():
        raise FileNotFoundError(f"Missing manual review file: {CLEANING_CANDIDATES}")

    pred_df = pd.read_csv(pred_path)
    candidates = pd.read_csv(CLEANING_CANDIDATES)
    if "Subjective Label" not in candidates.columns:
        raise ValueError(
            "data_cleaning_candidates.csv must include a `Subjective Label` column."
        )

    cleaned = pred_df.copy()
    cleaned["split"] = "test"
    cleaned["original_true_label"] = cleaned["true_label"]
    cleaned["cleaned_true_label"] = cleaned["true_label"]
    cleaned["subjective_label"] = ""
    cleaned["label_review_status"] = "not_reviewed"

    for _, row in candidates.iterrows():
        mask = (
            (cleaned["sample_index"] == row["sample_index"])
            & (cleaned["track_id"] == row["track_id"])
        )
        if not mask.any():
            continue
        cleaned_label, status = normalize_subjective_label(row["Subjective Label"])
        cleaned.loc[mask, "subjective_label"] = row["Subjective Label"]
        cleaned.loc[mask, "label_review_status"] = status
        if status == "in_domain":
            cleaned.loc[mask, "cleaned_true_label"] = cleaned_label
        for col in ["review_rank", "review_priority"]:
            if col in candidates.columns:
                cleaned.loc[mask, col] = row[col]

    cleaned["cleaned_true_id"] = cleaned["cleaned_true_label"].map(CLASS_TO_ID)
    cleaned["cleaned_correct"] = cleaned["cleaned_true_label"] == cleaned["pred_label"]
    cleaned.to_csv(OUT_DIR / "cleaned_test_predictions.csv", index=False)

    strict_df = cleaned[
        ~cleaned["label_review_status"].isin(["out_of_taxonomy", "unresolved"])
    ].copy()
    conservative_df = cleaned.copy()

    rows = []
    for name, df_eval, true_col in [
        ("original_test_labels", pred_df, "true_label"),
        ("cleaned_test_labels_in_domain_only", strict_df, "cleaned_true_label"),
        (
            "cleaned_test_labels_unresolved_kept_original",
            conservative_df,
            "cleaned_true_label",
        ),
    ]:
        rows.append(
            {
                "evaluation": name,
                "n_samples": len(df_eval),
                "accuracy": accuracy_score(df_eval[true_col], df_eval["pred_label"]),
                "f1_macro": f1_score(
                    df_eval[true_col],
                    df_eval["pred_label"],
                    labels=CLASS_NAMES,
                    average="macro",
                    zero_division=0,
                ),
            }
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_DIR / "cleaned_test_metrics.csv", index=False)

    report = classification_report(
        strict_df["cleaned_true_label"],
        strict_df["pred_label"],
        labels=CLASS_NAMES,
        zero_division=0,
    )
    lines = [
        "5.5 Cleaned-label test evaluation",
        f"Source predictions: {pred_path.relative_to(RESULTS_ROOT.parent)}",
        f"Manual review file: {CLEANING_CANDIDATES.relative_to(RESULTS_ROOT.parent)}",
        "",
        "Metrics:",
        metrics.to_string(index=False),
        "",
        "Label mapping rule:",
        "- 8-class subjective labels replace the original test label.",
        "- Experimental/Speech -> Experimental.",
        "- Pure Speech labels are excluded from the strict 8-class cleaned-label metric.",
        "",
        "Detailed per-class results are saved in cleaned_vs_original_per_class.csv.",
    ]
    (OUT_DIR / "cleaned_test_evaluation_summary.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    plot_cleaned_confusion(strict_df, OUT_DIR)

    original_per_class = per_class_scores(pred_df, "true_label").add_prefix("original_")
    cleaned_per_class = per_class_scores(strict_df, "cleaned_true_label").add_prefix("cleaned_")
    comparison = pd.concat(
        [
            original_per_class.rename(columns={"original_class": "class"}),
            cleaned_per_class.drop(columns=["cleaned_class"]),
        ],
        axis=1,
    )
    for metric in ["precision", "recall", "f1", "support"]:
        comparison[f"{metric}_delta"] = (
            comparison[f"cleaned_{metric}"] - comparison[f"original_{metric}"]
        )
    comparison.to_csv(OUT_DIR / "cleaned_vs_original_per_class.csv", index=False)
    plot_per_class_comparison(comparison, OUT_DIR)
    save_label_noise_impact(cleaned, strict_df, metrics, OUT_DIR)

    print_section("Cleaned-label test evaluation")
    print(metrics.to_string(index=False))
    return metrics


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
    evaluate_cleaned_test_labels(pred_path)


if __name__ == "__main__":
    main()
