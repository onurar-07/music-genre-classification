"""Shared train/validation/test split and reporting helpers."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20

ROOT = Path(__file__).parent
RESULTS_ROOT = ROOT / "results"
RESULTS_ROOT.mkdir(exist_ok=True)


def split_indices(y, random_state=RANDOM_STATE, test_size=TEST_SIZE, val_size=VAL_SIZE):
    """Return shared train/val/test indices: 64% / 16% / 20%, stratified."""
    idx = np.arange(len(y))
    idx_train_val, idx_test = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    idx_train, idx_val = train_test_split(
        idx_train_val,
        test_size=val_size,
        random_state=random_state,
        stratify=np.asarray(y)[idx_train_val],
    )
    return idx_train, idx_val, idx_test


def experiment_dir(name: str) -> Path:
    path = RESULTS_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def compute_scores(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }


def result_record(model, split, y_true, y_pred, **extra):
    row = {"model": model, "split": split}
    row.update(compute_scores(y_true, y_pred))
    row.update(extra)
    return row


def save_metrics(results, out_dir: Path) -> pd.DataFrame:
    rows = []
    for result in results:
        common = {
            "best_epoch": result.get("best_epoch", ""),
            "best_val_f1": result.get("best_val_f1", ""),
        }
        if "val_true" in result and "val_pred" in result:
            rows.append(
                result_record(
                    result["label"],
                    "val",
                    result["val_true"],
                    result["val_pred"],
                    **common,
                )
            )
        rows.append(
            result_record(
                result["label"],
                "test",
                result["test_true"],
                result["test_pred"],
                **common,
            )
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics.csv", index=False)
    return df


def write_classification_report(result, class_names, out_dir: Path):
    report = classification_report(
        result["test_true"],
        result["test_pred"],
        target_names=class_names,
        zero_division=0,
    )
    lines = [
        result["label"],
        f"Test accuracy: {accuracy_score(result['test_true'], result['test_pred']):.4f}",
        f"Test F1-macro: {f1_score(result['test_true'], result['test_pred'], average='macro'):.4f}",
    ]
    if result.get("best_epoch", "") != "":
        lines.append(f"Best epoch: {result['best_epoch']}")
    if result.get("best_val_f1", "") != "":
        lines.append(f"Best validation F1-macro: {result['best_val_f1']:.4f}")
    lines.extend(["", report])
    (out_dir / "classification_report.txt").write_text("\n".join(lines), encoding="utf-8")


def plot_confusion_matrix(y_true, y_pred, class_names, out_dir: Path, title: str):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=cm,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.4,
        ax=ax,
        cbar=True,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()


def save_history(results, out_dir: Path):
    rows = []
    for result in results:
        history = result.get("history")
        if not history:
            continue
        n_epochs = len(next(iter(history.values())))
        for epoch in range(n_epochs):
            row = {"model": result["label"], "epoch": epoch + 1}
            for key, values in history.items():
                row[key] = values[epoch]
            rows.append(row)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "training_history.csv", index=False)
    return df


def plot_training_history(results, out_dir: Path):
    histories = [r for r in results if r.get("history")]
    if not histories:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for result in histories:
        h = result["history"]
        axes[0].plot(h["train_loss"], label=result["label"])
        if "val_f1" in h:
            axes[1].plot(h["val_f1"], label=f"{result['label']} val F1")
        elif "val_acc" in h:
            axes[1].plot(h["val_acc"], label=f"{result['label']} val acc")
        if "train_acc" in h:
            axes[1].plot(h["train_acc"], linestyle="--", alpha=0.35)

    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].set_title("Validation selection")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_dir / "training_history.png", dpi=150)
    plt.close()


def plot_metrics(metrics_df: pd.DataFrame, out_dir: Path, title: str):
    test_df = metrics_df[metrics_df["split"] == "test"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, ylabel in zip(axes, ["accuracy", "f1_macro"], ["Accuracy", "F1-macro"]):
        bars = ax.bar(test_df["model"], test_df[col], color="#5B8DB8", edgecolor="white")
        ax.set_ylim(0, 1.0)
        ax.set_title(ylabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)
        for bar, val in zip(bars, test_df[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", fontsize=8)
    fig.suptitle(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "metrics.png", dpi=150)
    plt.close()


def update_global_comparison():
    rows = []
    for path in sorted(RESULTS_ROOT.glob("*/metrics.csv")):
        df = pd.read_csv(path)
        test_df = df[df["split"] == "test"].copy()
        test_df["experiment"] = path.parent.name
        rows.append(test_df)
    if not rows:
        return None

    all_df = pd.concat(rows, ignore_index=True)
    all_df = all_df[["experiment", "model", "split", "accuracy", "f1_macro", "best_epoch", "best_val_f1"]]
    all_df.to_csv(RESULTS_ROOT / "model_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    labels = all_df["model"]
    for ax, col, ylabel in zip(axes, ["accuracy", "f1_macro"], ["Accuracy", "F1-macro"]):
        bars = ax.bar(labels, all_df[col], color="#5B8DB8", edgecolor="white")
        ax.set_ylim(0, 1.0)
        ax.set_title(ylabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)
        for bar, val in zip(bars, all_df[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", fontsize=8)
    fig.suptitle("Unified test-set model comparison", fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_ROOT / "model_comparison.png", dpi=150)
    plt.close()
    return all_df
