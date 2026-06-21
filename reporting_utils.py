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


def print_section(title: str):
    print(f"\n[{title}]")


def format_seconds(seconds):
    if seconds == "" or pd.isna(seconds):
        return ""
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f}m"


def print_metrics_summary(metrics_df: pd.DataFrame, title="Summary"):
    test_df = metrics_df[metrics_df["split"] == "test"].copy()
    val_df = metrics_df[metrics_df["split"] == "val"][["model", "f1_macro"]].rename(
        columns={"f1_macro": "val_f1"}
    )
    if test_df.empty:
        return
    summary = test_df.merge(val_df, on="model", how="left")
    rename = {
        "accuracy": "test_acc",
        "f1_macro": "test_f1",
        "training_seconds": "time",
    }
    summary = summary.rename(columns=rename)
    summary = summary.sort_values("test_f1", ascending=False)
    for col in ["val_f1", "test_f1", "test_acc"]:
        if col in summary:
            summary[col] = summary[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    if "time" in summary:
        summary["time"] = summary["time"].map(format_seconds)
    keep = [
        "model",
        "val_f1",
        "test_f1",
        "test_acc",
        "best_epoch",
        "epochs_run",
        "time",
        "param_count",
    ]
    keep = [col for col in keep if col in summary.columns]
    summary = summary[keep]
    print_section(title)
    print(summary.to_string(index=False))


def print_saved_outputs(out_dir: Path, filenames):
    existing = [name for name in filenames if name and (out_dir / name).exists()]
    if not existing:
        return
    try:
        display_dir = out_dir.relative_to(ROOT)
    except ValueError:
        display_dir = out_dir
    print_section(f"Saved to {display_dir}")
    print(", ".join(existing))


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
            "param_count": result.get("param_count", ""),
            "trainable_param_count": result.get("trainable_param_count", ""),
            "training_seconds": result.get("training_seconds", ""),
            "epochs_run": result.get("epochs_run", ""),
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


def save_branch_probabilities(results, out_dir: Path):
    proba_results = [
        result for result in results
        if "val_proba" in result and "test_proba" in result
    ]
    if not proba_results:
        return False

    branch_labels = np.asarray([r["label"] for r in proba_results], dtype=object)
    np.savez_compressed(
        out_dir / "branch_probabilities.npz",
        branch_labels=branch_labels,
        val_proba=np.stack([r["val_proba"] for r in proba_results]),
        test_proba=np.stack([r["test_proba"] for r in proba_results]),
        val_true=proba_results[0]["val_true"],
        test_true=proba_results[0]["test_true"],
        test_indices=proba_results[0].get("test_indices", np.array([])),
    )
    pd.DataFrame({
        "branch_index": np.arange(len(branch_labels)),
        "model": branch_labels,
    }).to_csv(out_dir / "branch_probability_index.csv", index=False)
    return True


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


def fma_audio_path(track_id):
    tid = int(track_id)
    return ROOT / "data" / "fma_small" / f"{tid // 1000:03d}" / f"{tid:06d}.mp3"


def save_predictions(result, class_names, out_dir: Path, test_indices=None, track_ids=None):
    rows = []
    y_true = np.asarray(result["test_true"])
    y_pred = np.asarray(result["test_pred"])
    proba = result.get("test_proba")
    proba = np.asarray(proba) if proba is not None else None
    for i, (true_id, pred_id) in enumerate(zip(y_true, y_pred)):
        sample_index = int(test_indices[i]) if test_indices is not None else i
        row = {
            "sample_index": sample_index,
            "true_id": int(true_id),
            "pred_id": int(pred_id),
            "true_label": class_names[int(true_id)],
            "pred_label": class_names[int(pred_id)],
            "correct": bool(true_id == pred_id),
        }
        if proba is not None:
            sorted_proba = np.sort(proba[i])
            top2 = sorted_proba[-2] if len(sorted_proba) > 1 else 0.0
            row["confidence"] = float(proba[i, int(pred_id)])
            row["true_confidence"] = float(proba[i, int(true_id)])
            row["confidence_margin"] = float(row["confidence"] - top2)
        if track_ids is not None:
            track_id = int(track_ids[sample_index])
            row["track_id"] = track_id
            row["mp3_path"] = str(fma_audio_path(track_id).relative_to(ROOT))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "predictions.csv", index=False)
    if "confidence" in df.columns:
        high_conf_errors = df[~df["correct"]].sort_values(
            ["confidence", "confidence_margin"],
            ascending=False,
        )
        high_conf_errors.to_csv(out_dir / "high_confidence_errors.csv", index=False)
    return df


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
        output_path = out_dir / "training_history.png"
        if output_path.exists():
            output_path.unlink()
        return False

    sns.set_theme(style="darkgrid")
    n_rows = len(histories)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4.5 * n_rows), squeeze=False)
    for row_idx, result in enumerate(histories):
        h = result["history"]
        epochs = np.arange(1, len(h["train_loss"]) + 1)
        loss_ax, acc_ax = axes[row_idx]

        loss_ax.plot(epochs, h["train_loss"], label="Training loss", linewidth=2)
        if "val_loss" in h:
            loss_ax.plot(epochs, h["val_loss"], label="Validation loss", linewidth=2)
        loss_ax.set_title(f"{result['label']}: loss")
        loss_ax.set_xlabel("Epoch")
        loss_ax.set_ylabel("Loss")
        loss_ax.legend()

        if "train_acc" in h:
            acc_ax.plot(epochs, h["train_acc"], label="Training accuracy", linewidth=2)
        if "val_acc" in h:
            acc_ax.plot(epochs, h["val_acc"], label="Validation accuracy", linewidth=2)
        acc_ax.set_title(f"{result['label']}: accuracy")
        acc_ax.set_xlabel("Epoch")
        acc_ax.set_ylabel("Accuracy")
        acc_ax.set_ylim(0, 1.05)
        acc_ax.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "training_history.png", dpi=150)
    plt.close()
    return True


def plot_metrics(metrics_df: pd.DataFrame, out_dir: Path, title: str):
    test_df = metrics_df[metrics_df["split"] == "test"].copy()
    output_path = out_dir / "metrics.png"
    if len(test_df) <= 1:
        if output_path.exists():
            output_path.unlink()
        return False

    test_df = test_df.sort_values("f1_macro")
    y = np.arange(len(test_df))
    height = 0.36
    fig, ax = plt.subplots(figsize=(10, max(4, 0.55 * len(test_df) + 1.5)))
    ax.barh(y - height / 2, test_df["f1_macro"], height, label="F1-macro", color="#4C78A8")
    ax.barh(y + height / 2, test_df["accuracy"], height, label="Accuracy", color="#59A14F")
    ax.set_yticks(y)
    ax.set_yticklabels(test_df["model"])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Score")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right")
    for row, (_, item) in enumerate(test_df.iterrows()):
        ax.text(item["f1_macro"] + 0.01, row - height / 2, f"{item['f1_macro']:.3f}", va="center", fontsize=8)
        ax.text(item["accuracy"] + 0.01, row + height / 2, f"{item['accuracy']:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


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
    optional_cols = ["param_count", "trainable_param_count", "training_seconds", "epochs_run"]
    for col in optional_cols:
        if col not in all_df.columns:
            all_df[col] = ""
    all_df = all_df[
        [
            "experiment",
            "model",
            "split",
            "accuracy",
            "f1_macro",
            "best_epoch",
            "best_val_f1",
            "param_count",
            "trainable_param_count",
            "training_seconds",
            "epochs_run",
        ]
    ]
    all_df.to_csv(RESULTS_ROOT / "model_comparison.csv", index=False)

    plot_df = all_df.sort_values("f1_macro")
    y = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(11, max(5, 0.45 * len(plot_df) + 1.5)))
    ax.barh(y, plot_df["f1_macro"], color="#4C78A8", edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["model"])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Test F1-macro")
    ax.set_title("Unified test-set model comparison", fontweight="bold")
    for row, (_, item) in enumerate(plot_df.iterrows()):
        ax.text(item["f1_macro"] + 0.01, row, f"{item['f1_macro']:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_ROOT / "model_comparison.png", dpi=150)
    plt.close()
    return all_df
