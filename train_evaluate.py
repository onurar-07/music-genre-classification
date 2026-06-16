"""
Music genre classification: feature comparison + model evaluation.
Run AFTER extract_features.py has produced features/features.npz.

What this script does:
  Phase 1 — Compare feature groups (Timbre / Harmony / Rhythm / Combined)
             using a Random Forest classifier.
  Phase 2 — Compare Random Forest vs. MLP on the best-performing feature set.
  Output  — Accuracy, F1-macro, per-genre classification report,
             confusion matrix, summary bar chart, feature importance plot.

Results are written to results/.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix,
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ── Paths (always relative to this script's location) ─────────────────────────
ROOT          = Path(__file__).parent
FEATURES_PATH = ROOT / "features" / "features.npz"
RESULTS_DIR   = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.20   # 80 % train  /  20 % test


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data():
    d  = np.load(FEATURES_PATH, allow_pickle=True)
    le = LabelEncoder()
    y  = le.fit_transform(d["labels"])
    feature_sets = {
        "Timbre":   d["timbre"],
        "Harmony":  d["harmony"],
        "Rhythm":   d["rhythm"],
        "Combined": d["combined"],
    }
    return feature_sets, y, le


def split(X, y):
    return train_test_split(
        X, y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y
    )


# ── Models ────────────────────────────────────────────────────────────────────
def rf_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=200,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def mlp_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            activation="relu",
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE,
        )),
    ])


# ── Evaluation ────────────────────────────────────────────────────────────────
def run_experiment(model, X, y, le, label: str) -> dict:
    X_tr, X_te, y_tr, y_te = split(X, y)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    acc = accuracy_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred, average="macro")

    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"  Accuracy : {acc:.4f}   |   F1-macro : {f1:.4f}")
    print(classification_report(
        y_te, y_pred, target_names=le.classes_, zero_division=0
    ))

    return {
        "label":  label,
        "acc":    acc,
        "f1":     f1,
        "y_te":   y_te,
        "y_pred": y_pred,
        "model":  model,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_summary(results: list):
    df = pd.DataFrame([
        {"Configuration": r["label"], "Accuracy": r["acc"], "F1-macro": r["f1"]}
        for r in results
    ])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["steelblue", "coral"]
    for ax, col, c in zip(axes, ["Accuracy", "F1-macro"], colors):
        bars = ax.bar(df["Configuration"], df[col], color=c, edgecolor="white")
        ax.set_ylim(0, 1.0)
        ax.set_title(col, fontsize=13)
        ax.set_ylabel(col)
        ax.tick_params(axis="x", rotation=45)
        for bar, val in zip(bars, df[col]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + 0.01, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=9)

    fig.suptitle("Feature set & model comparison — FMA-small (8 genres)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_comparison.png", dpi=150)
    plt.close()

    return df


def plot_confusion_matrix(y_te, y_pred, class_names, title: str, path: Path):
    cm      = confusion_matrix(y_te, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=cm,          # show raw counts
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.4,
        ax=ax,
        cbar=True,
    )
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_feature_importance(rf_result: dict, feature_names: list, top_n: int = 30):
    """Bar chart of the top-N most important features in the Random Forest."""
    clf = rf_result["model"].named_steps["clf"]
    importances = clf.feature_importances_

    # Pair and sort
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    names, vals = zip(*pairs[:top_n])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(list(reversed(names)), list(reversed(vals)), color="teal")
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title(f"Top {top_n} feature importances — {rf_result['label']}")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
    plt.close()


# ── Feature name generation ────────────────────────────────────────────────────
def make_feature_names() -> dict:
    t_names = (
        [f"mfcc_mean_{i}"     for i in range(40)] +
        [f"mfcc_std_{i}"      for i in range(40)] +
        [f"contrast_mean_{i}" for i in range(7)]  +
        [f"contrast_std_{i}"  for i in range(7)]  +
        ["rolloff_mean", "rolloff_std"] +
        ["zcr_mean",    "zcr_std"]
    )
    h_names = (
        [f"chroma_mean_{i}"   for i in range(12)] +
        [f"chroma_std_{i}"    for i in range(12)] +
        [f"tonnetz_mean_{i}"  for i in range(6)]  +
        [f"tonnetz_std_{i}"   for i in range(6)]
    )
    r_names = (
        ["tempo"] +
        [f"tempogram_mean_{i}" for i in range(128)] +
        [f"tempogram_std_{i}"  for i in range(128)]
    )
    return {
        "Timbre":   t_names,
        "Harmony":  h_names,
        "Rhythm":   r_names,
        "Combined": t_names + h_names + r_names,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    feature_sets, y, le = load_data()
    feat_names = make_feature_names()
    results = []

    # ── Phase 1: feature set comparison (Random Forest) ──────────────────────
    print("\n" + "=" * 60)
    print("  Phase 1 — Feature set comparison  (Random Forest)")
    print("=" * 60)

    for name, X in feature_sets.items():
        res = run_experiment(rf_pipeline(), X, y, le, f"RF – {name}")
        results.append(res)

    best_rf   = max(results, key=lambda r: r["f1"])
    best_name = best_rf["label"].replace("RF – ", "")
    best_X    = feature_sets[best_name]
    print(f"\n→ Best feature set: {best_name}  (F1-macro = {best_rf['f1']:.4f})")

    # ── Phase 2: RF vs MLP on the best feature set ───────────────────────────
    print("\n" + "=" * 60)
    print(f"  Phase 2 — RF vs MLP  on '{best_name}' features")
    print("=" * 60)

    res_mlp = run_experiment(mlp_pipeline(), best_X, y, le, f"MLP – {best_name}")
    results.append(res_mlp)

    # ── Outputs ───────────────────────────────────────────────────────────────
    best_overall = max(results, key=lambda r: r["f1"])

    plot_confusion_matrix(
        best_overall["y_te"],
        best_overall["y_pred"],
        le.classes_,
        f"Confusion matrix — {best_overall['label']}",
        RESULTS_DIR / "confusion_matrix_best.png",
    )

    # Feature importance for the RF run on the best feature set
    rf_on_best = next(r for r in results if r["label"] == f"RF – {best_name}")
    plot_feature_importance(rf_on_best, feat_names[best_name], top_n=30)

    df_summary = plot_summary(results)
    df_summary.to_csv(RESULTS_DIR / "results_summary.csv", index=False)

    print("\n" + "=" * 60)
    print("  Final summary")
    print("=" * 60)
    print(df_summary.to_string(index=False))
    print(f"\nSaved to results/")
    print("  feature_comparison.png")
    print("  confusion_matrix_best.png")
    print("  feature_importance.png")
    print("  results_summary.csv")


if __name__ == "__main__":
    main()
