"""
Handcrafted feature baselines for FMA-small genre classification.

Uses the shared train/validation/test split from experiment_utils.py. Model and
feature selection use validation F1-macro; final numbers are reported on the
held-out test split.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from experiment_utils import (
    ROOT,
    RANDOM_STATE,
    experiment_dir,
    plot_confusion_matrix,
    plot_metrics,
    result_record,
    save_metrics,
    split_indices,
    update_global_comparison,
    write_classification_report,
)

warnings.filterwarnings("ignore")

FEATURES_PATH = ROOT / "features" / "features.npz"
OUT_DIR = experiment_dir("Random Forest vs MLP")


def load_data():
    d = np.load(FEATURES_PATH, allow_pickle=True)
    le = LabelEncoder()
    y = le.fit_transform(d["labels"])
    feature_sets = {
        "Timbre": d["timbre"],
        "Harmony": d["harmony"],
        "Rhythm": d["rhythm"],
        "Combined": d["combined"],
    }
    return feature_sets, y, le


def rf_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def mlp_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            activation="relu",
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE,
        )),
    ])


def evaluate_model(model, X, y, idx_train, idx_val, idx_test, label):
    model.fit(X[idx_train], y[idx_train])
    val_pred = model.predict(X[idx_val])
    test_pred = model.predict(X[idx_test])
    val_row = result_record(label, "val", y[idx_val], val_pred)
    test_row = result_record(label, "test", y[idx_test], test_pred)

    print(f"\n{'-' * 60}")
    print(f"  {label}")
    print(f"  Val  accuracy={val_row['accuracy']:.4f}  F1={val_row['f1_macro']:.4f}")
    print(f"  Test accuracy={test_row['accuracy']:.4f}  F1={test_row['f1_macro']:.4f}")

    return {
        "label": label,
        "model": model,
        "val_true": y[idx_val],
        "val_pred": val_pred,
        "test_true": y[idx_test],
        "test_pred": test_pred,
        "val_f1": val_row["f1_macro"],
    }


def make_feature_names():
    t_names = (
        [f"mfcc_mean_{i}" for i in range(40)] +
        [f"mfcc_std_{i}" for i in range(40)] +
        [f"contrast_mean_{i}" for i in range(7)] +
        [f"contrast_std_{i}" for i in range(7)] +
        ["rolloff_mean", "rolloff_std"] +
        ["zcr_mean", "zcr_std"]
    )
    h_names = (
        [f"chroma_mean_{i}" for i in range(12)] +
        [f"chroma_std_{i}" for i in range(12)] +
        [f"tonnetz_mean_{i}" for i in range(6)] +
        [f"tonnetz_std_{i}" for i in range(6)]
    )
    r_names = (
        ["tempo"] +
        [f"tempogram_mean_{i}" for i in range(128)] +
        [f"tempogram_std_{i}" for i in range(128)]
    )
    return {
        "Timbre": t_names,
        "Harmony": h_names,
        "Rhythm": r_names,
        "Combined": t_names + h_names + r_names,
    }


def plot_feature_importance(rf_result, feature_names, top_n=30):
    clf = rf_result["model"].named_steps["clf"]
    pairs = sorted(zip(feature_names, clf.feature_importances_), key=lambda x: x[1], reverse=True)
    names, vals = zip(*pairs[:top_n])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(list(reversed(names)), list(reversed(vals)), color="#2E8B8B")
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title(f"Top {top_n} feature importances - {rf_result['label']}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_importance.png", dpi=150)
    plt.close()


def main():
    feature_sets, y, le = load_data()
    feat_names = make_feature_names()
    idx_train, idx_val, idx_test = split_indices(y)

    print(f"Split sizes: train={len(idx_train)}  val={len(idx_val)}  test={len(idx_test)}")

    results = []
    print("\n" + "=" * 60)
    print("  Phase 1 - Feature set comparison (Random Forest)")
    print("=" * 60)

    for name, X in feature_sets.items():
        results.append(
            evaluate_model(rf_pipeline(), X, y, idx_train, idx_val, idx_test, f"RF - {name}")
        )

    best_rf = max(results, key=lambda r: r["val_f1"])
    best_name = best_rf["label"].replace("RF - ", "")
    best_X = feature_sets[best_name]
    print(f"\nBest RF feature set by validation F1: {best_name}")

    print("\n" + "=" * 60)
    print(f"  Phase 2 - MLP on {best_name} features")
    print("=" * 60)
    results.append(
        evaluate_model(mlp_pipeline(), best_X, y, idx_train, idx_val, idx_test, f"MLP - {best_name}")
    )

    metrics_df = save_metrics(results, OUT_DIR)
    plot_metrics(metrics_df, OUT_DIR, "Handcrafted feature baselines")

    best_overall = max(results, key=lambda r: r["val_f1"])
    plot_confusion_matrix(
        best_overall["test_true"],
        best_overall["test_pred"],
        le.classes_,
        OUT_DIR,
        f"Confusion matrix - {best_overall['label']}",
    )
    write_classification_report(best_overall, le.classes_, OUT_DIR)

    rf_on_best = next(r for r in results if r["label"] == f"RF - {best_name}")
    plot_feature_importance(rf_on_best, feat_names[best_name])

    global_df = update_global_comparison()

    print("\n=== Metrics ===")
    print(metrics_df.to_string(index=False))
    if global_df is not None:
        print("\nUpdated results/model_comparison.csv")
    print("\nSaved to results/Random Forest vs MLP/:")
    print("  metrics.csv")
    print("  metrics.png")
    print("  classification_report.txt")
    print("  confusion_matrix.png")
    print("  feature_importance.png")


if __name__ == "__main__":
    main()
