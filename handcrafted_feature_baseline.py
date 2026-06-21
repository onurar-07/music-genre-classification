"""
Handcrafted feature baselines for FMA-small genre classification.

Uses the shared train/validation/test split from reporting_utils.py. Model and
feature selection use validation F1-macro; final numbers are reported on the
held-out test split.
"""

import warnings
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from reporting_utils import (
    ROOT,
    RANDOM_STATE,
    experiment_dir,
    plot_confusion_matrix,
    plot_metrics,
    print_metrics_summary,
    print_saved_outputs,
    print_section,
    result_record,
    save_metrics,
    split_indices,
    update_global_comparison,
    write_classification_report,
)

warnings.filterwarnings("ignore")

FEATURES_PATH = ROOT / "features" / "features.npz"
OUT_DIR = experiment_dir("1 Random Forest vs MLP")


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


def model_complexity(model):
    clf = model.named_steps["clf"]
    if isinstance(clf, RandomForestClassifier):
        total_nodes = sum(est.tree_.node_count for est in clf.estimators_)
        return total_nodes, "", "total RF tree nodes"
    if isinstance(clf, MLPClassifier):
        total_params = sum(w.size for w in clf.coefs_) + sum(b.size for b in clf.intercepts_)
        return total_params, total_params, "MLP weights and biases"
    return "", "", ""


def evaluate_model(model, X, y, idx_train, idx_val, idx_test, label):
    start_time = time.perf_counter()
    model.fit(X[idx_train], y[idx_train])
    val_pred = model.predict(X[idx_val])
    test_pred = model.predict(X[idx_test])
    val_proba = model.predict_proba(X[idx_val])
    test_proba = model.predict_proba(X[idx_test])
    training_seconds = time.perf_counter() - start_time
    param_count, trainable_param_count, complexity_note = model_complexity(model)
    val_row = result_record(label, "val", y[idx_val], val_pred)
    test_row = result_record(label, "test", y[idx_test], test_pred)

    print(
        f"{label}: val_f1={val_row['f1_macro']:.4f}  "
        f"test_f1={test_row['f1_macro']:.4f}  time={training_seconds:.1f}s"
    )

    return {
        "label": label,
        "model": model,
        "val_true": y[idx_val],
        "val_pred": val_pred,
        "val_proba": val_proba,
        "test_true": y[idx_test],
        "test_pred": test_pred,
        "test_proba": test_proba,
        "val_f1": val_row["f1_macro"],
        "param_count": param_count,
        "trainable_param_count": trainable_param_count,
        "training_seconds": training_seconds,
        "epochs_run": getattr(model.named_steps["clf"], "n_iter_", ""),
    }


def save_branch_probabilities(results, out_dir, idx_val, idx_test):
    branch_labels = np.asarray([r["label"] for r in results], dtype=object)
    val_proba = np.stack([r["val_proba"] for r in results])
    test_proba = np.stack([r["test_proba"] for r in results])

    np.savez_compressed(
        out_dir / "branch_probabilities.npz",
        branch_labels=branch_labels,
        val_proba=val_proba,
        test_proba=test_proba,
        val_true=results[0]["val_true"],
        test_true=results[0]["test_true"],
        val_indices=idx_val,
        test_indices=idx_test,
    )
    pd.DataFrame({
        "branch_index": np.arange(len(branch_labels)),
        "model": branch_labels,
    }).to_csv(out_dir / "branch_probability_index.csv", index=False)


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


def plot_feature_importance(rf_result, feature_names, top_n=20):
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

    print_section("1 Random Forest vs MLP")
    print(f"Split: train={len(idx_train)}  val={len(idx_val)}  test={len(idx_test)}")

    results = []
    print_section("Feature set comparison")

    for name, X in feature_sets.items():
        results.append(
            evaluate_model(rf_pipeline(), X, y, idx_train, idx_val, idx_test, f"RF - {name}")
        )

    best_rf = max(results, key=lambda r: r["val_f1"])
    best_name = best_rf["label"].replace("RF - ", "")
    best_X = feature_sets[best_name]
    print(f"Selected RF feature set: {best_name}")

    print_section(f"MLP on {best_name} features")
    results.append(
        evaluate_model(mlp_pipeline(), best_X, y, idx_train, idx_val, idx_test, f"MLP - {best_name}")
    )

    metrics_df = save_metrics(results, OUT_DIR)
    save_branch_probabilities(results, OUT_DIR, idx_val, idx_test)
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

    update_global_comparison()

    print_metrics_summary(metrics_df)
    print_saved_outputs(
        OUT_DIR,
        [
            "metrics.csv",
            "metrics.png",
            "classification_report.txt",
            "confusion_matrix.png",
            "feature_importance.png",
            "branch_probabilities.npz",
            "branch_probability_index.csv",
        ],
    )


if __name__ == "__main__":
    main()
