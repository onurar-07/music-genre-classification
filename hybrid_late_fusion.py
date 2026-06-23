"""Part 3: hybrid late fusion between the best CNN-Mel and handcrafted branches."""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from cnn_training_utils import (
    finalize_experiment,
    load_mel_cache,
    make_split,
)
from reporting_utils import RESULTS_ROOT, experiment_dir, print_saved_outputs, print_section

OUT_DIR = experiment_dir("3 Hybrid Modal")
HANDCRAFTED_DIR = RESULTS_ROOT / "1 Random Forest vs MLP"


def display_model_name(label):
    return str(label).split(" (from ")[0]


def load_best_cnn_branch():
    preferred_paths = [
        RESULTS_ROOT / "2.6 Segment Averaging" / "segment_comparison.csv",
        RESULTS_ROOT / "2.5 Augmentation ablation" / "augmentation_comparison.csv",
    ]
    fallback_paths = [
        RESULTS_ROOT / "2.4 Multi-shape CNN" / "metrics.csv",
        RESULTS_ROOT / "2.3 ResNet CNN" / "metrics.csv",
        RESULTS_ROOT / "2.2 Regularisation ablation" / "metrics.csv",
        RESULTS_ROOT / "2.1 Plain CNN" / "metrics.csv",
    ]
    preferred = next((p for p in preferred_paths if p.exists()), None)
    paths = [preferred] if preferred is not None else [p for p in fallback_paths if p.exists()]
    if not paths:
        raise FileNotFoundError("Run the CNN experiments before hybrid_late_fusion.py.")

    rows = []
    for path in paths:
        df = pd.read_csv(path)
        val_df = df[df["split"] == "val"].copy()
        val_df["source"] = path.parent.name
        if "probability_source_dir" not in val_df.columns:
            val_df["probability_source_dir"] = path.parent.name
        if "probability_model" not in val_df.columns:
            val_df["probability_model"] = val_df["model"]
        rows.append(val_df)
    candidates = pd.concat(rows, ignore_index=True)
    selected = candidates.sort_values("f1_macro", ascending=False).iloc[0]
    return selected, candidates


def load_cnn_branch_probabilities(selected):
    source_dir = RESULTS_ROOT / selected["probability_source_dir"]
    proba_path = source_dir / "branch_probabilities.npz"
    if not proba_path.exists():
        raise FileNotFoundError(
            f"Missing {proba_path}. Rerun the selected CNN experiment with the updated code "
            "before hybrid_late_fusion.py."
        )
    probability_model = selected["probability_model"]
    with np.load(proba_path, allow_pickle=True) as data:
        branch_labels = [str(label) for label in data["branch_labels"]]
        if probability_model not in branch_labels:
            raise ValueError(
                f"CNN branch {probability_model!r} is missing from {proba_path}. "
                "Rerun the selected CNN experiment to refresh branch probabilities."
            )
        branch_idx = branch_labels.index(probability_model)
        result = {
            "label": display_model_name(selected["model"]),
            "val_proba": data["val_proba"][branch_idx],
            "test_proba": data["test_proba"][branch_idx],
            "val_true": data["val_true"],
            "test_true": data["test_true"],
            "test_indices": data["test_indices"] if "test_indices" in data.files else np.array([]),
            "best_epoch": selected.get("best_epoch", ""),
            "best_val_f1": selected["f1_macro"],
            "param_count": selected.get("param_count", ""),
            "trainable_param_count": selected.get("trainable_param_count", ""),
            "epochs_run": selected.get("epochs_run", ""),
        }
    return result


def load_best_handcrafted_branch():
    metrics_path = HANDCRAFTED_DIR / "metrics.csv"
    proba_path = HANDCRAFTED_DIR / "branch_probabilities.npz"
    if not metrics_path.exists():
        raise FileNotFoundError("Run python3 handcrafted_feature_baseline.py before hybrid_late_fusion.py.")
    if not proba_path.exists():
        raise FileNotFoundError(
            "Missing handcrafted branch_probabilities.npz. "
            "Rerun python3 handcrafted_feature_baseline.py once with the updated script to save fusion probabilities."
        )

    metrics_df = pd.read_csv(metrics_path)
    candidates = metrics_df[metrics_df["split"] == "val"].copy()
    candidates["source"] = HANDCRAFTED_DIR.name
    selected = candidates.sort_values("f1_macro", ascending=False).iloc[0]
    selected_label = selected["model"]

    with np.load(proba_path, allow_pickle=True) as data:
        branch_labels = [str(label) for label in data["branch_labels"]]
        if selected_label not in branch_labels:
            raise ValueError(
                f"Selected handcrafted branch {selected_label!r} is missing from {proba_path}. "
                "Rerun python3 handcrafted_feature_baseline.py to refresh Part 1 outputs."
            )
        branch_idx = branch_labels.index(selected_label)
        result = {
            "label": selected_label,
            "val_proba": data["val_proba"][branch_idx],
            "test_proba": data["test_proba"][branch_idx],
            "val_true": data["val_true"],
            "test_true": data["test_true"],
            "val_f1": selected["f1_macro"],
        }
    return result, candidates


def check_branch_alignment(cnn_result, handcrafted_result):
    if not np.array_equal(cnn_result["val_true"], handcrafted_result["val_true"]):
        raise ValueError("CNN and handcrafted validation splits do not align.")
    if not np.array_equal(cnn_result["test_true"], handcrafted_result["test_true"]):
        raise ValueError("CNN and handcrafted test splits do not align.")


def search_fusion_weight(y_true, cnn_proba, handcrafted_proba):
    rows = []
    best = None
    for alpha in np.linspace(0.0, 1.0, 21):
        fused = alpha * cnn_proba + (1 - alpha) * handcrafted_proba
        pred = fused.argmax(1)
        row = {
            "cnn_weight": alpha,
            "handcrafted_weight": 1 - alpha,
            "accuracy": accuracy_score(y_true, pred),
            "f1_macro": f1_score(y_true, pred, average="macro"),
        }
        rows.append(row)
        if best is None or row["f1_macro"] > best["f1_macro"]:
            best = row
    return best, pd.DataFrame(rows)


def main():
    start_time = time.perf_counter()
    selected_cnn, cnn_candidates = load_best_cnn_branch()
    cnn_result = load_cnn_branch_probabilities(selected_cnn)

    _mels, labels, track_ids = load_mel_cache()
    le, _y, _idx_train, _idx_val, idx_test = make_split(labels)

    print_section("3 Hybrid Modal")
    print(f"Selected CNN-Mel branch: {cnn_result['label']}")

    handcrafted_result, handcrafted_candidates = load_best_handcrafted_branch()
    check_branch_alignment(cnn_result, handcrafted_result)
    print(f"Selected handcrafted branch: {handcrafted_result['label']}")

    best_weight, weight_curve = search_fusion_weight(
        cnn_result["val_true"],
        cnn_result["val_proba"],
        handcrafted_result["val_proba"],
    )
    test_proba = (
        best_weight["cnn_weight"] * cnn_result["test_proba"]
        + best_weight["handcrafted_weight"] * handcrafted_result["test_proba"]
    )
    test_pred = test_proba.argmax(1)

    hybrid_result = {
        "label": f"Hybrid late fusion ({cnn_result['label']} + {handcrafted_result['label']})",
        "val_true": cnn_result["val_true"],
        "val_pred": (
            best_weight["cnn_weight"] * cnn_result["val_proba"]
            + best_weight["handcrafted_weight"] * handcrafted_result["val_proba"]
        ).argmax(1),
        "val_proba": (
            best_weight["cnn_weight"] * cnn_result["val_proba"]
            + best_weight["handcrafted_weight"] * handcrafted_result["val_proba"]
        ),
        "test_true": cnn_result["test_true"],
        "test_pred": test_pred,
        "test_proba": test_proba,
        "test_indices": cnn_result["test_indices"] if len(cnn_result["test_indices"]) else idx_test,
        "best_epoch": "",
        "best_val_f1": best_weight["f1_macro"],
        "param_count": cnn_result.get("param_count", ""),
        "trainable_param_count": cnn_result.get("trainable_param_count", ""),
        "training_seconds": time.perf_counter() - start_time,
        "epochs_run": "",
    }

    cnn_candidates.to_csv(OUT_DIR / "cnn_branch_candidates.csv", index=False)
    handcrafted_candidates.to_csv(OUT_DIR / "handcrafted_branch_candidates.csv", index=False)
    weight_curve.to_csv(OUT_DIR / "fusion_weight_search.csv", index=False)
    pd.DataFrame([best_weight]).to_csv(OUT_DIR / "selected_fusion_weight.csv", index=False)

    finalize_experiment(
        [hybrid_result],
        OUT_DIR,
        le.classes_,
        "3 Hybrid Modal",
        track_ids=track_ids,
    )
    print_section("Fusion weight")
    print(pd.DataFrame([best_weight]).to_string(index=False))
    print_saved_outputs(
        OUT_DIR,
        [
            "cnn_branch_candidates.csv",
            "handcrafted_branch_candidates.csv",
            "fusion_weight_search.csv",
            "selected_fusion_weight.csv",
        ],
    )


if __name__ == "__main__":
    main()
