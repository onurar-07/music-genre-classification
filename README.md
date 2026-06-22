# Music Genre Classification — FMA-small

Politecnico di Milano — Selected Topics in Music and Acoustic Engineering  
Task V: Music Genre Classification

Classifies 30-second music tracks into 8 genres using the FMA-small dataset.  
Compares handcrafted features (MFCC, chroma, rhythm) with CNN-based approaches on mel spectrograms.

---

## Project structure

```
term_project/
├── extract_features.py              # Part 1 — extract handcrafted features
├── handcrafted_feature_baseline.py  # Part 1 — RF/MLP handcrafted baseline
├── extract_mel_specs.py             # Part 2 — extract cached mel spectrograms
├── extract_mel_segments.py          # Part 2.5 — extract cached mel segments locally
├── plain_cnn.py                     # Part 2.1 — Plain CNN
├── regularisation_ablation.py       # Part 2.2 — regularisation ablation
├── resnet_cnn.py                    # Part 2.3 — ResNet CNN
├── augmentation_ablation.py         # Part 2.4 — augmentation ablation
├── segment_averaging.py             # Part 2.5 — segment training + track averaging
├── run_GPU.ipynb                    # Colab GPU notebook for model training
├── hybrid_late_fusion.py            # Part 3 — Hybrid Modal
├── error_analysis.py                # Part 4 — error analysis
├── model_complexity_summary.py      # Summarise parameter counts and training runtimes
├── cnn_training_utils.py # Shared CNN models, augmentations, training loop
├── reporting_utils.py  # Shared split, metrics, plots, and reports
├── requirements.txt
├── data/                # ← put dataset here (see below)
├── features/            # auto-created when you run the scripts
└── results/             # auto-created — all plots and CSVs saved here
```

---

## 1. Install dependencies

```bash
pip3 install -r requirements.txt
```

---

## 2. Download the dataset

You need two downloads from the [FMA GitHub page](https://github.com/mdeff/fma):

### Metadata (~342 MB)
```bash
cd data/
curl -O https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
7z x fma_metadata.zip -o.
```

### Audio — FMA-small (~7.2 GB)
```bash
curl -O https://os.unil.cloud.switch.ch/fma/fma_small.zip
7z x fma_small.zip -o.
```

```
data/
├── fma_small/
│   ├── 000/
│   │   ├── 000002.mp3
│   │   └── ...
│   ├── 001/
│   └── ...
└── fma_metadata/
    ├── tracks.csv
    ├── genres.csv
    └── ...
```

---

## 3. Run the pipeline

Recommended workflow:

1. **Local machine:** keep the raw FMA audio in `data/fma_small/` and run the
   feature extraction scripts once:

   ```bash
   python3 extract_features.py
   python3 extract_mel_specs.py
   python3 extract_mel_segments.py
   ```

2. **Colab GPU:** copy the project with the generated `features/*.npz` files,
   then run `run_GPU.ipynb` to train and evaluate the models.

Do not rerun feature extraction on Colab unless the raw FMA audio is also
available there.

Run Part 1 first for handcrafted baselines, Part 2 for CNN optimisation,
Part 3 for Hybrid Modal, then Part 4 for error analysis.

All model scripts use the same stratified train/validation/test split:
64% train, 16% validation, 20% test. Validation F1-macro selects the best
model/epoch; final metrics are reported on the held-out test split.

## Part 1 — Handcrafted Baseline

### Step 1 — Extract handcrafted features locally (~30 min, runs once)
```bash
python3 extract_features.py
```
Saves `features/features.npz` with timbre, harmony, rhythm, and combined feature vectors.

### Step 2 — Random Forest vs MLP on handcrafted features (Colab or local)
```bash
python3 handcrafted_feature_baseline.py
```
Compares feature groups (timbre / harmony / rhythm / combined) with Random Forest,  
then RF vs MLP on the best feature set.  
Saves unified outputs and branch probabilities for later hybrid fusion to
`results/1 Random Forest vs MLP/`.

## Part 2 — CNN Optimisation

### Step 0 — Extract mel spectrograms locally (~30 min, runs once)
```bash
python3 extract_mel_specs.py
```
Extracts mel spectrograms and caches them to `features/mel_specs.npz`.

### 2.1 — Plain CNN
```bash
python3 plain_cnn.py
```
Trains a plain CNN without augmentation.
Saves unified outputs to `results/2.1 Plain CNN/`.

### 2.2 — Regularisation ablation
```bash
python3 regularisation_ablation.py
```
Compares a heavily regularised CNN against a moderately regularised CNN.
This demonstrates under-learning when regularisation is too aggressive.
No augmentation is used here, so this stage isolates regularisation effects.
Saves unified outputs to `results/2.2 Regularisation ablation/`.

### 2.3 — ResNet CNN
```bash
python3 resnet_cnn.py
```
Trains a stronger ResNet-style CNN without augmentation, so this stage isolates
architecture effects.
Saves unified outputs to `results/2.3 ResNet CNN/`.

### 2.4 — Augmentation ablation on the selected model
```bash
python3 augmentation_ablation.py
```
Reads the validation results from Part 2.1, Part 2.2, and Part 2.3, selects the
best model architecture by validation F1-macro, reuses that model's no-augmentation result,
then trains SpecAugment, Mixup, and SpecAugment + Mixup on the selected model.
Saves unified outputs to `results/2.4 Augmentation ablation/`.

### 2.5 — Segment Averaging
Requires the local segment cache:
```bash
python3 extract_mel_segments.py
```
This writes `features/mel_segments.npz`.

```bash
python3 segment_averaging.py
```
Selects the current best CNN branch by validation F1-macro, trains that model on
multiple mel segments per track, and averages segment probabilities for
track-level validation/test prediction.
Saves unified outputs to `results/2.5 Segment Averaging/`.

## Part 3 — Hybrid Modal

```bash
python3 hybrid_late_fusion.py
```
Selects the best CNN-Mel branch and the best handcrafted-feature branch, then
searches a validation-set late-fusion weight and evaluates the fused model on
the test split. This step reads saved CNN and handcrafted branch probabilities
instead of retraining either branch. If `branch_probabilities.npz` is missing,
rerun the corresponding earlier experiment once with the updated code.
Saves unified outputs to `results/3 Hybrid Modal/`.

The final selected model is the best validation-selected model after Part 3.

## Part 4 — Error Analysis

```bash
python3 error_analysis.py
```
Analyses the selected model's `predictions.csv`, reports weakest classes and
most common confusion pairs, and writes next-improvement notes to
`results/4 Error analysis/`.

### Optional — Model complexity summary
```bash
python3 model_complexity_summary.py
```
Writes architecture parameter counts to `results/model_parameter_counts.csv`.
After experiments have been rerun, it also collects recorded training runtimes from
each `metrics.csv` into `results/model_training_runtimes.csv`.

---

## Results

Each experiment writes the same core output files inside its own subdirectory:

| File | Description |
|---|---|
| `metrics.csv` | Validation/test accuracy, F1, parameter counts, epochs run, and training runtime |
| `metrics.png` | Compact multi-model test accuracy/F1 comparison; skipped for single-model experiments |
| `classification_report.txt` | Test classification report for the validation-selected best model |
| `confusion_matrix.png` | Test confusion matrix for the validation-selected best model |
| `training_history.csv` | Per-epoch training/validation history for neural models |
| `training_history.png` | Training vs validation loss and accuracy curves for neural models |
| `predictions.csv` | Test-set true/predicted labels, confidence, track id, and mp3 path for error analysis |
| `high_confidence_errors.csv` | Misclassified test tracks sorted by prediction confidence |
| `selection_candidates.csv` | Part 2.4 only: Part 2.1/2.2/2.3 candidates ranked by validation F1 |
| `selected_model.txt` | Part 2.4 only: selected model used for augmentation ablation |
| `no_augmentation_baseline.csv` | Part 2.4 only: reused no-augmentation result from Part 2.1/2.2/2.3 |
| `augmentation_comparison.csv` | Part 2.4 only: reused baseline plus newly trained augmentations |
| `previous_best_baseline.csv` | Part 2.5 only: selected pre-segment baseline |
| `segment_comparison.csv` | Part 2.5 only: previous best vs segment averaging |
| `cnn_branch_candidates.csv` | Part 3 only: CNN branches considered for fusion |
| `handcrafted_branch_candidates.csv` | Part 3 only: handcrafted branches considered for fusion |
| `fusion_weight_search.csv` | Part 3 only: validation F1 across fusion weights |
| `selected_fusion_weight.csv` | Part 3 only: selected CNN/handcrafted fusion weights |
| `branch_probabilities.npz` | Validation/test probabilities saved by branch-producing experiments |
| `branch_probability_index.csv` | Branch labels stored in `branch_probabilities.npz` |
| `features/mel_segments.npz` | Part 2.5 only: cached fixed-width mel segments for segment training |

The cross-experiment leaderboard is updated automatically after each model
script runs:

| File | Description |
|---|---|
| `results/model_comparison.csv` | Unified test metrics across all experiments that have been rerun |
| `results/model_comparison.png` | Unified comparison plot |

The handcrafted baseline also writes `feature_importance.png` for the best RF
feature set.

---

## Dataset

**FMA-small** — Free Music Archive  
Defferrard et al., "FMA: A Dataset for Music Analysis", ISMIR 2017  
8000 tracks · 30 seconds each · 8 genres · Creative Commons licensed  
[https://github.com/mdeff/fma](https://github.com/mdeff/fma)
