# Music Genre Classification — FMA-small

Politecnico di Milano — Selected Topics in Music and Acoustic Engineering  
Task V: Music Genre Classification

Classifies 30-second music tracks into 8 genres using the FMA-small dataset.  
Compares handcrafted features (MFCC, chroma, rhythm) with CNN-based approaches on mel spectrograms.

---

## Project structure

```
term_project/
├── extract_features.py  # Step 1 — extract handcrafted features from audio
├── train_evaluate.py    # Step 2 — Random Forest vs MLP baseline
├── extract_mel_specs.py # Step 3 — extract cached mel spectrograms
├── cnn_plain.py         # Step 4 — plain CNN on mel spectrograms (+ SpecAugment)
├── cnn_regularized.py   # Step 5 — heavily regularised CNN (reference)
├── cnn_balanced.py      # Step 6 — balanced regularisation CNN for comparison
├── cnn_resnet.py        # Step 7 — full ResNet-style CNN experiment
├── experiment_utils.py  # Shared split, metrics, plots, and reports
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

> **macOS note:** the built-in `unzip` does not support the zip format used here.  
> Use `7z` instead: `brew install p7zip` if you don't have it.

After extraction your `data/` folder should look like this:

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

Run the scripts in order. Each one builds on the previous.

All model scripts use the same stratified train/validation/test split:
64% train, 16% validation, 20% test. Validation F1-macro selects the best
model/epoch; final metrics are reported on the held-out test split.

### Step 1 — Extract handcrafted features (~30 min, runs once)
```bash
python3 extract_features.py
```
Saves `features/features.npz` with timbre, harmony, rhythm, and combined feature vectors.

### Step 2 — Baseline: Random Forest vs MLP
```bash
python3 train_evaluate.py
```
Compares feature groups (timbre / harmony / rhythm / combined) with Random Forest,  
then RF vs MLP on the best feature set.  
Saves unified outputs to `results/Random Forest vs MLP/`.

### Step 3 — Extract mel spectrograms (~30 min, runs once)
```bash
python3 extract_mel_specs.py
```
Extracts mel spectrograms and caches them to `features/mel_specs.npz`.

### Step 4 — Plain CNN on mel spectrograms
```bash
python3 cnn_plain.py
```
Trains CNN without augmentation, then CNN with SpecAugment.
Saves unified outputs to `results/Plain CNN/`.

### Step 5 — Heavily regularised CNN
```bash
python3 cnn_regularized.py
```
Aggressive regularisation: strong dropout, large SpecAugment masks, Mixup, label smoothing.  
Useful for comparison — shows what happens when regularisation is too strong.
Saves unified outputs to `results/Regularised CNN/`.

### Step 6 — Balanced CNN regularisation experiment
```bash
python3 cnn_balanced.py
```
Tuned regularisation that avoids both overfitting and under-learning.  
Saves unified outputs to `results/Balanced CNN/`.

### Step 7 — ResNet-style CNN experiment
```bash
python3 cnn_resnet.py
```
Full residual CNN using the cached mel spectrograms.
Saves unified outputs to `results/ResNet CNN/`.

---

## Results

Each experiment writes the same core output files inside its own subdirectory:

| File | Description |
|---|---|
| `metrics.csv` | Validation and test accuracy/F1 for each model in that script |
| `metrics.png` | Bar chart of test accuracy and F1-macro |
| `classification_report.txt` | Test classification report for the validation-selected best model |
| `confusion_matrix.png` | Test confusion matrix for the validation-selected best model |
| `training_history.csv` | Per-epoch training/validation history for neural models |
| `training_history.png` | Training curves for neural models |

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
