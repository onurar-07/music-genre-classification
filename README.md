# Music Genre Classification — FMA-small

Politecnico di Milano — Selected Topics in Music and Acoustic Engineering  
Task V: Music Genre Classification

Classifies 30-second music tracks into 8 genres using the FMA-small dataset.  
Compares handcrafted features (MFCC, chroma, rhythm) with CNN-based approaches on mel spectrograms.

---

## Project structure

```
term_project/
├── extract_features.py     # Step 1 — extract handcrafted features from audio
├── train_evaluate.py       # Step 2 — Random Forest vs MLP baseline
├── cnn_classifier.py       # Step 3 — CNN on mel spectrograms (+ SpecAugment)
├── cnn_regularized.py      # Step 4 — heavily regularised CNN (reference)
├── cnn_balanced.py         # Step 5 — balanced regularisation CNN (best model)
├── requirements.txt
├── data/                   # ← put dataset here (see below)
├── features/               # auto-created when you run the scripts
└── results/                # auto-created — all plots and CSVs saved here
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
Saves plots and a results table to `results/`.

### Step 3 — CNN on mel spectrograms (~30 min first run, then fast)
```bash
python3 cnn_classifier.py
```
Extracts mel spectrograms and caches them to `features/mel_specs.npz` (first run only).  
Trains CNN without augmentation, then CNN with SpecAugment.

### Step 4 — Heavily regularised CNN
```bash
python3 cnn_regularized.py
```
Aggressive regularisation: strong dropout, large SpecAugment masks, Mixup, label smoothing.  
Useful for comparison — shows what happens when regularisation is too strong.

### Step 5 — Balanced CNN (best model)
```bash
python3 cnn_balanced.py
```
Tuned regularisation that avoids both overfitting and under-learning.  
Produces `results/all_models_final.png` — all models compared side by side.

---

## Results

All plots and CSVs are saved to `results/` after each script:

| File | Description |
|---|---|
| `feature_comparison.png` | RF accuracy across feature groups |
| `confusion_matrix_best.png` | Confusion matrix for best RF/MLP model |
| `cnn_training_history.png` | CNN training curves |
| `full_comparison.png` | RF/MLP vs CNN vs CNN+SpecAugment |
| `regularised_training.png` | Overfitting analysis |
| `balanced_training.png` | Balanced CNN training curves |
| `all_models_final.png` | All models compared |
| `all_models_final.csv` | Numbers for all models |

---

## Dataset

**FMA-small** — Free Music Archive  
Defferrard et al., "FMA: A Dataset for Music Analysis", ISMIR 2017  
8000 tracks · 30 seconds each · 8 genres · Creative Commons licensed  
[https://github.com/mdeff/fma](https://github.com/mdeff/fma)
