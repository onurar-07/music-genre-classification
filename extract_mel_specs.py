"""
Extract cached log-mel spectrograms for FMA-small.

Run after extract_features.py has created features/features.npz. This script
reads track IDs from that file, loads the corresponding MP3 files, and writes
features/mel_specs.npz for the CNN experiments.
"""

import warnings

import librosa
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
FEATURES_DIR = ROOT / "features"
FEATURES_DIR.mkdir(exist_ok=True)
FEATURES_PATH = FEATURES_DIR / "features.npz"
MEL_CACHE = FEATURES_DIR / "mel_specs.npz"
FMA_AUDIO = ROOT / "data" / "fma_small"

SR = 22050
DURATION = 29.0
N_MELS = 128
HOP_LEN = 512
MEL_W = 128


def audio_path(track_id: int):
    tid = f"{track_id:06d}"
    return FMA_AUDIO / tid[:3] / f"{tid}.mp3"


def extract_mel(y: np.ndarray, sr: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LEN, fmax=sr // 2
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] != MEL_W:
        scale = (N_MELS / mel_db.shape[0], MEL_W / mel_db.shape[1])
        mel_db = zoom(mel_db, scale, order=1)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    return mel_db.astype(np.float32)


def build_mel_cache():
    assert FEATURES_PATH.exists(), "Run extract_features.py first to create features/features.npz."

    data = np.load(FEATURES_PATH, allow_pickle=True)
    track_ids = data["track_ids"]
    labels = data["labels"]

    print(f"Extracting mel spectrograms for {len(track_ids)} tracks...")
    mels, valid_idx = [], []
    for i, tid in enumerate(tqdm(track_ids, desc="Mel extraction")):
        path = audio_path(int(tid))
        if not path.exists():
            continue
        try:
            y, sr = librosa.load(path, sr=SR, duration=DURATION, mono=True)
            mels.append(extract_mel(y, sr))
            valid_idx.append(i)
        except Exception as exc:
            tqdm.write(f"skip {tid}: {exc}")

    mels = np.stack(mels)
    labels = labels[valid_idx]
    track_ids = track_ids[valid_idx]

    np.savez_compressed(
        MEL_CACHE,
        mels=mels.astype(np.float16),
        labels=labels,
        track_ids=track_ids,
    )
    print(f"Saved {len(mels)} mel spectrograms to {MEL_CACHE.relative_to(ROOT)}")
    print(f"Shape: {mels.shape}  dtype: float16 on disk")


def main():
    build_mel_cache()


if __name__ == "__main__":
    main()
