"""
Feature extraction for FMA-small music genre classification.
Run this script ONCE; it saves all features to features/features.npz.

Expected directory layout before running:
    data/
        fma_small/          ← download from https://github.com/mdeff/fma
        fma_metadata/       ← download from the same repo (metadata zip)

Three feature groups are extracted per track:
    - Timbre  : MFCCs, spectral contrast, spectral roll-off, ZCR
    - Harmony : Chroma (STFT), Tonnetz
    - Rhythm  : Tempo + tempogram summary (beat histogram)
The script also saves their concatenation as "combined".
"""

import warnings
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm

from reporting_utils import print_section

warnings.filterwarnings("ignore")

# ── Paths (always relative to this script's location) ─────────────────────────
ROOT          = Path(__file__).parent
FMA_AUDIO_DIR = ROOT / "data" / "fma_small"
METADATA_DIR  = ROOT / "data" / "fma_metadata"
FEATURES_DIR  = ROOT / "features"
FEATURES_DIR.mkdir(exist_ok=True)

# ── Audio parameters ──────────────────────────────────────────────────────────
SR       = 22050   # resample everything to 22.05 kHz
DURATION = 29.0    # slightly under 30 s to avoid edge artefacts
N_MFCC   = 40
HOP_LEN  = 512


# ── Dataset helpers ───────────────────────────────────────────────────────────
def load_tracks() -> pd.Series:
    """Return a Series mapping track_id → genre_top for FMA-small tracks."""
    tracks = pd.read_csv(METADATA_DIR / "tracks.csv", index_col=0, header=[0, 1])
    small  = tracks[tracks[("set", "subset")] == "small"]
    genre  = small[("track", "genre_top")].dropna()
    return genre


def audio_path(track_id: int) -> Path:
    tid = f"{track_id:06d}"
    return FMA_AUDIO_DIR / tid[:3] / f"{tid}.mp3"


# ── Feature groups ────────────────────────────────────────────────────────────
def timbre_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Captures timbral characteristics.
    Features: MFCCs (40 coeffs) + spectral contrast (7 bands) +
              spectral roll-off + ZCR  →  98-dim vector (mean + std each).
    """
    mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LEN)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=HOP_LEN)
    rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LEN)
    zcr      = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LEN)
    return np.concatenate([
        mfcc.mean(1),     mfcc.std(1),       # 80
        contrast.mean(1), contrast.std(1),   # 14
        rolloff.mean(1),  rolloff.std(1),     #  2
        zcr.mean(1),      zcr.std(1),         #  2
    ])  # 98 dims total


def harmony_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Captures harmonic/tonal content.
    Features: Chroma STFT (12 pitch classes) + Tonnetz (6 dims)
              →  36-dim vector (mean + std each).
    """
    chroma  = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LEN)
    harm    = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=harm, sr=sr)
    return np.concatenate([
        chroma.mean(1),   chroma.std(1),    # 24
        tonnetz.mean(1),  tonnetz.std(1),   # 12
    ])  # 36 dims total


def rhythm_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Captures rhythmic patterns.
    Features: estimated tempo (BPM) + tempogram summary (beat histogram)
              →  257-dim vector.
    The tempogram encodes the periodic energy distribution over tempo values,
    acting as a beat histogram across the 30-second clip.
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LEN)
    tempo, _  = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=HOP_LEN
    )
    tempogram = librosa.feature.tempogram(
        onset_envelope=onset_env, sr=sr,
        hop_length=HOP_LEN, win_length=384
    )
    tg_crop = tempogram[:128]          # first 128 BPM bins
    return np.concatenate([
        [float(np.squeeze(tempo))],    #   1
        tg_crop.mean(1),               # 128
        tg_crop.std(1),                # 128
    ])  # 257 dims total


def extract_all(y: np.ndarray, sr: int) -> dict:
    t = timbre_features(y, sr)
    h = harmony_features(y, sr)
    r = rhythm_features(y, sr)
    return {
        "timbre":   t,
        "harmony":  h,
        "rhythm":   r,
        "combined": np.concatenate([t, h, r]),
    }


# ── Main extraction loop ──────────────────────────────────────────────────────
def main():
    tracks = load_tracks()
    print_section("Extract handcrafted features")
    print(f"Tracks with genre label: {len(tracks)}")
    print_section("Genre distribution")
    print(tracks.value_counts().to_string(), "\n")

    buffers = {"timbre": [], "harmony": [], "rhythm": [], "combined": []}
    labels, track_ids = [], []
    skipped = 0

    for tid, genre in tqdm(tracks.items(), total=len(tracks), desc="Extracting"):
        path = audio_path(tid)
        if not path.exists():
            skipped += 1
            continue
        try:
            y, sr = librosa.load(path, sr=SR, duration=DURATION, mono=True)
            feats = extract_all(y, sr)
            for k, v in feats.items():
                buffers[k].append(v)
            labels.append(genre)
            track_ids.append(tid)
        except Exception as exc:
            tqdm.write(f"  skip {tid}: {exc}")
            skipped += 1

    n = len(labels)
    print_section("Extraction summary")
    print(f"Processed: {n} / {len(tracks)} tracks  skipped={skipped}")

    np.savez_compressed(
        FEATURES_DIR / "features.npz",
        timbre    = np.vstack(buffers["timbre"]),
        harmony   = np.vstack(buffers["harmony"]),
        rhythm    = np.vstack(buffers["rhythm"]),
        combined  = np.vstack(buffers["combined"]),
        labels    = np.array(labels),
        track_ids = np.array(track_ids),
    )
    print("Saved: features/features.npz")
    print(f"Shapes: timbre={np.vstack(buffers['timbre']).shape}  "
          f"harmony={np.vstack(buffers['harmony']).shape}  "
          f"rhythm={np.vstack(buffers['rhythm']).shape}  "
          f"combined={np.vstack(buffers['combined']).shape}")


if __name__ == "__main__":
    main()
