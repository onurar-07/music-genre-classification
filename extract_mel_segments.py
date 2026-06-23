"""
Extract cached fixed-width log-mel segments for segment averaging.

Run this locally where the original FMA-small MP3 files are available. The
resulting features/mel_segments.npz can be copied to Colab for Part 2.6
training without copying the raw audio.
"""

import warnings
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

from reporting_utils import print_section

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
FEATURES_DIR = ROOT / "features"
FEATURES_DIR.mkdir(exist_ok=True)
FEATURES_PATH = FEATURES_DIR / "features.npz"
MEL_CACHE = FEATURES_DIR / "mel_specs.npz"
SEGMENT_CACHE = FEATURES_DIR / "mel_segments.npz"
FMA_AUDIO = ROOT / "data" / "fma_small"

SR = 22050
DURATION = 29.0
N_MELS = 128
HOP_LEN = 512
SEGMENT_FRAMES = 128
SEGMENTS_PER_TRACK = 4


def audio_path(track_id):
    tid = f"{int(track_id):06d}"
    return FMA_AUDIO / tid[:3] / f"{tid}.mp3"


def load_track_index():
    assert FEATURES_PATH.exists(), "Run extract_features.py first to create features/features.npz."
    features = np.load(FEATURES_PATH, allow_pickle=True)
    track_ids = features["track_ids"]
    labels = features["labels"]

    if MEL_CACHE.exists():
        mel_data = np.load(MEL_CACHE, allow_pickle=True)
        if "track_ids" in mel_data.files:
            return mel_data["track_ids"], mel_data["labels"]
        if "labels" in mel_data.files:
            same_labels = (
                len(mel_data["labels"]) == len(labels)
                and np.array_equal(mel_data["labels"], labels)
            )
            if not same_labels:
                raise ValueError(
                    "features/mel_specs.npz has no track_ids and its labels do not match "
                    "features/features.npz. Re-run extract_mel_specs.py before extracting segments."
                )

    return track_ids, labels


def extract_track_segments(track_id):
    path = audio_path(track_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing audio file: {path}")
    y, sr = librosa.load(path, sr=SR, duration=DURATION, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        hop_length=HOP_LEN,
        fmax=sr // 2,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)

    if mel_db.shape[1] < SEGMENT_FRAMES:
        pad = SEGMENT_FRAMES - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode="edge")

    max_start = mel_db.shape[1] - SEGMENT_FRAMES
    starts = np.linspace(0, max_start, SEGMENTS_PER_TRACK).round().astype(int)
    return np.stack([mel_db[:, start:start + SEGMENT_FRAMES] for start in starts]).astype(np.float32)


def build_segment_cache():
    track_ids, labels = load_track_index()
    missing = [track_id for track_id in track_ids if not audio_path(track_id).exists()]
    if missing:
        examples = ", ".join(str(int(track_id)) for track_id in missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} FMA-small audio files under {FMA_AUDIO}. "
            f"Examples: {examples}. Download/extract fma_small.zip locally before running this script."
        )

    print_section("Extract mel segments")
    print(f"Tracks: {len(track_ids)}  segments_per_track={SEGMENTS_PER_TRACK}")
    segments = [
        extract_track_segments(track_id)
        for track_id in tqdm(track_ids, desc="Segment extraction")
    ]
    segments = np.stack(segments).astype(np.float16)

    np.savez_compressed(
        SEGMENT_CACHE,
        segments=segments,
        labels=labels,
        track_ids=track_ids,
        segments_per_track=np.array(SEGMENTS_PER_TRACK),
        segment_frames=np.array(SEGMENT_FRAMES),
    )
    print_section("Extraction summary")
    print(f"Saved: {SEGMENT_CACHE.relative_to(ROOT)}")
    print(f"Mel segments: {segments.shape}  dtype=float16 on disk")


def main():
    build_segment_cache()


if __name__ == "__main__":
    main()
