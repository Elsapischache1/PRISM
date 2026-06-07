"""
STEP 2 — FEATURE EXTRACTION
============================
Reads audio files from ASVspoof 2019 LA dataset,
extracts handcrafted features using librosa,
saves a feature matrix + labels as .npy files.

Install dependencies first:
    pip install librosa soundfile numpy scikit-learn tqdm

Run:
    python step2_feature_extraction.py
"""

import os
import numpy as np
import librosa
from tqdm import tqdm

# ─── CONFIG ───────────────────────────────────────────────────────────────────

DATASET_DIR   = "dataset/LA"
OUTPUT_DIR    = "features"

TRAIN_AUDIO   = os.path.join(DATASET_DIR, "ASVspoof2019_LA_train/flac")
DEV_AUDIO     = os.path.join(DATASET_DIR, "ASVspoof2019_LA_dev/flac")

TRAIN_PROTO   = os.path.join(DATASET_DIR, "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt")
DEV_PROTO     = os.path.join(DATASET_DIR, "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt")

SAMPLE_RATE   = 16000   # ASVspoof standard
MAX_DURATION  = 4.0     # seconds — clip/pad all audio to this length
N_MFCC        = 40      # number of MFCC coefficients

# Limit samples for faster CPU training (set to None to use full dataset)
MAX_TRAIN_SAMPLES = 5000
MAX_DEV_SAMPLES   = 1000

# ─── LABEL PARSING ────────────────────────────────────────────────────────────

def parse_protocol(proto_path):
    """Returns dict: { file_id -> 0 (genuine) or 1 (spoof) }"""
    labels = {}
    with open(proto_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            file_id = parts[1]
            label   = parts[4]   # 'genuine' or 'spoof'
            labels[file_id] = 1 if label == "spoof" else 0
    return labels

# ─── FEATURE EXTRACTION ───────────────────────────────────────────────────────

def extract_features(audio_path, sr=SAMPLE_RATE, max_duration=MAX_DURATION):
    """
    Extracts a fixed-size feature vector from a single audio file.

    Features:
      - MFCC mean & std (40 coeffs × 2 = 80)
      - Chroma mean & std (12 × 2 = 24)
      - Spectral contrast mean & std (7 × 2 = 14)
      - Zero crossing rate mean & std (1 × 2 = 2)
      - RMS energy mean & std (1 × 2 = 2)
      ─────────────────────────────────────────
      Total: 122 features per audio clip
    """
    max_samples = int(sr * max_duration)

    try:
        y, _ = librosa.load(audio_path, sr=sr, duration=max_duration)
    except Exception as e:
        print(f"  Warning: Could not load {audio_path}: {e}")
        return None

    # Pad if shorter than max_duration
    if len(y) < max_samples:
        y = np.pad(y, (0, max_samples - len(y)))

    features = []

    # 1. MFCCs — captures vocal tract shape
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    features.extend(np.mean(mfcc, axis=1))   # mean across time
    features.extend(np.std(mfcc, axis=1))    # std across time

    # 2. Chroma — pitch class distribution
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    # 3. Spectral contrast — difference between peaks and valleys in spectrum
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features.extend(np.mean(contrast, axis=1))
    features.extend(np.std(contrast, axis=1))

    # 4. Zero crossing rate — how often signal crosses zero (smoothness indicator)
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    # 5. RMS energy — volume dynamics
    rms = librosa.feature.rms(y=y)
    features.append(np.mean(rms))
    features.append(np.std(rms))

    return np.array(features, dtype=np.float32)


# ─── DATASET BUILDER ──────────────────────────────────────────────────────────

def build_dataset(audio_dir, proto_path, max_samples=None, split_name="train"):
    """
    Iterates over all labeled audio files, extracts features, returns X and y.
    """
    print(f"\n[{split_name.upper()}] Parsing labels...")
    labels = parse_protocol(proto_path)

    file_ids = list(labels.keys())
    if max_samples:
        # Balance classes when subsampling
        genuine_ids = [fid for fid in file_ids if labels[fid] == 0]
        spoof_ids   = [fid for fid in file_ids if labels[fid] == 1]
        n = max_samples // 2
        np.random.seed(42)
        genuine_ids = np.random.choice(genuine_ids, min(n, len(genuine_ids)), replace=False)
        spoof_ids   = np.random.choice(spoof_ids,   min(n, len(spoof_ids)),   replace=False)
        file_ids    = list(genuine_ids) + list(spoof_ids)
        np.random.shuffle(file_ids)
        print(f"  Subsampled to {len(genuine_ids)} genuine + {len(spoof_ids)} spoof = {len(file_ids)} total")
    else:
        genuine_count = sum(1 for fid in file_ids if labels[fid] == 0)
        spoof_count   = sum(1 for fid in file_ids if labels[fid] == 1)
        print(f"  Total: {len(file_ids)} files ({genuine_count} genuine, {spoof_count} spoof)")

    X, y = [], []
    failed = 0

    for file_id in tqdm(file_ids, desc=f"Extracting features [{split_name}]"):
        audio_path = os.path.join(audio_dir, file_id + ".flac")

        if not os.path.exists(audio_path):
            failed += 1
            continue

        feats = extract_features(audio_path)
        if feats is None:
            failed += 1
            continue

        X.append(feats)
        y.append(labels[file_id])

    if failed > 0:
        print(f"  Warning: {failed} files skipped (missing or unreadable)")

    return np.array(X), np.array(y)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PRISM — Audio Deepfake Detection")
    print("Step 2: Feature Extraction")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Extract training features
    X_train, y_train = build_dataset(
        TRAIN_AUDIO, TRAIN_PROTO,
        max_samples=MAX_TRAIN_SAMPLES,
        split_name="train"
    )

    # Extract dev/validation features
    X_dev, y_dev = build_dataset(
        DEV_AUDIO, DEV_PROTO,
        max_samples=MAX_DEV_SAMPLES,
        split_name="dev"
    )

    # Save
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "X_dev.npy"),   X_dev)
    np.save(os.path.join(OUTPUT_DIR, "y_dev.npy"),   y_dev)

    print(f"\n✓ Saved to '{OUTPUT_DIR}/'")
    print(f"  X_train shape : {X_train.shape}")
    print(f"  y_train shape : {y_train.shape}  (0=genuine, 1=spoof)")
    print(f"  X_dev shape   : {X_dev.shape}")
    print(f"  y_dev shape   : {y_dev.shape}")
    print(f"\n  Feature vector size: {X_train.shape[1]}")
    print("\nProceed to step3_train_model.py")

if __name__ == "__main__":
    main()