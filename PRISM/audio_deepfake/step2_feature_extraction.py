"""
STEP 2 — FEATURE EXTRACTION
============================
Reads REAL/ and FAKE/ folders from DEEP-VOICE dataset,
extracts audio features using librosa,
saves feature matrix + labels as .npy files.

Install dependencies first:
    pip install librosa numpy scikit-learn tqdm soundfile

Run:
    python step2_feature_extraction.py
"""

import os
import numpy as np
import librosa
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ─── CONFIG ───────────────────────────────────────────────────────────────────

REAL_DIR     = "../dataset/DEEP-VOICE/REAL"
FAKE_DIR     = "../dataset/DEEP-VOICE/FAKE"
OUTPUT_DIR   = "features"

SAMPLE_RATE  = 16000   # resample all audio to this
MAX_DURATION = 4.0     # seconds — clip/pad all audio to this
N_MFCC       = 40      # number of MFCC coefficients

TEST_SIZE    = 0.2     # 80% train, 20% test split
RANDOM_SEED  = 42

# ─── FEATURE EXTRACTION ───────────────────────────────────────────────────────

def extract_features(audio_path, sr=SAMPLE_RATE, max_duration=MAX_DURATION):
    """
    Extracts a fixed-size 122-feature vector from one audio file.

    Features:
      - MFCC mean & std        (40 × 2 = 80)
      - Chroma mean & std      (12 × 2 = 24)
      - Spectral contrast mean & std (7 × 2 = 14)
      - Zero crossing rate mean & std      (2)
      - RMS energy mean & std              (2)
      ─────────────────────────────────────────
      Total: 122 features
    """
    max_samples = int(sr * max_duration)

    try:
        y, _ = librosa.load(audio_path, sr=sr, duration=max_duration, mono=True)
    except Exception as e:
        return None

    # Pad if shorter than max_duration
    if len(y) < max_samples:
        y = np.pad(y, (0, max_samples - len(y)))

    features = []

    # 1. MFCCs — captures vocal tract shape (most important for deepfake detection)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # 2. Chroma — pitch class energy distribution
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    # 3. Spectral contrast — difference between peaks and valleys in spectrum
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features.extend(np.mean(contrast, axis=1))
    features.extend(np.std(contrast, axis=1))

    # 4. Zero crossing rate — how often signal crosses zero (smoothness)
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    # 5. RMS energy — volume/loudness dynamics
    rms = librosa.feature.rms(y=y)
    features.append(np.mean(rms))
    features.append(np.std(rms))

    return np.array(features, dtype=np.float32)


# ─── LOAD ALL FILES ───────────────────────────────────────────────────────────

def load_folder(folder_path, label, label_name):
    """Load all audio files from a folder and extract features."""
    supported = (".mp3", ".wav", ".flac", ".ogg")
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(supported)]

    print(f"\n[{label_name}] Found {len(files)} files in {folder_path}")

    X, y = [], []
    failed = 0

    for fname in tqdm(files, desc=f"Extracting [{label_name}]"):
        fpath = os.path.join(folder_path, fname)
        feats = extract_features(fpath)

        if feats is None:
            failed += 1
            continue

        X.append(feats)
        y.append(label)

    if failed > 0:
        print(f"  Warning: {failed} files skipped (unreadable or corrupt)")

    print(f"  Successfully extracted: {len(X)} files")
    return X, y


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PRISM — Audio Deepfake Detection")
    print("Step 2: Feature Extraction")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load REAL files (label = 0)
    X_real, y_real = load_folder(REAL_DIR, label=0, label_name="REAL")

    # Load FAKE files (label = 1)
    X_fake, y_fake = load_folder(FAKE_DIR, label=1, label_name="FAKE")

    # Combine
    X = np.array(X_real + X_fake, dtype=np.float32)
    y = np.array(y_real + y_fake, dtype=np.int32)

    print(f"\n  Total samples     : {len(X)}")
    print(f"  Feature size      : {X.shape[1]}")
    print(f"  REAL (label 0)    : {(y==0).sum()}")
    print(f"  FAKE (label 1)    : {(y==1).sum()}")

    # Train / test split (stratified — keeps class balance in both splits)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )

    print(f"\n  Train split : {len(X_train)} samples")
    print(f"  Test split  : {len(X_test)} samples")

    # Save
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"),  y_test)

    print(f"\n✓ Features saved to '{OUTPUT_DIR}/'")
    print("  X_train.npy  y_train.npy  X_test.npy  y_test.npy")
    print("\nProceed to step3_train_model.py")

if __name__ == "__main__":
    main()