"""
STEP 2 — FEATURE EXTRACTION (FoR-norm)
========================================
Reads the pre-split FoR-norm dataset (training / validation / testing),
extracts audio features using librosa, and saves feature matrices as .npy.

Key differences from DEEP-VOICE pipeline:
  • FoR-norm is already split — we respect those splits, no random re-splitting
  • Files are already 16 kHz mono WAV — matches our SAMPLE_RATE exactly
  • Three output splits: train, val, test (not just train/test)
  • Validation set is saved separately for use in step 3 (threshold tuning)

Run:
    python step2_feature_extraction.py
"""

import os
import numpy as np
import librosa
from tqdm import tqdm

# ─── CONFIG ───────────────────────────────────────────────────────────────────

DATASET_ROOT = "../dataset/for-norm"
OUTPUT_DIR   = "features"

# FoR-norm is already at 16 kHz mono — we match that exactly
SAMPLE_RATE  = 16000
MAX_DURATION = 4.0    # seconds — clip or pad to this length
N_MFCC       = 40     # number of MFCC coefficients

# Labels
LABEL_REAL = 0
LABEL_FAKE = 1

# ─── FEATURE EXTRACTION ───────────────────────────────────────────────────────

def extract_features(audio_path, sr=SAMPLE_RATE, max_duration=MAX_DURATION):
    """
    Extracts a fixed-size 122-feature vector from one audio file.

    Features:
      - MFCC mean & std           (40 × 2 = 80)
      - Chroma mean & std         (12 × 2 = 24)
      - Spectral contrast mean & std (7 × 2 = 14)
      - Zero crossing rate mean & std          (2)
      - RMS energy mean & std                  (2)
      ──────────────────────────────────────────────
      Total: 122 features
    """
    max_samples = int(sr * max_duration)

    try:
        # FoR-norm WAVs are already 16 kHz mono — librosa will still work fine
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


# ─── LOAD ONE SPLIT ───────────────────────────────────────────────────────────

def load_split(split_name):
    """
    Loads all real + fake files from one FoR-norm split folder.
    Returns X (feature matrix) and y (labels).
    """
    supported = (".wav", ".mp3", ".flac", ".ogg")

    real_dir = os.path.join(DATASET_ROOT, split_name, "real")
    fake_dir = os.path.join(DATASET_ROOT, split_name, "fake")

    for d in [real_dir, fake_dir]:
        if not os.path.exists(d):
            raise FileNotFoundError(
                f"Expected folder not found: {d}\n"
                f"Run step1_dataset_setup.py to verify your dataset."
            )

    X, y = [], []

    for folder, label, label_name in [
        (real_dir, LABEL_REAL, "real"),
        (fake_dir, LABEL_FAKE, "fake"),
    ]:
        files   = [f for f in os.listdir(folder) if f.lower().endswith(supported)]
        failed  = 0

        print(f"  [{split_name}/{label_name}]  {len(files):,} files")

        for fname in tqdm(files, desc=f"  Extracting {split_name}/{label_name}"):
            fpath = os.path.join(folder, fname)
            feats = extract_features(fpath)
            if feats is None:
                failed += 1
                continue
            X.append(feats)
            y.append(label)

        if failed:
            print(f"    ⚠  {failed} files skipped (unreadable / corrupt)")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    real_n = (y == LABEL_REAL).sum()
    fake_n = (y == LABEL_FAKE).sum()
    print(f"  → {split_name}: {len(X):,} samples  (real={real_n:,}, fake={fake_n:,})\n")
    return X, y


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PRISM — Audio Deepfake Detection")
    print("Step 2: Feature Extraction (FoR-norm)")
    print("=" * 60)
    print(f"\n  Dataset : {DATASET_ROOT}")
    print(f"  Sample rate : {SAMPLE_RATE} Hz  |  Max duration : {MAX_DURATION}s")
    print(f"  Features per clip : 122\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Training split ──────────────────────────────────────────────────────
    print("── training ────────────────────────────────────────────────")
    X_train, y_train = load_split("training")

    # ── Validation split ────────────────────────────────────────────────────
    print("── validation ──────────────────────────────────────────────")
    X_val, y_val = load_split("validation")

    # ── Testing split ───────────────────────────────────────────────────────
    print("── testing ─────────────────────────────────────────────────")
    X_test, y_test = load_split("testing")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("─" * 60)
    print(f"  Feature vector size : {X_train.shape[1]}")
    print(f"  Train   : {len(X_train):>7,} samples")
    print(f"  Val     : {len(X_val):>7,} samples")
    print(f"  Test    : {len(X_test):>7,} samples")
    print(f"  Total   : {len(X_train)+len(X_val)+len(X_test):>7,} samples")

    # ── Save ────────────────────────────────────────────────────────────────
    saves = {
        "X_train.npy": X_train, "y_train.npy": y_train,
        "X_val.npy":   X_val,   "y_val.npy":   y_val,
        "X_test.npy":  X_test,  "y_test.npy":  y_test,
    }
    for fname, arr in saves.items():
        np.save(os.path.join(OUTPUT_DIR, fname), arr)

    print(f"\n✓ Features saved to '{OUTPUT_DIR}/'")
    print("  X_train / y_train  — used in step3 for training")
    print("  X_val   / y_val    — used in step3 for threshold tuning")
    print("  X_test  / y_test   — used in step4 for final evaluation")
    print("\nProceed to step3_train_model.py")

if __name__ == "__main__":
    main()