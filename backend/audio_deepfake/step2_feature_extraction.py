"""
STEP 2 — FEATURE EXTRACTION (FoR-norm) — IMPROVED
====================================================
Changes from original:
  • Added spectral rolloff, mel-spectrogram stats, delta-MFCCs
  • Added tempo and spectral bandwidth
  • Feature vector: 122 → ~200 features
  • Better temporal info via delta (velocity) of MFCCs

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

SAMPLE_RATE  = 16000
MAX_DURATION = 4.0
N_MFCC       = 40
N_MELS       = 64

LABEL_REAL = 0
LABEL_FAKE = 1

# ─── FEATURE EXTRACTION ───────────────────────────────────────────────────────

def extract_features(audio_path, sr=SAMPLE_RATE, max_duration=MAX_DURATION):
    """
    Extracts a fixed-size feature vector from one audio file.

    Features:
      - MFCC mean & std                     (40 × 2 = 80)
      - Delta-MFCC mean & std               (40 × 2 = 80)  ← NEW: captures dynamics
      - Chroma mean & std                   (12 × 2 = 24)
      - Spectral contrast mean & std        ( 7 × 2 = 14)
      - Mel-spectrogram mean & std          (64 × 2 = 128) ← NEW: richer freq info
      - Spectral rolloff mean & std                     (2) ← NEW
      - Spectral bandwidth mean & std                   (2) ← NEW
      - Zero crossing rate mean & std                   (2)
      - RMS energy mean & std                           (2)
      ────────────────────────────────────────────────────
      Total: ~334 features
    """
    max_samples = int(sr * max_duration)

    try:
        y, _ = librosa.load(audio_path, sr=sr, duration=max_duration, mono=True)
    except Exception:
        return None

    if len(y) < max_samples:
        y = np.pad(y, (0, max_samples - len(y)))

    features = []

    # 1. MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # 2. Delta-MFCCs — captures how MFCCs change over time (temporal dynamics)
    #    Deepfakes often have unnatural transitions — this catches them
    delta_mfcc = librosa.feature.delta(mfcc)
    features.extend(np.mean(delta_mfcc, axis=1))
    features.extend(np.std(delta_mfcc, axis=1))

    # 3. Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    # 4. Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features.extend(np.mean(contrast, axis=1))
    features.extend(np.std(contrast, axis=1))

    # 5. Mel-spectrogram — finer frequency resolution than MFCCs
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features.extend(np.mean(mel_db, axis=1))
    features.extend(np.std(mel_db, axis=1))

    # 6. Spectral rolloff — frequency below which 85% of energy is concentrated
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(np.mean(rolloff))
    features.append(np.std(rolloff))

    # 7. Spectral bandwidth — width of the spectral distribution
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(np.mean(bandwidth))
    features.append(np.std(bandwidth))

    # 8. Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    # 9. RMS energy
    rms = librosa.feature.rms(y=y)
    features.append(np.mean(rms))
    features.append(np.std(rms))

    return np.array(features, dtype=np.float32)


# ─── LOAD ONE SPLIT ───────────────────────────────────────────────────────────

def load_split(split_name):
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
        files  = [f for f in os.listdir(folder) if f.lower().endswith(supported)]
        failed = 0

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
    print("Step 2: Feature Extraction (FoR-norm) — IMPROVED")
    print("=" * 60)
    print(f"\n  Dataset     : {DATASET_ROOT}")
    print(f"  Sample rate : {SAMPLE_RATE} Hz  |  Max duration : {MAX_DURATION}s")
    print(f"  New features: delta-MFCCs, mel-spectrogram, rolloff, bandwidth\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("── training ────────────────────────────────────────────────")
    X_train, y_train = load_split("training")

    print("── validation ──────────────────────────────────────────────")
    X_val, y_val = load_split("validation")

    print("── testing ─────────────────────────────────────────────────")
    X_test, y_test = load_split("testing")

    print("─" * 60)
    print(f"  Feature vector size : {X_train.shape[1]}  (was 122)")
    print(f"  Train   : {len(X_train):>7,} samples")
    print(f"  Val     : {len(X_val):>7,} samples")
    print(f"  Test    : {len(X_test):>7,} samples")

    saves = {
        "X_train.npy": X_train, "y_train.npy": y_train,
        "X_val.npy":   X_val,   "y_val.npy":   y_val,
        "X_test.npy":  X_test,  "y_test.npy":  y_test,
    }
    for fname, arr in saves.items():
        np.save(os.path.join(OUTPUT_DIR, fname), arr)

    print(f"\n✓ Features saved to '{OUTPUT_DIR}/'")
    print("\nProceed to step3_train_model.py")

if __name__ == "__main__":
    main()