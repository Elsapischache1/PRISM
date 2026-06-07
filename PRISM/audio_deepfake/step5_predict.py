"""
STEP 5 — SINGLE FILE PREDICTION
=================================
Test the trained model on any audio file.
The predict_audio() function here is imported directly by the FastAPI backend.

Usage:
    python step5_predict.py path/to/audio.mp3
    python step5_predict.py path/to/audio.wav
"""

import sys
import os
import numpy as np
import joblib
import librosa

# ─── CONFIG ───────────────────────────────────────────────────────────────────

MODEL_PATH   = "model/audio_deepfake_model.joblib"
SAMPLE_RATE  = 16000
MAX_DURATION = 4.0
N_MFCC       = 40

# ─── FEATURE EXTRACTION (must be identical to step2) ──────────────────────────

def extract_features(audio_path, sr=SAMPLE_RATE, max_duration=MAX_DURATION):
    max_samples = int(sr * max_duration)

    try:
        y, _ = librosa.load(audio_path, sr=sr, duration=max_duration, mono=True)
    except Exception as e:
        raise ValueError(f"Could not load audio file: {e}")

    if len(y) < max_samples:
        y = np.pad(y, (0, max_samples - len(y)))

    features = []

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features.extend(np.mean(contrast, axis=1))
    features.extend(np.std(contrast, axis=1))

    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    rms = librosa.feature.rms(y=y)
    features.append(np.mean(rms))
    features.append(np.std(rms))

    return np.array(features, dtype=np.float32)


# ─── INFERENCE FUNCTION (imported by FastAPI backend) ─────────────────────────

def predict_audio(audio_path: str) -> dict:
    """
    Main inference function — called by the FastAPI backend.

    Returns:
        {
            "verdict"    : "REAL" or "FAKE",
            "confidence" : float (0–100),
            "p_real"     : float (0–100),
            "p_fake"     : float (0–100),
            "file"       : filename string
        }
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. Run step3_train_model.py first."
        )

    model    = joblib.load(MODEL_PATH)
    features = extract_features(audio_path).reshape(1, -1)

    prediction  = model.predict(features)[0]        # 0=REAL, 1=FAKE
    probability = model.predict_proba(features)[0]  # [p_real, p_fake]

    label      = "FAKE" if prediction == 1 else "REAL"
    confidence = float(probability[prediction]) * 100
    p_real     = float(probability[0]) * 100
    p_fake     = float(probability[1]) * 100

    return {
        "verdict"    : label,
        "confidence" : round(confidence, 2),
        "p_real"     : round(p_real, 2),
        "p_fake"     : round(p_fake, 2),
        "file"       : os.path.basename(audio_path)
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PRISM — Audio Deepfake Detection")
    print("Step 5: Single File Prediction")
    print("=" * 60)

    if len(sys.argv) < 2:
        print("\nUsage: python step5_predict.py <audio_file>")
        print("Supported: .mp3  .wav  .flac  .ogg")
        sys.exit(1)

    audio_path = sys.argv[1]

    if not os.path.exists(audio_path):
        print(f"✗ File not found: {audio_path}")
        sys.exit(1)

    print(f"\nAnalyzing: {audio_path}")
    print("Extracting features...")

    try:
        result = predict_audio(audio_path)
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)

    print("\n" + "─" * 40)
    icon = "🔴 DEEPFAKE DETECTED" if result["verdict"] == "FAKE" else "🟢 GENUINE AUDIO"
    print(f"  Verdict     : {icon}")
    print(f"  Confidence  : {result['confidence']:.1f}%")
    print(f"  P(Real)     : {result['p_real']:.1f}%")
    print(f"  P(Fake)     : {result['p_fake']:.1f}%")
    print("─" * 40)

if __name__ == "__main__":
    main()