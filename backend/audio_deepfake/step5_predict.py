"""
STEP 5 — SINGLE FILE PREDICTION
=========================================
Uses the saved model package (scaler + calibrated model + optimal threshold).

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

MODEL_PATH   = os.path.join(os.path.dirname(__file__), "model", "audio_deepfake_model.joblib")
SAMPLE_RATE  = 16000
MAX_DURATION = 4.0
N_MFCC       = 40
N_MELS       = 64

# ─── FEATURE EXTRACTION (must match step2 exactly — 334 features) ─────────────

def extract_features(audio_path, sr=SAMPLE_RATE, max_duration=MAX_DURATION):
    max_samples = int(sr * max_duration)

    try:
        y, _ = librosa.load(audio_path, sr=sr, duration=max_duration, mono=True)
    except Exception as e:
        raise ValueError(f"Could not load audio file: {e}")

    if len(y) < max_samples:
        y = np.pad(y, (0, max_samples - len(y)))

    features = []

    # 1. MFCCs (80)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # 2. Delta-MFCCs (80)
    delta_mfcc = librosa.feature.delta(mfcc)
    features.extend(np.mean(delta_mfcc, axis=1))
    features.extend(np.std(delta_mfcc, axis=1))

    # 3. Chroma (24)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    # 4. Spectral contrast (14)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features.extend(np.mean(contrast, axis=1))
    features.extend(np.std(contrast, axis=1))

    # 5. Mel-spectrogram (128)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features.extend(np.mean(mel_db, axis=1))
    features.extend(np.std(mel_db, axis=1))

    # 6. Spectral rolloff (2)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(np.mean(rolloff))
    features.append(np.std(rolloff))

    # 7. Spectral bandwidth (2)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(np.mean(bandwidth))
    features.append(np.std(bandwidth))

    # 8. Zero crossing rate (2)
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    # 9. RMS energy (2)
    rms = librosa.feature.rms(y=y)
    features.append(np.mean(rms))
    features.append(np.std(rms))

    return np.array(features, dtype=np.float32)


# ─── INFERENCE FUNCTION (imported by api.py) ──────────────────────────────────

def predict_audio(audio_path: str, threshold_override: float = None) -> dict:
    """
    Main inference function — called by FastAPI backend (api.py).

    Returns a dict with these keys (frontend-ready):
        label      : "DEEPFAKE" or "AUTHENTIC"
        confidence : float, 0–100  (probability of predicted class)
        status     : human-readable string
        p_real     : float, 0–100
        p_fake     : float, 0–100
        threshold  : float (threshold used)
        borderline : bool (True if close call)
        file       : filename only
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. Run step3_train_model.py first."
        )

    # Load package: scaler + calibrated model + threshold
    package   = joblib.load(MODEL_PATH)
    scaler    = package["scaler"]
    model     = package["model"]
    threshold = package["threshold"]

    if threshold_override is not None:
        threshold = threshold_override

    # Extract and scale features
    features        = extract_features(audio_path).reshape(1, -1)
    features_scaled = scaler.transform(features)

    # Predict
    probability = model.predict_proba(features_scaled)[0]   # [p_real, p_fake]
    p_fake      = float(probability[1])
    p_real      = float(probability[0])

    # Apply threshold
    is_fake    = p_fake >= threshold
    borderline = abs(p_fake - threshold) < 0.10

    # Frontend-compatible keys
    label      = "DEEPFAKE"   if is_fake else "AUTHENTIC"
    status     = "Manipulation detected" if is_fake else "No manipulation found"
    confidence = (p_fake if is_fake else p_real) * 100

    return {
        # ── frontend keys ──────────────────────────────
        "label"      : label,
        "confidence" : round(confidence, 2),
        "status"     : status,
        # ── extra detail (shown in result card) ────────
        "p_real"     : round(p_real * 100, 2),
        "p_fake"     : round(p_fake * 100, 2),
        "threshold"  : round(threshold, 2),
        "borderline" : borderline,
        "file"       : os.path.basename(audio_path),
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PRISM — Audio Deepfake Detection")
    print("Step 5: Single File Prediction")
    print("=" * 60)

    if len(sys.argv) < 2:
        print("\nUsage: python step5_predict.py <audio_file> [threshold]")
        print("  threshold : optional override, e.g. 0.5  (default: use saved threshold)")
        print("Supported: .mp3  .wav  .flac  .ogg  .mpeg")
        sys.exit(1)

    audio_path         = sys.argv[1]
    threshold_override = float(sys.argv[2]) if len(sys.argv) >= 3 else None

    if not os.path.exists(audio_path):
        print(f"✗ File not found: {audio_path}")
        sys.exit(1)

    print(f"\nAnalyzing: {audio_path}")
    print("Extracting features...")

    try:
        result = predict_audio(audio_path, threshold_override=threshold_override)
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)

    print("\n" + "─" * 40)
    icon = "🔴 DEEPFAKE DETECTED" if result["label"] == "DEEPFAKE" else "🟢 GENUINE AUDIO"
    print(f"  Verdict     : {icon}")
    if result["borderline"]:
        print(f"  ⚠ Borderline — close to threshold, treat with caution")
    print(f"  Confidence  : {result['confidence']:.1f}%")
    print(f"  P(Real)     : {result['p_real']:.1f}%")
    print(f"  P(Fake)     : {result['p_fake']:.1f}%")
    print(f"  Threshold   : {result['threshold']}  (DEEPFAKE if P(Fake) >= threshold)")
    print("─" * 40)

if __name__ == "__main__":
    main()