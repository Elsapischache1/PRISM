"""
STEP 3 — MODEL TRAINING
========================
Loads extracted features, trains a Random Forest classifier,
evaluates on test set, saves the trained model.

Run:
    python step3_train_model.py
"""

import os
import numpy as np
import joblib
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, confusion_matrix
)

# ─── CONFIG ───────────────────────────────────────────────────────────────────

FEATURES_DIR = "features"
MODEL_DIR    = "model"
MODEL_PATH   = os.path.join(MODEL_DIR, "audio_deepfake_model.joblib")

# ─── LOAD DATA ────────────────────────────────────────────────────────────────

def load_features():
    print("Loading features...")
    X_train = np.load(os.path.join(FEATURES_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(FEATURES_DIR, "y_train.npy"))
    X_test  = np.load(os.path.join(FEATURES_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(FEATURES_DIR, "y_test.npy"))

    print(f"  Train : {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Test  : {X_test.shape[0]} samples")
    print(f"  Train — REAL: {(y_train==0).sum()}, FAKE: {(y_train==1).sum()}")
    return X_train, y_train, X_test, y_test

# ─── EVALUATION ───────────────────────────────────────────────────────────────

def evaluate(model, X, y, split_name="Test"):
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)
    cm  = confusion_matrix(y, y_pred)

    print(f"\n── {split_name} Results ──────────────────────────────")
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"                Predicted")
    print(f"                REAL    FAKE")
    print(f"  Actual REAL   {cm[0][0]:5d}   {cm[0][1]:5d}")
    print(f"  Actual FAKE   {cm[1][0]:5d}   {cm[1][1]:5d}")
    print(f"\n  Classification Report:")
    print(classification_report(y, y_pred, target_names=["REAL", "FAKE"]))

    return acc, auc

# ─── FEATURE IMPORTANCE ───────────────────────────────────────────────────────

def print_feature_importance(model, top_n=10):
    rf = model.named_steps["clf"]
    importances = rf.feature_importances_

    feature_names = (
        [f"MFCC_mean_{i}"        for i in range(40)] +
        [f"MFCC_std_{i}"         for i in range(40)] +
        [f"Chroma_mean_{i}"      for i in range(12)] +
        [f"Chroma_std_{i}"       for i in range(12)] +
        [f"SpContrast_mean_{i}"  for i in range(7)]  +
        [f"SpContrast_std_{i}"   for i in range(7)]  +
        ["ZCR_mean", "ZCR_std", "RMS_mean", "RMS_std"]
    )

    paired = sorted(zip(importances, feature_names), reverse=True)

    print(f"\n── Top {top_n} Most Important Features ─────────────────")
    for i, (imp, name) in enumerate(paired[:top_n]):
        bar = "█" * int(imp * 300)
        print(f"  {i+1:2d}. {name:<25s}  {imp:.4f}  {bar}")

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PRISM — Audio Deepfake Detection")
    print("Step 3: Model Training")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)

    X_train, y_train, X_test, y_test = load_features()

    print("\nBuilding model pipeline...")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
            verbose=1
        ))
    ])

    print(f"\nTraining on {len(X_train)} samples...")
    print("(Takes ~2–5 min on CPU ☕)")
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"\nTraining complete in {elapsed:.1f}s")

    train_acc, train_auc = evaluate(model, X_train, y_train, "Train")
    test_acc,  test_auc  = evaluate(model, X_test,  y_test,  "Test")

    print_feature_importance(model, top_n=10)

    joblib.dump(model, MODEL_PATH)
    model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"\n✓ Model saved to '{MODEL_PATH}' ({model_size:.1f} MB)")

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Train Accuracy : {train_acc*100:.2f}%")
    print(f"  Test  Accuracy : {test_acc*100:.2f}%")
    print(f"  Test  ROC-AUC  : {test_auc:.4f}")
    print(f"  Model saved at : {MODEL_PATH}")
    print("=" * 60)

    if test_acc < 0.80:
        print("\n⚠  Accuracy below 80%. The dataset may be small.")
        print("   Try collecting more audio samples.")
    else:
        print("\n✓ Good accuracy! Proceed to step4_evaluate.py")

if __name__ == "__main__":
    main()