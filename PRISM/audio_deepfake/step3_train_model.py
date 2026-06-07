"""
STEP 3 — MODEL TRAINING
========================
Loads extracted features, trains a Random Forest classifier,
evaluates on dev set, saves the trained model.

Run:
    python step3_train_model.py
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, confusion_matrix
)
import time

# ─── CONFIG ───────────────────────────────────────────────────────────────────

FEATURES_DIR = "features"
MODEL_DIR    = "model"
MODEL_PATH   = os.path.join(MODEL_DIR, "audio_deepfake_model.joblib")

# ─── LOAD DATA ────────────────────────────────────────────────────────────────

def load_features():
    print("Loading features...")
    X_train = np.load(os.path.join(FEATURES_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(FEATURES_DIR, "y_train.npy"))
    X_dev   = np.load(os.path.join(FEATURES_DIR, "X_dev.npy"))
    y_dev   = np.load(os.path.join(FEATURES_DIR, "y_dev.npy"))

    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Dev  : {X_dev.shape[0]} samples")
    print(f"  Train class balance — Genuine: {(y_train==0).sum()}, Spoof: {(y_train==1).sum()}")
    return X_train, y_train, X_dev, y_dev

# ─── EVALUATION ───────────────────────────────────────────────────────────────

def evaluate(model, X, y, split_name="Dev"):
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]  # probability of being spoof

    acc     = accuracy_score(y, y_pred)
    auc     = roc_auc_score(y, y_proba)
    cm      = confusion_matrix(y, y_pred)

    print(f"\n── {split_name} Results ──────────────────────────────")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"              Predicted")
    print(f"              Real   Fake")
    print(f"  Actual Real  {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"  Actual Fake  {cm[1][0]:5d}  {cm[1][1]:5d}")
    print(f"\n  Classification Report:")
    print(classification_report(y, y_pred, target_names=["Genuine", "Spoof"]))

    return acc, auc

# ─── FEATURE IMPORTANCE ───────────────────────────────────────────────────────

def print_feature_importance(model, top_n=10):
    """Print top N most important features from the Random Forest."""
    rf = model.named_steps["clf"]
    importances = rf.feature_importances_

    feature_names = (
        [f"MFCC_mean_{i}"     for i in range(40)] +
        [f"MFCC_std_{i}"      for i in range(40)] +
        [f"Chroma_mean_{i}"   for i in range(12)] +
        [f"Chroma_std_{i}"    for i in range(12)] +
        [f"SpContrast_mean_{i}" for i in range(7)] +
        [f"SpContrast_std_{i}"  for i in range(7)] +
        ["ZCR_mean", "ZCR_std", "RMS_mean", "RMS_std"]
    )

    # Pair and sort
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

    X_train, y_train, X_dev, y_dev = load_features()

    # ── Build pipeline ──────────────────────────────────────────
    # StandardScaler normalizes features (important for some features having
    # very different scales). Random Forest is then trained on normalized data.

    print("\nBuilding model pipeline...")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,       # number of trees
            max_depth=20,           # prevent overfitting
            min_samples_leaf=2,
            class_weight="balanced",# handles class imbalance automatically
            n_jobs=-1,              # use all CPU cores
            random_state=42,
            verbose=1
        ))
    ])

    # ── Train ───────────────────────────────────────────────────
    print(f"\nTraining Random Forest on {len(X_train)} samples...")
    print("(This may take 2–5 minutes on CPU — grab a coffee ☕)")
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"\nTraining complete in {elapsed:.1f}s")

    # ── Evaluate ────────────────────────────────────────────────
    train_acc, train_auc = evaluate(model, X_train, y_train, split_name="Train")
    dev_acc,   dev_auc   = evaluate(model, X_dev,   y_dev,   split_name="Dev")

    # ── Feature importance ──────────────────────────────────────
    print_feature_importance(model, top_n=10)

    # ── Save model ──────────────────────────────────────────────
    joblib.dump(model, MODEL_PATH)
    model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"\n✓ Model saved to '{MODEL_PATH}' ({model_size:.1f} MB)")

    # ── Summary card ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Train Accuracy  : {train_acc*100:.2f}%")
    print(f"  Dev   Accuracy  : {dev_acc*100:.2f}%")
    print(f"  Dev   ROC-AUC   : {dev_auc:.4f}")
    print(f"  Model           : {MODEL_PATH}")
    print("=" * 60)

    if dev_acc < 0.80:
        print("\n⚠  Dev accuracy below 80%. Consider:")
        print("   - Increasing MAX_TRAIN_SAMPLES in step2 and re-extracting")
        print("   - Trying GradientBoosting (swap clf in this script)")
    else:
        print("\n✓ Good accuracy! Proceed to step4_evaluate.py for detailed analysis.")

if __name__ == "__main__":
    main()