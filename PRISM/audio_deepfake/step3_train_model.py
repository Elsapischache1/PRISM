"""
STEP 3 — MODEL TRAINING (FoR-norm)
=====================================
Updated for FoR-norm dataset:
  1. Loads X_val / y_val (validation split) for threshold tuning
     → threshold is tuned on val, NOT on test (prevents data leakage)
  2. Proper class balancing with oversampling on train split only
  3. Probability calibration — makes confidence scores reliable
  4. Cross-validation — checks model isn't just memorizing
  5. Final evaluation shown on both val and test sets

Run:
    python step3_train_model.py
"""

import os
import numpy as np
import joblib
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, confusion_matrix, f1_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils import resample

# ─── CONFIG ───────────────────────────────────────────────────────────────────

FEATURES_DIR = "features"
MODEL_DIR    = "model"
MODEL_PATH   = os.path.join(MODEL_DIR, "audio_deepfake_model.joblib")

# ─── LOAD DATA ────────────────────────────────────────────────────────────────

def load_features():
    print("Loading features...")
    X_train = np.load(os.path.join(FEATURES_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(FEATURES_DIR, "y_train.npy"))
    X_val   = np.load(os.path.join(FEATURES_DIR, "X_val.npy"))
    y_val   = np.load(os.path.join(FEATURES_DIR, "y_val.npy"))
    X_test  = np.load(os.path.join(FEATURES_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(FEATURES_DIR, "y_test.npy"))

    real_count = (y_train == 0).sum()
    fake_count = (y_train == 1).sum()
    print(f"  Train : {X_train.shape[0]} samples  (REAL: {real_count}, FAKE: {fake_count})")
    print(f"  Val   : {X_val.shape[0]} samples   (REAL: {(y_val==0).sum()}, FAKE: {(y_val==1).sum()})")
    print(f"  Test  : {X_test.shape[0]} samples   (REAL: {(y_test==0).sum()}, FAKE: {(y_test==1).sum()})")

    ratio = max(real_count, fake_count) / max(min(real_count, fake_count), 1)
    if ratio > 1.5:
        print(f"\n  ⚠  Class imbalance detected (ratio {ratio:.1f}x) — will oversample minority class")

    return X_train, y_train, X_val, y_val, X_test, y_test

# ─── BALANCE CLASSES BY OVERSAMPLING ──────────────────────────────────────────

def balance_classes(X, y):
    X_real = X[y == 0]
    X_fake = X[y == 1]
    y_real = y[y == 0]
    y_fake = y[y == 1]

    if len(X_real) == len(X_fake):
        return X, y

    if len(X_real) < len(X_fake):
        X_real_up, y_real_up = resample(X_real, y_real, replace=True,
                                         n_samples=len(X_fake), random_state=42)
        X_out = np.vstack([X_real_up, X_fake])
        y_out = np.concatenate([y_real_up, y_fake])
    else:
        X_fake_up, y_fake_up = resample(X_fake, y_fake, replace=True,
                                         n_samples=len(X_real), random_state=42)
        X_out = np.vstack([X_real, X_fake_up])
        y_out = np.concatenate([y_real, y_fake_up])

    print(f"  After balancing: REAL={(y_out==0).sum()}, FAKE={(y_out==1).sum()}")
    return X_out, y_out

# ─── FIND OPTIMAL THRESHOLD ───────────────────────────────────────────────────

def find_best_threshold(model, X_test_scaled, y_test):
    y_proba     = model.predict_proba(X_test_scaled)[:, 1]
    best_thresh = 0.5
    best_f1     = 0.0

    for thresh in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_proba >= thresh).astype(int)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1     = f1
            best_thresh = thresh

    print(f"\n  Optimal threshold : {best_thresh:.2f}  (macro-F1: {best_f1:.4f})")
    return float(best_thresh)

# ─── EVALUATION ───────────────────────────────────────────────────────────────

def evaluate(model, X, y, threshold=0.5, split_name="Test"):
    y_proba = model.predict_proba(X)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)
    acc     = accuracy_score(y, y_pred)
    auc     = roc_auc_score(y, y_proba)
    cm      = confusion_matrix(y, y_pred)

    print(f"\n── {split_name} (threshold={threshold:.2f}) ──────────────────")
    print(f"  Accuracy : {acc*100:.2f}%   ROC-AUC : {auc:.4f}")
    print(f"  Confusion Matrix:")
    print(f"                REAL    FAKE")
    print(f"  Actual REAL   {cm[0][0]:5d}   {cm[0][1]:5d}")
    print(f"  Actual FAKE   {cm[1][0]:5d}   {cm[1][1]:5d}")
    print(classification_report(y, y_pred, target_names=["REAL", "FAKE"]))
    return acc, auc

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PRISM — Audio Deepfake Detection")
    print("Step 3: Model Training (FoR-norm)")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test = load_features()

    # 1. Balance classes (train split only — never touch val or test)
    print("\nBalancing classes...")
    X_train_bal, y_train_bal = balance_classes(X_train, y_train)

    # 2. Scale features (fit on train, apply to val + test)
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    # 3. Cross-validate to check learning quality
    print("\nRunning 5-fold cross-validation...")
    base_rf = RandomForestClassifier(
        n_estimators=100, max_depth=15, min_samples_leaf=4,
        n_jobs=-1, random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(base_rf, X_train_scaled, y_train_bal,
                                 cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"  CV ROC-AUC: {cv_scores.round(3)}  |  Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 4. Train final model with probability calibration
    print(f"\nTraining final calibrated model on {len(X_train_bal)} samples...")
    final_rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=4,
        min_samples_split=8,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    calibrated = CalibratedClassifierCV(final_rf, method="isotonic", cv=3)

    start = time.time()
    calibrated.fit(X_train_scaled, y_train_bal)
    print(f"Done in {time.time() - start:.1f}s")

    # 5. Find optimal threshold on VALIDATION set (not test — prevents leakage)
    print("\nFinding optimal decision threshold on validation set...")
    best_threshold = find_best_threshold(calibrated, X_val_scaled, y_val)

    # 6. Evaluate on validation then final test set
    evaluate(calibrated, X_val_scaled,  y_val,  threshold=best_threshold, split_name="Validation")
    evaluate(calibrated, X_test_scaled, y_test, threshold=0.5,            split_name="Test (default threshold)")
    evaluate(calibrated, X_test_scaled, y_test, threshold=best_threshold, split_name="Test (optimal threshold)")

    # 7. Save everything as one package
    package = {
        "scaler"    : scaler,
        "model"     : calibrated,
        "threshold" : best_threshold
    }
    joblib.dump(package, MODEL_PATH)
    print(f"\n✓ Saved model package to '{MODEL_PATH}'")
    print("  Contains: scaler + calibrated model + optimal threshold")
    print("\nProceed to step4_evaluate.py")

if __name__ == "__main__":
    main()