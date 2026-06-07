"""
STEP 3 — MODEL TRAINING (FoR-norm) — IMPROVED (FAST)
======================================================
Key improvements over original:
  1. SMOTE-style oversampling via manual interpolation
  2. RandomForest with class_weight="balanced" — fast, no GBM slowdown
  3. Threshold tuned to maximize FAKE recall (target >= 80%)
  4. No cross-validation during training (saves ~10-15 min)
     → val set already gives honest performance estimate

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
    roc_auc_score, confusion_matrix, f1_score, recall_score
)
from sklearn.calibration import CalibratedClassifierCV

# ─── CONFIG ───────────────────────────────────────────────────────────────────

FEATURES_DIR = "features"
MODEL_DIR    = "model"
MODEL_PATH   = os.path.join(MODEL_DIR, "audio_deepfake_model.joblib")

# Target minimum FAKE recall — tune threshold until this is met
TARGET_FAKE_RECALL = 0.78
# Minimum REAL recall — don't let it drop below this while chasing FAKE recall
MIN_REAL_RECALL = 0.78

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

    return X_train, y_train, X_val, y_val, X_test, y_test

# ─── SYNTHETIC MINORITY OVERSAMPLING (manual SMOTE-like) ──────────────────────

def synthetic_oversample(X, y, random_state=42):
    """
    Generates synthetic samples for the minority class by interpolating
    between real samples. Better than plain duplication because it adds
    variety instead of exact copies.
    """
    rng = np.random.RandomState(random_state)

    X_real = X[y == 0]
    X_fake = X[y == 1]
    y_real = y[y == 0]
    y_fake = y[y == 1]

    minority, majority = (X_fake, y_fake, 1), (X_real, y_real, 0)
    if len(X_real) < len(X_fake):
        minority, majority = (X_real, y_real, 0), (X_fake, y_fake, 1)

    X_min, y_min_arr, min_label = minority
    X_maj, y_maj_arr, _         = majority
    n_needed = len(X_maj) - len(X_min)

    print(f"  Generating {n_needed:,} synthetic samples for class {min_label}...")

    synthetic_X = []
    for _ in range(n_needed):
        # Pick two random minority samples and interpolate
        idx1, idx2 = rng.choice(len(X_min), 2, replace=False)
        alpha = rng.uniform(0.2, 0.8)           # blend ratio
        new_sample = X_min[idx1] * alpha + X_min[idx2] * (1 - alpha)
        synthetic_X.append(new_sample)

    synthetic_X = np.array(synthetic_X, dtype=np.float32)
    synthetic_y = np.full(n_needed, min_label, dtype=np.int32)

    X_out = np.vstack([X, synthetic_X])
    y_out = np.concatenate([y, synthetic_y])

    # Shuffle
    idx = rng.permutation(len(X_out))
    print(f"  After oversampling: REAL={(y_out==0).sum():,}, FAKE={(y_out==1).sum():,}")
    return X_out[idx], y_out[idx]

# ─── THRESHOLD TUNING ─────────────────────────────────────────────────────────

def find_best_threshold(model, X_val_scaled, y_val,
                        target_fake_recall=TARGET_FAKE_RECALL,
                        min_real_recall=MIN_REAL_RECALL):
    """
    Finds threshold that:
    1. FAKE recall >= target_fake_recall  (catch deepfakes)
    2. REAL recall >= min_real_recall     (don't kill real audio)
    3. Among qualifying thresholds, picks best macro-F1

    Falls back to minimizing |fake_recall - real_recall| if no threshold
    satisfies both constraints simultaneously.
    """
    y_proba = model.predict_proba(X_val_scaled)[:, 1]
    candidates = []

    for thresh in np.arange(0.05, 0.95, 0.01):
        y_pred      = (y_proba >= thresh).astype(int)
        fake_recall = recall_score(y_val, y_pred, pos_label=1, zero_division=0)
        real_recall = recall_score(y_val, y_pred, pos_label=0, zero_division=0)
        macro_f1    = f1_score(y_val, y_pred, average="macro", zero_division=0)
        candidates.append((thresh, fake_recall, real_recall, macro_f1))

    # Tier 1: both recall constraints met
    qualified = [(t, fr, rr, f) for t, fr, rr, f in candidates
                 if fr >= target_fake_recall and rr >= min_real_recall]

    if qualified:
        best = max(qualified, key=lambda x: x[3])  # best macro-F1
        print(f"  ✓ Threshold {best[0]:.2f}  |  FAKE recall: {best[1]:.3f}  REAL recall: {best[2]:.3f}  macro-F1: {best[3]:.4f}")
        return float(best[0])

    # Tier 2: minimize gap between fake and real recall (most balanced)
    print(f"  ⚠ No threshold meets both constraints — finding most balanced...")
    best = min(candidates, key=lambda x: abs(x[1] - x[2]))
    print(f"  → Threshold {best[0]:.2f}  |  FAKE recall: {best[1]:.3f}  REAL recall: {best[2]:.3f}  macro-F1: {best[3]:.4f}")
    return float(best[0])

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
    print("Step 3: Model Training — IMPROVED")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test = load_features()

    # 1. Merge train + val for final model training
    #    More data = better generalization to test set
    #    We'll use a small stratified split of the combined set for threshold tuning
    print("\nMerging train + val for final training...")
    X_all = np.vstack([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])
    print(f"  Combined: {len(X_all):,} samples  (REAL: {(y_all==0).sum():,}, FAKE: {(y_all==1).sum():,})")

    # Hold out 15% of combined set for threshold tuning (stratified)
    from sklearn.model_selection import train_test_split
    X_fit, X_thresh, y_fit, y_thresh = train_test_split(
        X_all, y_all, test_size=0.15, stratify=y_all, random_state=42
    )
    print(f"  Fit set   : {len(X_fit):,} samples")
    print(f"  Thresh set: {len(X_thresh):,} samples")

    # 2. Synthetic oversampling on fit set only
    print("\nBalancing classes with synthetic oversampling...")
    X_fit_bal, y_fit_bal = synthetic_oversample(X_fit, y_fit)

    # 3. Scale (fit on training data only)
    scaler        = StandardScaler()
    X_fit_scaled  = scaler.fit_transform(X_fit_bal)
    X_thresh_scaled = scaler.transform(X_thresh)
    X_test_scaled = scaler.transform(X_test)

    # 3. Build RF with class_weight="balanced"
    #    - Faster than GBM ensemble (parallelizes across all cores)
    #    - class_weight="balanced" penalizes FAKE misses more heavily
    #    - 150 trees is enough for 53k samples; diminishing returns beyond that
    print("\nBuilding RandomForest (class_weight=balanced)...")

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,              # was 20 — shallower = less memorization
        min_samples_leaf=8,        # was 2 — harder to overfit individual samples
        min_samples_split=16,      # was 5 — more conservative splits
        max_features="sqrt",
        class_weight={0: 1, 1: 1.4},
        n_jobs=-1,
        random_state=42,
    )

    # sigmoid calibration is more stable than isotonic when training data
    # and val data come from slightly different distributions (FoR-norm quirk)
    print(f"\nTraining calibrated RF on {len(X_fit_bal):,} samples...")
    calibrated = CalibratedClassifierCV(rf, method="sigmoid", cv=5)

    start = time.time()
    calibrated.fit(X_fit_scaled, y_fit_bal)
    print(f"Done in {time.time() - start:.1f}s")

    # 5. Tune threshold on the held-out threshold set (never seen during training)
    print(f"\nFinding threshold targeting FAKE recall >= {TARGET_FAKE_RECALL:.0%}...")
    best_threshold = find_best_threshold(calibrated, X_thresh_scaled, y_thresh)

    # 6. Evaluate on test only (no val leakage)
    evaluate(calibrated, X_thresh_scaled, y_thresh, threshold=best_threshold, split_name="Threshold-tuning set")
    evaluate(calibrated, X_test_scaled,   y_test,   threshold=0.5,            split_name="Test (default 0.5)")
    evaluate(calibrated, X_test_scaled,   y_test,   threshold=best_threshold, split_name="Test (tuned threshold)")

    # 7. Save
    package = {
        "scaler"    : scaler,
        "model"     : calibrated,
        "threshold" : best_threshold
    }
    joblib.dump(package, MODEL_PATH)
    print(f"\n✓ Saved model package to '{MODEL_PATH}'")
    print("  Contains: scaler + calibrated RF + tuned threshold")
    print("\nProceed to step4_evaluate.py")

if __name__ == "__main__":
    main()