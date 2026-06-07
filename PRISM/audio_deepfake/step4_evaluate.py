"""
STEP 4 — EVALUATION & PLOTS
=============================
Loads trained model, runs full evaluation,
saves plots as plots/evaluation.png

Run:
    python step4_evaluate.py
"""

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    confusion_matrix, classification_report,
    accuracy_score
)

# ─── CONFIG ───────────────────────────────────────────────────────────────────

FEATURES_DIR = "features"
MODEL_PATH   = "model/audio_deepfake_model.joblib"
PLOTS_DIR    = "plots"

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PRISM — Audio Deepfake Detection")
    print("Step 4: Evaluation")
    print("=" * 60)

    print("\nLoading model and test features...")
    model  = joblib.load(MODEL_PATH)
    X_test = np.load(os.path.join(FEATURES_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(FEATURES_DIR, "y_test.npy"))

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm  = confusion_matrix(y_test, y_pred)

    print(f"\n  Accuracy : {acc*100:.2f}%")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['REAL', 'FAKE'])}")

    # ── Plots ────────────────────────────────────────────────────
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig = plt.figure(figsize=(16, 5))
    fig.suptitle("PRISM — Audio Deepfake Detection: Evaluation", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(1, 3, figure=fig)

    # Plot 1: Confusion Matrix
    ax1 = fig.add_subplot(gs[0])
    im  = ax1.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax1.set_title("Confusion Matrix")
    ax1.set_xlabel("Predicted Label")
    ax1.set_ylabel("True Label")
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["REAL", "FAKE"])
    ax1.set_yticks([0, 1]); ax1.set_yticklabels(["REAL", "FAKE"])
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm[i, j]),
                     ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black",
                     fontsize=16, fontweight="bold")
    plt.colorbar(im, ax=ax1)

    # Plot 2: ROC Curve
    ax2 = fig.add_subplot(gs[1])
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax2.plot(fpr, tpr, color="#4f46e5", lw=2, label=f"AUC = {auc:.4f}")
    ax2.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax2.fill_between(fpr, tpr, alpha=0.1, color="#4f46e5")
    ax2.set_title("ROC Curve")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend()
    ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1.02])

    # Plot 3: Score Distribution
    ax3 = fig.add_subplot(gs[2])
    ax3.hist(y_proba[y_test == 0], bins=30, alpha=0.6, color="#22c55e", label="REAL")
    ax3.hist(y_proba[y_test == 1], bins=30, alpha=0.6, color="#ef4444", label="FAKE")
    ax3.axvline(x=0.5, color="black", linestyle="--", lw=1.5, label="Threshold (0.5)")
    ax3.set_title("Prediction Score Distribution")
    ax3.set_xlabel("P(FAKE) — model confidence")
    ax3.set_ylabel("Count")
    ax3.legend()

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "evaluation.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✓ Plots saved to '{out_path}'")
    print("\nProceed to step5_predict.py to test on a single audio file.")

if __name__ == "__main__":
    main()