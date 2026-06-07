"""
STEP 4 — EVALUATION & PLOTS
=============================
Loads trained model, runs full evaluation,
saves ROC curve and confusion matrix as PNG files.

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

# ─── LOAD ─────────────────────────────────────────────────────────────────────

def load_all():
    print("Loading model and features...")
    model = joblib.load(MODEL_PATH)
    X_dev = np.load(os.path.join(FEATURES_DIR, "X_dev.npy"))
    y_dev = np.load(os.path.join(FEATURES_DIR, "y_dev.npy"))
    print(f"  Dev set: {len(X_dev)} samples")
    return model, X_dev, y_dev

# ─── PLOTS ────────────────────────────────────────────────────────────────────

def plot_all(model, X_dev, y_dev):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    y_pred  = model.predict(X_dev)
    y_proba = model.predict_proba(X_dev)[:, 1]

    fig = plt.figure(figsize=(16, 5))
    fig.suptitle("PRISM — Audio Deepfake Detection: Evaluation", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(1, 3, figure=fig)

    # ── Plot 1: Confusion Matrix ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    cm  = confusion_matrix(y_dev, y_pred)
    im  = ax1.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax1.set_title("Confusion Matrix")
    ax1.set_xlabel("Predicted Label")
    ax1.set_ylabel("True Label")
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["Genuine", "Spoof"])
    ax1.set_yticks([0, 1]); ax1.set_yticklabels(["Genuine", "Spoof"])
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm[i, j]),
                     ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black",
                     fontsize=16, fontweight="bold")
    plt.colorbar(im, ax=ax1)

    # ── Plot 2: ROC Curve ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    fpr, tpr, _ = roc_curve(y_dev, y_proba)
    auc = roc_auc_score(y_dev, y_proba)
    ax2.plot(fpr, tpr, color="#4f46e5", lw=2, label=f"AUC = {auc:.4f}")
    ax2.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax2.fill_between(fpr, tpr, alpha=0.1, color="#4f46e5")
    ax2.set_title("ROC Curve")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend()
    ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1.02])

    # ── Plot 3: Score Distribution ───────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    genuine_scores = y_proba[y_dev == 0]
    spoof_scores   = y_proba[y_dev == 1]
    ax3.hist(genuine_scores, bins=30, alpha=0.6, color="#22c55e", label="Genuine")
    ax3.hist(spoof_scores,   bins=30, alpha=0.6, color="#ef4444", label="Spoof")
    ax3.axvline(x=0.5, color="black", linestyle="--", lw=1.5, label="Threshold (0.5)")
    ax3.set_title("Prediction Score Distribution")
    ax3.set_xlabel("P(Spoof) — model confidence")
    ax3.set_ylabel("Count")
    ax3.legend()

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "evaluation.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Evaluation plots saved to '{out_path}'")

    return y_pred, y_proba

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PRISM — Audio Deepfake Detection")
    print("Step 4: Evaluation")
    print("=" * 60)

    model, X_dev, y_dev = load_all()

    y_pred  = model.predict(X_dev)
    y_proba = model.predict_proba(X_dev)[:, 1]

    acc = accuracy_score(y_dev, y_pred)
    auc = roc_auc_score(y_dev, y_proba)

    print(f"\n  Accuracy  : {acc*100:.2f}%")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\n{classification_report(y_dev, y_pred, target_names=['Genuine', 'Spoof'])}")

    plot_all(model, X_dev, y_dev)

    print("\n✓ Done. Proceed to step5_predict.py to test on a single file.")

if __name__ == "__main__":
    main()