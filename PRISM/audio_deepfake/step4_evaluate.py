"""
STEP 4 — EVALUATION & PLOTS (FIXED)
=====================================
Updated to use the new model package (scaler + model + threshold).

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
    confusion_matrix, classification_report, accuracy_score
)

FEATURES_DIR = "features"
MODEL_PATH   = "model/audio_deepfake_model.joblib"
PLOTS_DIR    = "plots"

def main():
    print("=" * 60)
    print("PRISM — Audio Deepfake Detection")
    print("Step 4: Evaluation")
    print("=" * 60)

    package   = joblib.load(MODEL_PATH)
    scaler    = package["scaler"]
    model     = package["model"]
    threshold = package["threshold"]

    X_test = np.load(os.path.join(FEATURES_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(FEATURES_DIR, "y_test.npy"))

    X_scaled = scaler.transform(X_test)
    y_proba  = model.predict_proba(X_scaled)[:, 1]
    y_pred   = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm  = confusion_matrix(y_test, y_pred)

    print(f"\n  Threshold : {threshold:.2f}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['REAL', 'FAKE'])}")

    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig = plt.figure(figsize=(16, 5))
    fig.suptitle("PRISM — Audio Deepfake Detection: Evaluation", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(1, 3, figure=fig)

    ax1 = fig.add_subplot(gs[0])
    im  = ax1.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax1.set_title("Confusion Matrix")
    ax1.set_xlabel("Predicted"); ax1.set_ylabel("True")
    ax1.set_xticks([0,1]); ax1.set_xticklabels(["REAL","FAKE"])
    ax1.set_yticks([0,1]); ax1.set_yticklabels(["REAL","FAKE"])
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm[i,j]), ha="center", va="center",
                     color="white" if cm[i,j] > cm.max()/2 else "black",
                     fontsize=16, fontweight="bold")
    plt.colorbar(im, ax=ax1)

    ax2 = fig.add_subplot(gs[1])
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax2.plot(fpr, tpr, color="#4f46e5", lw=2, label=f"AUC = {auc:.4f}")
    ax2.plot([0,1],[0,1],"k--",lw=1,label="Random")
    ax2.fill_between(fpr, tpr, alpha=0.1, color="#4f46e5")
    ax2.set_title("ROC Curve"); ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
    ax2.legend(); ax2.set_xlim([0,1]); ax2.set_ylim([0,1.02])

    ax3 = fig.add_subplot(gs[2])
    ax3.hist(y_proba[y_test==0], bins=30, alpha=0.6, color="#22c55e", label="REAL")
    ax3.hist(y_proba[y_test==1], bins=30, alpha=0.6, color="#ef4444", label="FAKE")
    ax3.axvline(x=threshold, color="black", linestyle="--", lw=1.5, label=f"Threshold ({threshold:.2f})")
    ax3.set_title("Score Distribution"); ax3.set_xlabel("P(FAKE)"); ax3.set_ylabel("Count")
    ax3.legend()

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "evaluation.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Plots saved to '{out_path}'")
    print("\nProceed to step5_predict.py")

if __name__ == "__main__":
    main()