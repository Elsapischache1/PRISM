"""
STEP 4 — EVALUATION & PLOTS — IMPROVED
========================================
Added:
  • Threshold sweep plot — shows how precision/recall trade off
  • Per-class score distribution with clear threshold line
  • Prints comparison: default vs tuned threshold

Run:
    python step4_evaluate.py
"""

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve,
    confusion_matrix, classification_report, accuracy_score,
    f1_score, recall_score, precision_score
)

FEATURES_DIR = "features"
MODEL_PATH   = "model/audio_deepfake_model.joblib"
PLOTS_DIR    = "plots"

def main():
    print("=" * 60)
    print("PRISM — Audio Deepfake Detection")
    print("Step 4: Evaluation — IMPROVED")
    print("=" * 60)

    package   = joblib.load(MODEL_PATH)
    scaler    = package["scaler"]
    model     = package["model"]
    threshold = package["threshold"]

    X_test = np.load(os.path.join(FEATURES_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(FEATURES_DIR, "y_test.npy"))

    X_scaled = scaler.transform(X_test)
    y_proba  = model.predict_proba(X_scaled)[:, 1]

    # ── Results at default (0.5) vs tuned threshold ──────────────────────────
    print("\n── Comparison: Default vs Tuned Threshold ──────────────────")
    print(f"{'Metric':<22} {'Default (0.50)':>15} {'Tuned ({:.2f})'.format(threshold):>15}")
    print("─" * 55)

    for t, label in [(0.5, "default"), (threshold, "tuned")]:
        y_pred = (y_proba >= t).astype(int)
        acc    = accuracy_score(y_test, y_pred)
        auc    = roc_auc_score(y_test, y_proba)
        fake_r = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        real_r = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
        f1     = f1_score(y_test, y_pred, average="macro", zero_division=0)
        if label == "default":
            d = {"acc": acc, "auc": auc, "fake_r": fake_r, "real_r": real_r, "f1": f1}
        else:
            t_data = {"acc": acc, "auc": auc, "fake_r": fake_r, "real_r": real_r, "f1": f1}

    rows = [
        ("Accuracy",       f"{d['acc']*100:.2f}%",   f"{t_data['acc']*100:.2f}%"),
        ("ROC-AUC",        f"{d['auc']:.4f}",         f"{t_data['auc']:.4f}"),
        ("FAKE Recall",    f"{d['fake_r']:.4f}",      f"{t_data['fake_r']:.4f}"),
        ("REAL Recall",    f"{d['real_r']:.4f}",      f"{t_data['real_r']:.4f}"),
        ("Macro F1",       f"{d['f1']:.4f}",          f"{t_data['f1']:.4f}"),
    ]
    for name, dv, tv in rows:
        print(f"  {name:<20} {dv:>15} {tv:>15}")

    # Full report at tuned threshold
    y_pred_tuned = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_tuned)
    print(f"\n── Full Report (threshold={threshold:.2f}) ──────────────────")
    print(f"  Threshold : {threshold:.2f}")
    print(f"  Accuracy  : {t_data['acc']*100:.2f}%")
    print(f"  ROC-AUC   : {t_data['auc']:.4f}")
    print(f"\n{classification_report(y_test, y_pred_tuned, target_names=['REAL', 'FAKE'])}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig = plt.figure(figsize=(20, 5))
    fig.suptitle("PRISM — Audio Deepfake Detection: Evaluation", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(1, 4, figure=fig)

    # Plot 1: Confusion matrix
    ax1 = fig.add_subplot(gs[0])
    im  = ax1.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax1.set_title(f"Confusion Matrix\n(threshold={threshold:.2f})")
    ax1.set_xlabel("Predicted"); ax1.set_ylabel("True")
    ax1.set_xticks([0,1]); ax1.set_xticklabels(["REAL","FAKE"])
    ax1.set_yticks([0,1]); ax1.set_yticklabels(["REAL","FAKE"])
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm[i,j]), ha="center", va="center",
                     color="white" if cm[i,j] > cm.max()/2 else "black",
                     fontsize=16, fontweight="bold")
    plt.colorbar(im, ax=ax1)

    # Plot 2: ROC curve
    ax2 = fig.add_subplot(gs[1])
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val = roc_auc_score(y_test, y_proba)
    ax2.plot(fpr, tpr, color="#4f46e5", lw=2, label=f"AUC = {auc_val:.4f}")
    ax2.plot([0,1],[0,1],"k--",lw=1,label="Random")
    ax2.fill_between(fpr, tpr, alpha=0.1, color="#4f46e5")
    ax2.set_title("ROC Curve"); ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
    ax2.legend(); ax2.set_xlim([0,1]); ax2.set_ylim([0,1.02])

    # Plot 3: Score distribution
    ax3 = fig.add_subplot(gs[2])
    ax3.hist(y_proba[y_test==0], bins=30, alpha=0.6, color="#22c55e", label="REAL")
    ax3.hist(y_proba[y_test==1], bins=30, alpha=0.6, color="#ef4444", label="FAKE")
    ax3.axvline(x=threshold, color="black", linestyle="--", lw=2,
                label=f"Threshold ({threshold:.2f})")
    ax3.axvline(x=0.5, color="gray", linestyle=":", lw=1.5, label="Default (0.50)")
    ax3.set_title("Score Distribution"); ax3.set_xlabel("P(FAKE)"); ax3.set_ylabel("Count")
    ax3.legend(fontsize=8)

    # Plot 4: NEW — Threshold sweep (precision/recall vs threshold)
    ax4 = fig.add_subplot(gs[3])
    thresholds = np.arange(0.05, 0.95, 0.01)
    fake_recalls   = []
    real_recalls   = []
    macro_f1s      = []

    for t in thresholds:
        yp = (y_proba >= t).astype(int)
        fake_recalls.append(recall_score(y_test, yp, pos_label=1, zero_division=0))
        real_recalls.append(recall_score(y_test, yp, pos_label=0, zero_division=0))
        macro_f1s.append(f1_score(y_test, yp, average="macro", zero_division=0))

    ax4.plot(thresholds, fake_recalls, color="#ef4444", lw=2, label="FAKE recall")
    ax4.plot(thresholds, real_recalls, color="#22c55e", lw=2, label="REAL recall")
    ax4.plot(thresholds, macro_f1s,   color="#4f46e5", lw=2, linestyle="--", label="Macro F1")
    ax4.axvline(x=threshold, color="black", linestyle="--", lw=1.5,
                label=f"Chosen ({threshold:.2f})")
    ax4.set_title("Threshold Sweep"); ax4.set_xlabel("Threshold"); ax4.set_ylabel("Score")
    ax4.legend(fontsize=8); ax4.set_xlim([0.05, 0.95]); ax4.set_ylim([0, 1.05])
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "evaluation.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Plots saved to '{out_path}'")
    print("\nProceed to step5_predict.py")

if __name__ == "__main__":
    main()