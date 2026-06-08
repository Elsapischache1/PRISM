"""
PRISM — test_remaining.py
Evaluates the model on the 100k images NOT used during training.

Training used:  20k real + 20k fake  (random seed 42, same as step3_train.py)
This script uses the remaining ~50k real + ~50k fake = ~100k images.

Run from backend/:
  python video_deepfake/test_remaining.py \
      --faces_dir dataset/faces \
      --checkpoint video_deepfake/model/checkpoints/calibrated_model.pth

Outputs:
  video_deepfake/results/remaining_report.txt
  video_deepfake/results/remaining_confusion.png
  video_deepfake/results/remaining_roc.png
  video_deepfake/results/remaining_score_dist.png
"""

import argparse
import random
import sys
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    confusion_matrix, roc_curve, classification_report
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

BACKEND_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(BACKEND_DIR / "video_deepfake"))
from model.deepfake_detector import build_model


EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# ---------------------------------------------------------------------------
# Reproduce exact train split from step3_train.py, return the COMPLEMENT
# ---------------------------------------------------------------------------

def get_unseen_files(faces_dir: Path, train_per_class: int = 20000, seed: int = 42):
    """
    Mirrors the sampling logic in step3_train.py exactly so we get
    only images the model has never seen.
    """
    real_all = sorted([p for p in (faces_dir / "real").iterdir()
                       if p.suffix.lower() in EXTS])
    fake_all = sorted([p for p in (faces_dir / "fake").iterdir()
                       if p.suffix.lower() in EXTS])

    random.seed(seed)
    # Reproduce step3 sampling
    real_used = set(random.sample(real_all, min(train_per_class, len(real_all))))
    fake_used = set(random.sample(fake_all, min(train_per_class, len(fake_all))))

    # Complement = everything NOT in the training sample
    real_unseen = [p for p in real_all if p not in real_used]
    fake_unseen = [p for p in fake_all if p not in fake_used]

    print(f"[Split] Total real : {len(real_all):,}  →  trained on {len(real_used):,}  →  unseen {len(real_unseen):,}")
    print(f"[Split] Total fake : {len(fake_all):,}  →  trained on {len(fake_used):,}  →  unseen {len(fake_unseen):,}")
    return real_unseen, fake_unseen


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class UnseenDataset(Dataset):
    def __init__(self, real_files, fake_files, transform, seq_len):
        self.transform   = transform
        self.seq_len     = seq_len
        self.real_files  = real_files
        self.fake_files  = fake_files
        self.samples     = [(p, 0) for p in real_files] + [(p, 1) for p in fake_files]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        pool   = self.real_files if label == 0 else self.fake_files
        chosen = [path] + random.choices(pool, k=self.seq_len - 1)
        frames = torch.stack([
            self.transform(Image.open(p).convert("RGB")) for p in chosen
        ])
        return frames, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PRISM] Device: {device}")

    ckpt_path   = Path(args.checkpoint)
    ckpt        = torch.load(ckpt_path, map_location=device)
    ta          = ckpt.get("args", {})
    temperature = ckpt.get("temperature", 1.0)
    seq_len     = ta.get("seq_len", 4)
    image_size  = ta.get("image_size", 224)

    print(f"[PRISM] Checkpoint: {ckpt_path.name}  T={temperature:.4f}  seq_len={seq_len}")

    model = build_model(
        pretrained=False,
        d_model=ta.get("d_model", 256),
        n_heads=ta.get("n_heads", 8),
        n_layers=ta.get("n_layers", 2),
        dropout=0.0, device=device
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Get unseen files
    real_unseen, fake_unseen = get_unseen_files(
        Path(args.faces_dir),
        train_per_class=args.train_per_class,
        seed=args.seed
    )

    if args.max_per_class:
        random.seed(args.seed + 1)
        real_unseen = random.sample(real_unseen, min(args.max_per_class, len(real_unseen)))
        fake_unseen = random.sample(fake_unseen, min(args.max_per_class, len(fake_unseen)))
        print(f"[PRISM] Capped to {len(real_unseen):,} real + {len(fake_unseen):,} fake for speed")

    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    ds     = UnseenDataset(real_unseen, fake_unseen, tf, seq_len)
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers,
                        pin_memory=True)

    print(f"[PRISM] Evaluating {len(ds):,} unseen images …\n")

    all_probs, all_labels = [], []
    for frames, labels in tqdm(loader, desc="Evaluating"):
        logits = model(frames.to(device))
        probs  = torch.sigmoid(logits / temperature).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.numpy().tolist())

    probs  = np.array(all_probs)
    labels = np.array(all_labels)
    preds  = (probs >= 0.5).astype(int)

    auc = roc_auc_score(labels, probs)
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds)
    cm  = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    report = "\n".join([
        f"PRISM — Unseen Data Evaluation Report",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Checkpoint: {ckpt_path.name}",
        f"Images    : {len(ds):,}  ({len(real_unseen):,} real + {len(fake_unseen):,} fake)",
        f"Note      : None of these images were seen during training",
        f"",
        f"{'='*48}",
        f"  AUC      : {auc:.4f}",
        f"  Accuracy : {acc:.4f}",
        f"  F1       : {f1:.4f}",
        f"{'='*48}",
        f"",
        f"  True  Positives (fake  → fake) : {tp:,}",
        f"  True  Negatives (real  → real) : {tn:,}",
        f"  False Positives (real  → fake) : {fp:,}",
        f"  False Negatives (fake  → real) : {fn:,}",
        f"",
        f"  Real accuracy : {tn/(tn+fp):.4f}",
        f"  Fake accuracy : {tp/(tp+fn):.4f}",
        f"",
        classification_report(labels, preds, target_names=["REAL", "FAKE"]),
        f"{'='*48}",
        f"  {'PASS — model generalises well' if auc >= 0.85 else 'WARN — AUC below 0.85'}  (AUC={auc:.4f})",
        f"{'='*48}",
    ])

    print(report)

    # Save outputs
    out_dir = Path("video_deepfake") / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "remaining_report.txt").write_text(report, encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Confusion matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["REAL","FAKE"],
                yticklabels=["REAL","FAKE"], ax=axes[0])
    axes[0].set_title("Confusion Matrix — Unseen 100k")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

    # ROC curve
    fpr, tpr, _ = roc_curve(labels, probs)
    axes[1].plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC={auc:.4f}")
    axes[1].plot([0,1],[0,1],"k--",alpha=0.4)
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
    axes[1].set_title("ROC Curve — Unseen 100k"); axes[1].legend()

    # Score distribution
    axes[2].hist(probs[labels==0], bins=50, alpha=0.6, color="steelblue", label="Real")
    axes[2].hist(probs[labels==1], bins=50, alpha=0.6, color="tomato",    label="Fake")
    axes[2].axvline(0.5, color="gray", linestyle="--")
    axes[2].set_xlabel("Fake probability"); axes[2].set_ylabel("Count")
    axes[2].set_title("Score Distribution"); axes[2].legend()

    plt.tight_layout()
    plot_path = out_dir / "remaining_plots.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"\n[PRISM] Report → {out_dir / 'remaining_report.txt'}")
    print(f"[PRISM] Plots  → {plot_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="PRISM — Test on unseen 100k images")
    p.add_argument("--faces_dir",       default="dataset/faces")
    p.add_argument("--checkpoint",      default="video_deepfake/model/checkpoints/calibrated_model.pth")
    p.add_argument("--train_per_class", type=int, default=20000,
                   help="Must match --max_per_class used in step3_train.py")
    p.add_argument("--max_per_class",   type=int, default=None,
                   help="Cap unseen images per class (omit = use all ~50k each)")
    p.add_argument("--batch_size",      type=int, default=64)
    p.add_argument("--workers",         type=int, default=4)
    p.add_argument("--seed",            type=int, default=42)
    args = p.parse_args()
    main(args)