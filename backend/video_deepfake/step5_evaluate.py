"""
PRISM — step5_evaluate.py
AUC, F1, confusion matrix, ROC curve on the val split.

Run from backend/:
  python video_deepfake/step5_evaluate.py \
      --faces_dir dataset/faces \
      --checkpoint video_deepfake/model/checkpoints/calibrated_model.pth
"""

import argparse, sys, random
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix, roc_curve, classification_report
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

BACKEND_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(BACKEND_DIR / "video_deepfake"))
from model.deepfake_detector import build_model


class FaceValDataset(Dataset):
    EXTS = {".jpg", ".jpeg", ".png", ".webp"}
    def __init__(self, real_files, fake_files, transform, seq_len):
        self.transform  = transform
        self.seq_len    = seq_len
        self.real_files = real_files
        self.fake_files = fake_files
        self.samples    = [(p, 0) for p in real_files] + [(p, 1) for p in fake_files]
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        pool   = self.real_files if label == 0 else self.fake_files
        chosen = [path] + random.choices(pool, k=self.seq_len - 1)
        frames = torch.stack([self.transform(Image.open(p).convert("RGB")) for p in chosen])
        return frames, torch.tensor(label, dtype=torch.float32)


@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(args.checkpoint, map_location=device)
    ta     = ckpt.get("args", {})
    T      = ckpt.get("temperature", 1.0)
    seq_len  = ta.get("seq_len", 4)
    img_size = ta.get("image_size", 224)

    model = build_model(pretrained=False,
                        d_model=ta.get("d_model", 256),
                        n_heads=ta.get("n_heads", 8),
                        n_layers=ta.get("n_layers", 2),
                        dropout=0.0, device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    EXTS = {".jpg", ".jpeg", ".png", ".webp"}
    real = [p for p in (Path(args.faces_dir)/"real").iterdir() if p.suffix.lower() in EXTS]
    fake = [p for p in (Path(args.faces_dir)/"fake").iterdir() if p.suffix.lower() in EXTS]
    random.seed(42)
    real_val = random.sample(real, min(20000, len(real)))[:int(min(20000,len(real))*0.1)]
    fake_val = random.sample(fake, min(20000, len(fake)))[:int(min(20000,len(fake))*0.1)]

    ds     = FaceValDataset(real_val, fake_val, tf, seq_len)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    print(f"[Evaluate] {len(ds):,} val samples  T={T:.4f}")

    probs, labels = [], []
    for frames, lbls in loader:
        p = torch.sigmoid(model(frames.to(device)) / T).cpu().numpy()
        probs.extend(p.tolist()); labels.extend(lbls.numpy().tolist())

    probs  = np.array(probs);  labels = np.array(labels)
    preds  = (probs > 0.5).astype(int)
    auc    = roc_auc_score(labels, probs)
    f1     = f1_score(labels, preds)
    acc    = accuracy_score(labels, preds)
    cm     = confusion_matrix(labels, preds)

    print(f"\n{'='*45}")
    print(f"  AUC:      {auc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1:       {f1:.4f}")
    print(f"{'='*45}")
    print(classification_report(labels, preds, target_names=["REAL","FAKE"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["REAL","FAKE"], yticklabels=["REAL","FAKE"], ax=axes[0])
    axes[0].set_title("Confusion Matrix"); axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

    fpr, tpr, _ = roc_curve(labels, probs)
    axes[1].plot(fpr, tpr, color="steelblue", label=f"AUC={auc:.4f}")
    axes[1].plot([0,1],[0,1],"k--",alpha=0.4)
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR"); axes[1].set_title("ROC Curve"); axes[1].legend()

    plt.tight_layout()
    out = Path(args.checkpoint).parent / "evaluation.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"\n[Evaluate] Plot → {out}")
    if auc >= 0.85:
        print(f"[Evaluate] ✓ AUC {auc:.4f} ≥ 0.85  — ready for API")
        print(f"  Next: uvicorn api:app --host 0.0.0.0 --port 8000 --reload")
    else:
        print(f"[Evaluate] ✗ AUC {auc:.4f} < 0.85  — add more epochs or data")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--faces_dir",  default="dataset/faces")
    p.add_argument("--checkpoint", default="video_deepfake/model/checkpoints/calibrated_model.pth")
    args = p.parse_args()
    main(args)