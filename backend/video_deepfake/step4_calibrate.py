"""
PRISM — step4_calibrate.py
Temperature scaling on the validation split.

Run from backend/:
  python video_deepfake/step4_calibrate.py \
      --faces_dir dataset/faces \
      --checkpoint video_deepfake/model/checkpoints/best_model.pth
"""

import argparse, sys, random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from scipy.optimize import minimize_scalar

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
def collect_logits(model, loader, device):
    model.eval()
    logits, labels = [], []
    for f, l in loader:
        logits.append(model(f.to(device)).cpu())
        labels.append(l)
    return torch.cat(logits), torch.cat(labels)


def nll(T, logits, labels):
    return nn.functional.binary_cross_entropy_with_logits(
        logits / max(T, 1e-6), labels).item()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    ckpt      = torch.load(ckpt_path, map_location=device)
    ta        = ckpt.get("args", {})
    seq_len   = ta.get("seq_len", 4)
    img_size  = ta.get("image_size", 224)

    model = build_model(pretrained=False,
                        d_model=ta.get("d_model", 256),
                        n_heads=ta.get("n_heads", 8),
                        n_layers=ta.get("n_layers", 2),
                        dropout=0.0, device=device)
    model.load_state_dict(ckpt["model_state_dict"])

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    EXTS = {".jpg", ".jpeg", ".png", ".webp"}
    real = [p for p in (Path(args.faces_dir)/"real").iterdir() if p.suffix.lower() in EXTS]
    fake = [p for p in (Path(args.faces_dir)/"fake").iterdir() if p.suffix.lower() in EXTS]
    random.seed(42)
    # use same val split as training (first 10%)
    real_val = random.sample(real, min(20000, len(real)))[:int(min(20000,len(real))*0.1)]
    fake_val = random.sample(fake, min(20000, len(fake)))[:int(min(20000,len(fake))*0.1)]

    ds     = FaceValDataset(real_val, fake_val, tf, seq_len)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    print(f"[Calibrate] Val samples: {len(ds):,}")
    logits, labels = collect_logits(model, loader, device)

    result      = minimize_scalar(lambda T: nll(T, logits, labels), bounds=(0.01, 10.0), method="bounded")
    temperature = float(result.x)
    print(f"[Calibrate] T={temperature:.4f}  NLL: {nll(1.0,logits,labels):.4f} → {nll(temperature,logits,labels):.4f}")

    save = ckpt_path.parent / "calibrated_model.pth"
    torch.save({**ckpt, "temperature": temperature}, save)
    print(f"[Calibrate] Saved → {save}")
    print(f"  Next: python video_deepfake/step5_evaluate.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--faces_dir",  default="dataset/faces")
    p.add_argument("--checkpoint", default="video_deepfake/model/checkpoints/best_model.pth")
    args = p.parse_args()
    main(args)