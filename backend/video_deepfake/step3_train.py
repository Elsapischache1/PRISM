"""
PRISM — step3_train.py  (Speed-optimised: EfficientNet-B0, 40k images, AMP)

RTX 3050 6GB target: ~2 to 2.5 hrs total for 10 epochs on 40k images.

Dataset layout (140k Real and Fake Faces — Kaggle xhlulu):
  dataset/faces/real/   ← 70k images total, we sample 20k
  dataset/faces/fake/   ← 70k images total, we sample 20k

Run from backend/:
  python video_deepfake/step3_train.py \
      --faces_dir dataset/faces \
      --max_per_class 20000 \
      --epochs 10 \
      --batch_size 32

Key speed wins vs original:
  1. EfficientNet-B0 backbone   (3× faster than B4)
  2. AMP (torch.cuda.amp)       (~1.5× faster on RTX 30xx Tensor Cores)
  3. 40k images instead of 140k (~3.5× fewer steps/epoch)
  4. seq_len=4 instead of 8    (halves VRAM per forward pass)
  5. Phase 1 only 3 frozen epochs (saves ~30 min vs 5)
"""

import argparse
import sys
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

BACKEND_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(BACKEND_DIR / "video_deepfake"))
from model.deepfake_detector import build_model


# ---------------------------------------------------------------------------
# Dataset — samples max_per_class images, builds pseudo-video sequences
# ---------------------------------------------------------------------------

class FaceDataset(Dataset):
    EXTS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, real_files, fake_files, transform, seq_len=4):
        self.transform   = transform
        self.seq_len     = seq_len
        self.real_files  = real_files
        self.fake_files  = fake_files
        self.samples     = [(p, 0) for p in real_files] + [(p, 1) for p in fake_files]
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        pool = self.real_files if label == 0 else self.fake_files

        # Pseudo-sequence: anchor frame + (seq_len-1) random same-class frames
        chosen = [path] + random.choices(pool, k=self.seq_len - 1)
        frames = torch.stack([
            self.transform(Image.open(p).convert("RGB")) for p in chosen
        ])  # (T, 3, H, W)
        return frames, torch.tensor(label, dtype=torch.float32)


def load_files(root: Path, max_per_class: int):
    EXTS = {".jpg", ".jpeg", ".png", ".webp"}
    real = [p for p in (root / "real").iterdir() if p.suffix.lower() in EXTS]
    fake = [p for p in (root / "fake").iterdir() if p.suffix.lower() in EXTS]
    if not real: raise FileNotFoundError(f"No images in {root/'real'}")
    if not fake: raise FileNotFoundError(f"No images in {root/'fake'}")
    random.seed(42)
    real = random.sample(real, min(max_per_class, len(real)))
    fake = random.sample(fake, min(max_per_class, len(fake)))
    print(f"[Dataset] Using real={len(real):,}  fake={len(fake):,}  "
          f"total={len(real)+len(fake):,}")
    return real, fake


def make_transforms(size=224):
    train = transforms.Compose([
        transforms.Resize((size + 16, size + 16)),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.1),
    ])
    val = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return train, val


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, scaler, device, train):
    model.train(train)
    total_loss = correct = total = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for frames, labels in tqdm(loader, leave=False, desc="train" if train else "val "):
            frames, labels = frames.to(device), labels.to(device)

            with autocast(enabled=(scaler is not None)):
                logits = model(frames)
                loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds   = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"[PRISM] Device={device}  AMP={'ON' if use_amp else 'OFF'}")

    train_tf, val_tf = make_transforms(args.image_size)

    real_files, fake_files = load_files(Path(args.faces_dir), args.max_per_class)

    # 90/10 split per class
    def split(files, val_frac=0.1):
        n = int(len(files) * val_frac)
        return files[n:], files[:n]

    real_train, real_val = split(real_files)
    fake_train, fake_val = split(fake_files)

    train_ds = FaceDataset(real_train, fake_train, train_tf, seq_len=args.seq_len)
    val_ds   = FaceDataset(real_val,   fake_val,   val_tf,   seq_len=args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.workers,
                              pin_memory=True, persistent_workers=args.workers>0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers,
                              pin_memory=True, persistent_workers=args.workers>0)

    print(f"[PRISM] Train={len(train_ds):,}  Val={len(val_ds):,}  "
          f"Steps/epoch={len(train_loader)}")

    model = build_model(pretrained=True, d_model=args.d_model,
                        n_heads=args.n_heads, n_layers=args.n_layers,
                        dropout=args.dropout, device=device)

    criterion = nn.BCEWithLogitsLoss()
    scaler    = GradScaler() if use_amp else None

    ckpt_dir = Path(__file__).parent / "model" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    phase1_epochs = min(args.freeze_epochs, args.epochs)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        if epoch == 1:
            print(f"\n[PRISM] Phase 1: backbone FROZEN ({phase1_epochs} epochs)")
            model.freeze_backbone()
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=args.lr,
                epochs=phase1_epochs, steps_per_epoch=len(train_loader))

        if epoch == phase1_epochs + 1:
            print(f"\n[PRISM] Phase 2: backbone UNFROZEN — fine-tuning all")
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr * 0.05,
                weight_decay=args.weight_decay)
            remaining = args.epochs - phase1_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining, eta_min=1e-7)

        train_loss, train_acc = run_epoch(model, train_loader, criterion,
                                          optimizer, scaler, device, train=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion,
                                          None,      None,   device, train=False)

        if epoch <= phase1_epochs:
            scheduler.step()           # OneCycleLR steps per epoch here
        else:
            scheduler.step()

        elapsed = time.time() - t0
        eta_epochs = args.epochs - epoch
        print(f"Ep {epoch:02d}/{args.epochs}  "
              f"tr_loss={train_loss:.4f} tr_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.1e}  "
              f"{elapsed:.0f}s  ETA≈{eta_epochs*elapsed/60:.0f}min")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            path = ckpt_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "args": {
                    "d_model": args.d_model, "n_heads": args.n_heads,
                    "n_layers": args.n_layers, "seq_len": args.seq_len,
                    "image_size": args.image_size,
                },
            }, path)
            print(f"  ✓ Best model saved  val_acc={val_acc:.4f}")

    print(f"\n[PRISM] Done. Best val_acc={best_val_acc:.4f}")
    print(f"  Next: python video_deepfake/step4_calibrate.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--faces_dir",      default="dataset/faces")
    p.add_argument("--max_per_class",  type=int,   default=20000,
                   help="Images per class to use (20000 real + 20000 fake = 40k total)")
    p.add_argument("--epochs",         type=int,   default=10)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--seq_len",        type=int,   default=4,
                   help="Frames per pseudo-video (keep at 4 for RTX 3050)")
    p.add_argument("--image_size",     type=int,   default=224)
    p.add_argument("--d_model",        type=int,   default=256)
    p.add_argument("--n_heads",        type=int,   default=8)
    p.add_argument("--n_layers",       type=int,   default=2)
    p.add_argument("--dropout",        type=float, default=0.1)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--weight_decay",   type=float, default=1e-4)
    p.add_argument("--freeze_epochs",  type=int,   default=3,
                   help="Epochs to freeze backbone (3 is optimal for 10-epoch run)")
    p.add_argument("--workers",        type=int,   default=4)
    args = p.parse_args()
    main(args)