# Run this from backend/ — creates demo_fake.mp4 and demo_real.mp4
# from your actual training data images

import cv2
import random
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png"}

def images_to_video(folder, out_path, n=60, fps=10):
    files = [p for p in Path(folder).iterdir() if p.suffix.lower() in EXTS]
    random.seed(99)
    chosen = random.sample(files, min(n, len(files)))
    img = cv2.imread(str(chosen[0]))
    h, w = img.shape[:2]
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for p in chosen:
        frame = cv2.imread(str(p))
        if frame is not None:
            out.write(frame)
    out.release()
    print(f"Saved {out_path}")

images_to_video("dataset/faces/real", "demo_real.mp4")
images_to_video("dataset/faces/fake", "demo_fake.mp4")