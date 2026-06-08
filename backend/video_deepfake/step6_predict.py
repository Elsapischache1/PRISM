"""
PRISM — step6_predict.py
Single-video inference — called by backend/api.py and usable as CLI.

CLI:
  python video_deepfake/step6_predict.py --video path/to/video.mp4
"""

import argparse, json, sys, time
from pathlib import Path

import cv2, torch
import numpy as np
from torchvision import transforms
from PIL import Image

BACKEND_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(BACKEND_DIR / "video_deepfake"))
from model.deepfake_detector import build_model


def extract_frames(video_path: Path, fps_target=5, max_frames=50):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(src_fps / fps_target))
    indices = list(range(0, total, step))
    if len(indices) > max_frames:
        indices = indices[:: len(indices) // max_frames][:max_frames]
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, f = cap.read()
        if ret: frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def crop_faces(frames, mtcnn, size=224):
    out = []
    for rgb in frames:
        try:
            face = mtcnn(Image.fromarray(rgb))
            if face is not None:
                arr = face.permute(1,2,0).numpy().astype(np.uint8)
                out.append(Image.fromarray(arr).resize((size, size)))
            else:
                out.append(None)
        except Exception:
            out.append(None)
    return out


def get_transform(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])


@torch.no_grad()
def predict_video(video_path, model, mtcnn, device, temperature=1.0,
                  fps_target=5, max_frames=50, seq_len=4, image_size=224):
    t0 = time.time()
    tf = get_transform(image_size)

    raw = extract_frames(video_path, fps_target, max_frames)
    if not raw:
        raise ValueError("No frames extracted.")

    faces = crop_faces(raw, mtcnn, image_size)
    detected = [f for f in faces if f is not None]
    n_detected = len(detected)
    imgs = detected if detected else [Image.fromarray(f).resize((image_size, image_size)) for f in raw]

    tensors = [tf(img) for img in imgs]
    window_probs = []
    for i in range(0, len(tensors), seq_len):
        w = tensors[i:i+seq_len]
        if len(w) < seq_len:
            w = w + [w[-1]] * (seq_len - len(w))
        batch  = torch.stack(w).unsqueeze(0).to(device)  # (1,T,C,H,W)
        logit  = model(batch).squeeze()
        prob   = torch.sigmoid(logit / temperature).item()
        window_probs.append(prob)

    fake_prob = float(np.mean(window_probs))
    real_prob = 1.0 - fake_prob
    verdict   = "FAKE" if fake_prob >= 0.5 else "REAL"
    confidence = fake_prob if verdict == "FAKE" else real_prob

    return {
        "verdict":          verdict,
        "confidence":       round(confidence, 4),
        "fake_prob":        round(fake_prob, 4),
        "real_prob":        round(real_prob, 4),
        "frames_analysed":  len(raw),
        "faces_detected":   n_detected,
        "processing_time":  round(time.time() - t0, 2),
        "frame_fake_probs": [round(p, 4) for p in window_probs],
    }


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(args.checkpoint, map_location=device)
    ta     = ckpt.get("args", {})
    T      = ckpt.get("temperature", 1.0)

    model = build_model(pretrained=False,
                        d_model=ta.get("d_model", 256),
                        n_heads=ta.get("n_heads", 8),
                        n_layers=ta.get("n_layers", 2),
                        dropout=0.0,
                        checkpoint_path=args.checkpoint, device=device)
    model.eval()

    from facenet_pytorch import MTCNN
    mtcnn = MTCNN(image_size=224, margin=30, min_face_size=80,
                  select_largest=True, keep_all=False, post_process=False, device=device)

    result = predict_video(
        Path(args.video), model, mtcnn, device,
        temperature=T, fps_target=args.fps_target,
        max_frames=args.max_frames,
        seq_len=ta.get("seq_len", 4),
        image_size=ta.get("image_size", 224),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video",      required=True)
    p.add_argument("--checkpoint", default="video_deepfake/model/checkpoints/calibrated_model.pth")
    p.add_argument("--fps_target", type=int, default=5)
    p.add_argument("--max_frames", type=int, default=50)
    args = p.parse_args()
    main(args)