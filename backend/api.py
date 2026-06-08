# backend/api.py  — PRISM unified gateway
#
# Routes:
#   POST /predict/audio   →  audio deepfake detection
#   POST /predict/video   →  video deepfake detection
#   GET  /health          →  liveness check
#
# Run from the backend/ folder:
#   uvicorn api:app --host 0.0.0.0 --port 8000 --reload

import os
import sys
import shutil
import tempfile
import importlib.util
from pathlib import Path

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from facenet_pytorch import MTCNN

# ---------------------------------------------------------------------------
# Path setup — makes all sub-modules importable regardless of cwd
# ---------------------------------------------------------------------------

BACKEND_DIR     = Path(__file__).parent.resolve()
AUDIO_DIR       = BACKEND_DIR / "audio_deepfake"
VIDEO_DIR       = BACKEND_DIR / "video_deepfake"
VIDEO_MODEL_DIR = VIDEO_DIR / "model"

for p in [str(BACKEND_DIR), str(AUDIO_DIR), str(VIDEO_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_module(name: str, file_path: Path):
    """Load a Python file as a module by absolute path — avoids package resolution issues."""
    spec   = importlib.util.spec_from_file_location(name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Import predict functions
# ---------------------------------------------------------------------------

_audio_mod    = _load_module("step5_predict_audio", AUDIO_DIR / "step5_predict.py")
predict_audio = _audio_mod.predict_audio

_detector_mod = _load_module("deepfake_detector", VIDEO_MODEL_DIR / "deepfake_detector.py")
build_model   = _detector_mod.build_model

_predict_mod  = _load_module("step6_predict_video", VIDEO_DIR / "step6_predict.py")
predict_video = _predict_mod.predict_video

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="PRISM Deepfake Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Video model — loaded once at startup
# ---------------------------------------------------------------------------

VIDEO_CHECKPOINT = VIDEO_DIR / "model" / "checkpoints" / "calibrated_model.pth"

_video_model = None
_mtcnn       = None
_device      = None
_temperature = 1.0
_seq_len     = 4
_image_size  = 224


@app.on_event("startup")
def load_video_model():
    global _video_model, _mtcnn, _device, _temperature, _seq_len, _image_size

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not VIDEO_CHECKPOINT.exists():
        print(f"[PRISM] WARNING: video checkpoint not found at {VIDEO_CHECKPOINT}")
        print(f"[PRISM] Run step3_train.py and step4_calibrate.py first.")
        return

    ckpt        = torch.load(VIDEO_CHECKPOINT, map_location=_device)
    train_args  = ckpt.get("args", {})
    _temperature = ckpt.get("temperature", 1.0)
    _seq_len     = train_args.get("seq_len", 4)
    _image_size  = train_args.get("image_size", 224)

    _video_model = build_model(
        pretrained=False,
        d_model=train_args.get("d_model", 256),
        n_heads=train_args.get("n_heads", 8),
        n_layers=train_args.get("n_layers", 2),
        dropout=0.0,
        checkpoint_path=str(VIDEO_CHECKPOINT),
        device=_device,
    )
    _video_model.eval()

    _mtcnn = MTCNN(
        image_size=_image_size, margin=30, min_face_size=60,
        select_largest=True, keep_all=False,
        post_process=False, device=_device,
    )

    print(f"[PRISM] Video model loaded  device={_device}  T={_temperature:.4f}  seq_len={_seq_len}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "video_model_loaded": _video_model is not None,
        "device": str(_device),
    }


@app.post("/predict/audio")
async def predict_audio_route(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        result = predict_audio(tmp_path)
    finally:
        os.remove(tmp_path)
    return result


@app.post("/predict/video")
async def predict_video_route(file: UploadFile = File(...)):
    if _video_model is None:
        raise HTTPException(
            status_code=503,
            detail="Video model not loaded. Run step3_train.py and step4_calibrate.py first."
        )

    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    suffix  = os.path.splitext(file.filename)[1].lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{suffix}'. Allowed: {allowed}"
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        result = predict_video(
            video_path=tmp_path,
            model=_video_model,
            mtcnn=_mtcnn,
            device=_device,
            temperature=_temperature,
            fps_target=5,
            max_frames=50,
            seq_len=_seq_len,
            image_size=_image_size,
        )
    finally:
        os.unlink(tmp_path)

    # Strip per-frame probs — not needed by frontend
    result.pop("frame_fake_probs", None)
    return result