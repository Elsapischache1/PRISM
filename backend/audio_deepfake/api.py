# audio_deepfake/api.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil, tempfile, os
from step5_predict import predict_audio   # your existing predict function

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/audio")
async def predict(file: UploadFile = File(...)):
    # Save upload to a temp file
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = predict_audio(tmp_path)   # returns dict with label + confidence
    finally:
        os.remove(tmp_path)

    return result