# PRISM — Audio Deepfake Detection Pipeline

## Folder Structure
```
audio_deepfake/
├── step1_dataset_setup.py      ← verify dataset is in place
├── step2_feature_extraction.py ← extract MFCC + other features
├── step3_train_model.py        ← train Random Forest classifier
├── step4_evaluate.py           ← detailed evaluation + plots
├── step5_predict.py            ← predict on a single audio file
├── requirements.txt
├── dataset/                    ← YOU place ASVspoof 2019 LA here
│   └── LA/
├── features/                   ← auto-created by step2
├── model/                      ← auto-created by step3
└── plots/                      ← auto-created by step4
```

## Run Order

```bash
# 0. Install dependencies
pip install -r requirements.txt

# 1. Download dataset manually from https://datashare.ed.ac.uk/handle/10283/3336
#    Place in dataset/LA/

# 2. Verify setup
python step1_dataset_setup.py

# 3. Extract features (takes ~10-20 min depending on MAX_TRAIN_SAMPLES)
python step2_feature_extraction.py

# 4. Train model (takes ~2-5 min on CPU)
python step3_train_model.py

# 5. Evaluate with plots
python step4_evaluate.py

# 6. Test on a file
python step5_predict.py path/to/audio.wav
```

## How It Works

```
Raw Audio (.wav/.flac/.mp3)
         ↓
Feature Extraction (librosa)
  - MFCCs (40 coeffs × mean+std = 80 features)
  - Chroma (12 × 2 = 24)
  - Spectral Contrast (7 × 2 = 14)
  - Zero Crossing Rate (2)
  - RMS Energy (2)
  ─────────────────────────────
  Total: 122 features per clip
         ↓
Random Forest Classifier (200 trees)
         ↓
Output: REAL / FAKE + confidence %
```

## Expected Performance
- Accuracy: ~85–92% on ASVspoof 2019 LA dev set
- ROC-AUC:  ~0.90–0.95
- Inference time: ~0.5s per file on CPU

## Next Step
After training, the `predict_audio()` function in `step5_predict.py`
is imported directly by the FastAPI backend.