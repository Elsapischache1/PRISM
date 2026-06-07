# PRISM — Audio Deepfake Detection Pipeline

## Folder Structure
```
PRISM/
├── audio_deepfake/                  ← run ALL scripts from here
│   ├── step1_dataset_setup.py
│   ├── step2_feature_extraction.py
│   ├── step3_train_model.py
│   ├── step4_evaluate.py
│   ├── step5_predict.py
│   ├── requirements.txt
│   ├── features/                    ← auto-created by step2
│   ├── model/                       ← auto-created by step3
│   └── plots/                       ← auto-created by step4
│
└── dataset/
    └── DEEP-VOICE/                  ← download from Kaggle
        ├── REAL/                    ← real human speech (.mp3)
        └── FAKE/                    ← AI generated speech (.mp3)
```

## Dataset
Download from Kaggle:
https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition

Unzip and place the DEEP-VOICE folder at: PRISM/dataset/DEEP-VOICE/

## Run Order

```bash
# 0. Install dependencies (run once)
pip install -r requirements.txt

# 1. Verify dataset is in place
python step1_dataset_setup.py

# 2. Extract features from all audio files (~5-10 min)
python step2_feature_extraction.py

# 3. Train the Random Forest model (~2-5 min on CPU)
python step3_train_model.py

# 4. Evaluate with plots
python step4_evaluate.py

# 5. Test on any audio file
python step5_predict.py path/to/audio.mp3
```

## How It Works

```
REAL/ or FAKE/ folder
        ↓
Raw Audio (.mp3 / .wav)
        ↓
Feature Extraction (librosa)
  ├── MFCCs          (80 features) ← most important
  ├── Chroma         (24 features)
  ├── Spectral Contrast (14 features)
  ├── Zero Crossing Rate (2 features)
  └── RMS Energy     (2 features)
  ─────────────────────────────
  Total: 122 features per clip
        ↓
Random Forest Classifier (200 trees)
        ↓
REAL / FAKE + confidence %
```

## Labels
- `0` = REAL (genuine human voice)
- `1` = FAKE (AI generated / voice cloned)

## Expected Performance
- Accuracy : ~88–95% on test split
- ROC-AUC  : ~0.92–0.97
- Inference : ~0.3–0.5s per file on CPU