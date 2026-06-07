"""
STEP 1 — DATASET SETUP
======================
ASVspoof 2019 Logical Access dataset.

MANUAL DOWNLOAD STEPS (do this before running other scripts):
--------------------------------------------------------------
1. Go to: https://datashare.ed.ac.uk/handle/10283/3336
2. Register/login (free)
3. Download these two files:
     - LA.zip  (the Logical Access partition — ~4GB)
4. Unzip into a folder called 'dataset/' in the same directory as these scripts.

Expected folder structure after unzipping:
    dataset/
    └── LA/
        ├── ASVspoof2019_LA_train/
        │   └── flac/          ← training audio files (.flac)
        ├── ASVspoof2019_LA_dev/
        │   └── flac/
        ├── ASVspoof2019_LA_eval/
        │   └── flac/
        └── ASVspoof2019_LA_cm_protocols/
            ├── ASVspoof2019.LA.cm.train.trn.txt   ← labels for train
            ├── ASVspoof2019.LA.cm.dev.trl.txt     ← labels for dev
            └── ASVspoof2019.LA.cm.eval.trl.txt    ← labels for eval

LABEL FILE FORMAT (space-separated):
    speaker_id  file_id  env  attack_type  label
    e.g.:
    LA_0079  LA_T_1138215  -  A01  spoof
    LA_0079  LA_T_1271820  -  -    genuine

'genuine' = REAL human voice
'spoof'   = FAKE / deepfake voice

Run this script to verify your setup is correct:
"""

import os
import sys

DATASET_DIR = "dataset/LA"

REQUIRED_PATHS = [
    "dataset/LA/ASVspoof2019_LA_train/flac",
    "dataset/LA/ASVspoof2019_LA_dev/flac",
    "dataset/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
    "dataset/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
]

def verify_setup():
    print("=" * 60)
    print("PRISM — Audio Deepfake Detection")
    print("Step 1: Dataset Verification")
    print("=" * 60)

    all_good = True
    for path in REQUIRED_PATHS:
        exists = os.path.exists(path)
        status = "✓" if exists else "✗ MISSING"
        print(f"  [{status}] {path}")
        if not exists:
            all_good = False

    print()
    if all_good:
        # Count files
        train_flac = len([f for f in os.listdir("dataset/LA/ASVspoof2019_LA_train/flac") if f.endswith(".flac")])
        dev_flac   = len([f for f in os.listdir("dataset/LA/ASVspoof2019_LA_dev/flac") if f.endswith(".flac")])
        print(f"  Train audio files : {train_flac}")
        print(f"  Dev audio files   : {dev_flac}")
        print()
        print("✓ Dataset is ready. Proceed to step2_feature_extraction.py")
    else:
        print("✗ Some paths are missing. Please follow the download instructions above.")
        sys.exit(1)

if __name__ == "__main__":
    verify_setup()