"""
STEP 1 — DATASET VERIFICATION
===============================
Verifies the DEEP-VOICE dataset is correctly placed.

Expected folder structure:
    PRISM/
    ├── audio_deepfake/     ← run scripts from here
    └── dataset/
        └── DEEP-VOICE/
            ├── REAL/       ← real human speech (.mp3)
            └── FAKE/       ← AI generated / voice cloned (.mp3)

Run:
    cd PRISM/audio_deepfake
    python step1_dataset_setup.py
"""

import os
import sys

DATASET_DIR = "../dataset/DEEP-VOICE"
REAL_DIR    = os.path.join(DATASET_DIR, "REAL")
FAKE_DIR    = os.path.join(DATASET_DIR, "FAKE")

def verify_setup():
    print("=" * 60)
    print("PRISM — Audio Deepfake Detection")
    print("Step 1: Dataset Verification")
    print("=" * 60)

    all_good = True

    # Check folders exist
    for path, name in [(DATASET_DIR, "DEEP-VOICE/"), (REAL_DIR, "DEEP-VOICE/REAL/"), (FAKE_DIR, "DEEP-VOICE/FAKE/")]:
        exists = os.path.exists(path)
        status = "✓" if exists else "✗ MISSING"
        print(f"  [{status}] dataset/{name}")
        if not exists:
            all_good = False

    if not all_good:
        print("\n✗ Some folders are missing.")
        print("  Make sure your structure looks like:")
        print("  PRISM/dataset/DEEP-VOICE/REAL/")
        print("  PRISM/dataset/DEEP-VOICE/FAKE/")
        sys.exit(1)

    # Count files
    real_files = [f for f in os.listdir(REAL_DIR) if f.endswith(".mp3") or f.endswith(".wav")]
    fake_files = [f for f in os.listdir(FAKE_DIR) if f.endswith(".mp3") or f.endswith(".wav")]

    print(f"\n  REAL audio files : {len(real_files)}")
    print(f"  FAKE audio files : {len(fake_files)}")
    print(f"  Total            : {len(real_files) + len(fake_files)}")

    if len(real_files) == 0 or len(fake_files) == 0:
        print("\n✗ One or both folders are empty. Check your unzip.")
        sys.exit(1)

    # Show a few sample filenames
    print(f"\n  Sample REAL files: {real_files[:3]}")
    print(f"  Sample FAKE files: {fake_files[:3]}")

    print("\n✓ Dataset looks good! Proceed to step2_feature_extraction.py")

if __name__ == "__main__":
    verify_setup()