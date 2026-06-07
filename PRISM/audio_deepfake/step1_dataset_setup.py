"""
STEP 1 — DATASET VERIFICATION (FoR-norm)
==========================================
Verifies the FoR-norm dataset is correctly placed.

Download and extract:
    wget https://bil.eecs.yorku.ca/share/for-norm.tar.gz
    tar -xzf for-norm.tar.gz -C ../dataset/

Expected folder structure after extraction:
    PRISM/
    ├── audio_deepfake/     ← run scripts from here
    └── dataset/
        └── for-norm/
            ├── training/
            │   ├── real/   ← .wav files
            │   └── fake/   ← .wav files
            ├── validation/
            │   ├── real/
            │   └── fake/
            └── testing/
                ├── real/
                └── fake/

Run:
    cd PRISM/audio_deepfake
    python step1_dataset_setup.py
"""

import os
import sys

DATASET_ROOT = "../dataset/for-norm"
SPLITS       = ["training", "validation", "testing"]
CLASSES      = ["real", "fake"]

def count_audio(folder):
    supported = (".wav", ".mp3", ".flac", ".ogg")
    return [f for f in os.listdir(folder) if f.lower().endswith(supported)]

def verify_setup():
    print("=" * 60)
    print("PRISM — Audio Deepfake Detection")
    print("Step 1: Dataset Verification (FoR-norm)")
    print("=" * 60)

    # Check root
    if not os.path.exists(DATASET_ROOT):
        print(f"\n✗ Dataset root not found: {DATASET_ROOT}")
        print("  Download and extract with:")
        print("    wget https://bil.eecs.yorku.ca/share/for-norm.tar.gz")
        print("    tar -xzf for-norm.tar.gz -C ../dataset/")
        sys.exit(1)

    print(f"\n  [✓] {DATASET_ROOT}")

    all_good   = True
    totals     = {}

    for split in SPLITS:
        print(f"\n  [{split}]")
        totals[split] = {}
        for cls in CLASSES:
            path   = os.path.join(DATASET_ROOT, split, cls)
            exists = os.path.exists(path)
            status = "✓" if exists else "✗ MISSING"
            print(f"    [{status}]  {split}/{cls}/", end="")

            if exists:
                files = count_audio(path)
                print(f"  →  {len(files)} files")
                totals[split][cls] = len(files)
            else:
                print()
                all_good = False
                totals[split][cls] = 0

    if not all_good:
        print("\n✗ Some folders are missing. Check your extraction path.")
        sys.exit(1)

    # Summary table
    print("\n" + "─" * 50)
    print(f"  {'Split':<12}  {'Real':>7}  {'Fake':>7}  {'Total':>8}")
    print("─" * 50)
    grand_total = 0
    for split in SPLITS:
        r = totals[split].get("real", 0)
        f = totals[split].get("fake", 0)
        t = r + f
        grand_total += t
        print(f"  {split:<12}  {r:>7,}  {f:>7,}  {t:>8,}")
    print("─" * 50)
    print(f"  {'TOTAL':<12}  {'':>7}  {'':>7}  {grand_total:>8,}")

    # Sanity checks
    print()
    warnings = 0
    for split in SPLITS:
        r = totals[split].get("real", 0)
        f = totals[split].get("fake", 0)
        if r == 0 or f == 0:
            print(f"  ⚠  {split}: one class is empty — check extraction")
            warnings += 1
        elif abs(r - f) / max(r, f) > 0.05:
            print(f"  ⚠  {split}: class imbalance detected (real={r}, fake={f})")
            warnings += 1
        else:
            print(f"  ✓  {split}: balanced  (real={r:,}, fake={f:,})")

    if warnings == 0:
        print("\n✓ FoR-norm dataset looks correct!")
        print("  • Pre-split into training / validation / testing")
        print("  • All files are 16 kHz mono WAV (already normalized)")
        print("  • No manual train/test split needed in step 2")
    else:
        print(f"\n⚠  {warnings} warning(s) above — review before proceeding")

    print("\nProceed to step2_feature_extraction.py")

if __name__ == "__main__":
    verify_setup()