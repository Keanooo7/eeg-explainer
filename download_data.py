#!/usr/bin/env python3
"""
Pre-cache the PhysioNet EEG data for one subject (run once, optional).
=====================================================================
`train_and_export.py` already downloads on demand via MNE, so this script is a
convenience: it fetches runs 4/8/12 (imagined LEFT/RIGHT fist) for a subject up
front so the first training run has no network wait, and prints where the files
landed. Mirrors the one-shot `download.py` pattern from the A2 FCN repo.

    python download_data.py               # subject 1
    python download_data.py --subject 7   # any 1..109

Data is cached under ~/mne_data/MNE-eegbci-data/.../eegmmidb/1.0.0/ (see README).
Dataset: EEG Motor Movement/Imagery Database, PhysioNet v1.0.0, ODC-By 1.0.
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", type=int, default=1, help="PhysioNet subject id (1..109)")
    args = ap.parse_args()

    from mne.datasets import eegbci

    runs = [4, 8, 12]  # task 2: imagine opening/closing LEFT or RIGHT fist
    print(f"fetching PhysioNet EEGMMIDB subject {args.subject}, runs {runs} ...")
    paths = eegbci.load_data(args.subject, runs)
    print("cached files:")
    for p in paths:
        print(f"  {p}")
    print("done — train with:  python train_and_export.py --subject "
          f"{args.subject} --epochs 200 --out ./assets")


if __name__ == "__main__":
    main()
