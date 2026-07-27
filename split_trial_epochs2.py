"""
split_trial_epochs.py

Splits 19-channel referential LORETA-ready .txt files (as produced by
bipolar_to_referential.py + batch_convert_folder.py) into trial-aligned epoch
files no longer than 1024 frames (LORETA-KEY's cross-spectral maker limit).

Each source file is 10,240 rows = 4 real trials (repetitions) of 2,560 rows
(10 s @ 256 Hz) each, concatenated back-to-back. Instead of chopping the whole
10,240-row file into 10 arbitrary sequential 1024-row blocks (which can straddle
a boundary between two separate repetitions), this script splits per-trial
first, then subdivides within each trial so no epoch ever crosses a repetition
boundary.

Two modes:
  --mode 1024  (default): 2 x 1024-frame epochs per trial (uses rows 0-2047 of
               each 2560-row trial, discards the last 512 rows/trial).
               8 epochs per source file. Best low-frequency (delta) resolution.
  --mode 640 : 4 x 640-frame epochs per trial (640 divides 2560 evenly, no
               discard). 16 epochs per source file. More epochs to average
               over, but coarser resolution below ~2-3 Hz.

Usage:
    python split_trial_epochs.py <referential_root> <epoch_output_root> [--mode 1024|640]

<referential_root> should mirror the Subject/Direction/<Type_Direction>.txt tree
produced by batch_convert_folder.py. Output mirrors the same tree, with one
subfolder per source file containing that file's epoch .txt files, plus a
top-level epoch_manifest.csv listing every epoch and where it came from.

NOTE: assumes each referential .txt is a plain whitespace-delimited matrix with
no header, loadable by numpy.loadtxt, shape (10240, 19). If your
bipolar_to_referential.py writes a different delimiter/format, adjust
load_referential_txt() accordingly.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

SAMPLES_PER_FILE = 10240
TRIALS_PER_FILE = 4
SAMPLES_PER_TRIAL = SAMPLES_PER_FILE // TRIALS_PER_FILE  # 2560
N_CHANNELS = 19


def load_referential_txt(path):
    data = np.loadtxt(path)

    if data.ndim != 2 or data.shape[1] != N_CHANNELS:
        raise ValueError(f"expected {N_CHANNELS} columns, got shape {data.shape}")

    n_rows = data.shape[0]
    if n_rows < SAMPLES_PER_FILE:
        raise ValueError(
            f"expected at least {SAMPLES_PER_FILE} rows, got {n_rows}"
        )
    if n_rows > SAMPLES_PER_FILE:
        # e.g. Subject 3's files have extra trials (30720 rows = 12 trials
        # instead of 4). Keep only the first 4 trials (10240 rows) so every
        # subject/condition contributes the same number of trials/epochs.
        print(
            f"  NOTE: {path} has {n_rows} rows ({n_rows // SAMPLES_PER_TRIAL} trials); "
            f"truncating to first {SAMPLES_PER_FILE} rows (first {TRIALS_PER_FILE} trials)."
        )
        data = data[:SAMPLES_PER_FILE]

    return data


def iter_epochs(data, mode):
    """Yield (trial_idx, epoch_idx, epoch_array) for one file's data."""
    epoch_len = 1024 if mode == "1024" else 640
    epochs_per_trial = SAMPLES_PER_TRIAL // epoch_len  # 2 for 1024, 4 for 640

    for trial_idx in range(TRIALS_PER_FILE):
        t0 = trial_idx * SAMPLES_PER_TRIAL
        trial_data = data[t0 : t0 + SAMPLES_PER_TRIAL]
        for epoch_idx in range(epochs_per_trial):
            e0 = epoch_idx * epoch_len
            yield trial_idx, epoch_idx, trial_data[e0 : e0 + epoch_len]


def process_tree(ref_root, out_root, mode):
    ref_root = Path(ref_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(ref_root.rglob("*.txt"))
    if not txt_files:
        print(f"No .txt files found under {ref_root}")
        return

    manifest_rows = []

    for txt_path in txt_files:
        rel = txt_path.relative_to(ref_root)
        subject = rel.parts[0] if len(rel.parts) > 0 else "unknown"
        direction = rel.parts[1] if len(rel.parts) > 1 else "unknown"
        condition_stem = txt_path.stem  # e.g. ARROW_Backward

        try:
            data = load_referential_txt(txt_path)
        except Exception as exc:
            print(f"SKIPPING {txt_path}: {exc}")
            continue

        out_dir = out_root / rel.parent / condition_stem
        out_dir.mkdir(parents=True, exist_ok=True)

        n_written = 0
        for trial_idx, epoch_idx, epoch in iter_epochs(data, mode):
            fname = f"{subject}_{condition_stem}_trial{trial_idx + 1}_ep{epoch_idx + 1}.txt"
            out_path = out_dir / fname
            np.savetxt(out_path, epoch, fmt="%.6f")
            n_written += 1
            manifest_rows.append(
                dict(
                    subject=subject,
                    direction=direction,
                    condition=condition_stem,
                    trial=trial_idx + 1,
                    epoch=epoch_idx + 1,
                    epoch_len=epoch.shape[0],
                    source_file=str(txt_path),
                    epoch_file=str(out_path),
                )
            )
        print(f"{txt_path} -> {n_written} epoch files in {out_dir}")

    if not manifest_rows:
        print("No epochs written — check input files.")
        return

    manifest_path = out_root / "epoch_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nWrote manifest with {len(manifest_rows)} epochs to {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("referential_root", help="Root folder of 19-channel referential .txt files")
    parser.add_argument("epoch_output_root", help="Where to write epoch .txt files + manifest")
    parser.add_argument(
        "--mode",
        choices=["1024", "640"],
        default="1024",
        help="1024 = 2 epochs/trial (discards last 512 samples/trial, best freq. resolution). "
        "640 = 4 epochs/trial (uses every sample, coarser low-frequency resolution).",
    )
    args = parser.parse_args()
    process_tree(args.referential_root, args.epoch_output_root, args.mode)


if __name__ == "__main__":
    main()
