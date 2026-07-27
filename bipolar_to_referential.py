"""
bipolar_to_referential.py

Converts one bipolar double-banana .xlsx trial (20 columns: 18 chain channels +
BP1-REF + BP2-REF) into a 19-electrode referential signal using a common
average reference, and writes it as a plain-text matrix LORETA can read.

--- Method ---
The 18 bipolar channels form 3 electrically disconnected chains ("islands"):

  Right lateral:      Fp2 -[FP2-F8]-> F8 -[F8-T4]-> T4 -[T4-T6]-> T6 -[T6-O2]-> O2
  Right parasagittal: Fp2 -[FP2-F4]-> F4 -[F4-C4]-> C4 -[C4-P4]-> P4 -[P4-O2]-> O2
  Left lateral:       Fp1 -[FP1-F7]-> F7 -[F7-T3]-> T3 -[T3-T5]-> T5 -[T5-O1]-> O1
  Left parasagittal:  Fp1 -[FP1-F3]-> F3 -[F3-C3]-> C3 -[C3-P3]-> P3 -[P3-O1]-> O1
  Midline:            Fz  -[FZ-CZ]->  Cz -[CZ-PZ]->  Pz

Since a bipolar channel "A-B" = V(A) - V(B), setting each island's anchor
electrode (Fp2, Fp1, or Fz) to 0 lets every other electrode in that island be
reconstructed by cumulative subtraction: V(next) = V(prev) - bipolar_value.

Fp2 and Fp1 each anchor two chains that both terminate at O2 / O1
respectively, giving two independent estimates of those two electrodes --
averaged here (verified against sample data: they agree exactly).

BP1-REF / BP2-REF are excluded: checked against real data and found to be
near-constant offsets, not a usable link between islands.

After reconstruction, a common average reference (CAR) is applied across all
19 electrodes (subtracting, at every time point, the mean across all 19
channels). This does NOT depend on knowing each island's true absolute offset
being correct in an absolute sense -- for frequency-domain analysis (LORETA's
band-power cross-spectrum), a constant per-island DC offset only affects 0 Hz
and has no effect on the delta/theta/alpha/beta bands actually used.

--- Output ---
Plain ASCII .txt, rows = time points, columns = 19 electrodes, fixed order:
    Fp1 Fp2 F3 F4 F7 F8 C3 C4 T3 T4 T5 T6 P3 P4 O1 O2 Fz Pz Cz
No header row, whitespace-delimited (numpy.loadtxt-compatible).
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Fixed output electrode order (confirmed from the LORETA coordinate-file build)
ELECTRODE_ORDER = [
    "Fp1", "Fp2", "F3", "F4", "F7", "F8", "C3", "C4", "T3", "T4",
    "T5", "T6", "P3", "P4", "O1", "O2", "Fz", "Pz", "Cz",
]

# Each chain: (anchor_electrode, [(bipolar_column_name, next_electrode), ...])
CHAINS = [
    ("Fp2", [("FP2-F8", "F8"), ("F8-T4", "T4"), ("T4-T6", "T6"), ("T6-O2", "O2")]),
    ("Fp2", [("FP2-F4", "F4"), ("F4-C4", "C4"), ("C4-P4", "P4"), ("P4-O2", "O2")]),
    ("Fp1", [("FP1-F7", "F7"), ("F7-T3", "T3"), ("T3-T5", "T5"), ("T5-O1", "O1")]),
    ("Fp1", [("FP1-F3", "F3"), ("F3-C3", "C3"), ("C3-P3", "P3"), ("P3-O1", "O1")]),
    ("Fz", [("FZ-CZ", "Cz"), ("CZ-PZ", "Pz")]),
]


def _normalize(name):
    """'FP2- F8' / 'F8 - T4' / 'FP2-F8' all -> 'FP2-F8'."""
    return "".join(str(name).split()).upper()


def load_bipolar_xlsx(path):
    """Load a bipolar .xlsx and return a dict of normalized_column_name -> np.array."""
    df = pd.read_excel(path)
    return {_normalize(col): df[col].values.astype(np.float64) for col in df.columns}


def reconstruct_referential(bipolar_data):
    """
    bipolar_data: dict of normalized bipolar column name -> 1D array (time series).
    Returns: (n_timepoints, 19) array in ELECTRODE_ORDER, after common average reference.
    """
    n_samples = len(next(iter(bipolar_data.values())))
    estimates = {}  # electrode -> list of reconstructed series

    for anchor, links in CHAINS:
        estimates.setdefault(anchor, []).append(np.zeros(n_samples))
        v_prev = np.zeros(n_samples)
        for col_name, next_electrode in links:
            key = _normalize(col_name)
            if key not in bipolar_data:
                raise KeyError(f"Expected bipolar column '{col_name}' not found in file")
            v_next = v_prev - bipolar_data[key]
            estimates.setdefault(next_electrode, []).append(v_next)
            v_prev = v_next

    missing = [e for e in ELECTRODE_ORDER if e not in estimates]
    if missing:
        raise ValueError(f"Could not reconstruct electrodes: {missing}")

    # Average duplicate estimates (e.g. O2 from both right chains)
    referential = {
        electrode: np.mean(np.stack(series_list, axis=0), axis=0)
        for electrode, series_list in estimates.items()
    }

    matrix = np.stack([referential[e] for e in ELECTRODE_ORDER], axis=1)  # (n_samples, 19)

    # Common average reference: subtract the across-channel mean at every time point
    car = matrix - matrix.mean(axis=1, keepdims=True)
    return car


def convert_file(xlsx_path, out_txt_path=None):
    """Convert one bipolar .xlsx trial to a referential .txt file. Returns the (n,19) array."""
    bipolar_data = load_bipolar_xlsx(xlsx_path)
    referential = reconstruct_referential(bipolar_data)
    if out_txt_path is not None:
        out_txt_path = Path(out_txt_path)
        out_txt_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(out_txt_path, referential, fmt="%.6f")
    return referential


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert one bipolar .xlsx trial to a referential .txt file")
    parser.add_argument("xlsx_path", help="Path to the bipolar .xlsx file")
    parser.add_argument("out_txt_path", help="Path to write the referential .txt file")
    args = parser.parse_args()

    result = convert_file(args.xlsx_path, args.out_txt_path)
    print(f"Wrote {result.shape[0]} time points x {result.shape[1]} electrodes -> {args.out_txt_path}")
