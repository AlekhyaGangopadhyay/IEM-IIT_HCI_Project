# LORETA Workflow — Full Record (v3)

This supersedes `LORETA_Workflow_Summary.md` and `LORETA_Workflow_v2_AllSubjects.md`.
It captures everything worked out so far: the corrected data understanding, the
three conversion/epoching scripts, the Subject 3 fix, and a step-by-step walkthrough
of the actual LORETA-KEY menu.

---

## 1. What the dataset actually is

- Raw format: bipolar double-banana montage, 20 columns per file (18 real chain
  channels + `BP1-REF` / `BP2-REF`), exported as `.xlsx`.
- Folder structure:
  ```
  Original Dataset/
    Subject 1/
      Backward/  ARROW_Backward.xlsx  LETTER_Backward.xlsx  WORD_Backward.xlsx
      Forward/   ARROW_Forward.xlsx   LETTER_Forward.xlsx   WORD_Forward.xlsx
      Left/      ...
      Right/     ...
    Subject 2/  (same structure)
    Subject 3/  (same structure, but longer files — see §4)
  ```
  3 subjects × 4 directions × 3 cue types = **36 files**.
- Each file (for Subjects 1–2) is **10,240 rows = 4 trials × 2,560 rows (10 s @
  256 Hz)** of the same condition, concatenated back-to-back — not one
  continuous 40-second block, which was the original (incorrect) assumption.
- `BP1-REF` / `BP2-REF` were checked directly against real data: both are
  near-constant offsets (std ≈ 3 µV), not a trigger/marker channel. Correctly
  excluded.
- No sharp discontinuity was found in any EEG channel at the presumed
  2,560-sample trial boundaries — consistent with 4 reps run back-to-back with
  no inserted rest, but not independently provable without a marker channel or
  the original protocol notes.
- The 18 bipolar channels form **3 disconnected chains ("islands")**:
  - Right hemisphere (8 electrodes, anchor Fp2): lateral chain
    `Fp2→F8→T4→T6→O2` and parasagittal chain `Fp2→F4→C4→P4→O2` (both reach O2 —
    verified to match exactly, mean abs diff = 0.0, in real data).
  - Left hemisphere (8 electrodes, anchor Fp1): same shape, mirrored, both
    reach O1 (verified exact match).
  - Midline (3 electrodes, anchor Fz): `Fz→Cz→Pz`.
  - Total: 19 standard 10-20 electrodes. Order confirmed from the built
    coordinates file: `Fp1 Fp2 F3 F4 F7 F8 C3 C4 T3 T4 T5 T6 P3 P4 O1 O2 Fz Pz Cz`
    (Cz, not CPz — corrected earlier from the midline chain `FZ-CZ`, `CZ-PZ`).

---

## 2. The three scripts

Keep all three in the same folder (e.g. directly in `HCI/`), since
`batch_convert_folder.py` imports from `bipolar_to_referential.py`.

### `bipolar_to_referential.py`
Converts one bipolar `.xlsx` file into a 19-electrode referential signal:
sets each island's anchor electrode (Fp2, Fp1, Fz) to 0, cumulatively
reconstructs every other electrode in that island by subtracting each bipolar
value in sequence, averages the two independent estimates of O2 and O1, then
applies a common average reference (CAR) across all 19 electrodes. Output:
plain-text matrix, rows = time points, columns = 19 electrodes in the fixed
order above, no header.

### `batch_convert_folder.py`
Walks the full `Original Dataset` tree and runs the above conversion on every
`.xlsx` file found, writing `.txt` files into a new output folder that mirrors
the same `Subject/Direction/` structure.

```
python batch_convert_folder.py "Original Dataset" "referential_output"
```

### `split_trial_epochs.py`
Splits each 19-channel referential `.txt` into LORETA-compatible epochs
(≤1024 frames — LORETA-KEY's cross-spectral maker hard limit), aligned to the
**real** trial boundaries rather than chopped arbitrarily:

- `--mode 1024` (default): 2 × 1,024-frame epochs per trial (uses rows
  0–2047 of each 2,560-row trial, discards the last 512 rows/trial) → **8
  epochs per file**, no epoch ever straddles a trial boundary, best low-frequency
  (delta) resolution.
- `--mode 640`: 4 × 640-frame epochs per trial (640 divides 2,560 evenly, no
  discard) → 16 epochs/file, more epochs to average but coarser resolution
  below ~2–3 Hz.

Also handles files longer than expected (see §4) by truncating to the first
4 trials, and writes `epoch_manifest.csv` tracking every epoch back to its
subject/condition/trial/source file.

```
python split_trial_epochs.py "referential_output" "epoch_output" --mode 1024
```

---

## 3. Why per-subject, per-condition — never pooled across subjects

When building cross-spectra (§5, step 4 below), always compute **one spectrum
per subject per condition** (36 total), never one shared spectrum pooling
epochs from multiple subjects. Pooling would make *epochs*, not *subjects*,
the unit of replication — with only 3 subjects, that lets whichever subject
contributes noisier/cleaner epochs dominate the result, and any later
"significant" group difference wouldn't actually reflect consistency across
people. This is the same leakage/pseudo-replication issue already flagged for
the motor-imagery classifier (window-level vs. file-level splits). It's also
what LORETA's own SnPM group tool expects: one input file = one replicate.

---

## 4. The Subject 3 fix

Subject 3's files turned out to have **30,720 rows (12 trials)** instead of
10,240 (4 trials) — `split_trial_epochs.py`'s shape check was rejecting them
outright ("skipped"). Fixed: the script now truncates any file longer than
10,240 rows down to the first 10,240 (first 4 trials), printing a `NOTE:` line
so it's visible when this happens, and keeps everything else identical so
Subject 3 lines up with Subjects 1–2 (8 epochs/file, same as everyone else).
No re-run of `batch_convert_folder.py` was needed — only the epoch-splitting
step.

---

## 5. LORETA-KEY menu walkthrough (current stage)

Matching the actual LORETA-KEY menu, in order:

1. **Electrode coordinates maker → Based on extended 10/10 system template**
   Only needed once — reuse the `.xyz` file from the earlier 6-file run if it
   still exists (electrode set/order hasn't changed).

2. **Transformation matrix (LORETA operator)**
   Also only needed once, built from the `.xyz` file above — reuse if you
   still have it.

3. **EEG cross-spectra → AllEEGs → 1Spec(man)** ← the main new work
   Run this **36 times** (3 subjects × 12 conditions). Each run:
   - Select that one subject's 8 epoch `.txt` files for that one condition
     (e.g. only Subject 1's `ARROW_Backward` epochs).
   - Settings: Electrodes = 19, Time frames/file = 1024, Sampling rate = 256,
     Frequencies = δ θ α1 α2 β1 β2 β3 Ω, Normalize = unchecked, Force Average
     Reference = checked.
   - Save with a name tracking subject + condition, e.g. `S1_ARROW_Backward.crs`.
   - Recommended: do one subject's 12 conditions per sitting rather than all 36
     at once.

4. **EEG cross-spectrum → LORETA**
   Combine each of the 36 `.crs` files with the `.xyz` + transformation matrix
   from steps 1–2 → 36 per-band `.lor` current-density files (one per subject
   per condition).

5. **LORETA viewer** *(optional spot-check)*
   Open a couple of `.lor` files to sanity-check they look like plausible
   brain maps before moving on.

6. **Utilities → List of "FileNames" Maker**
   Build group file lists for each contrast of interest, e.g. all 3 subjects'
   `Forward` `.lor` files vs. all 3 subjects' `Backward` `.lor` files. One list
   per side of each contrast:
   - Forward vs. Backward
   - Left vs. Right
   - *(optional)* Forward+Backward vs. Left+Right
   - Cue-type: Arrow vs. Letter vs. Word (collapsed across direction) — useful
     as a spatial check on the frontal/EOG confound question raised separately
     for the classifier: if this contrast localizes mostly to
     occipital/frontal sources rather than sensorimotor cortex, that supports
     the frontal-channel dominance seen in the classifier being a visual/ocular
     artifact rather than real motor-imagery signal.

7. **Statistical non-Parametric Mapping (SnPM) → Voxel-wise LORETA (text
   format) comparisons**
   Run each contrast from step 6, band by band, using the file lists built
   there.

**Honest caveat carried through every step above:** N = 3 subjects per
condition is a very small sample for any voxel-wise group statistic — treat
SnPM results as hypothesis-generating, not confirmatory.

---

## 6. Status checklist

- [x] Bipolar → referential conversion script built and verified against real
      data (O1/O2 cross-check exact match)
- [x] Batch conversion scaled to all 3 subjects, tested against the real
      folder structure
- [x] Trial-aware epoch splitter built, verified end-to-end
- [x] Subject 3's 12-trial files handled (truncated to first 4 trials)
- [x] Electrode coordinates + transformation matrix (reused from earlier run,
      confirm still on hand)
- [ ] 36 cross-spectra (`AllEEGs → 1Spec(man)`, one per subject per condition)
- [ ] 36 cross-spectrum → LORETA conversions (`.lor` files)
- [ ] Group file lists for each contrast (`List of "FileNames" Maker`)
- [ ] SnPM voxel-wise comparisons, band by band, for each contrast
