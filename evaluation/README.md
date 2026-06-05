# Robustness Evaluation Suite

Runnable scripts that produce every number in the **Experimental Results** and
**Planned Robustness Evaluation** sections of `EEG_Paper.tex`. Each script writes
a JSON to `results/`; `make_tables.py` aggregates them into markdown tables whose
cells map 1:1 onto the `--` placeholders in the paper.

> No values are invented. Cells you have not yet computed render as `--` or
> `PENDING` with the exact command needed to fill them.

## Layout
```
evaluation/
├── config.py                  # paths + constants (edit here if your layout differs)
├── common.py                  # architectures, filtering, CSP/FBCSP, EA, metrics
├── run_baselines.py           # -> tab:baselines   (CSP, FBCSP, EEGNet, EA, ours)
├── run_ablation.py            # -> tab:ablation
├── run_statistics.py          # -> Statistical Validation (Wilcoxon)
├── make_tables.py             # aggregate JSON -> results/RESULTS_TABLES.md
├── build_dataset.py           # helper: build .npy dataset from xlsx folders
├── train_decoders.py          # helper: retrain 1D-CNN/ConvLSTM on a tagged dataset
├── train_wgan_no_spectral.py  # helper: WGAN-GP with lambda_spec=0 (ablation)
├── data/                      # <- put offline .npy here (see below)
└── results/                   # JSON + RESULTS_TABLES.md (auto-created)
```

## Prerequisites (data the scripts read)
| Need | Where | Used by |
|---|---|---|
| `models/EEG_pure_1DCNN_classifier.pth`, `EEG_ConvLSTM_classifier.pth` | already in repo | all |
| `data_for_testing/raw/{LE,RY,For}.xlsx` | already referenced by `src/benchmarking.py` | cross-session |
| `X_train_500.npy`, `X_test_500.npy`, `y_train_cls_500.npy`, `y_test_cls_500.npy` | copy into `evaluation/data/` (the Kaggle processed dataset) | offline cols, CSP/FBCSP/EEGNet |

```bash
pip install -r requirements.txt
```

## Quick start (the cheap 90%)
Run from inside `evaluation/`:
```bash
python run_statistics.py     # Wilcoxon static-vs-adaptive  (no offline data needed)
python run_baselines.py      # baseline table (CSP/FBCSP/EEGNet need data/*.npy)
python run_ablation.py       # ablation table (2 rows auto, 2 marked PENDING)
python make_tables.py        # -> results/RESULTS_TABLES.md
```
This fills: the whole Statistical Validation subsection, the EA + ours rows of
`tab:baselines` (and CSP/FBCSP/EEGNet if `data/*.npy` is present), and 4 of the 6
ablation rows.

## The two retraining rows (only heavy step)
These fill `- spectral loss` and `- generative augmentation` in `tab:ablation`.

**(a) `- generative augmentation`** — train on the 24 originals only:
```bash
python build_dataset.py --source "<Chebyshev Filtered Data>" --tag orig24 --max-per-class 6
python train_decoders.py --tag orig24 --models cnn
python run_ablation.py            # row now auto-fills
```

**(b) `- spectral loss`** — regenerate the corpus with `lambda_spec=0`:
```bash
python train_wgan_no_spectral.py --input "<originals: class subfolders>" --output data/synthetic_nospec
python build_dataset.py --source data/synthetic_nospec --tag nospec
python train_decoders.py --tag nospec --models cnn
python run_ablation.py            # row now auto-fills
```
`train_wgan_no_spectral.py` is the only GPU-heavy step (~same cost as your
original synthesis run).

## Notes
- CSP/FBCSP are implemented from scratch (numpy/scipy) — no `mne` dependency.
- `EA + 1D-CNN` is online session-level Euclidean whitening reusing the trained
  1D-CNN (no retraining); offline column is `--` because EA is a cross-session
  method.
- The ablation reference decoder is the 1D-CNN; edit `run_ablation.py` to switch.
- `config.GLOBAL_TRAIN_MEAN/STD` are the values from `src/benchmarking.py`.
  `build_dataset.py` prints the exact scaler stats for any tag if you want to
  replace them.
