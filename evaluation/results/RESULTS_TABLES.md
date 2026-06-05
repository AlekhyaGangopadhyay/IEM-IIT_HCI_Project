# Robustness Evaluation Results

### Table: Baseline Comparison (tab:baselines)

| Method | Offline | Cross-Ses. | Params | Lat. (ms) |
|---|---|---|---|---|
| CSP + LDA | 39.14% | 37.04% | -- | -- |
| FBCSP + LDA | 43.69% | 37.04% | -- | -- |
| EEGNet | 49.17% | 33.33% | 1,668 | 1.496 |
| EA + 1D-CNN | -- | 25.93% | 158,788 | 1.273 |
| Ours: 1D-CNN + Calib. | 84.53% | 33.33% | 158,788 | 1.475 |
| Ours: ConvLSTM + Calib. | 85.18% | 25.93% | 253,700 | 5.428 |

### Table: Ablation Study (tab:ablation)

| Configuration | Offline | Cross-Ses. | Shift. Rate |
|---|---|---|---|
| Full pipeline | 84.53% | 33.33% | 37.04% |
| - spectral loss | PENDING | PENDING | PENDING |
| - generative augmentation | PENDING | PENDING | PENDING |
| - linear detrending | 84.53% | 29.63% | 33.33% |
| - adaptive calibration (static) | 84.53% | 29.63% | 7.41% |
| - safety-margin filter | 84.53% | 33.33% | off (no gating) |

### Statistical Validation (Wilcoxon signed-rank)

- Mean accuracy: static **25.93%** vs adaptive **29.63%** (gain **+3.70%**)
- Per-file Wilcoxon (Static vs Adaptive): W=1.5, p=0.375
- Rank-biserial effect size: 0.800
- Bootstrap 95% CI of mean gain: [-3.7037037037037006, 9.259259259259268]
