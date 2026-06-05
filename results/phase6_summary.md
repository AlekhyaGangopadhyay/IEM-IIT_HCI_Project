# Phase 6 Summary: Scientific Validation

## Completed Deliverables

- Generated comparative benchmark results in `results/benchmark_metrics.md`.
- Produced visual artifacts:
  - `results/comparative_confusion_matrices.png`
  - `results/trajectory_comparison.png`
- Generated evaluation outputs in `evaluation/results/`:
  - `baselines_results.json`
  - `stats_results.json`
  - `ablation_results.json`
  - `RESULTS_TABLES.md`

## Key Findings

- **Static vs Adaptive Calibration**:
  - Mean static accuracy: **25.93%**
  - Mean adaptive accuracy: **29.63%**
  - Mean accuracy gain: **+3.70%**
  - Wilcoxon signed-rank p-value: **0.375**
  - Rank-biserial effect size: **0.800**

- **Benchmark highlights**:
  - `Ours: 1D-CNN + Calib.` achieved **84.53%** offline accuracy and **33.33%** cross-session accuracy.
  - `Ours: ConvLSTM + Calib.` achieved **85.18%** offline accuracy and **25.93%** cross-session accuracy.
  - `EEGNet` achieved **49.17%** offline and **33.33%** cross-session.
  - `CSP + LDA` and `FBCSP + LDA` both achieved **37.04%** cross-session.

- **Ablation study**:
  - Full pipeline cross-session accuracy: **33.33%** with a **37.04%** shift rate.
  - Removing linear detrending lowers cross-session accuracy to **29.63%**.
  - Static normalization (no adaptive calibration) lowers cross-session accuracy to **29.63%** and reduces shift detection.
  - The safety-margin filter is currently analyzed with gating disabled as a control condition.

## Next Steps

1. Incorporate `results/RESULTS_TABLES.md` into the manuscript or export the tables to LaTeX.
2. Add the generated confusion matrix and trajectory plots to the paper figures section.
3. Refine the manuscript text using the summary findings above.
4. Complete the remaining paper draft and reinforce the narrative around cross-session stability.
