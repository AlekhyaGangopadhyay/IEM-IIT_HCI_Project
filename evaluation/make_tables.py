"""
Aggregate results/*.json into markdown tables that map 1:1 onto the '--'
cells of EEG_Paper.tex (tab:baselines, tab:ablation, Statistical Validation).
Writes results/RESULTS_TABLES.md and prints to stdout.
"""
import common as K


def acc(v):
    if v is None:
        return "--"
    if isinstance(v, str):
        return v
    return f"{v:.2f}%"


def num(v):
    if v is None:
        return "--"
    if isinstance(v, str):
        return v
    return f"{v:,}" if isinstance(v, int) else f"{v:.3f}"


def baselines_md(d):
    order = ["CSP + LDA", "FBCSP + LDA", "EEGNet", "EA + 1D-CNN",
             "Ours: 1D-CNN + Calib.", "Ours: ConvLSTM + Calib."]
    L = ["### Table: Baseline Comparison (tab:baselines)", "",
         "| Method | Offline | Cross-Ses. | Params | Lat. (ms) |",
         "|---|---|---|---|---|"]
    for k in order:
        v = d.get(k, {})
        L.append(f"| {k} | {acc(v.get('offline'))} | {acc(v.get('cross'))} "
                 f"| {num(v.get('params'))} | {num(v.get('latency_ms'))} |")
    return "\n".join(L)


def ablation_md(d):
    order = ["Full pipeline", "- spectral loss", "- generative augmentation",
             "- linear detrending", "- adaptive calibration (static)",
             "- safety-margin filter"]
    L = ["### Table: Ablation Study (tab:ablation)", "",
         "| Configuration | Offline | Cross-Ses. | Shift. Rate |",
         "|---|---|---|---|"]
    for k in order:
        v = d.get(k, {})
        sh = v.get("shift")
        sh = sh if isinstance(sh, str) else (acc(sh) if sh is not None else "--")
        L.append(f"| {k} | {acc(v.get('offline'))} | {acc(v.get('cross'))} | {sh} |")
    return "\n".join(L)


def stats_md(d):
    w = d.get("per_file_wilcoxon_static_vs_adaptive", {})
    return "\n".join([
        "### Statistical Validation (Wilcoxon signed-rank)", "",
        f"- Mean accuracy: static **{d.get('mean_static_acc'):.2f}%** vs "
        f"adaptive **{d.get('mean_adaptive_acc'):.2f}%** "
        f"(gain **{d.get('mean_gain'):+.2f}%**)",
        f"- Per-file Wilcoxon (Static vs Adaptive): "
        f"W={w.get('statistic')}, p={w.get('pvalue')}",
        f"- Rank-biserial effect size: {d.get('rank_biserial_effect_size'):.3f}",
        f"- Bootstrap 95% CI of mean gain: {d.get('bootstrap_95ci_mean_gain')}",
    ])


def main():
    blocks = []
    for name, fn in [("baselines_results.json", baselines_md),
                     ("ablation_results.json", ablation_md),
                     ("stats_results.json", stats_md)]:
        d = K.load_json(name)
        blocks.append(fn(d) if d else f"_({name} not found - run the matching script)_")
    md = "# Robustness Evaluation Results\n\n" + "\n\n".join(blocks) + "\n"
    K.save_json  # noqa (keep import used)
    import os, config as C
    fp = os.path.join(C.RESULTS_DIR, "RESULTS_TABLES.md")
    open(fp, "w", encoding="utf-8").write(md)
    print(md); print(f"[saved] {fp}")


if __name__ == "__main__":
    main()
