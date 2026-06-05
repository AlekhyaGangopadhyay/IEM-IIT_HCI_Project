"""
Statistical validation -> fills the `Statistical Validation` subsection.

Primary test: Wilcoxon signed-rank on paired per-(file, model) accuracies,
Static (Condition A) vs Adaptive (Condition B) calibration -- the headline
claim of the paper. Also reports a per-window correctness test for the 1D-CNN,
the rank-biserial effect size, and a bootstrap CI of the mean accuracy gain.

Fully reproducible from models/*.pth + the raw cross-session files.
Writes: results/stats_results.json
"""
import numpy as np
from scipy.stats import wilcoxon
import common as K
import config as C


def per_file_acc(model, f, mode):
    Xn = K.normalize(f["X"], mode)
    pr, _ = K.predict(model, Xn, f["X"].shape[0], C.XSESSION_TIMESTEPS)
    yt = np.full(f["X"].shape[0], f["true_label"])
    return K.accuracy(yt, pr), (pr == f["true_label"]).astype(int)


def safe_wilcoxon(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.all(a - b == 0):
        return {"statistic": None, "pvalue": 1.0, "note": "all differences zero"}
    try:
        s, p = wilcoxon(a, b)
        return {"statistic": float(s), "pvalue": float(p)}
    except ValueError as e:
        return {"statistic": None, "pvalue": None, "note": str(e)}


def rank_biserial(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = np.argsort(np.argsort(np.abs(d))) + 1
    rp, rn = r[d > 0].sum(), r[d < 0].sum()
    return float((rp - rn) / r.sum())


def bootstrap_ci(diffs, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    means = [rng.choice(diffs, len(diffs), replace=True).mean() for _ in range(n)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main():
    xs = K.load_cross_session()
    models = {"1D-CNN": K.load_decoder("cnn"), "ConvLSTM": K.load_decoder("lstm")}

    static_accs, adapt_accs = [], []
    cnn_static_w, cnn_adapt_w = [], []
    for name, m in models.items():
        for f in xs:
            sa, sw = per_file_acc(m, f, "static")
            aa, aw = per_file_acc(m, f, "adaptive")
            static_accs.append(sa); adapt_accs.append(aa)
            if name == "1D-CNN":
                cnn_static_w.append(sw); cnn_adapt_w.append(aw)

    diffs = list(np.array(adapt_accs) - np.array(static_accs))
    out = {
        "n_pairs_per_file_test": len(diffs),
        "mean_static_acc": float(np.mean(static_accs)),
        "mean_adaptive_acc": float(np.mean(adapt_accs)),
        "mean_gain": float(np.mean(diffs)),
        "per_file_wilcoxon_static_vs_adaptive": safe_wilcoxon(adapt_accs, static_accs),
        "rank_biserial_effect_size": rank_biserial(adapt_accs, static_accs),
        "bootstrap_95ci_mean_gain": bootstrap_ci(diffs),
        "per_window_wilcoxon_cnn": safe_wilcoxon(
            np.concatenate(cnn_adapt_w), np.concatenate(cnn_static_w)),
    }
    K.save_json("stats_results.json", out)
    print(f"  mean static={out['mean_static_acc']:.2f}%  "
          f"adaptive={out['mean_adaptive_acc']:.2f}%  gain={out['mean_gain']:+.2f}%")
    print(f"  Wilcoxon (per-file) p={out['per_file_wilcoxon_static_vs_adaptive']['pvalue']}")
    print(f"  effect size (rank-biserial)={out['rank_biserial_effect_size']:.3f}  "
          f"95% CI gain={out['bootstrap_95ci_mean_gain']}")


if __name__ == "__main__":
    main()
