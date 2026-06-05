"""
Ablation study -> fills Table `tab:ablation` in EEG_Paper.tex.
Reference decoder = 1D-CNN. One component removed per row.

Rows & how they are produced:
  Full pipeline               cnn (adaptive calib)                 [auto]
  - spectral loss             cnn trained on lambda_spec=0 synth   [needs EEG_1DCNN_nospec.pth]
  - generative augmentation   cnn trained on 24 originals only     [needs EEG_1DCNN_orig24.pth]
  - linear detrending         cnn, adaptive, detrend OFF           [auto]
  - adaptive calibration      cnn, static global norm (Cond. A)    [auto]
  - safety-margin filter      cnn, adaptive, dispatch gate OFF     [auto]

Offline column for the two retrained variants needs their own test split at
data/<tag>/X_test.npy (+ y_test). Cross-session always uses the raw files.
Writes: results/ablation_results.json
"""
import os
import numpy as np
import torch
import common as K
import config as C


def pooled(model, xs, mode, do_detrend=True):
    yt, yp, probs = [], [], []
    for f in xs:
        Xn = K.normalize(f["X"], mode, do_detrend=do_detrend)
        pr, pb = K.predict(model, Xn, f["X"].shape[0], C.XSESSION_TIMESTEPS)
        yt += [f["true_label"]] * f["X"].shape[0]; yp += list(pr); probs.append(pb)
    return K.accuracy(yt, yp), K.shifting_rate(np.concatenate(probs))


def offline_acc(model, tag=None):
    """Offline test accuracy. tag=None -> main dataset; else data/<tag>/."""
    try:
        if tag is None:
            _, _, Xte, yte = K.load_offline()
        else:
            d = os.path.join(C.DATA_DIR, tag)
            Xte, yte = np.load(d + "/X_test.npy"), np.load(d + "/y_test.npy")
    except (FileNotFoundError, OSError):
        return None
    pr, _ = K.predict(model, Xte.astype(np.float32).reshape(-1, 3),
                      len(Xte), C.TRAIN_SEQ_LEN)
    return K.accuracy(yte, pr)


def load_alt(weights_name):
    fp = os.path.join(C.MODELS_DIR, weights_name)
    if not os.path.exists(fp):
        return None
    m = K.EEG_Pure_1DCNN_Classifier().to(K.device)
    m.load_state_dict(torch.load(fp, map_location=K.device)); m.eval()
    return m


def main():
    xs = K.load_cross_session()
    cnn = K.load_decoder("cnn")
    rows = {}

    acc, shift = pooled(cnn, xs, "adaptive")
    rows["Full pipeline"] = {"offline": offline_acc(cnn), "cross": acc, "shift": shift}

    nospec = load_alt("EEG_1DCNN_nospec.pth")
    if nospec:
        a, s = pooled(nospec, xs, "adaptive")
        rows["- spectral loss"] = {"offline": offline_acc(nospec, "nospec"), "cross": a, "shift": s}
    else:
        rows["- spectral loss"] = {"offline": "PENDING", "cross": "PENDING", "shift": "PENDING",
                                   "_note": "run train_wgan_no_spectral.py -> build_dataset.py -> train_decoders.py --tag nospec"}

    orig = load_alt("EEG_1DCNN_orig24.pth")
    if orig:
        a, s = pooled(orig, xs, "adaptive")
        rows["- generative augmentation"] = {"offline": offline_acc(orig, "orig24"), "cross": a, "shift": s}
    else:
        rows["- generative augmentation"] = {"offline": "PENDING", "cross": "PENDING", "shift": "PENDING",
                                             "_note": "build_dataset.py --source originals -> train_decoders.py --tag orig24"}

    a, s = pooled(cnn, xs, "adaptive", do_detrend=False)
    rows["- linear detrending"] = {"offline": rows["Full pipeline"]["offline"], "cross": a, "shift": s}

    a, s = pooled(cnn, xs, "static")
    rows["- adaptive calibration (static)"] = {"offline": rows["Full pipeline"]["offline"], "cross": a, "shift": s}

    rows["- safety-margin filter"] = {"offline": rows["Full pipeline"]["offline"],
                                      "cross": rows["Full pipeline"]["cross"], "shift": "off (no gating)"}

    K.save_json("ablation_results.json", rows)
    for k, v in rows.items():
        print(f"  {k:34s} off={v['offline']} cross={v['cross']} shift={v['shift']}")


if __name__ == "__main__":
    main()
