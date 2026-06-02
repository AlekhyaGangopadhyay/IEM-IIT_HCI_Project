"""
Shared building blocks for the evaluation suite:
  * decoder architectures (identical to the trained ones) + EEGNet
  * Chebyshev filtering, sequence/window loading
  * static / adaptive normalization (mirrors src/benchmarking.py)
  * lightweight from-scratch CSP / FBCSP, Euclidean Alignment
  * metrics, parameter count, inference-latency timing
"""
import os, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import cheby1, butter, sosfiltfilt, detrend
from scipy.linalg import eigh

import config as C

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================================
# Architectures (must match the saved .pth state dicts)
# =========================================================================
class EEG_Pure_1DCNN_Classifier(nn.Module):
    def __init__(self, input_dim=3, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_dim, 64, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(64),
            nn.MaxPool1d(2), nn.Dropout1d(0.3),
            nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(128),
            nn.MaxPool1d(2), nn.Dropout1d(0.3),
            nn.Conv1d(128, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(256),
            nn.AdaptiveAvgPool1d(1))
        self.classifier_head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Dropout(0.4), nn.Linear(64, num_classes))

    def forward(self, x):                      # x: (N, time, ch)
        feats = self.features(x.transpose(1, 2)).squeeze(-1)
        return self.classifier_head(feats)


class EEG_ConvLSTM_Classifier(nn.Module):
    def __init__(self, input_dim=3, cnn_channels=64, lstm_hidden_dim=128,
                 num_layers=2, num_classes=4):
        super().__init__()
        self.conv_features = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, 5, padding=2), nn.ReLU(),
            nn.BatchNorm1d(cnn_channels), nn.MaxPool1d(2), nn.Dropout1d(0.3),
            nn.Conv1d(cnn_channels, cnn_channels, 3, padding=1), nn.ReLU(),
            nn.BatchNorm1d(cnn_channels))
        self.lstm = nn.LSTM(cnn_channels, lstm_hidden_dim, num_layers,
                            batch_first=True, dropout=0.4 if num_layers > 1 else 0.0)
        self.classifier_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 64), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Dropout(0.4), nn.Linear(64, num_classes))

    def forward(self, x):                      # x: (N, time, ch)
        cnn_out = self.conv_features(x.transpose(1, 2))
        _, (hidden, _) = self.lstm(cnn_out.transpose(1, 2))
        return self.classifier_head(hidden[-1])


class EEGNet(nn.Module):
    """Compact EEGNet (Lawhern et al. 2018), input (N, 1, ch, time)."""
    def __init__(self, n_ch=3, n_time=256, n_classes=4, F1=8, D=2, F2=16, drop=0.25):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False), nn.BatchNorm2d(F1))
        self.depth = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(drop))
        self.sep = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(drop))
        with torch.no_grad():
            n = self.sep(self.depth(self.block1(torch.zeros(1, 1, n_ch, n_time)))).numel()
        self.head = nn.Linear(n, n_classes)

    def forward(self, x):                      # x: (N, time, ch)
        x = x.transpose(1, 2).unsqueeze(1)     # (N, 1, ch, time)
        x = self.sep(self.depth(self.block1(x)))
        return self.head(x.flatten(1))


def load_decoder(kind):
    """kind in {'cnn','lstm'}; loads trained weights."""
    if kind == "cnn":
        m = EEG_Pure_1DCNN_Classifier().to(device)
        m.load_state_dict(torch.load(C.CNN_WEIGHTS, map_location=device))
    else:
        m = EEG_ConvLSTM_Classifier().to(device)
        m.load_state_dict(torch.load(C.LSTM_WEIGHTS, map_location=device))
    m.eval()
    return m


# =========================================================================
# Signal / data loading
# =========================================================================
def chebyshev_filter(data, fs=C.FS, lo=C.LOWCUT, hi=C.HIGHCUT, order=C.ORDER, rp=C.RP):
    sos = cheby1(order, rp, [lo / (0.5 * fs), hi / (0.5 * fs)],
                 btype="bandpass", output="sos")
    return sosfiltfilt(sos, data, axis=0)


def load_cross_session():
    """Returns list of dicts: {name, true_label, X:(n,255,3)} for LE/RY/For."""
    out = []
    for sc in C.TEST_SCENARIOS:
        fp = os.path.join(C.RAW_TEST_DIR, sc["filename"])
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Missing cross-session file: {fp}")
        df = pd.read_excel(fp); df.columns = df.columns.str.strip()
        vals = df[C.SELECTED_CHANNELS].values.astype(np.float32)
        vals = chebyshev_filter(vals)
        T = C.XSESSION_TIMESTEPS
        n = vals.shape[0] // T
        X = vals[:n * T].reshape(n, T, C.N_CHANNELS)
        out.append({"name": sc["filename"], "true_label": sc["true_label"], "X": X})
    return out


def load_offline():
    """Returns X_train,y_train,X_test,y_test from DATA_DIR (.npy)."""
    p = {k: os.path.join(C.DATA_DIR, v) for k, v in C.NPY.items()}
    missing = [v for v in p.values() if not os.path.exists(v)]
    if missing:
        raise FileNotFoundError(
            "Offline .npy dataset not found. Place these in evaluation/data/:\n  "
            + "\n  ".join(C.NPY.values()))
    return (np.load(p["X_train"]), np.load(p["y_train"]),
            np.load(p["X_test"]),  np.load(p["y_test"]))


# =========================================================================
# Normalization (mirrors benchmarking.py)
# =========================================================================
def normalize(X, mode, do_detrend=True):
    """X:(n,T,3) -> normalized. mode in {'static','adaptive'}."""
    Xp = detrend(X, axis=1) if do_detrend else X.copy()
    if mode == "static":
        mu = np.array(C.GLOBAL_TRAIN_MEAN, np.float32)
        sd = np.array(C.GLOBAL_TRAIN_STD, np.float32)
        return (Xp.reshape(-1, 3) - mu) / sd
    elif mode == "adaptive":
        mu = Xp.mean(axis=(0, 1)); sd = Xp.std(axis=(0, 1))
        sd = np.where(sd == 0, 1e-8, sd)
        return ((Xp - mu) / sd).reshape(-1, 3)
    raise ValueError(mode)


def predict(model, X_norm, n, T):
    Xt = torch.tensor(X_norm.reshape(n, T, 3), dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(Xt)
        probs = torch.softmax(logits, 1).cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()
    return preds, probs


# =========================================================================
# Euclidean Alignment (online, session-level whitening)
# =========================================================================
def euclidean_align(X):
    """X:(n,T,3) -> EA-whitened (n,T,3). Whitens each session by R^{-1/2}."""
    E = np.transpose(X, (0, 2, 1))             # (n, ch, T)
    R = np.mean([e @ e.T / e.shape[1] for e in E], axis=0)
    w, v = eigh(R); w = np.clip(w, 1e-12, None)
    R_isqrt = v @ np.diag(w ** -0.5) @ v.T
    E = np.stack([R_isqrt @ e for e in E])
    return np.transpose(E, (0, 2, 1))


# =========================================================================
# CSP / FBCSP (from scratch, numpy only)
# =========================================================================
def _cov(E):                                   # E:(ch,time)
    E = E - E.mean(1, keepdims=True)
    c = E @ E.T
    return c / (np.trace(c) + 1e-12)


def _csp_filters(epochs, labels, target, m=1):
    A = np.mean([_cov(epochs[i]) for i in np.where(labels == target)[0]], axis=0)
    B = np.mean([_cov(epochs[i]) for i in np.where(labels != target)[0]], axis=0)
    w, v = eigh(A, A + B)
    order = np.argsort(w)
    v = v[:, order]
    return np.concatenate([v[:, :m].T, v[:, -m:].T], axis=0)  # (2m, ch)


def _csp_feat(epochs, filt_list):
    feats = []
    for E in epochs:
        row = []
        for W in filt_list:
            z = W @ E
            vz = np.var(z, 1); vz = vz / (vz.sum() + 1e-12)
            row.extend(np.log(vz + 1e-12))
        feats.append(row)
    return np.asarray(feats)


def _bandpass_epochs(E, lo, hi, fs=C.FS, order=4):
    sos = butter(order, [lo / (0.5 * fs), hi / (0.5 * fs)], btype="band", output="sos")
    return np.stack([sosfiltfilt(sos, e, axis=1) for e in E])


def fit_csp_lda(Xtr, ytr, m=1):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    E = np.transpose(Xtr, (0, 2, 1))           # (N,ch,T)
    filt = [_csp_filters(E, ytr, c, m) for c in range(C.N_CLASSES)]
    lda = LinearDiscriminantAnalysis().fit(_csp_feat(E, filt), ytr)
    return {"filt": filt, "lda": lda}


def predict_csp_lda(model, X):
    E = np.transpose(X, (0, 2, 1))
    return model["lda"].predict(_csp_feat(E, model["filt"]))


FBCSP_BANDS = [(8, 12), (12, 16), (16, 22), (22, 30)]


def fit_fbcsp_lda(Xtr, ytr, m=1):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    E = np.transpose(Xtr, (0, 2, 1))
    bands = []
    feats_all = []
    for lo, hi in FBCSP_BANDS:
        Eb = _bandpass_epochs(E, lo, hi)
        filt = [_csp_filters(Eb, ytr, c, m) for c in range(C.N_CLASSES)]
        bands.append(filt)
        feats_all.append(_csp_feat(Eb, filt))
    lda = LinearDiscriminantAnalysis().fit(np.concatenate(feats_all, 1), ytr)
    return {"bands": bands, "lda": lda}


def predict_fbcsp_lda(model, X):
    E = np.transpose(X, (0, 2, 1))
    feats_all = []
    for (lo, hi), filt in zip(FBCSP_BANDS, model["bands"]):
        Eb = _bandpass_epochs(E, lo, hi)
        feats_all.append(_csp_feat(Eb, filt))
    return model["lda"].predict(np.concatenate(feats_all, 1))


# =========================================================================
# Metrics / profiling
# =========================================================================
def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)) * 100)


def shifting_rate(probs, thr=0.15):
    s = np.sort(probs, axis=1)[:, ::-1]
    margin = s[:, 0] - s[:, 1]
    return float(np.mean(margin < thr) * 100)


def count_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def latency_ms(model, T=C.XSESSION_TIMESTEPS, iters=200):
    model.eval()
    x = torch.zeros(1, T, 3, dtype=torch.float32).to(device)
    with torch.no_grad():
        for _ in range(10):
            model(x)
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return float((time.perf_counter() - t0) / iters * 1000)


def save_json(name, obj):
    fp = os.path.join(C.RESULTS_DIR, name)
    with open(fp, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[saved] {fp}")
    return fp


def load_json(name):
    fp = os.path.join(C.RESULTS_DIR, name)
    return json.load(open(fp)) if os.path.exists(fp) else None
