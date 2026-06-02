"""
Build an offline .npy dataset (file-level split + StandardScaler) from a
directory of class sub-folders of .xlsx recordings. Mirrors
src/filesequenicng.py but is local and parameterized.

Usage:
  # ablation: 24 originals only (Chebyshev-filtered)
  python build_dataset.py --source "<Chebyshev Filtered Data>" --tag orig24
  # ablation: lambda_spec=0 synthetic corpus
  python build_dataset.py --source "<synthetic_nospec dir>" --tag nospec

Each <source> must contain sub-folders: Right/ Left/ Forward/ Backward/
Writes data/<tag>/{X_train,X_test,y_train,y_test}.npy
"""
import os, glob, argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import config as C

LABELS = {"Right": 0, "Left": 1, "Forward": 2, "Backward": 3}


def sequences(data, n):
    return [data[i:i + n] for i in range(0, len(data) - n + 1, n)]


def collect(source, max_per_class):
    files = []
    for cls, lab in LABELS.items():
        fs = sorted(glob.glob(os.path.join(source, cls, "*.xlsx")))
        if max_per_class:
            fs = fs[:max_per_class]
        if not fs:
            print(f"[warn] no files for class {cls} in {source}")
        files += [(f, lab) for f in fs]
    return files


def build(files, seq):
    X, y = [], []
    for fp, lab in files:
        try:
            df = pd.read_excel(fp); df.columns = df.columns.str.strip()
            data = df[C.SELECTED_CHANNELS].values.astype(np.float32)
            for s in sequences(data, seq):
                X.append(s); y.append(lab)
        except Exception as e:
            print(f"[skip] {fp}: {e}")
    return np.asarray(X, np.float32), np.asarray(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seq", type=int, default=C.TRAIN_SEQ_LEN)
    ap.add_argument("--max-per-class", type=int, default=0)
    a = ap.parse_args()

    files = collect(a.source, a.max_per_class)
    tr, te = [], []
    for lab in LABELS.values():
        cf = [f for f in files if f[1] == lab]
        a_tr, a_te = train_test_split(cf, test_size=0.2, random_state=42)
        tr += a_tr; te += a_te
    print(f"[ok] train files={len(tr)} test files={len(te)}")

    Xtr, ytr = build(tr, a.seq)
    Xte, yte = build(te, a.seq)
    ch = Xtr.shape[2]
    sc = StandardScaler().fit(Xtr.reshape(-1, ch))
    Xtr = sc.transform(Xtr.reshape(-1, ch)).reshape(Xtr.shape)
    Xte = sc.transform(Xte.reshape(-1, ch)).reshape(Xte.shape)

    out = os.path.join(C.DATA_DIR, a.tag); os.makedirs(out, exist_ok=True)
    np.save(out + "/X_train.npy", Xtr); np.save(out + "/y_train.npy", ytr)
    np.save(out + "/X_test.npy",  Xte); np.save(out + "/y_test.npy",  yte)
    print(f"[saved] {out}  Xtrain={Xtr.shape} Xtest={Xte.shape}")
    print(f"[scaler] mean={sc.mean_}  std={sc.scale_}  "
          f"(use these as GLOBAL_TRAIN_MEAN/STD for tag={a.tag} if needed)")


if __name__ == "__main__":
    main()
