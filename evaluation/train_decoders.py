"""
Train the 1D-CNN and/or ConvLSTM decoders on a tagged dataset built by
build_dataset.py. Used for the two ablation variants that need retraining.

Usage:
  python train_decoders.py --tag nospec  --models cnn
  python train_decoders.py --tag orig24  --models cnn lstm

Reads data/<tag>/{X_train,y_train,X_test,y_test}.npy
Saves models/EEG_1DCNN_<tag>.pth and/or models/EEG_ConvLSTM_<tag>.pth
"""
import os, argparse
import numpy as np
import torch
import torch.nn as nn
import common as K
import config as C


def train(model, Xtr, ytr, Xte, yte, epochs, clip=None):
    dev = K.device
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                                       torch.tensor(ytr, dtype=torch.long)),
        batch_size=256, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=2, factor=0.5)
    lossf = nn.CrossEntropyLoss()
    Xte_t = torch.tensor(Xte, dtype=torch.float32).to(dev)
    best = 0.0; best_state = None
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward()
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = K.accuracy(yte, model(Xte_t).argmax(1).cpu().numpy())
        sch.step(acc)
        if acc > best:
            best, best_state = acc, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  epoch {ep+1:02d}/{epochs}  test_acc={acc:.2f}%")
    model.load_state_dict(best_state)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--models", nargs="+", default=["cnn"], choices=["cnn", "lstm"])
    ap.add_argument("--epochs", type=int, default=20)
    a = ap.parse_args()

    d = os.path.join(C.DATA_DIR, a.tag)
    Xtr, ytr = np.load(d + "/X_train.npy"), np.load(d + "/y_train.npy")
    Xte, yte = np.load(d + "/X_test.npy"), np.load(d + "/y_test.npy")
    print(f"[ok] {a.tag}: Xtrain={Xtr.shape} Xtest={Xte.shape}")

    if "cnn" in a.models:
        m = K.EEG_Pure_1DCNN_Classifier().to(K.device)
        best = train(m, Xtr, ytr, Xte, yte, a.epochs)
        fp = os.path.join(C.MODELS_DIR, f"EEG_1DCNN_{a.tag}.pth")
        torch.save(m.state_dict(), fp); print(f"[saved] {fp}  best={best:.2f}%")
    if "lstm" in a.models:
        m = K.EEG_ConvLSTM_Classifier().to(K.device)
        best = train(m, Xtr, ytr, Xte, yte, a.epochs, clip=2.0)
        fp = os.path.join(C.MODELS_DIR, f"EEG_ConvLSTM_{a.tag}.pth")
        torch.save(m.state_dict(), fp); print(f"[saved] {fp}  best={best:.2f}%")


if __name__ == "__main__":
    main()
