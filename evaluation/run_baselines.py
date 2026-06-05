"""
Baseline comparison -> fills Table `tab:baselines` in EEG_Paper.tex.

Methods: CSP+LDA, FBCSP+LDA, EEGNet, EA+1D-CNN, and our 1D-CNN / ConvLSTM.
Columns: Offline test acc, pooled Cross-session acc, #params, per-window latency.

Requires:
  * evaluation/data/*.npy  (offline processed dataset) for CSP/FBCSP/EEGNet/offline
  * data_for_testing/raw/{LE,RY,For}.xlsx for cross-session
  * models/*.pth for our decoders
Writes: results/baselines_results.json
"""
import numpy as np
import torch
import common as K
import config as C


def pooled_nn(model, xs, mode="adaptive", ea=False):
    yt, yp = [], []
    for f in xs:
        X = K.euclidean_align(f["X"]) if ea else f["X"]
        Xn = K.normalize(X, mode)
        preds, _ = K.predict(model, Xn, f["X"].shape[0], C.XSESSION_TIMESTEPS)
        yt += [f["true_label"]] * f["X"].shape[0]; yp += list(preds)
    return K.accuracy(yt, yp)


def pooled_classic(model, predict_fn, xs):
    yt, yp = [], []
    for f in xs:
        yp += list(predict_fn(model, f["X"]))
        yt += [f["true_label"]] * f["X"].shape[0]
    return K.accuracy(yt, yp)


def train_eegnet(Xtr, ytr, Xte, yte, epochs=20):
    import time
    m = K.EEGNet(n_time=C.TRAIN_SEQ_LEN).to(K.device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    lossf = torch.nn.CrossEntropyLoss()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr_t, ytr_t), batch_size=256, shuffle=True)
    for ep in range(epochs):
        m.train()
        for xb, yb in dl:
            xb, yb = xb.to(K.device), yb.to(K.device)
            opt.zero_grad(); loss = lossf(m(xb), yb); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        pr = m(torch.tensor(Xte, dtype=torch.float32).to(K.device)).argmax(1).cpu().numpy()
    torch.save(m.state_dict(), C.MODELS_DIR + "/EEGNet_classifier.pth")
    return m, K.accuracy(yte, pr)


def main():
    rows = {}
    xs = K.load_cross_session()
    print(f"[ok] cross-session files: {[f['name'] for f in xs]}")

    # ---- our decoders ----------------------------------------------------
    cnn, lstm = K.load_decoder("cnn"), K.load_decoder("lstm")
    try:
        Xtr, ytr, Xte, yte = K.load_offline()
        cnn_off = K.accuracy(yte, K.predict(cnn, Xte.reshape(-1, 3), len(Xte), C.TRAIN_SEQ_LEN)[0])
        lstm_off = K.accuracy(yte, K.predict(lstm, Xte.reshape(-1, 3), len(Xte), C.TRAIN_SEQ_LEN)[0])
        have_offline = True
    except FileNotFoundError as e:
        print(f"[warn] {e}\n[warn] offline columns -> PENDING (place .npy in evaluation/data/)")
        cnn_off = lstm_off = None; have_offline = False

    rows["Ours: 1D-CNN + Calib."] = {
        "offline": cnn_off, "cross": pooled_nn(cnn, xs, "adaptive"),
        "params": K.count_params(cnn), "latency_ms": round(K.latency_ms(cnn), 3)}
    rows["Ours: ConvLSTM + Calib."] = {
        "offline": lstm_off, "cross": pooled_nn(lstm, xs, "adaptive"),
        "params": K.count_params(lstm), "latency_ms": round(K.latency_ms(lstm), 3)}

    # ---- EA + 1D-CNN (cross-session technique) ---------------------------
    rows["EA + 1D-CNN"] = {
        "offline": None, "cross": pooled_nn(cnn, xs, "adaptive", ea=True),
        "params": K.count_params(cnn), "latency_ms": round(K.latency_ms(cnn), 3)}

    # ---- classical + EEGNet (need offline dataset) -----------------------
    if have_offline:
        Xtr = Xtr.astype(np.float32); Xte = Xte.astype(np.float32)
        print("[run] fitting CSP+LDA ...")
        csp = K.fit_csp_lda(Xtr, ytr)
        rows["CSP + LDA"] = {
            "offline": K.accuracy(yte, K.predict_csp_lda(csp, Xte)),
            "cross": pooled_classic(csp, K.predict_csp_lda, xs),
            "params": None, "latency_ms": None}
        print("[run] fitting FBCSP+LDA ...")
        fb = K.fit_fbcsp_lda(Xtr, ytr)
        rows["FBCSP + LDA"] = {
            "offline": K.accuracy(yte, K.predict_fbcsp_lda(fb, Xte)),
            "cross": pooled_classic(fb, K.predict_fbcsp_lda, xs),
            "params": None, "latency_ms": None}
        print("[run] training EEGNet ...")
        eeg, eeg_off = train_eegnet(Xtr, ytr, Xte, yte)
        rows["EEGNet"] = {
            "offline": eeg_off, "cross": pooled_nn(eeg, xs, "adaptive"),
            "params": K.count_params(eeg), "latency_ms": round(K.latency_ms(eeg), 3)}
    else:
        for k in ["CSP + LDA", "FBCSP + LDA", "EEGNet"]:
            rows[k] = {"offline": "PENDING", "cross": "PENDING",
                       "params": "PENDING", "latency_ms": "PENDING"}

    K.save_json("baselines_results.json", rows)
    for k, v in rows.items():
        print(f"  {k:26s} off={v['offline']} cross={v['cross']} "
              f"params={v['params']} lat={v['latency_ms']}")


if __name__ == "__main__":
    main()
