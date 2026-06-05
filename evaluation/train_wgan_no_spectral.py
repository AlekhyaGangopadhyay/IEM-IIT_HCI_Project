"""
WGAN-GP synthesizer with the spectral loss DISABLED (lambda_spec = 0).
Local, dependency-light adaptation of src/Synthetic Data Creation.py used
to produce the augmented corpus for the '- spectral loss' ablation row.

Usage:
  python train_wgan_no_spectral.py --input "<originals: class subfolders>" \
         --output data/synthetic_nospec --per-file 100 --epochs 100

<input> must contain sub-folders Right/ Left/ Forward/ Backward/ of .xlsx
recordings with the three bipolar columns. Output mirrors that structure.
Then: build_dataset.py --source data/synthetic_nospec --tag nospec
"""
import os, glob, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import config as C

WIN, STRIDE, LATENT = 128, 32, 100
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Generator(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.fc = nn.Linear(LATENT, 256 * 16)
        self.net = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 4, 2, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 4, 2, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 4, 2, 1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, nc, 3, padding=1))

    def forward(self, z):
        return self.net(self.fc(z).view(-1, 256, 16))[:, :, :WIN]


class Critic(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(nc, 32, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Flatten(), nn.Linear(128 * 16, 1))

    def forward(self, x):
        return self.net(x)


def grad_penalty(C_, real, fake):
    a = torch.rand(real.size(0), 1, 1, device=dev)
    mix = (a * real + (1 - a) * fake).requires_grad_(True)
    g = torch.autograd.grad(C_(mix), mix, torch.ones(real.size(0), 1, device=dev),
                            create_graph=True, retain_graph=True)[0]
    return ((g.view(real.size(0), -1).norm(2, 1) - 1) ** 2).mean()


def windows(sig, w, s):
    return np.stack([sig[i:i + w] for i in range(0, len(sig) - w + 1, s)])


def overlap_add(win, s, w, nc):
    n = win.shape[0]; tot = (n - 1) * s + w
    sig = np.zeros((tot, nc)); cnt = np.zeros((tot, nc))
    for i in range(n):
        sig[i * s:i * s + w] += win[i]; cnt[i * s:i * s + w] += 1
    return sig / np.maximum(cnt, 1)


def process(fp, out_dir, lam, epochs, per_file):
    df = pd.read_excel(fp); df.columns = df.columns.str.strip()
    sig = df[C.SELECTED_CHANNELS].values.astype(np.float32)
    cols = C.SELECTED_CHANNELS; nc = len(cols)
    win = windows(sig, WIN, STRIDE)
    sc = StandardScaler(); flat = sc.fit_transform(win.reshape(-1, nc))
    win = np.transpose(flat.reshape(win.shape), (0, 2, 1))
    X = torch.tensor(win, dtype=torch.float32)
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X),
                                     batch_size=64, shuffle=True, drop_last=True)
    G, Cr = Generator(nc).to(dev), Critic(nc).to(dev)
    oG = torch.optim.Adam(G.parameters(), 1e-4, betas=(0.0, 0.9))
    oC = torch.optim.Adam(Cr.parameters(), 1e-4, betas=(0.0, 0.9))
    for ep in range(1, epochs + 1):
        for (rx,) in dl:
            rx = rx.to(dev); bs = rx.size(0)
            for _ in range(3):
                fx = G(torch.randn(bs, LATENT, device=dev))
                closs = -(Cr(rx).mean() - Cr(fx.detach()).mean()) + 10 * grad_penalty(Cr, rx, fx.detach())
                oC.zero_grad(); closs.backward(); oC.step()
            fx = G(torch.randn(bs, LATENT, device=dev))
            gloss = -Cr(fx).mean()
            if lam > 0:                       # disabled for this ablation (lam=0)
                sp = torch.mean(torch.abs(torch.abs(torch.fft.rfft(rx, dim=-1))
                                          - torch.abs(torch.fft.rfft(fx, dim=-1))))
                gloss = gloss + lam * sp
            oG.zero_grad(); gloss.backward(); oG.step()
    base = os.path.splitext(os.path.basename(fp))[0]
    os.makedirs(out_dir, exist_ok=True)
    for k in range(1, per_file + 1):
        with torch.no_grad():
            fake = G(torch.randn(len(win), LATENT, device=dev)).cpu().numpy()
        fake = np.transpose(fake, (0, 2, 1)).reshape(-1, nc)
        fake = sc.inverse_transform(fake).reshape(len(win), WIN, nc)
        rec = overlap_add(fake, STRIDE, WIN, nc)
        pd.DataFrame(rec, columns=cols).to_excel(
            os.path.join(out_dir, f"{base}_nospec_{k}.xlsx"), index=False)
    print(f"  [done] {base}: {per_file} files")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=os.path.join(C.DATA_DIR, "synthetic_nospec"))
    ap.add_argument("--per-file", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lambda-spec", type=float, default=0.0)   # 0 = no spectral loss
    a = ap.parse_args()
    t0 = time.time()
    for cls in ["Right", "Left", "Forward", "Backward"]:
        for fp in sorted(glob.glob(os.path.join(a.input, cls, "*.xlsx"))):
            process(fp, os.path.join(a.output, cls), a.lambda_spec, a.epochs, a.per_file)
    print(f"[all done] {(time.time()-t0)/60:.1f} min -> {a.output}")


if __name__ == "__main__":
    main()
