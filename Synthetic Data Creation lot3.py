# ============================================================
# EEG WGAN-GP SYNTHETIC DATASET GENERATOR
# AUTOMATIC GOOGLE DRIVE FOLDER PROCESSING
# ============================================================

# Folder Structure Expected:
#
# BASE_FOLDER/
# ├── Subject 1/
# │   ├── Right/
# │   ├── Left/
# │   ├── Forward/
# │   └── Backward/
# └── Subject 2/
#     ├── Right/
#     ├── Left/
#     ├── Forward/
#     └── Backward/
#
# Each movement folder contains 3 .xlsx EEG files
#
# Output:
#
# Synthetic Dataset/
# ├── Subject 1/
# │   ├── Right/
# │   ├── Left/
# │   ├── Forward/
# │   └── Backward/
# └── Subject 2/
#     ├── Right/
#     ├── Left/
#     ├── Forward/
#     └── Backward/
#
# ============================================================

!pip install openpyxl -q
!pip install tenacity -q

import os
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from google.colab import drive
from tenacity import retry, stop_after_attempt, wait_fixed, wait_random, retry_if_exception_type

# ============================================================
# MOUNT DRIVE
# ============================================================

drive.mount('/content/drive')

# ============================================================
# IMPORTANT
# ============================================================

# CHANGE THIS PATH
BASE_PATH = "/content/drive/My Drive/Human Computer Interface (HCI)/Original Dataset"

# Output folder
OUTPUT_BASE = "/content/drive/My Drive/Human Computer Interface (HCI)/Synthetic Data"

# ============================================================
# SETTINGS
# ============================================================

WINDOW_SIZE = 128
STRIDE = 32

LATENT_DIM = 100

BATCH_SIZE = 64
NUM_EPOCHS = 150

LR = 1e-4

N_CRITIC = 3
LAMBDA_GP = 10

N_SYNTHETIC = 150

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {device}")

# ============================================================
# EEG WINDOWING
# ============================================================

def create_windows(signal, win_size, stride):

    windows = []

    for start in range(0, len(signal) - win_size + 1, stride):

        segment = signal[start:start + win_size]

        windows.append(segment)

    return np.stack(windows)

# ============================================================
# GENERATOR
# ============================================================

class Generator(nn.Module):

    def __init__(self, n_channels):

        super().__init__()

        self.fc = nn.Linear(LATENT_DIM, 256 * 16)

        self.net = nn.Sequential(

            nn.ConvTranspose1d(256, 128, 4, 2, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.ConvTranspose1d(128, 64, 4, 2, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.ConvTranspose1d(64, 32, 4, 2, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, n_channels, 3, padding=1)
        )

    def forward(self, z):

        x = self.fc(z)

        x = x.view(-1, 256, 16)

        x = self.net(x)

        return x[:, :, :WINDOW_SIZE]

# ============================================================
# CRITIC
# ============================================================

class Critic(nn.Module):

    def __init__(self, n_channels):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv1d(n_channels, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv1d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv1d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Flatten(),

            nn.Linear(128 * 16, 1)
        )

    def forward(self, x):

        return self.net(x)

# ============================================================
# GRADIENT PENALTY
# ============================================================

def gradient_penalty(C, real, fake):

    bs = real.size(0)

    alpha = torch.rand(bs, 1, 1, device=device)

    interpolated = alpha * real + (1 - alpha) * fake

    interpolated.requires_grad_(True)

    mixed_scores = C(interpolated)

    gradient = torch.autograd.grad(
        inputs=interpolated,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient = gradient.view(bs, -1)

    gp = ((gradient.norm(2, dim=1) - 1) ** 2).mean()

    return gp

# ============================================================
# OVERLAP ADD RECONSTRUCTION
# ============================================================

def overlap_add(windows, stride, window_size, n_channels):

    n_windows = windows.shape[0]

    total_len = (n_windows - 1) * stride + window_size

    signal = np.zeros((total_len, n_channels))

    counter = np.zeros((total_len, n_channels))

    for i in range(n_windows):

        start = i * stride

        end = start + window_size

        signal[start:end] += windows[i]

        counter[start:end] += 1

    signal /= counter

    return signal

# ============================================================
# TRAIN + GENERATE FUNCTION
# ============================================================

def process_file(file_path, output_folder, subject_name):

    print("\n===================================================")
    print(f"Processing: {file_path}")
    print("===================================================")

    df = pd.read_excel(file_path, engine='openpyxl')

    signal = df.values.astype(np.float32)

    col_names = list(df.columns)

    total_samples = signal.shape[0]

    n_channels = signal.shape[1]

    # ========================================================
    # CREATE WINDOWS
    # ========================================================

    windows = create_windows(
        signal,
        WINDOW_SIZE,
        STRIDE
    )

    # normalize
    temp = windows.reshape(-1, n_channels)

    scaler = StandardScaler()

    temp = scaler.fit_transform(temp)

    windows = temp.reshape(windows.shape)

    # [batch, channels, time]
    windows = np.transpose(windows, (0, 2, 1))

    X = torch.tensor(windows, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    # ========================================================
    # MODELS
    # ========================================================

    G = Generator(n_channels).to(device)

    C = Critic(n_channels).to(device)

    opt_G = optim.Adam(
        G.parameters(),
        lr=LR,
        betas=(0.0, 0.9)
    )

    opt_C = optim.Adam(
        C.parameters(),
        lr=LR,
        betas=(0.0, 0.9)
    )

    # ========================================================
    # TRAINING
    # ========================================================

    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):

        for (real_x,) in loader:

            real_x = real_x.to(device)

            bs = real_x.size(0)

            # =================================================
            # TRAIN CRITIC
            # =================================================

            for _ in range(N_CRITIC):

                noise = torch.randn(
                    bs,
                    LATENT_DIM,
                    device=device
                )

                fake_x = G(noise)

                critic_real = C(real_x).mean()

                critic_fake = C(fake_x.detach()).mean()

                gp = gradient_penalty(
                    C,
                    real_x,
                    fake_x.detach()
                )

                c_loss = (
                    -(critic_real - critic_fake)
                    + LAMBDA_GP * gp
                )

                opt_C.zero_grad()

                c_loss.backward()

                opt_C.step()

            # =================================================
            # TRAIN GENERATOR
            # =================================================

            noise = torch.randn(
                bs,
                LATENT_DIM,
                device=device
            )

            fake_x = G(noise)

            adv_loss = -C(fake_x).mean()

            # spectral loss
            fft_real = torch.fft.rfft(real_x, dim=-1)

            fft_fake = torch.fft.rfft(fake_x, dim=-1)

            spectral_loss = torch.mean(
                torch.abs(
                    torch.abs(fft_real)
                    - torch.abs(fft_fake)
                )
            )

            g_loss = adv_loss + 0.1 * spectral_loss

            opt_G.zero_grad()

            g_loss.backward()

            opt_G.step()

        if epoch % 50 == 0:

            elapsed = (time.time() - start_time) / 60

            print(
                f"Epoch {epoch}/{NUM_EPOCHS} | "
                f"W-dist={(critic_real - critic_fake).item():+.4f} | "
                f"G={g_loss.item():.4f} | "
                f"C={c_loss.item():.4f} | "
                f"{elapsed:.1f} min"
            )

    print("\nTraining Completed!")

    # ========================================================
    # GENERATE SYNTHETIC FILES
    # ========================================================

    base_filename = os.path.splitext(
        os.path.basename(file_path)
    )[0]

    n_windows = len(windows)

    os.makedirs(output_folder, exist_ok=True)

    print(f"\nGenerating {N_SYNTHETIC} synthetic files...")

    # Define retry function for to_excel
    @retry(
        stop=stop_after_attempt(5), # Try up to 5 times
        wait=wait_fixed(2) + wait_random(0, 3), # Wait 2-5 seconds between retries
        retry=retry_if_exception_type(OSError), # Only retry on OSError
        reraise=True # Re-raise the exception if all retries fail
    )
    def save_to_excel_with_retry(df_to_save, path, columns):
        pd.DataFrame(
            df_to_save,
            columns=columns
        ).to_excel(
            path,
            index=False,
            engine='openpyxl'
        )

    for synth_num in range(1, N_SYNTHETIC + 1):

        with torch.no_grad():

            noise = torch.randn(
                n_windows,
                LATENT_DIM,
                device=device
            )

            fake = G(noise).cpu().numpy()

        fake = np.transpose(fake, (0, 2, 1))

        fake_reshaped = fake.reshape(-1, n_channels)

        fake_reshaped = scaler.inverse_transform(fake_reshaped)

        fake = fake_reshaped.reshape(fake.shape)

        fake_signal = overlap_add(
            fake,
            STRIDE,
            WINDOW_SIZE,
            n_channels
        )

        # ====================================================
        # SAVE NAME FORMAT
        # ====================================================

        save_name = (
            f"{base_filename}_"
            f"{subject_name}_"
            f"synthetic_{synth_num}.xlsx"
        )

        save_path = os.path.join(
            output_folder,
            save_name
        )

        try:
            save_to_excel_with_retry(
                fake_signal,
                save_path,
                col_names
            )
        except OSError as e:
            print(f"Failed to save {save_path} after multiple retries: {e}")
            raise # Re-raise to stop processing if persistent failure

        if synth_num % 25 == 0:

            print(f"Saved {synth_num}/{N_SYNTHETIC}")

    print(f"\nFinished: {file_path}")

# ============================================================
# PROCESS ALL SUBJECTS + MOVEMENTS
# ============================================================

subjects = ["Subject 1", "Subject 2"]

movements = ["Right", "Left", "Forward", "Backward"]

for subject in subjects:

    for movement in movements:

        input_folder = os.path.join(
            BASE_PATH,
            subject,
            movement
        )

        output_folder = os.path.join(
            OUTPUT_BASE,
            subject,
            movement
        )

        if not os.path.exists(input_folder):

            print(f"Missing folder: {input_folder}")

            continue

        files_list = sorted([
            f for f in os.listdir(input_folder)
            if f.endswith(".xlsx")
        ])

        print("\n===================================================")
        print(f"Subject: {subject}")
        print(f"Movement: {movement}")
        print(f"Files Found: {len(files_list)}")
        print("===================================================")

        for file_name in files_list:

            full_path = os.path.join(
                input_folder,
                file_name
            )

            process_file(
                full_path,
                output_folder,
                subject.replace(" ", "_lot2")
            )

print("\n===================================================")
print("ALL SYNTHETIC DATA GENERATION COMPLETED")
print("===================================================")