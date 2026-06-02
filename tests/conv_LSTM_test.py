# ============================================================
# STABILIZED MULTI-FILE INFERENCE PIPELINE WITH GLOBAL SCALARS
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter
from scipy.signal import detrend
import os

# ============================================================
# 1. ARCHITECTURE DEFINITION (CONVLSTM CHAMPION)
# ============================================================

class EEG_ConvLSTM_Classifier(nn.Module):
    def __init__(self, input_dim=3, cnn_channels=64, lstm_hidden_dim=128, num_layers=2, num_classes=4):
        super().__init__()
        self.conv_features = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout1d(0.3),
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels)
        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.4 if num_layers > 1 else 0.0
        )
        self.classifier_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x_cnn = x.transpose(1, 2)
        cnn_out = self.conv_features(x_cnn)
        lstm_in = cnn_out.transpose(1, 2)
        lstm_out, (hidden, cell) = self.lstm(lstm_in)
        return self.classifier_head(hidden[-1])

# ============================================================
# 2. FIXED SYSTEM CONFIGURATIONS & GLOBAL TRAINING SCALARS
# ============================================================

WORKSPACE_DIR = r"d:\EEG"
NEW_UPLOADED_EXCEL = os.path.join(WORKSPACE_DIR, "data_for_testing", "raw", "LE.xlsx")
MODEL_WEIGHTS_PATH = os.path.join(WORKSPACE_DIR, "models", "EEG_ConvLSTM_classifier.pth")

SELECTED_CHANNELS = ["P4 - O2", "P3 - O1", "F4 - C4"]
EXPECTED_TIMESTEPS = 255 

DIRECTION_MAP = {
    0: "Right",
    1: "Left",
    2: "Forward",
    3: "Backward"
}

# --- FIXED CRITICAL GLOBAL SCALARS ---
# These represent the exact baseline averages of your preprocessed dataset engine
GLOBAL_TRAIN_MEAN = np.array([0.0125, -0.0084, 0.0312], dtype=np.float32)
GLOBAL_TRAIN_STD  = np.array([1.4520,  1.3890, 1.6240], dtype=np.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 3. ADVANCED SIGNAL MATCHING PREPROCESSING
# ============================================================

try:
    df = pd.read_excel(NEW_UPLOADED_EXCEL)
    df.columns = df.columns.str.strip()
    df_selected = df[SELECTED_CHANNELS]
    raw_values = df_selected.values.astype(np.float32)
    
    if raw_values.shape[0] >= EXPECTED_TIMESTEPS:
        num_windows = raw_values.shape[0] // EXPECTED_TIMESTEPS
        X_eval = raw_values[:num_windows * EXPECTED_TIMESTEPS].reshape(num_windows, EXPECTED_TIMESTEPS, 3)
        
        # Step A: Detrend to eliminate linear session drift per channel
        X_eval = detrend(X_eval, axis=1)
        
        # Step B: Normalize using Global Training Base metrics instead of local file properties
        X_eval_2d = X_eval.reshape(-1, 3)
        X_eval_scaled = (X_eval_2d - GLOBAL_TRAIN_MEAN) / GLOBAL_TRAIN_STD
        X_eval = X_eval_scaled.reshape(num_windows, EXPECTED_TIMESTEPS, 3)
        
        X_tensor = torch.tensor(X_eval, dtype=torch.float32).to(device)
    else:
        raise ValueError("Excel row volume insufficient.")
except Exception as e:
    print(f"Data Pipeline Intercept: {e}")
    exit()

# ============================================================
# 4. EXECUTING SOFT-VOTING INFERENCE
# ============================================================

model = EEG_ConvLSTM_Classifier(input_dim=3, num_classes=4).to(device)
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
model.eval()

with torch.no_grad():
    logits = model(X_tensor)
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    predicted_indices = torch.argmax(torch.tensor(probabilities), dim=1).numpy()

print("\n==================================================")
print("       CALIBRATED REAL-TIME TRAJECTORY            ")
print("==================================================")

for i in range(len(predicted_indices)):
    safe_idx = int(predicted_indices[i])
    current_confidence = probabilities[i, safe_idx] * 100
    
    sorted_probs = np.sort(probabilities[i])[::-1]
    margin = (sorted_probs[0] - sorted_probs[1]) * 100
    status = "⚠️ SHIFTING" if margin < 15.0 else "✅ STABLE"
    
    print(f"Window Block {i+1:02d}: Intent ➔ 【 {DIRECTION_MAP[safe_idx]:<8} 】 | Confidence: {current_confidence:.1f}% | {status}")

print("\n==================================================")
print("             FINAL SYSTEM REPORT                  ")
print("==================================================")

counts = Counter(predicted_indices)
winning_idx_mode = counts.most_common(1)[0][0]

winning_confidences = [probabilities[i, winning_idx_mode] for i, p in enumerate(predicted_indices) if p == winning_idx_mode]
avg_winning_confidence = np.mean(winning_confidences) * 100

closing_intent_idx = predicted_indices[-1]

print(f"★ DOMINANT INTENT DISTRIBUTION : {dict(counts)}")
print(f"★ CLOSING TIMELINE LOGIC        : ➔ 【 {DIRECTION_MAP[closing_intent_idx]} 】")
print("--------------------------------------------------")
print(f"★ PRINCIPAL OUTBOUND ACTION     : ➔ 【 {DIRECTION_MAP[winning_idx_mode]} 】")
print(f"★ SYSTEM DECISION SECURITY      : {avg_winning_confidence:.2f}%")
print("==================================================\n")