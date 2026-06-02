# ===========================================================================
# ADAPTIVE LIVE-CALIBRATION PIPELINE: PURE 1D-CNN RE-BALANCER
# ===========================================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter
from scipy.signal import detrend
import os

# ===========================================================================
# 1. MODEL ARCHITECTURE DEFINITION
# ===========================================================================

class EEG_Pure_1DCNN_Classifier(nn.Module):
    def __init__(self, input_dim=3, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.3),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),
            nn.Dropout1d(0.3),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        feats = self.features(x).squeeze(-1)
        return self.classifier_head(feats)

# ===========================================================================
# 2. CONFIGURATIONS
# ===========================================================================

# Local workspace paths
WORKSPACE_DIR = r"d:\EEG"
NEW_UPLOADED_EXCEL = os.path.join(WORKSPACE_DIR, "data_for_testing", "raw", "LE.xlsx")
MODEL_WEIGHTS_PATH = os.path.join(WORKSPACE_DIR, "models", "EEG_pure_1DCNN_classifier.pth")

SELECTED_CHANNELS = ["P4 - O2", "P3 - O1", "F4 - C4"]
EXPECTED_TIMESTEPS = 255  

DIRECTION_MAP = {
    0: "RIGHT", 
    1: "LEFT", 
    2: "FORWARD", 
    3: "BACKWARD"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================================================================
# 3. THE FIX: ACTIVE SESSION CALIBRATION ENGINE
# ===========================================================================

try:
    df = pd.read_excel(NEW_UPLOADED_EXCEL)
    df.columns = df.columns.str.strip()
    df_selected = df[SELECTED_CHANNELS]
    raw_values = df_selected.values.astype(np.float32)
    
    if raw_values.shape[0] >= EXPECTED_TIMESTEPS:
        num_windows = raw_values.shape[0] // EXPECTED_TIMESTEPS
        X_eval = raw_values[:num_windows * EXPECTED_TIMESTEPS].reshape(num_windows, EXPECTED_TIMESTEPS, 3)
        
        # Step A: Remove linear trend drift from this recording session
        X_eval = detrend(X_eval, axis=1)
        
        # Step B: FIX - Compute dynamic adaptive calibration scalars for THIS specific file session
        # This stops the background impedance changes from forcing a 'FORWARD' collapse
        session_mean = np.mean(X_eval, axis=(0, 1))
        session_std  = np.std(X_eval, axis=(0, 1))
        session_std  = np.where(session_std == 0, 1e-8, session_std)
        
        # Step C: Normalize file using its own calibrated scale boundaries
        X_eval_calibrated = (X_eval - session_mean) / session_std
        X_tensor = torch.tensor(X_eval_calibrated, dtype=torch.float32).to(device)
        
        print("==================================================")
        print("         ADAPTIVE CALIBRATION LOGS                ")
        print("==================================================")
        print(f"✔ Active Session Mean across channels: {session_mean}")
        print(f"✔ Active Session Standard Deviation   : {session_std}")
        print(f"✔ Successfully balanced {num_windows} evaluation windows.")
    else:
        raise ValueError(f"Excel row depth must be at least {EXPECTED_TIMESTEPS} rows.")
except Exception as e:
    print(f"❌ Calibration Pipeline Error: {e}")
    exit()

# ===========================================================================
# 4. RUN PROBABILITY-WEIGHTED INFERENCE
# ===========================================================================

model = EEG_Pure_1DCNN_Classifier(input_dim=3, num_classes=4).to(device)
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
model.eval()

with torch.no_grad():
    logits = model(X_tensor)
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    predicted_indices = torch.argmax(logits, dim=1).cpu().numpy()

# ===========================================================================
# 5. DISPATCH TIMELINE TRAJECTORY
# ===========================================================================

print("\n==================================================")
print("       CALIBRATED REAL-TIME TRAJECTORY            ")
print("==================================================")

for i in range(len(predicted_indices)):
    safe_idx = int(predicted_indices[i])
    current_confidence = probabilities[i, safe_idx] * 100
    
    # Calculate how stable the decision is compared to the runner-up class
    sorted_probs = np.sort(probabilities[i])[::-1]
    margin = (sorted_probs[0] - sorted_probs[1]) * 100
    status = "⚠️ SHIFTING" if margin < 15.0 else "✅ STABLE"
    
    print(f"Window Block {i+1:02d}: Intent ➔ 【 {DIRECTION_MAP[safe_idx]:<8} 】 | Confidence: {current_confidence:.1f}% | {status}")

# ===========================================================================
# 6. MODE-VOTED TRAJECTORY DISPATCH (THE CHAMPION RESOLVER)
# ===========================================================================

print("\n==================================================")
print("             FINAL SYSTEM REPORT                  ")
print("==================================================")

# 1. Calculate the standard mathematical mode (The dominant active state)
counts = Counter(predicted_indices)
winning_idx_mode = counts.most_common(1)[0][0]

# 2. Calculate the average confidence specifically for that winning intent
winning_confidences = [probabilities[i, winning_idx_mode] for i, p in enumerate(predicted_indices) if p == winning_idx_mode]
avg_winning_confidence = np.mean(winning_confidences) * 100

# 3. Check the closing window state (Real-world system execution response)
closing_intent_idx = predicted_indices[-1]

print(f"★ DOMINANT INTENT DISTRIBUTION : {dict(counts)}")
print(f"★ CLOSING TIMELINE LOGIC        : ➔ 【 {DIRECTION_MAP[closing_intent_idx]} 】")
print("--------------------------------------------------")
print(f"★ PRINCIPAL OUTBOUND ACTION     : ➔ 【 {DIRECTION_MAP[winning_idx_mode]} 】")
print(f"★ SYSTEM DECISION SECURITY      : {avg_winning_confidence:.2f}%")
print("==================================================\n")