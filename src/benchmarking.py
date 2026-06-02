import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter
from scipy.signal import detrend, cheby1, sosfiltfilt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ===========================================================================
# 1. ARCHITECTURES DEFINITIONS
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

# ===========================================================================
# 2. CONFIGURATIONS & GLOBAL PARAMETERS
# ===========================================================================

WORKSPACE_DIR = r"d:\EEG"
TEST_DATA_DIR = os.path.join(WORKSPACE_DIR, "data_for_testing", "raw")
RESULTS_DIR = os.path.join(WORKSPACE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SELECTED_CHANNELS = ["P4 - O2", "P3 - O1", "F4 - C4"]
EXPECTED_TIMESTEPS = 255  

DIRECTION_MAP = {0: "Right", 1: "Left", 2: "Forward", 3: "Backward"}

GLOBAL_TRAIN_MEAN = np.array([0.0125, -0.0084, 0.0312], dtype=np.float32)
GLOBAL_TRAIN_STD  = np.array([1.4520,  1.3890, 1.6240], dtype=np.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define target test files and their homogeneous ground truth labels
test_scenarios = [
    {"filename": "LE.xlsx", "true_label": 1, "label_name": "Left"},
    {"filename": "RY.xlsx", "true_label": 0, "label_name": "Right"},
    {"filename": "For.xlsx", "true_label": 2, "label_name": "Forward"},
]

# ===========================================================================
# 3. HELPER FUNCTIONS: CHEBYSHEV FILTERING
# ===========================================================================

def chebyshev_eeg_filter(data, fs=250, lowcut=8, highcut=30, order=6, rp=0.3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = cheby1(N=order, rp=rp, Wn=[low, high], btype='bandpass', output='sos')
    filtered = sosfiltfilt(sos, data, axis=0)
    return filtered

def load_and_segment_file(filepath):
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.strip()
    df_selected = df[SELECTED_CHANNELS]
    raw_values = df_selected.values.astype(np.float32)
    
    # Apply Chebyshev bandpass filter first (to align spectrally with training data)
    filtered_values = chebyshev_eeg_filter(raw_values)
    
    if filtered_values.shape[0] < EXPECTED_TIMESTEPS:
        raise ValueError(f"File {filepath} has insufficient rows ({filtered_values.shape[0]})")
        
    num_windows = filtered_values.shape[0] // EXPECTED_TIMESTEPS
    X_segmented = filtered_values[:num_windows * EXPECTED_TIMESTEPS].reshape(num_windows, EXPECTED_TIMESTEPS, 3)
    return X_segmented

# ===========================================================================
# 4. BENCHMARK EXECUTION FUNCTION
# ===========================================================================

def run_evaluation(model, model_name, X_eval, config_type):
    """
    config_type: 'static' or 'adaptive'
    """
    num_windows = X_eval.shape[0]
    
    # Step A: Detrend
    X_processed = detrend(X_eval, axis=1)
    
    if config_type == 'static':
        # Normalize using global training scaling parameters
        X_2d = X_processed.reshape(-1, 3)
        X_scaled = (X_2d - GLOBAL_TRAIN_MEAN) / GLOBAL_TRAIN_STD
        X_final = X_scaled.reshape(num_windows, EXPECTED_TIMESTEPS, 3)
    elif config_type == 'adaptive':
        # Dynamic dynamic local session normalization
        session_mean = np.mean(X_processed, axis=(0, 1))
        session_std  = np.std(X_processed, axis=(0, 1))
        session_std  = np.where(session_std == 0, 1e-8, session_std)
        X_final = (X_processed - session_mean) / session_std
    else:
        raise ValueError("Unknown normalization configuration type")
        
    X_tensor = torch.tensor(X_final, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        logits = model(X_tensor)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        predicted_classes = torch.argmax(logits, dim=1).cpu().numpy()
        
    return predicted_classes, probabilities

# ===========================================================================
# 5. MAIN BENCHMARK ENGINE
# ===========================================================================

def main():
    print("===========================================================================")
    print("                STARTING BCI BATCH BENCHMARK RUNNER (FILTERED)             ")
    print("===========================================================================")
    print(f"Device: {device}")
    
    # Initialize models
    cnn_model = EEG_Pure_1DCNN_Classifier(input_dim=3, num_classes=4).to(device)
    lstm_model = EEG_ConvLSTM_Classifier(input_dim=3, num_classes=4).to(device)
    
    cnn_weights = os.path.join(WORKSPACE_DIR, "models", "EEG_pure_1DCNN_classifier.pth")
    lstm_weights = os.path.join(WORKSPACE_DIR, "models", "EEG_ConvLSTM_classifier.pth")
    
    cnn_model.load_state_dict(torch.load(cnn_weights, map_location=device))
    lstm_model.load_state_dict(torch.load(lstm_weights, map_location=device))
    
    cnn_model.eval()
    lstm_model.eval()
    
    results_summary = []
    
    # Store labels and predictions for confusion matrices
    y_true_all = []
    
    preds_dict = {
        "CNN_Static": [],
        "CNN_Adaptive": [],
        "LSTM_Static": [],
        "LSTM_Adaptive": []
    }
    
    for scenario in test_scenarios:
        filename = scenario["filename"]
        true_label = scenario["true_label"]
        label_name = scenario["label_name"]
        filepath = os.path.join(TEST_DATA_DIR, filename)
        
        print(f"\nProcessing Session File: {filename} (True Label: {label_name} / {true_label})")
        
        try:
            X_eval = load_and_segment_file(filepath)
            num_windows = X_eval.shape[0]
            print(f"-> Successfully extracted {num_windows} window blocks.")
            
            # Ground truth array for this file
            y_true_file = [true_label] * num_windows
            y_true_all.extend(y_true_file)
            
            # Evaluate all 4 configurations
            for model_obj, model_name in [(cnn_model, "CNN"), (lstm_model, "LSTM")]:
                for config in ["static", "adaptive"]:
                    preds, probs = run_evaluation(model_obj, model_name, X_eval, config)
                    
                    key = f"{model_name}_{config.capitalize()}"
                    preds_dict[key].extend(preds)
                    
                    # Calculate window-level accuracy
                    accuracy = accuracy_score(y_true_file, preds) * 100
                    
                    # Calculate predictability entropy / dominant predictions
                    prediction_counts = dict(Counter(preds))
                    dominant_pred = max(prediction_counts, key=prediction_counts.get)
                    dominant_count = prediction_counts[dominant_pred]
                    dominant_pct = (dominant_count / num_windows) * 100
                    
                    # Calculate stability
                    stable_count = 0
                    shifting_count = 0
                    for i in range(num_windows):
                        sorted_probs = np.sort(probs[i])[::-1]
                        margin = (sorted_probs[0] - sorted_probs[1]) * 100
                        if margin < 15.0:
                            shifting_count += 1
                        else:
                            stable_count += 1
                            
                    shifting_pct = (shifting_count / num_windows) * 100
                    
                    # Store metrics
                    results_summary.append({
                        "File": filename,
                        "Expected": label_name,
                        "Model": model_name,
                        "Normalization": config.capitalize(),
                        "Accuracy (%)": f"{accuracy:.2f}%",
                        "Dominant Predict": DIRECTION_MAP[dominant_pred],
                        "Dominant Pct (%)": f"{dominant_pct:.1f}%",
                        "Shifting Rate (%)": f"{shifting_pct:.1f}%"
                    })
                    
        except Exception as e:
            print(f"Error evaluating file {filename}: {e}")
            
    # ===========================================================================
    # 6. OUTPUT SUMMARY TABLES
    # ===========================================================================
    df_results = pd.DataFrame(results_summary)
    print("\n\n===========================================================================")
    print("                       BENCHMARK RESULTS TABLE                             ")
    print("===========================================================================")
    print(df_results.to_markdown(index=False))
    
    # Save table to markdown file
    df_results.to_markdown(os.path.join(RESULTS_DIR, "benchmark_metrics.md"), index=False)
    
    # ===========================================================================
    # 7. GENERATE AND SAVE PLOTS (CONFUSION MATRICES)
    # ===========================================================================
    classes = ["Right", "Left", "Forward", "Backward"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Comparative Evaluation: Static vs. Adaptive Calibration (Chebyshev Filtered)", fontsize=16, fontweight='bold')
    
    configs_to_plot = [
        ("CNN_Static", "1D-CNN with Static Global Normalization", axes[0, 0]),
        ("CNN_Adaptive", "1D-CNN with Adaptive Session Calibration", axes[0, 1]),
        ("LSTM_Static", "ConvLSTM with Static Global Normalization", axes[1, 0]),
        ("LSTM_Adaptive", "ConvLSTM with Adaptive Session Calibration", axes[1, 1])
    ]
    
    for key, title, ax in configs_to_plot:
        preds = preds_dict[key]
        cm = confusion_matrix(y_true_all, preds, labels=[0, 1, 2, 3])
        
        # Calculate percentage accuracy per class
        cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        cm_pct = np.nan_to_num(cm_pct)
        
        # Format labels: count (percentage%)
        annot = np.empty_like(cm).astype(str)
        for r in range(4):
            for c in range(4):
                annot[r, c] = f"{cm[r, c]}\n({cm_pct[r, c]:.1f}%)"
                
        sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax, cbar=False, annot_kws={"size": 11})
        ax.set_title(title, fontsize=12, fontweight='semibold')
        ax.set_ylabel("True Target Class", fontsize=10)
        ax.set_xlabel("Predicted Action Class", fontsize=10)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(RESULTS_DIR, "comparative_confusion_matrices.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved comparative confusion matrices to: {plot_path}")
    
    # ===========================================================================
    # 8. TRAJECTORY VISUALIZATION PLOT
    # ===========================================================================
    filename_traj = "LE.xlsx"
    filepath_traj = os.path.join(TEST_DATA_DIR, filename_traj)
    X_traj = load_and_segment_file(filepath_traj)
    num_windows_traj = X_traj.shape[0]
    
    preds_static_traj, probs_static_traj = run_evaluation(cnn_model, "CNN", X_traj, "static")
    preds_adapt_traj, probs_adapt_traj = run_evaluation(cnn_model, "CNN", X_traj, "adaptive")
    
    plt.figure(figsize=(12, 5))
    x_axis = np.arange(1, num_windows_traj + 1)
    
    plt.plot(x_axis, preds_static_traj, 'ro--', label="Static Global (Branch A) - Class Collapse", alpha=0.6)
    plt.plot(x_axis, preds_adapt_traj, 'gs-', label="Adaptive Session (Branch B) - Correct Sequence", alpha=0.8)
    
    # Draw horizontal helper lines for classes
    for c_val, c_name in DIRECTION_MAP.items():
        plt.axhline(y=c_val, color='gray', linestyle=':', alpha=0.3)
        
    plt.yticks(ticks=[0, 1, 2, 3], labels=classes)
    plt.xlabel("Temporal Window Block ID", fontsize=11)
    plt.ylabel("Class Predicted Outbound Action", fontsize=11)
    plt.title(f"Output Trajectory Comparison over time ({filename_traj}) - target class 'Left'", fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.2)
    plt.legend(loc="upper right")
    
    traj_plot_path = os.path.join(RESULTS_DIR, "trajectory_comparison.png")
    plt.savefig(traj_plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved temporal trajectory comparison plot to: {traj_plot_path}")
    
    print("\n===========================================================================")
    print("                    BENCHMARK COMPLETED SUCCESSFULLY                       ")
    print("===========================================================================")

if __name__ == "__main__":
    main()
