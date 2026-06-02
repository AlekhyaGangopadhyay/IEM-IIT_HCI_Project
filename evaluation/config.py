"""
Central configuration for the robustness-evaluation suite.

All paths are resolved relative to this file so the suite is portable
(no hard-coded d:\\EEG paths). Override any path with an environment
variable of the same name if your layout differs.
"""
import os

# ---- repo layout ---------------------------------------------------------
EVAL_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(EVAL_DIR)                       # IEM-IIT_HCI_Project
MODELS_DIR = os.environ.get("EEG_MODELS_DIR", os.path.join(REPO_ROOT, "models"))
RAW_TEST_DIR = os.environ.get("EEG_RAW_TEST_DIR",
                              os.path.join(REPO_ROOT, "data_for_testing", "raw"))

# Offline processed dataset (the Kaggle .npy files). Place them here, or set
# EEG_DATA_DIR to wherever X_train_500.npy etc. live.
DATA_DIR    = os.environ.get("EEG_DATA_DIR", os.path.join(EVAL_DIR, "data"))
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- trained decoder weights --------------------------------------------
CNN_WEIGHTS  = os.path.join(MODELS_DIR, "EEG_pure_1DCNN_classifier.pth")
LSTM_WEIGHTS = os.path.join(MODELS_DIR, "EEG_ConvLSTM_classifier.pth")

# ---- signal / dataset constants (must match the trained pipeline) -------
FS = 250
LOWCUT, HIGHCUT, ORDER, RP = 8, 30, 6, 0.3
SELECTED_CHANNELS = ["P4 - O2", "P3 - O1", "F4 - C4"]
N_CHANNELS = 3
N_CLASSES = 4
TRAIN_SEQ_LEN = 256          # offline sequence length used in training
XSESSION_TIMESTEPS = 255     # window length used by benchmarking.py
DIRECTION_MAP = {0: "Right", 1: "Left", 2: "Forward", 3: "Backward"}

# Global training scaler stats (from src/benchmarking.py). Replace with the
# exact StandardScaler mean_/scale_ from filesequenicng.py if you have them.
GLOBAL_TRAIN_MEAN = [0.0125, -0.0084, 0.0312]
GLOBAL_TRAIN_STD  = [1.4520,  1.3890,  1.6240]

# Cross-session test files and their homogeneous ground-truth labels.
TEST_SCENARIOS = [
    {"filename": "LE.xlsx",  "true_label": 1, "label_name": "Left"},
    {"filename": "RY.xlsx",  "true_label": 0, "label_name": "Right"},
    {"filename": "For.xlsx", "true_label": 2, "label_name": "Forward"},
]

# Offline .npy filenames expected inside DATA_DIR.
NPY = {
    "X_train": "X_train_500.npy", "X_test": "X_test_500.npy",
    "y_train": "y_train_cls_500.npy", "y_test": "y_test_cls_500.npy",
}
