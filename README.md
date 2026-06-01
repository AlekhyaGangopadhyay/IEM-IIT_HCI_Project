# EEG Direction Classification using WGAN-GP and Spatial-Temporal Deep Networks

A deep learning framework for classifying directional movement intent from non-invasive multi-channel EEG signals using Chebyshev spectral filtering, synthetic data augmentation via Wasserstein GAN with Gradient Penalty (WGAN-GP), and deep temporal/convolutional sequence models (LSTM, 1D-CNN, and ConvLSTM).

> **Status:** Active research project. Core preprocessing pipelines, generative models, and spatial-temporal classifiers are implemented, validated, and packaged with adaptive real-time inference calibration.

---

## Table of Contents
- [Pipeline Architecture](#pipeline-architecture)
- [Raw Dataset and Preprocessing](#raw-dataset-and-preprocessing)
- [Generative Augmentation: WGAN-GP](#generative-augmentation-wgan-gp)
- [Dataset Sequencing and Splitting](#dataset-sequencing-and-splitting)
- [Classifier Architectures](#classifier-architectures)
- [Adaptive Calibration and Simulated Inference](#adaptive-calibration-and-simulated-inference)
- [Experimental Results](#experimental-results)
- [Technical Stack](#technical-stack)
- [Open Research Directions](#open-research-directions)

---

## Pipeline Architecture

The framework follows two distinct flows: an **offline training pipeline** (raw data → augmentation → filtering → sequencing → model fitting) and a **live inference pipeline** (new recording → calibration → classification). Note that generative augmentation runs first on the *raw* dataset, and Chebyshev filtering is then applied to the synthetic output — i.e. the order is **WGAN-GP → Chebyshev**, not the reverse.

### Offline Training Pipeline

```
[ Raw Excel EEG Recordings ]
               │
               ▼
┌──────────────────────────────┐
│  WGAN-GP Generative Stage    │ ──► Trains per-file on raw (all-channel) signals and
│                              │     synthesizes realistic EEG trials for augmentation
└──────────────────────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Chebyshev Bandpass Filter    │ ──► Applied to synthetic output; isolates the
│                              │     mu/beta sensorimotor band (8 - 30 Hz)
└──────────────────────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Bipolar Channel Isolation   │ ──► Selects: ['P4 - O2', 'P3 - O1', 'F4 - C4']
└──────────────────────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Sequence Generation & Scale  │ ──► Slices into 256-step windows & scales globally
└──────────────────────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Spatial-Temporal Classifier │ ──► Trains 1D-CNN / ConvLSTM
└──────────────────────────────┘
```

### Live Inference Pipeline

```
[ Live EEG Stream / Uploaded Excel ]
               │
               ▼
┌──────────────────────────────┐
│  Bipolar Channel Isolation   │ ──► Selects: ['P4 - O2', 'P3 - O1', 'F4 - C4']
└──────────────────────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Linear Trend Detrending     │ ──► Removes DC offset & electrode drift noise
│                              │     (inference-only calibration step)
└──────────────────────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Adaptive / Global Scaling   │ ──► 1D-CNN: per-session stats; ConvLSTM: fixed
│                              │     global training scalars (see Calibration)
└──────────────────────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Spatial-Temporal Classifier │ ──► Evaluates 255-step windows via 1D-CNN / ConvLSTM
└──────────────────────────────┘
               │
               ▼
┌──────────────────────────────┐
│   Dual-Output System Head    │
├──────────────────────────────┴────────────────────────────────┐
│ 1. Principal Outbound Action ──► Majority Mode Voting (Safety)│
│ 2. Closing Timeline Logic    ──► Real-Time Terminal Trigger   │
└───────────────────────────────────────────────────────────────┘
```

---

## Raw Dataset and Preprocessing

### Dataset Structure
The experimental EEG data is organized subject-wise and categorized by directional movement intent. The dataset hierarchy is structured as follows:

```
EEG Dataset/
├── Subject 1/
│   ├── Right/
│   ├── Left/
│   ├── Forward/
│   └── Backward/
│
└── Subject 2/
    ├── Right/
    ├── Left/
    ├── Forward/
    └── Backward/
```
Each directional subfolder contains multiple multi-channel EEG recordings stored as openpyxl-compliant `.xlsx` files.

### Selected EEG Channels
We isolate three specific differential channel configurations that measure neural oscillations over cortical regions associated with motor execution, visual-spatial attention, and planning:

* **P4 - O2**: Parieto-occipital right hemisphere channel.
* **P3 - O1**: Parieto-occipital left hemisphere channel.
* **F4 - C4**: Frontocentral right hemisphere channel.

These configurations exploit bipolar spatial filtering to suppress common-mode electrical noise across the scalp and improve spatial resolution.

### Chebyshev Type-I Bandpass Filtering
To preserve the highly relevant motor imagery oscillations while rejecting eye-blink artifacts (delta band, <4 Hz) and high-frequency powerline interference (50/60 Hz), we pass the raw signals through a Chebyshev Type-I bandpass filter. 

```
Type-I Chebyshev bandpass filters feature a steep roll-off in the transition band,
achieved by allowing a controlled ripple in the passband.
```

The filter settings are summarized below:

| Preprocessing Parameter | Configured Value |
|---|---|
| Sampling Rate ($f_s$) | 250 Hz |
| Lower Cutoff Frequency | 8 Hz (Alpha band boundary) |
| Upper Cutoff Frequency | 30 Hz (Beta band boundary) |
| Filter Order ($N$) | 6 (Highly selective transition) |
| Passband Ripple ($g_p$) | 0.3 dB |

This specific passband (8–30 Hz) isolates sensorimotor rhythms—specifically the event-related desynchronization (ERD) and event-related synchronization (ERS) patterns that occur during motor imagery.

---

## Generative Augmentation: WGAN-GP

### Motivation for Generative Augmentation
Supervised deep learning models trained on small EEG datasets struggle to generalize due to inter-session and inter-subject variability. Classical data augmentation (e.g., adding Gaussian noise or horizontal shifting) fails to capture the underlying temporal and phase relationships of bio-signals. We implement a Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP) to model the true statistical distribution of the EEG channels and synthesize novel, high-fidelity trials.

### Generator and Critic Network Architectures
To stabilize training and avoid mode collapse, we use the Wasserstein distance with a 1-Lipschitz constraint enforced via a gradient penalty.

#### Generator Structure
Accepts a 100-dimensional latent noise vector and generates multi-channel sequences:
```
Latent z (100) ──► Fully Connected (256 * 16) ──► Reshape (256, 16)
                      │
                      ▼
               ConvTranspose1d (256 ──► 128, Kernel=4, Stride=2, Pad=1) + BatchNorm + ReLU
                      │
                      ▼
               ConvTranspose1d (128 ──► 64,  Kernel=4, Stride=2, Pad=1) + BatchNorm + ReLU
                      │
                      ▼
               ConvTranspose1d (64  ──► 32,  Kernel=4, Stride=2, Pad=1) + BatchNorm + ReLU
                      │
                      ▼
               Conv1d (32 ──► 3 Channels, Kernel=3, Pad=1) ──► Synthetic Window (3, 128)
```

#### Critic Structure
Evaluates real or synthetic sequences and estimates the Wasserstein distance:
```
Input (3, 128) ──► Conv1d (3 ──► 32, Kernel=4, Stride=2, Pad=1) + LeakyReLU(0.2)
                      │
                      ▼
               Conv1d (32 ──► 64, Kernel=4, Stride=2, Pad=1) + LeakyReLU(0.2)
                      │
                      ▼
               Conv1d (64 ──► 128, Kernel=4, Stride=2, Pad=1) + LeakyReLU(0.2)
                      │
                      ▼
               Flatten ──► Linear (128 * 16 ──► 1) ──► Critic Scalar Score
```

*Note: In compliance with WGAN-GP theory, the Critic uses Layer Normalization (or no normalization) rather than Batch Normalization to prevent correlation between samples in the same batch.*

### Spectral Loss and Overlap-Add Reconstruction
To enforce phase alignment and spectral matching, the generator loss is augmented with an $L_1$ frequency-domain penalty:

$$\mathcal{L}_{\text{spectral}} = \mathbb{E} \left[ \left| \left| \mathcal{F}(x_{\text{real}}) \right| - \left| \mathcal{F}(x_{\text{fake}}) \right| \right| \right]$$

where $\mathcal{F}$ represents the Real Fast Fourier Transform (RFFT) along the temporal dimension.

Synthesized EEG windows of length 128 with a stride of 32 are reconstructed back into continuous-time signals using an **Overlap-Add (OLA) algorithm** to eliminate edge discontinuities:

$$x_{\text{reconstructed}}[t] = \frac{\sum_{i} w_i[t - i \cdot S] \cdot x_i[t - i \cdot S]}{\sum_{i} w_i[t - i \cdot S]}$$

where $S$ is the stride, $x_i$ is the $i$-th generated window, and $w_i$ is the corresponding window weight.

---

## Dataset Sequencing and Splitting

### File-Level Splitting
To prevent data leakage during temporal windowing, we split our data at the **file level** instead of the sequence level.
* **Train Set (80% of files)**: Real and synthetic files are windowed to train the networks.
* **Test Set (20% of files)**: Left out completely during windowing, normalization fitting, and training to ensure unbiased evaluation.

### Sequential Windowing and Global Normalization
Signals are segmented into distinct, fixed-size temporal windows of **256 timesteps** (`SEQUENCE_LENGTH = 256` in `src/filesequenicng.py`). We apply a global normalization scheme where a single `StandardScaler` is fitted on the training set and applied to the test set:

$$x_{\text{normalized}} = \frac{x - \mu_{\text{global}}}{\sigma_{\text{global}}}$$

The compiled feature arrays are saved as high-performance NumPy files: `X_train_500.npy` and `X_test_500.npy`. The corresponding label arrays consumed by the classifier scripts are `y_train_cls_500.npy` and `y_test_cls_500.npy`.

> **Note on window length:** Training sequences are 256 timesteps, while the live-inference scripts reshape incoming recordings into **255-timestep** windows (`EXPECTED_TIMESTEPS = 255`). Because both classifiers reduce the temporal axis via `AdaptiveAvgPool1d` (1D-CNN) or the LSTM's final hidden state (ConvLSTM), they accept either length without modification, but the off-by-one is intentional to flag here rather than implied to be identical.

---

## Classifier Architectures

We implement and evaluate two principal spatial-temporal network backbones:

### Pure 1D-CNN Classifier
This model extracts spatial-temporal features directly through nested 1D convolutional layers, bypasses recurrent connections, and achieves fast inference times.

```
Input Tensor (Batch, 256, 3) ──► Transpose to (Batch, 3, 256)
                                      │
                                      ▼
                               [ Conv1D Block 1 ]
                               64 Filters, Kernel=7, Padding=3, ReLU
                               BatchNorm1d + MaxPool1d(2) + Dropout(0.3)
                                      │
                                      ▼
                               [ Conv1D Block 2 ]
                               128 Filters, Kernel=5, Padding=2, ReLU
                               BatchNorm1d + MaxPool1d(2) + Dropout(0.3)
                                      │
                                      ▼
                               [ Conv1D Block 3 ]
                               256 Filters, Kernel=3, Padding=1, ReLU
                               BatchNorm1d + AdaptiveAvgPool1d(1)
                                      │
                                      ▼
                               [ Classifier Head ]
                               Linear(256 ──► 64) + ReLU + BatchNorm1d + Dropout(0.4)
                               Linear(64 ──► 4 Classes) ──► Softmax
```

### ConvLSTM Hybrid Classifier
This architecture combines 1D convolutions for spatial feature extraction with a multi-layer Long Short-Term Memory (LSTM) network to track temporal sequence dynamics.

```
Input Tensor (Batch, 256, 3) ──► Transpose to (Batch, 3, 256)
                                      │
                                      ▼
                               [ Conv1D Feature Map ]
                               Conv1d(3 ──► 64, K=5, P=2) + ReLU + BatchNorm1d + MaxPool1d(2)
                               Dropout1d(0.3) + Conv1d(64 ──► 64, K=3, P=1) + ReLU + BatchNorm1d
                                      │
                                      ▼
                               Transpose back to (Batch, Timesteps_Reduced, 64)
                                      │
                                      ▼
                               [ Recurrent Engine ]
                               LSTM Layer 1 (Hidden=128, batch_first=True)
                               LSTM Layer 2 (Hidden=128) + Internal Dropout(0.4)
                                      │
                                      ▼
                               Extract Last Hidden State h_T (Batch, 128)
                                      │
                                      ▼
                               [ Classifier Head ]
                               Linear(128 ──► 64) + ReLU + BatchNorm1d + Dropout(0.4)
                               Linear(64 ──► 4 Classes) ──► Softmax
```

---

## Adaptive Calibration and Simulated Inference

### Detrending and Impedance Compensation
During deployment, variations in scalp contact impedance introduce significant DC offset shifts and electrode drift. This causes standard classifiers to collapse, often predicting a single class (such as "Forward") continuously. To counter this, both inference scripts first apply **detrending** — a linear least-squares detrending operator that eliminates session-specific microvolt drift across each temporal window. They then diverge in how they normalize, providing two complementary calibration strategies:

1. **Adaptive Per-Session Scaling (`tests/1D_CNN_test.py`)**: Rather than relying on training scale boundaries, this engine dynamically estimates the active session's own statistics to normalize the incoming window block:

$$X_{\text{calibrated}} = \frac{X_{\text{eval}} - \mu_{\text{session}}}{\sigma_{\text{session}}}$$

This stabilizes the feature space and maintains class balance even under changing noise conditions, at the cost of assuming the session is class-balanced.

2. **Fixed Global Scalars (`tests/conv_LSTM_test.py`)**: This engine instead normalizes against precomputed global training statistics, preserving the absolute scale the model was trained on:

$$X_{\text{calibrated}} = \frac{X_{\text{eval}} - \mu_{\text{global}}}{\sigma_{\text{global}}}$$

where `GLOBAL_TRAIN_MEAN` and `GLOBAL_TRAIN_STD` are hardcoded baseline vectors representing the preprocessed training distribution.

### Dual-Output Decision Logic
During testing and live inference, the framework segments the EEG data stream using File Sequencing (generating distinct, fixed-size window matrices, such as the 255-timestep sequences). The framework outputs two key decision metrics:
* **The Principal Outbound Action (The Majority Mode)**: Aggregates classifications across all sliding windows in the active session and performs a majority vote. This acts as a safety filter to prevent false triggers from spurious, high-frequency noise.
* **The Closing Timeline Logic (The Temporal Recency Window)**: Directly tracks the classification of the final temporal window. This minimizes control latency and allows real-time execution triggers when persistent movement intent is maintained.

### Decision Security Assessment
For every executed prediction, the engine evaluates the difference between the primary class probability and the runner-up class probability:

$$\text{Margin} = (P_{\text{winning}} - P_{\text{runner-up}}) \times 100$$

If $\text{Margin} < 15\%$, the output is flagged as `⚠️ SHIFTING` due to high class ambiguity. If $\text{Margin} \ge 15\%$, it is flagged as `✅ STABLE`, confirming high decision security.

---

## Experimental Results

The models were trained and validated using PyTorch on NVIDIA Tesla T4 graphics accelerators. By employing File Sequencing (segmenting continuous data into distinct, fixed-size window matrices of 256-timestep blocks) alongside a rigorous file-level split, we obtained the following generalization performance:

| Classifier Model | Training Accuracy | Test Accuracy | Generalization Gap |
|---|---|---|---|
| Pure 1D-CNN | 77.00% | 84.53% | +7.53% (Excellent Out-of-Distribution Generalization) |
| Hybrid ConvLSTM | 86.00% | 85.18% | -0.82% (High Temporal Consistency & Balanced Fit) |

Performance visualization assets are tracked under the `results/` directory:
* `results/1D_cnn_accuracy_curve.png`: Convergence profile of the Convolutional model.
* `results/LSTM_eeg_seq_class_train_test_accuracy_plot.png`: Epoch-by-epoch validation curves.
* `results/LSTM_confusion_matrix_eeg_seq_classification.png`: Class-wise precision and recall matrix showing clean separations between Left, Right, Forward, and Backward.

---

## Technical Stack

| Area | Software Library | Purpose in Framework |
|---|---|---|
| Deep Learning | PyTorch >= 2.0 | Architecture definitions, backpropagation, and weight loading |
| Mathematical Operations | NumPy >= 1.22 | Array transformations, OLA reconstruction, and vector scaling |
| Preprocessing & DSP | SciPy >= 1.10 | Chebyshev Type-I filter implementation and linear detrending |
| Data Processing | Pandas >= 1.5 | Excel parsing and session structuring |
| Model Validation | Scikit-Learn >= 1.1 | File-level splitting, standard scaling, and metrics evaluation |
| Visualization | Matplotlib >= 3.5 | Generating training curves and confusion matrices |
| Excel Interface | OpenPyXL >= 3.0 | Reading and writing raw and synthetic multi-channel worksheets |

---

## Open Research Directions

* **Curriculum Training**: Train WGAN-GP progressively, beginning with coarse temporal shapes and moving to fine-grained phase patterns.
* **Attention Mechanisms**: Integrate multi-head self-attention (Transformer blocks) after the 1D-CNN layers to model long-range temporal dependencies.
* **Cross-Subject Transfer Learning**: Implement domain adversarial training to minimize feature representation variance between different subjects.
* **Closed-Loop Hardware Integration**: Extend the adaptive calibration engine into a real-time ROS (Robot Operating System) node to control physical mobile platforms.
