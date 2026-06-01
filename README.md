# EEG Direction Classification using WGAN-GP and LSTM

A deep learning pipeline for classifying directional intent from EEG signals using
Chebyshev bandpass filtering, synthetic data augmentation via Wasserstein GAN with
Gradient Penalty, and temporal sequence learning with LSTM networks.

---

## Abstract

This work presents an end-to-end framework for EEG-based directional movement
classification across four classes: Right, Left, Forward, and Backward. The pipeline
combines classical signal preprocessing with modern generative and sequence models.
Raw EEG recordings are denoised using a Chebyshev Type-I bandpass filter to retain
the alpha and beta bands relevant to motor imagery. A Wasserstein GAN with Gradient
Penalty (WGAN-GP) is trained to generate synthetic EEG windows for data augmentation.
The combined real and synthetic data is segmented into temporal sequences and fed
into a two-layer LSTM classifier. The model achieves a final train accuracy of 95.41%
and test accuracy of 95.72%.

---

## 1. Introduction

EEG-based brain-computer interfaces (BCIs) face two persistent challenges: signal
noise and limited sample size. This project addresses both. Noise is handled through
narrow-band filtering targeting the motor imagery frequency range. Data scarcity is
addressed through adversarial synthesis using a stabilized GAN variant. The
classification task is framed as a multi-class temporal sequence problem and solved
using a recurrent neural network.

---

## 2. Dataset

EEG recordings were collected subject-wise and organized by directional task. Each
direction folder contains multiple multichannel EEG recordings stored as `.xlsx`
files.

**Classes:** Right, Left, Forward, Backward

**Dataset structure:**

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

**EEG channels used:**

```
P4 - O2
P3 - O1
F4 - C4
```

These channels were selected for their relevance to directional cortical activity
patterns.

---

## 3. Methodology

### 3.1 Pipeline Overview

```
Raw EEG
   │
   ▼
Chebyshev Bandpass Filtering
   │
   ▼
WGAN-GP Synthetic EEG Generation
   │
   ▼
Sequence Generation and Normalization
   │
   ▼
LSTM Training
   │
   ▼
Direction Classification
```

### 3.2 Chebyshev Bandpass Filtering

A Chebyshev Type-I bandpass filter was applied to suppress low-frequency drift,
high-frequency noise, and unwanted artifacts while preserving EEG bands relevant
to cognitive and motor imagery activity.

**Filter parameters:**

| Parameter         | Value  |
| ----------------- | ------ |
| Sampling rate     | 250 Hz |
| Low cutoff        | 8 Hz   |
| High cutoff       | 30 Hz  |
| Filter order      | 4      |
| Passband ripple   | 0.5 dB |

The passband (8 to 30 Hz) covers the alpha and beta bands, which are well
established in motor imagery literature.

### 3.3 Synthetic EEG Generation (WGAN-GP)

A Wasserstein GAN with Gradient Penalty was trained to learn the distribution of
real EEG windows and generate synthetic samples for augmentation.

**Generator:** Accepts latent noise vectors and outputs synthetic EEG windows
through fully connected layers with BatchNorm and LeakyReLU activations.

**Critic:** Distinguishes real from synthetic EEG and estimates the Wasserstein
distance. LayerNorm is used in place of BatchNorm for training stability.

**Training behavior:** The Wasserstein distance increased steadily and stabilized
at a positive value, indicating healthy adversarial learning without mode collapse.

```
Epoch 300/300
W-dist = +38.44
```

Synthetic recordings are reconstructed from generated windows into continuous
signals and saved in `.xlsx` format using the convention:

```
originalfilename_subject_synthetic_number.xlsx
```

### 3.4 Sequence Generation

Filtered EEG signals were segmented into overlapping windows for temporal learning.

| Parameter        | Value |
| ---------------- | ----- |
| Sequence length  | 256   |
| Stride           | 128   |

### 3.5 LSTM Classifier

A two-layer LSTM network was trained on the normalized sequences for four-class
classification.

**Architecture:**

```
Input
  │
  ▼
LSTM Layer 1
  │
  ▼
LSTM Layer 2
  │
  ▼
Fully Connected Layer
  │
  ▼
Softmax (4 classes)
```

Dropout regularization was applied to mitigate overfitting.

**Class labels:**

| Label | Class    |
| ----- | -------- |
| 0     | Right    |
| 1     | Left     |
| 2     | Forward  |
| 3     | Backward |

---

## 4. Results

The final trained model achieved the following performance:

| Metric              | Value   |
| ------------------- | ------- |
| Train Accuracy      | 95.41%  |
| Test Accuracy       | 95.72%  |

The close gap between training and test accuracy suggests good generalization
across the held-out evaluation set.

---

## 5. Implementation

### 5.1 Repository Structure

```
IEM-IIT_HCI_Project/
│
├── Dataset/
├── Synthetic Dataset/
├── Chebyshev Filtered Data/
│
├── Filtering/
│   └── chebyshev_filter.py
│
├── GAN/
│   ├── train_wgan_gp.py
│   └── generate_synthetic.py
│
├── LSTM/
│   └── lstm-eeg-sequence-classification.ipynb
│
├── Models/
│   └── eeg_lstm_model.pth
│
├── Results/
│   ├── accuracy_plot.png
│   └── confusion_matrix.png
│
├── requirements.txt
└── README.md
```

### 5.2 Component Summary

| Module                          | Function                              |
| ------------------------------- | ------------------------------------- |
| `chebyshev_filter.py`           | EEG noise removal and band selection  |
| `train_wgan_gp.py`              | Synthetic EEG generation training     |
| `generate_synthetic.py`         | Synthetic EEG inference and export    |
| `lstm-eeg-sequence-classification.ipynb` | LSTM training and evaluation |

### 5.3 Technologies

Python, PyTorch, NumPy, Pandas, Scikit-learn, SciPy, Matplotlib, OpenPyXL.

### 5.4 Training Environment

Training was carried out on Kaggle GPU and Google Colab environments using
Tesla T4 accelerators, with CPU fallback where required.

---

## 6. Limitations and Future Work

The current implementation uses sequence-level train-test splitting, which can
introduce mild leakage between adjacent windows drawn from the same recording.
For research-grade evaluation, file-level splitting is recommended.

**Planned extensions:**

- File-level train-test splitting to eliminate sequence leakage
- Bidirectional LSTM and CNN-LSTM hybrid architectures
- Transformer-based EEG classifiers with self-attention
- Inclusion of additional EEG channels
- Attention mechanisms over temporal sequences
- Real-time inference pipeline for online BCI applications

---

## 7. Conclusion

The proposed pipeline demonstrates that combining classical EEG preprocessing,
GAN-based data augmentation, and temporal sequence modeling yields a robust
framework for directional intent classification. The achieved test accuracy of
95.72% supports the viability of this approach for downstream BCI applications,
with clear pathways for further improvement through stricter evaluation protocols
and architectural extensions.
