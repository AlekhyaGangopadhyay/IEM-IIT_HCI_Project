# EEG Direction Classification using GAN + LSTM

## Overview

This project focuses on EEG-based direction classification using deep learning and signal processing techniques. The complete pipeline includes:

* EEG preprocessing and cleaning
* Chebyshev bandpass filtering
* Synthetic EEG generation using WGAN-GP (Wasserstein GAN with Gradient Penalty)
* Sequence generation for temporal learning
* LSTM-based EEG direction classification
* Performance evaluation using train and test accuracy

The project was developed as part of the Human Computer Interface (HCI) research workflow.

---

# Dataset Description

The dataset consists of EEG recordings corresponding to four directional classes:

* Right
* Left
* Forward
* Backward

The EEG data was organized subject-wise and task-wise.

## Original Dataset Structure

```text
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

Each direction folder contains multiple EEG `.xlsx` recordings.

---

# EEG Channels Used

The following EEG channels were used for classification:

```python
[
    'P4 - O2',
    'P3 - O1',
    'F4 - C4'
]
```

These channels were selected for capturing directional brain activity patterns.

---

# Signal Processing Pipeline

The complete processing pipeline:

```text
Raw EEG
→ Chebyshev Filtering
→ GAN-based Synthetic EEG Generation
→ Sequence Generation
→ Normalization
→ LSTM Training
→ Direction Classification
```

---

# Project Directory Structure

```text
IEM-IIT_HCI_Project/
│
├── Dataset/
│
├── Synthetic Dataset/
│
├── Chebyshev Filtered Data/
│
├── GAN/
│   ├── train_wgan_gp.py
│   ├── generate_synthetic.py
│
├── Filtering/
│   ├── chebyshev_filter.py
│
├── LSTM/
│   ├── lstm-eeg-sequence-classification.ipynb
│
├── Models/
│   ├── eeg_lstm_model.pth
│
├── Results/
│   ├── accuracy_plot.png
│   ├── confusion_matrix.png
│
├── requirements.txt
├── README.md
```

---

# File Descriptions

# 1. EEG Dataset Files

These files contain the original EEG recordings collected for directional movement classification.

Each file contains multichannel EEG data stored in `.xlsx` format.

Classes:

* Right
* Left
* Forward
* Backward

---

# 2. GAN Training Files

## `train_wgan_gp.py`

This file implements a Wasserstein GAN with Gradient Penalty (WGAN-GP) for synthetic EEG generation.

### Features

* Wasserstein distance optimization
* Gradient penalty stabilization
* EEG window-based training
* Temporal feature learning
* Synthetic EEG augmentation

### GAN Architecture

#### Generator

The generator:

* takes latent noise vectors
* generates synthetic EEG windows
* uses fully connected layers with BatchNorm and LeakyReLU

#### Critic

The critic:

* distinguishes real vs synthetic EEG
* uses LayerNorm for stability
* estimates Wasserstein distance

### Training Observations

Healthy GAN training showed:

```text
W-distance increasing and stabilizing positively
```

Example:

```text
Epoch 300/300
W-dist = +38.44
```

This indicates stable adversarial learning.

---

## `generate_synthetic.py`

This file:

* loads the trained GAN generator
* generates synthetic EEG recordings
* reconstructs EEG windows into continuous signals
* saves synthetic EEG files in `.xlsx` format

### Output Structure

```text
Synthetic Dataset/
├── Subject 1/
│   ├── Right/
│   ├── Left/
│   ├── Forward/
│   └── Backward/
```

### Naming Convention

Synthetic files are stored as:

```text
originalfilename_subject_synthetic_number.xlsx
```

Example:

```text
ARROW_Right_Subject1_synthetic_25.xlsx
```

---

# 3. Filtering Files

## `chebyshev_filter.py`

This file performs EEG signal preprocessing using a Chebyshev Type-I Bandpass Filter.

### Purpose

The filter removes:

* low-frequency drift
* high-frequency noise
* unwanted artifacts

while preserving EEG frequency bands relevant to cognitive activity.

### Filter Parameters

```python
fs = 250
lowcut = 8
highcut = 30
order = 4
rp = 0.5
```

### Frequency Range

The filter preserves:

```text
8 Hz – 30 Hz
```

which covers:

* Alpha waves
* Beta waves

important for motor imagery and directional EEG analysis.

### Output Structure

```text
Chebyshev Filtered Data/
├── Right/
├── Left/
├── Forward/
└── Backward/
```

### Output Naming

```text
chebyshev_originalfilename.xlsx
```

Example:

```text
chebyshev_ARROW_Right.xlsx
```

---

# 4. LSTM Training File

## `train_lstm.py`

This file trains an LSTM-based deep learning model for EEG direction classification.

### Input

The model receives:

* filtered EEG sequences
* normalized temporal EEG windows

### Sequence Parameters

```python
SEQUENCE_LENGTH = 256
STRIDE = 128
```

### LSTM Architecture

```text
Input Layer
→ 2-Layer LSTM
→ Fully Connected Layers
→ Softmax Output
```

### Model Features

* Temporal EEG learning
* Sequential pattern extraction
* Multi-class classification
* Regularization using Dropout

### Classes

```text
0 → Right
1 → Left
2 → Forward
3 → Backward
```

---

# Training Results

Final Results:

```text
Final Train Accuracy : 95.41%
Final Test Accuracy  : 95.72%
```

The model successfully learned temporal EEG patterns for directional classification.

---

# Performance Summary

| Component           | Purpose                      |
| ------------------- | ---------------------------- |
| Chebyshev Filter    | EEG Noise Removal            |
| WGAN-GP             | Synthetic EEG Generation     |
| Sequence Generator  | Temporal EEG Window Creation |
| LSTM                | EEG Direction Classification |
| Accuracy Evaluation | Performance Measurement      |

---

# Technologies Used

* Python
* PyTorch
* NumPy
* Pandas
* Scikit-learn
* SciPy
* Matplotlib
* OpenPyXL
* Kaggle GPU
* Google Colab

---

# Hardware and Training Environment

The project was trained using:

* Kaggle GPU environment
* Google Colab
* Tesla T4 / CPU environment

---

# Future Improvements

Possible future enhancements:

* Bidirectional LSTM
* CNN-LSTM Hybrid Models
* Transformer-based EEG Classification
* File-level train-test splitting
* Real-time EEG classification
* Attention mechanisms
* More EEG channels

---

# Research Notes

The current implementation uses sequence-level train-test splitting.

For research-grade evaluation, future work should use:

```text
File-level splitting
```

to avoid sequence leakage between train and test datasets.

---

# Results

The project demonstrates:

* successful EEG preprocessing
* stable GAN training
* synthetic EEG generation
* high-accuracy temporal EEG classification
* end-to-end EEG deep learning workflow

---

# Author

Developed by Alekhya Gangopadhyay as part of the IEM-IIT HCI Project.
