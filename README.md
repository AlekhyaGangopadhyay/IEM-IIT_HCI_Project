# EEG Data Augmentation and Classification for Robotic Movement

## Overview

This project presents an end-to-end pipeline for EEG signal analysis, synthetic data generation, and classification for robotic movement applications. The dataset used in this work was collected from two subjects and processed to remove noise and improve signal quality before being used for downstream modeling.

The core idea of the project is to address the limited availability of EEG samples by generating synthetic EEG data using a GAN-based approach. The original and synthetic datasets are then combined and used for EEG signal classification. The final objective is to support robotic movement through brain signal interpretation.

## Objectives

- Preprocess raw EEG signals collected from two subjects
- Extract cleaner and more usable neural signal patterns
- Train a GAN model to generate synthetic EEG data
- Combine real and synthetic EEG samples for augmentation
- Classify EEG signals for movement-related interpretation
- Use the classified output for robotic movement control

## Project Workflow

1. EEG data acquisition from two subjects
2. Preprocessing of raw signals
3. Noise reduction and filtering
4. Synthetic data generation using the GAN model
5. Dataset augmentation using real and generated samples
6. EEG signal classification
7. Mapping classification results to robotic movement commands

## Methodology

### 1. Data Acquisition
EEG recordings were obtained from two subjects. Since EEG datasets are often small and difficult to collect in large quantities, this project focuses on making effective use of limited data.

### 2. Preprocessing
The raw EEG signals were preprocessed to improve quality and remove unwanted noise. This stage is essential because EEG signals are highly susceptible to artifacts and interference.

The preprocessing pipeline includes signal filtering and other cleaning operations designed to prepare the data for modeling.

### 3. Synthetic Data Generation
The preprocessed EEG data were used as input to the GAN model implemented in `Implementing_GAN.ipynb`. The GAN learns the statistical distribution of the EEG signals and generates synthetic samples that resemble the original data.

This augmentation step is important because it helps reduce data scarcity and supports more robust downstream learning.

### 4. Classification
After combining the real and synthetic EEG data, the augmented dataset is used for classification. The classification model is trained to distinguish patterns in EEG signals relevant to the intended task.

### 5. Robotic Movement Application
The final classified output is intended to support robotic movement control. In this context, EEG signals serve as an interface between brain activity and external machine response.

## Repository Structure

```text
.
├── Chebyshev_filtering.ipynb
├── Implementing_GAN.ipynb
├── LICENSE
└── README.md
```

## Notebooks

### Chebyshev_filtering.ipynb
Notebook used for EEG preprocessing and signal filtering.

### Implementing_GAN.ipynb
Notebook used for GAN-based synthetic EEG data generation.

## Significance of the Work

This project demonstrates how generative modeling can be applied to EEG data in low-sample settings. By combining preprocessing, synthetic data generation, and classification, the workflow provides a practical approach for EEG-based human-computer interaction and robotic control.

## Future Work

- Expand the dataset with more subjects  
- Improve preprocessing and artifact removal methods  
- Explore more advanced generative models  
- Evaluate classification performance on larger datasets  
- Integrate the pipeline with real-time robotic systems  
