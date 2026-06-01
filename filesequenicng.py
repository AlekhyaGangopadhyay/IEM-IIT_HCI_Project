# ============================================================
# EEG DATASET PREPARATION
# 500 FILES / CLASS
# FILE-LEVEL SPLIT
# MANDATORY ORIGINAL FILES INCLUDED
# ============================================================

import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIGURATION
# ============================================================

BASE_PATH = "/kaggle/input/datasets/alekhya7gangopadhyay/eeg-dataset/content/drive/My Drive/Human Computer Interface (HCI)/Chebyshev Filtered Data"

SEQUENCE_LENGTH = 256

MAX_FILES_PER_CLASS = 500

SELECTED_CHANNELS = [
    "P4 - O2",
    "P3 - O1",
    "F4 - C4"
]

label_map = {
    "Right": 0,
    "Left": 1,
    "Forward": 2,
    "Backward": 3
}

random.seed(42)

# ============================================================
# MANDATORY FILES
# ============================================================

mandatory_files = {

    "Right": [
        "chebyshev_ARROW_Right.xlsx",
        "chebyshev_LETTER_Right.xlsx",
        "chebyshev_WORD_Right.xlsx",
        "chebyshev_Right ARROW2.xlsx",
        "chebyshev_Right LETTER2.xlsx",
        "chebyshev_Right WORD2.xlsx"
    ],

    "Left": [
        "chebyshev_ARROW_Left.xlsx",
        "chebyshev_LETTER_Left.xlsx",
        "chebyshev_WORD_Left.xlsx",
        "chebyshev_ARROW_Left_2.xlsx",
        "chebyshev_LETTER_Left_2.xlsx",
        "chebyshev_WORD_Left_2.xlsx"
    ],

    "Forward": [
        "chebyshev_ARROW_Forward.xlsx",
        "chebyshev_LETTER_Forward.xlsx",
        "chebyshev_WORD_Forward.xlsx",
        "chebyshev_ARROW_Forward_2.xlsx",
        "chebyshev_LETTER_Forward_2.xlsx",
        "chebyshev_WORD_Forward_2.xlsx"
    ],

    "Backward": [
        "chebyshev_ARROW_Backward.xlsx",
        "chebyshev_LETTER_Backward.xlsx",
        "chebyshev_WORD_Backword.xlsx",
        "chebyshev_ARROW_Backword_2.xlsx",
        "chebyshev_LETTER_Backward_2.xlsx",
        "chebyshev_WORD_Backward_2.xlsx"
    ]
}

# ============================================================
# SELECT FILES
# ============================================================

selected_files = []

print("\nSelecting Files...\n")

for class_name, label in label_map.items():

    folder = os.path.join(BASE_PATH, class_name)

    all_files = sorted([
        f for f in os.listdir(folder)
        if f.endswith(".xlsx")
    ])

    mandatory = []

    for f in mandatory_files[class_name]:

        path = os.path.join(folder, f)

        if os.path.exists(path):
            mandatory.append(f)
        else:
            print(f"Missing Mandatory File: {f}")

    remaining = [
        f for f in all_files
        if f not in mandatory
    ]

    n_needed = MAX_FILES_PER_CLASS - len(mandatory)

    sampled = random.sample(
        remaining,
        n_needed
    )

    final_files = mandatory + sampled

    print(
        f"{class_name}: {len(final_files)} files selected"
    )

    for f in final_files:

        selected_files.append(
            (
                os.path.join(folder, f),
                label
            )
        )

print("\nTotal Selected Files:", len(selected_files))

# ============================================================
# FILE LEVEL SPLIT
# ============================================================

train_files = []
test_files = []

for class_name, label in label_map.items():

    class_files = [
        x
        for x in selected_files
        if x[1] == label
    ]

    train_cls, test_cls = train_test_split(
        class_files,
        test_size=0.20,
        random_state=42
    )

    train_files.extend(train_cls)
    test_files.extend(test_cls)

print("\nTrain Files:", len(train_files))
print("Test Files :", len(test_files))

# ============================================================
# SEQUENCE CREATION
# ============================================================

def create_sequences(data, seq_len):

    sequences = []

    for start in range(
        0,
        len(data) - seq_len + 1,
        seq_len
    ):

        sequences.append(
            data[start:start + seq_len]
        )

    return sequences

# ============================================================
# BUILD TRAIN DATASET
# ============================================================

print("\nCreating Training Dataset...\n")

X_train = []
y_train = []

total_train = len(train_files)

for idx, (filepath, label) in enumerate(train_files, start=1):

    if idx % 25 == 0 or idx == total_train:

        print(
            f"Train Progress: "
            f"{idx}/{total_train} "
            f"({100*idx/total_train:.1f}%)"
        )

    try:

        df = pd.read_excel(filepath)

        df = df[SELECTED_CHANNELS]

        data = df.values.astype(np.float32)

        sequences = create_sequences(
            data,
            SEQUENCE_LENGTH
        )

        for seq in sequences:

            X_train.append(seq)
            y_train.append(label)

    except Exception as e:

        print("ERROR:", filepath)
        print(e)

# ============================================================
# BUILD TEST DATASET
# ============================================================

print("\nCreating Test Dataset...\n")

X_test = []
y_test = []

total_test = len(test_files)

for idx, (filepath, label) in enumerate(test_files, start=1):

    if idx % 25 == 0 or idx == total_test:

        print(
            f"Test Progress: "
            f"{idx}/{total_test} "
            f"({100*idx/total_test:.1f}%)"
        )

    try:

        df = pd.read_excel(filepath)

        df = df[SELECTED_CHANNELS]

        data = df.values.astype(np.float32)

        sequences = create_sequences(
            data,
            SEQUENCE_LENGTH
        )

        for seq in sequences:

            X_test.append(seq)
            y_test.append(label)

    except Exception as e:

        print("ERROR:", filepath)
        print(e)

# ============================================================
# CONVERT TO NUMPY
# ============================================================

X_train = np.array(
    X_train,
    dtype=np.float32
)

X_test = np.array(
    X_test,
    dtype=np.float32
)

y_train = np.array(y_train)
y_test = np.array(y_test)

print("\nBefore Normalization")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

# ============================================================
# NORMALIZATION
# ============================================================

channels = X_train.shape[2]

scaler = StandardScaler()

X_train_2d = X_train.reshape(
    -1,
    channels
)

X_test_2d = X_test.reshape(
    -1,
    channels
)

X_train_2d = scaler.fit_transform(
    X_train_2d
)

X_test_2d = scaler.transform(
    X_test_2d
)

X_train = X_train_2d.reshape(
    X_train.shape
)

X_test = X_test_2d.reshape(
    X_test.shape
)

print("\nNormalization Check")
print("Mean:", X_train.mean())
print("Std :", X_train.std())

# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n==========================")
print("DATASET SUMMARY")
print("==========================")

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

print("\nTrain Distribution")

for c in range(4):

    print(
        f"Class {c}:",
        np.sum(y_train == c)
    )

print("\nTest Distribution")

for c in range(4):

    print(
        f"Class {c}:",
        np.sum(y_test == c)
    )

# ============================================================
# SAVE DATASET
# ============================================================

np.save(
    "/kaggle/working/X_train_500.npy",
    X_train
)

np.save(
    "/kaggle/working/X_test_500.npy",
    X_test
)

np.save(
    "/kaggle/working/y_train_500.npy",
    y_train
)

np.save(
    "/kaggle/working/y_test_500.npy",
    y_test
)

print("\nDataset Saved Successfully")

print("/kaggle/working/X_train_500.npy")
print("/kaggle/working/X_test_500.npy")
print("/kaggle/working/y_train_500.npy")
print("/kaggle/working/y_test_500.npy")

import zipfile

zip_path = "/kaggle/working/EEG_Dataset_500_FileSplit.zip"

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:

    zipf.write(
        "/kaggle/working/X_train_500.npy",
        arcname="X_train_500.npy"
    )

    zipf.write(
        "/kaggle/working/X_test_500.npy",
        arcname="X_test_500.npy"
    )

    zipf.write(
        "/kaggle/working/y_train_500.npy",
        arcname="y_train_500.npy"
    )

    zipf.write(
        "/kaggle/working/y_test_500.npy",
        arcname="y_test_500.npy"
    )

print("ZIP Created:")
print(zip_path)

