from google.colab import drive
drive.mount('/content/drive')

import os

base_path = '/content/drive/My Drive/Human Computer Interface (HCI)/Chebyshev Filtered Data'

leaf_folder_file_counts = {}

for root, dirs, files in os.walk(base_path):
    if not dirs:
        file_count = len([f for f in files if os.path.isfile(os.path.join(root, f))])
        leaf_folder_file_counts[root] = file_count

print("File counts in leaf folders:")
for folder, count in leaf_folder_file_counts.items():
    print(f"  {folder}: {count} files")


!pip install openpyxl scipy -q

import os
import numpy as np
import pandas as pd

from scipy.signal import cheby1
from scipy.signal import sosfiltfilt


INPUT_FOLDER ="/content/drive/My Drive/Human Computer Interface (HCI)/Synthetic Data/Subject 1/Right/Right"

OUTPUT_FOLDER = "/content/drive/My Drive/Human Computer Interface (HCI)/Chebyshev Filtered Data/Right"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

FS = 250

LOWCUT = 8
HIGHCUT = 30

ORDER = 6

RP = 0.3

def chebyshev_eeg_filter(
    data,
    fs=250,
    lowcut=8,
    highcut=30,
    order=6,
    rp=0.3
):
    nyquist = 0.5 * fs

    low = lowcut / nyquist
    high = highcut / nyquist

    sos = cheby1(
        N=order,
        rp=rp,
        Wn=[low, high],
        btype='bandpass',
        output='sos'
    )

   
    filtered = sosfiltfilt(
        sos,
        data,
        axis=0
    )

    return filtered

files_list = sorted([
    f for f in os.listdir(INPUT_FOLDER)
    if f.endswith(".xlsx")
])

print("\n================================================")
print(f"FILES FOUND: {len(files_list)}")
print("================================================\n")

processed = 0

for file_name in files_list:

    try:

        file_path = os.path.join(
            INPUT_FOLDER,
            file_name
        )

        print(f"Processing: {file_name}")

        
        df = pd.read_excel(
            file_path,
            engine='openpyxl'
        )

        data = df.values.astype(np.float32)

        columns = list(df.columns)

    
        filtered_data = chebyshev_eeg_filter(
            data=data,
            fs=FS,
            lowcut=LOWCUT,
            highcut=HIGHCUT,
            order=ORDER,
            rp=RP
        )

    

        output_name = (
            f"chebyshev_{file_name}"
        )

        output_path = os.path.join(
            OUTPUT_FOLDER,
            output_name
        )

        pd.DataFrame(
            filtered_data,
            columns=columns
        ).to_excel(
            output_path,
            index=False,
            engine='openpyxl'
        )

        processed += 1

        print(f"Saved: {output_name}\n")

    except Exception as e:

        print(f"Error in file: {file_name}")
        print(e)



print("\n================================================")
print("CHEBYSHEV FILTERING COMPLETED")
print(f"TOTAL FILES PROCESSED: {processed}")
print("================================================")