'''Best Test Accuracy: 33.33%'''

import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    labels_map = {'LEFT': 0, 'RIGHT': 1, 'FORWARD': 2, 'BACKWARD': 3}
    X_all = []
    y_all = []

    print("Reading Excel files...", flush=True)
    
    # Path logic: Look for folders in the Processed_Filtered_Data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'Processed_Filtered_Data')
    
    # We search for .xlsx files in subfolders named LEFT, RIGHT, FORWARD, BACKWARD
    file_list = []
    for folder in labels_map.keys():
        folder_path = os.path.join(data_dir, folder)
        if os.path.exists(folder_path):
            files = glob.glob(os.path.join(folder_path, '*.xlsx'))
            file_list.extend(files)
        else:
            print(f"Warning: Folder {folder} not found in {script_dir}")

    for file in file_list:
        # Get the immediate parent folder name to assign label
        dir_name = os.path.basename(os.path.dirname(file)).upper()
        
        if dir_name in labels_map:
            try:
                # Read columns 7, 14, 15 (0-indexed)
                df = pd.read_excel(file, usecols=[7, 14, 15], engine='openpyxl')
                data = df.values
                
                if data.shape[1] == 3:
                    # Downsampling to prevent vanishing gradients (adjust as needed)
                    # Taking every 20th sample helps the LSTM focus on the overall 'shape' of the EEG
                    downsampled_data = data[::20, :] 
                    X_all.append(downsampled_data)
                    y_all.append(labels_map[dir_name])
                else:
                    print(f"Skipped {file} due to wrong column count: {data.shape[1]}")
            except Exception as e:
                print(f"Error reading {file}: {e}", flush=True)

    if len(X_all) == 0:
        print("No valid files found. Check your folder names and .xlsx contents.", flush=True)
        return

    print(f"Loaded {len(X_all)} files.", flush=True)

    # Pad sequences to the same length
    max_len = max([x.shape[0] for x in X_all])
    X_padded = []
    for x in X_all:
        if x.shape[0] < max_len:
            pad = np.zeros((max_len - x.shape[0], x.shape[1]))
            x = np.vstack((x, pad))
        X_padded.append(x)

    X = np.stack(X_padded)
    y = np.array(y_all)

    # Stratified split ensures balanced classes in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Normalization (StandardScaler approach)
    train_mean = np.mean(X_train, axis=(0, 1), keepdims=True)
    train_std = np.std(X_train, axis=(0, 1), keepdims=True)
    X_train = (X_train - train_mean) / (train_std + 1e-8)
    X_test = (X_test - train_mean) / (train_std + 1e-8)
    
    # Convert to Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=8, shuffle=True)

    class EEG_LSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes):
            super(EEG_LSTM, self).__init__()
            self.lstm = nn.LSTM(input_size=input_size, 
                                hidden_size=hidden_size, 
                                num_layers=num_layers, 
                                batch_first=True, 
                                dropout=0.5 if num_layers > 1 else 0)
            self.dropout = nn.Dropout(0.5)
            self.fc = nn.Linear(hidden_size, num_classes)
            
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            last_out = lstm_out[:, -1, :] 
            out = self.dropout(last_out)
            return self.fc(out)

    model = EEG_LSTM(input_size=3, hidden_size=64, num_layers=2, num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)

    epochs = 80
    best_test_acc = 0.0

    print(f"\nTraining for {epochs} epochs...", flush=True)
    for epoch in range(epochs):
        model.train()
        correct, total_loss = 0, 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == batch_y).sum().item()
            
        train_acc = correct / len(X_train)

        # Validation
        model.eval()
        with torch.no_grad():
            t_outputs = model(X_test_t)
            _, t_preds = torch.max(t_outputs, 1)
            test_acc = (t_preds == y_test_t).sum().item() / len(X_test_t)
            
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")

    print("\n" + "="*30)
    print(f"Best Test Accuracy: {best_test_acc * 100:.2f}%")
    print("="*30)

if __name__ == '__main__':
    main()