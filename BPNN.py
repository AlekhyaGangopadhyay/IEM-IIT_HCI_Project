'''Final Train Accuracy: 78.26%
   Final Test Accuracy:  20.83%
Best Test Accuracy:   33.33% '''

import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def main():
    
    torch.manual_seed(42)
    np.random.seed(42)

    labels_map = {'LEFT': 0, 'RIGHT': 1, 'FORWARD': 2, 'BACKWARD': 3}
    X_all = []
    y_all = []

    print("Reading Excel files...", flush=True)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_list = glob.glob(os.path.join(script_dir, '*/*.xlsx'))

    for file in file_list:
        dir_name = os.path.basename(os.path.dirname(file)).upper()
        if dir_name in labels_map:
            try:
                # Read only columns 7, 14, and 15 to significantly speed up loading
                df = pd.read_excel(file, usecols=[7, 14, 15], engine='openpyxl')
                data = df.values
                
                if data.shape[1] == 3:
                    X_all.append(data)
                    y_all.append(labels_map[dir_name])
                else:
                    print(f"Skipped {file} due to wrong column count: {data.shape[1]}")
            except Exception as e:
                print(f"Error reading {file}: {e}", flush=True)

    if len(X_all) == 0:
        print("No valid files found.", flush=True)
        return

    print(f"Loaded {len(X_all)} files.", flush=True)

    print("Extracting statistical features (Mean, Std, Var, Min, Max, PTP) from EEG signals to prevent overfitting...", flush=True)
    
    X_features = []
    for x in X_all:
        mean_vals = np.mean(x, axis=0)
        std_vals = np.std(x, axis=0)
        var_vals = np.var(x, axis=0)
        min_vals = np.min(x, axis=0)
        max_vals = np.max(x, axis=0)
        ptp_vals = max_vals - min_vals
        
        # Concatenate features into a single vector (18 features total)
        features = np.concatenate([mean_vals, std_vals, var_vals, min_vals, max_vals, ptp_vals])
        X_features.append(features)
        
    X = np.stack(X_features) 
    y = np.array(y_all)
    
    # Train test split (80-20)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_idx = int(0.8 * len(indices))
    
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_train = (X_train - X_mean) / (X_std + 1e-8)
    X_test = (X_test - X_mean) / (X_std + 1e-8)
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}", flush=True)
    
    # Convert vectors to PyTorch Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Define BPNN (Multi-Layer Perceptron) architecture tailored for small feature set
    class BPNN(nn.Module):
        def __init__(self, input_size, num_classes=4):
            super(BPNN, self).__init__()
            # Simpler network to avoid overfitting with small datasets
            self.net = nn.Sequential(
                nn.Linear(input_size, 32),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(16, num_classes)
            )
            
        def forward(self, x):
            return self.net(x)
            
    input_size = X_train_t.shape[1]
    model = BPNN(input_size=input_size, num_classes=4)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)
    
    # Train
    epochs = 100
    print(f"Training for {epochs} epochs...", flush=True)
    
    best_test_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == batch_y).item()
            
        train_acc = correct / len(train_dataset)
        
        # Validation on test to keep track of overfitting
        model.eval()
        with torch.no_grad():
            t_outputs = model(X_test_t)
            _, t_preds = torch.max(t_outputs, 1)
            test_acc = torch.sum(t_preds == y_test_t).item() / len(y_test_t)
            
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}", flush=True)

    # Test Evaluation
    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train_t)
        _, train_preds = torch.max(train_outputs, 1)
        train_accuracy = torch.sum(train_preds == y_train_t).item() / len(y_train_t)
        
        test_outputs = model(X_test_t)
        _, test_preds = torch.max(test_outputs, 1)
        test_accuracy = torch.sum(test_preds == y_test_t).item() / len(y_test_t)
        
    print("-" * 40, flush=True)
    print(f"Final Train Accuracy: {train_accuracy * 100:.2f}%", flush=True)
    print(f"Final Test Accuracy:  {test_accuracy * 100:.2f}%", flush=True)
    print(f"Best Test Accuracy:   {best_test_acc * 100:.2f}%", flush=True)

if __name__ == '__main__':
    main()
