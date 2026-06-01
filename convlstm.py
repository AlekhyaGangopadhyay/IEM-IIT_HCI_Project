# ============================================================
# END-TO-END CONVLSTM EEG INTENT CLASSIFIER
# ============================================================

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score


X_train = np.load("/kaggle/input/datasets/alekhya7gangopadhyay/eeg-processed/X_train_500.npy")
X_test  = np.load("/kaggle/input/datasets/alekhya7gangopadhyay/eeg-processed/X_test_500.npy")
y_train = np.load("/kaggle/input/datasets/alekhya7gangopadhyay/eeg-processed/y_train_cls_500.npy")
y_test  = np.load("/kaggle/input/datasets/alekhya7gangopadhyay/eeg-processed/y_test_cls_500.npy")

print("Dataset Domain Arrays Loaded.")
print(f"X_train Shape: {X_train.shape} -> y_train Labels: {y_train.shape}")
print(f"X_test Shape : {X_test.shape}  -> y_test Labels : {y_test.shape}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training ConvLSTM Classifier on platform:", device)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)
y_test_t  = torch.tensor(y_test, dtype=torch.long)



BATCH_SIZE = 256
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)



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
        
        final_state = hidden[-1]
        
        logits = self.classifier_head(final_state)
        return logits



model = EEG_ConvLSTM_Classifier(input_dim=3, num_classes=4).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)



EPOCHS = 20
train_losses, test_losses = [], []
train_accs, test_accs = [], []
best_test_acc = 0.0

print("\nStarting ConvLSTM Classifier Model Training...")
for epoch in range(EPOCHS):
    
    model.train()
    total_train_loss = 0
    correct_train = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        logits = model(X_batch)
        
        loss = criterion(logits, y_batch)
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0) 
        optimizer.step()
        
        total_train_loss += loss.item() * X_batch.size(0)
        preds = torch.argmax(logits, dim=1)
        correct_train += (preds == y_batch).sum().item()
        
    avg_train_loss = total_train_loss / len(train_loader.dataset)
    epoch_train_acc = (correct_train / len(train_loader.dataset)) * 100
    
    model.eval()
    total_test_loss = 0
    correct_test = 0
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            total_test_loss += loss.item() * X_batch.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_test += (preds == y_batch).sum().item()
            
    avg_test_loss = total_test_loss / len(test_loader.dataset)
    epoch_test_acc = (correct_test / len(test_loader.dataset)) * 100
    
    scheduler.step(epoch_test_acc)
    
    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)
    train_accs.append(epoch_train_acc)
    test_accs.append(epoch_test_acc)
    
    if epoch_test_acc > best_test_acc:
        best_test_acc = epoch_test_acc
        torch.save(model.state_dict(), "/kaggle/working/EEG_ConvLSTM_classifier.pth")
        
    print(f"Epoch [{epoch+1:02d}/{EPOCHS}] "
          f"| Train Loss: {avg_train_loss:.4f} ({epoch_train_acc:.2f}% Acc) "
          f"| Test Loss: {avg_test_loss:.4f} ({epoch_test_acc:.2f}% Test Acc)")

print(f"\nTraining Complete. Best ConvLSTM Test Accuracy: {best_test_acc:.2f}%")
print("Saved production weights to: /kaggle/working/EEG_ConvLSTM_classifier.pth")