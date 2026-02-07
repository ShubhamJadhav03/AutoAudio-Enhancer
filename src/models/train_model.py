import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# Get project root directory (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# --- DEFINE THE CLASS AT THE TOP (Global) ---
class AudioCNN_Pro(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # Block 1: Detect basic edges/frequencies
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Block 2: Detect textures
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Block 3: Detect complex patterns
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Compute Flatten Size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 40, 130)
            out = self.layer3(self.layer2(self.layer1(dummy)))
            self.flatten_dim = out.view(1, -1).shape[1]

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4), # Drop 40% of neurons to prevent memorization
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# --- THE GATEKEEPER ---
if __name__ == "__main__":
    # ⚠️ INDENT EVERYTHING BELOW THIS LINE
    
    print("⚔️ Starting Training Mode...")
    
    # --- SETUP ---
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data/processed")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Training on: {device}")

    # --- DATA LOADING ---
    X = np.load(os.path.join(PROCESSED_DIR, "X.npy"))
    y = np.load(os.path.join(PROCESSED_DIR, "y.npy"))

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    np.save(os.path.join(MODEL_DIR, "label_encoder.npy"), label_encoder.classes_)

    # Reshape for CNN: (N, 1, 40, 130)
    X = X[:, np.newaxis, :, :]

    # Tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y_tensor
    )

    # Dataloaders (Increased batch size to 32 for stability)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

    # --- TRAINING ---
    model = AudioCNN_Pro(len(label_encoder.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 25 

    print(" Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Track training accuracy to spot overfitting
            _, predicted = torch.max(outputs.data, 1)
            total += yb.size(0)
            correct += (predicted == yb).sum().item()

        train_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {running_loss/len(train_loader):.4f} - Train Acc: {train_acc:.2f}%")

    # --- EVALUATION ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            _, predicted = torch.max(outputs, 1)
            total += yb.size(0)
            correct += (predicted == yb).sum().item()

    accuracy = 100 * correct / total
    print(f"\n Final Test Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "audio_cnn.pth"))
    print("💾 Model Saved.")