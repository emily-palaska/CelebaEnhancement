import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# Neural Network Class
class ImageEnhancementNet(nn.Module):
    def __init__(self):
        super(ImageEnhancementNet, self).__init__()
        # Encoder
        self.enc1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        # Decoder
        self.dec1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec3 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

    def forward(self, x):
        # Encoder
        x = F.relu(self.bn1(self.enc1(x)))
        x = F.relu(self.bn2(self.enc2(x)))
        x = F.relu(self.bn3(self.enc3(x)))
        # Decoder
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = torch.sigmoid(self.dec3(x))
        return x

# Custom Dataset Class
class ImageEnhancementDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.tensor(x_data, dtype=torch.float32).permute(0, 3, 1, 2)  # NHWC to NCHW
        self.y_data = torch.tensor(y_data, dtype=torch.float32).permute(0, 3, 1, 2)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

# Train Function
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for degraded, clear in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            degraded, clear = degraded.to(device), clear.to(device)
            
            # Forward Pass
            outputs = model(degraded)
            loss = criterion(outputs, clear)
            
            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}")

# Evaluate Function
def evaluate_model(model, test_loader):
    model.eval()
    mse_scores = []
    with torch.no_grad():
        for degraded, clear in test_loader:
            degraded, clear = degraded.to(device), clear.to(device)
            outputs = model(degraded)
            mse = mean_squared_error(clear.cpu().numpy().flatten(), outputs.cpu().numpy().flatten())
            mse_scores.append(mse)
    avg_mse = np.mean(mse_scores)
    print(f"Average MSE on Test Set: {avg_mse:.4f}")

# Example Usage
if __name__ == "__main__":
    # Simulated Data
    train_x = np.random.rand(100, 178, 218, 3)  # 100 degraded images
    train_y = np.random.rand(100, 178, 218, 3)  # 100 clear images
    test_x = np.random.rand(20, 178, 218, 3)    # 20 degraded images
    test_y = np.random.rand(20, 178, 218, 3)    # 20 clear images
    
    # Datasets and Loaders
    train_dataset = ImageEnhancementDataset(train_x, train_y)
    test_dataset = ImageEnhancementDataset(test_x, test_y)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Model, Criterion, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageEnhancementNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train and Evaluate
    train_model(model, train_loader, criterion, optimizer, epochs=10)
    evaluate_model(model, test_loader)
