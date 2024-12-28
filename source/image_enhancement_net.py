import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
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

# Train Function
def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
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

# Custom Dataset Class
class ImageEnhancementDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.tensor(x_data, dtype=torch.float32).permute(0, 3, 1, 2)  # NHWC to NCHW
        self.y_data = torch.tensor(y_data, dtype=torch.float32).permute(0, 3, 1, 2)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]
    