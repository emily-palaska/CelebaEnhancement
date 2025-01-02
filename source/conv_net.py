import torch
import torch.nn as nn
import torch.nn.functional as f

class ImageEnhancementConvNet(nn.Module):
    def __init__(self):
        super(ImageEnhancementConvNet, self).__init__()
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
        x = f.relu(self.bn1(self.enc1(x)))
        x = f.relu(self.bn2(self.enc2(x)))
        x = f.relu(self.bn3(self.enc3(x)))
        # Decoder
        x = f.relu(self.dec1(x))
        x = f.relu(self.dec2(x))
        x = torch.sigmoid(self.dec3(x))    
        return x
