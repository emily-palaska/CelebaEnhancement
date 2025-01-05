import torch.nn as nn
import torchvision.models as models
import torch

class Generator(nn.Module):
    def __init__(self, backbone=None):
        super(Generator, self).__init__()
        if backbone == "resnet18":
            # Use ResNet18 backbone
            self.feature_extractor = models.resnet18(weights='IMAGENET1K_V1')  # Use latest weights API
            self.feature_extractor.fc = nn.Identity()  # Remove classifier
            # Now add layers to convert feature map to 180x220x3
            self.conv = nn.Conv2d(512, 3, kernel_size=3, stride=1, padding=1)  # Ensure 3 output channels
            self.upsample = nn.Upsample(size=(180, 220), mode='bilinear', align_corners=False)

        elif backbone == "vgg16":
            # Use VGG16 backbone
            self.feature_extractor = models.vgg16(weights='IMAGENET1K_V1')  # Use latest weights API
            self.feature_extractor.classifier = nn.Identity()  # Remove classifier
            # Now add layers to convert feature map to 180x220x3
            self.conv = nn.Conv2d(512, 3, kernel_size=3, stride=1, padding=1)  # Ensure 3 output channels
            self.upsample = nn.Upsample(size=(180, 220), mode='bilinear', align_corners=False)

        else:
            # Default Generator Implementation
            self.model = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
                nn.Tanh()
            )

    def forward(self, x):
        if hasattr(self, 'feature_extractor'):
            features = self.feature_extractor(x)  # Extract features
            features = features.view(features.size(0), 512, 1, 1)  # Reshape for transpose conv
            features = self.conv(features)  # Generate the image
            return self.upsample(features)  # Upsample to 180x220
        return self.model(x)  # Use default model


class Discriminator(nn.Module):
    def __init__(self, backbone=None):
        super(Discriminator, self).__init__()
        if backbone == "resnet18":
            # Use ResNet18 backbone
            self.feature_extractor = models.resnet18(weights='IMAGENET1K_V1')  # Use latest weights API
            self.feature_extractor.fc = nn.Identity()  # Remove classifier
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(512, 1)

        elif backbone == "vgg16":
            # Use VGG16 backbone
            self.feature_extractor = models.vgg16(weights='IMAGENET1K_V1')  # Use latest weights API
            self.feature_extractor.classifier = nn.Identity()  # Remove classifier
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(512, 1)

        else:
            # Default Discriminator Implementation
            self.model = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                nn.Flatten(),
                nn.Linear(256 * 644, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        if hasattr(self, 'feature_extractor'):
            features = self.feature_extractor(x)  # Extract features from ResNet18 or VGG16
            features = self.flatten(features)  # Flatten features for the final layer
            return torch.sigmoid(self.fc(features))  # Output a single probability
        return self.model(x)  # Use default model