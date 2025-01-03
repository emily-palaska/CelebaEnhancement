import torch.nn as nn
import torchvision.models as models

# Generator class with ResNet18, VGG16, or default implementation
class Generator(nn.Module):
    def __init__(self, backbone=None):
        super(Generator, self).__init__()
        if backbone == "resnet18":
            self.feature_extractor = models.resnet18(pretrained=True)
            self.feature_extractor.fc = nn.Identity()  # Remove classifier
        elif backbone == "vgg16":
            self.feature_extractor = models.vgg16(pretrained=True)
            self.feature_extractor.classifier = nn.Identity()  # Remove classifier
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
            features = self.feature_extractor(x)
            features = features.view(features.size(0), 512, 1, 1)  # Reshape for transpose conv
            return self.model(features)
        return self.model(x)

# Discriminator class with ResNet18, VGG16, or default implementation
class Discriminator(nn.Module):
    def __init__(self, backbone=None):
        super(Discriminator, self).__init__()
        if backbone == "resnet18":
            self.feature_extractor = models.resnet18(pretrained=True)
            self.feature_extractor.fc = nn.Identity()  # Remove classifier
        elif backbone == "vgg16":
            self.feature_extractor = models.vgg16(pretrained=True)
            self.feature_extractor.classifier = nn.Identity()  # Remove classifier
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
            features = self.feature_extractor(x)
            return nn.Sigmoid()(nn.Flatten()(features))
        return self.model(x)
