import torch
import torch.nn as nn

class ImageEnhancementRBFNet(nn.Module):
    def __init__(self, num_rbf_neurons=256, img_size=(178, 218), num_channels=3):
        super(ImageEnhancementRBFNet, self).__init__()
        self.num_rbf_neurons = num_rbf_neurons
        self.img_size = img_size
        self.num_channels = num_channels

        # RBF centers and widths
        self.centers = nn.Parameter(torch.randn(num_rbf_neurons, num_channels, *img_size))
        self.log_sigmas = nn.Parameter(torch.zeros(num_rbf_neurons))

        # Fully connected layer to map RBF outputs back to an image
        self.fc = nn.Linear(num_rbf_neurons, num_channels * img_size[0] * img_size[1])
        
    def rbf(self, x, centers, log_sigmas):
        """
        Compute RBF activations.
        """
        x = x.view(x.size(0), self.num_channels, -1)  # Flatten spatial dimensions
        centers = centers.view(self.num_rbf_neurons, self.num_channels, -1)
        diff = x.unsqueeze(1) - centers.unsqueeze(0)  # [batch_size, num_rbf_neurons, num_channels, num_pixels]
        diff = diff.pow(2).sum(dim=2)  # Squared L2 norm across channels
        sigma_squared = torch.exp(2 * log_sigmas).unsqueeze(0).unsqueeze(-1)  # Expand for broadcasting
        activations = torch.exp(-diff / (2 * sigma_squared))
        return activations.sum(dim=2)  # Sum across pixels

    def forward(self, x):
        # Compute RBF activations
        rbf_outputs = self.rbf(x, self.centers, self.log_sigmas)

        # Map activations back to an image
        out = self.fc(rbf_outputs)
        out = out.view(x.size(0), self.num_channels, *self.img_size)

        # Apply activation function to match image range
        out = torch.sigmoid(out)
        return out
