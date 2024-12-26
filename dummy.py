from torchvision.datasets import CelebA
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

# Set up transformation
transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor()
])

# Load dataset from the manually downloaded folder
dataset = CelebA(
    root='data/',
    split='train',
    download=True,
    transform=transform
)

# Create dataloader
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
