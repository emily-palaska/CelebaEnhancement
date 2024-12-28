from celeba import CelebADataset
from image_enhancement_net import ImageEnhancementDataset, ImageEnhancementNet, train_model
from utils import evaluate_image_quality
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

def main():
    # Initiliaze dataset
    dataset = CelebADataset(noise=True, num_samples=10000)
    dataset.load()    

    # Datasets and Loaders
    x_train, y_train, x_test, y_test = dataset.get_train_test_split()
    train_dataset = ImageEnhancementDataset(x_train, y_train)
    test_dataset = ImageEnhancementDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Model, Criterion, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageEnhancementNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train and evaluate
    train_model(model, train_loader, criterion, optimizer, device, epochs=1)
    metrics = evaluate_image_quality(model, test_loader, device, json_path="..\results\10000samples_10epochs.json")
    print(metrics)


if __name__ == '__main__':
    main()


