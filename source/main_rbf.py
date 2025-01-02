from celeba import CelebADataset
from rbf_net import ImageEnhancementRBFNet
from utils import evaluate_image_quality, save_metrics_to_json, ImageEnhancementDataset
from train_loops import train_model
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

def main():
    # Training parameters
    num_samples=10000
    num_epochs=100
    batch_size=16
    lr = 0.001
    file_name = f"../results/rbf_s{num_samples}_e{num_epochs}_bs{batch_size}_lr{lr}.json"

    # Initialize dataset
    dataset = CelebADataset(noise=True, num_samples=num_samples)
    dataset.load()

    # Datasets and Loaders
    x_train, y_train, x_test, y_test = dataset.get_train_test_split()
    train_dataset = ImageEnhancementDataset(x_train, y_train)
    test_dataset = ImageEnhancementDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, Criterion, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageEnhancementRBFNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train and evaluate
    results = train_model(model, train_loader, criterion, optimizer, device, epochs=num_epochs)
    results['test'] = evaluate_image_quality(model, test_loader, device)
    
    # Save results to JSON file
    print(results)
    save_metrics_to_json(results, file_name)


if __name__ == '__main__':
    main()


