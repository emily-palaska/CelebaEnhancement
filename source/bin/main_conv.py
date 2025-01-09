from celeba import CelebADataset
from conv_net import ImageEnhancementConvNet
from utils import evaluate_image_quality, save_metrics_to_json, ImageEnhancementDataset, plot_examples_with_predictions
from train_loops import train_model
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

def main():
    # Training Parameters
    num_samples = 10000
    num_epochs = 50
    batch_size = 32
    noise = False
    lr = 0.0001
    file_name = 'noise' if noise else 'celeba'
    file_name += f"_gan_s{num_samples}_e{num_epochs}_bs{batch_size}_lr{lr}"

    # Dataset and DatLoader
    dataset = CelebADataset(noise=noise, num_samples=num_samples)
    dataset.load()
    x_train, y_train, x_test, y_test = dataset.get_train_test_split()
    train_dataset = ImageEnhancementDataset(x_train, y_train, resize=True)
    test_dataset = ImageEnhancementDataset(x_test, y_test, resize=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize Models, Optimizers, and Criterion
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageEnhancementConvNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train and evaluate
    results = train_model(model, train_loader, criterion, optimizer, device, epochs=num_epochs)
    metrics, x, y, y_hat = evaluate_image_quality(model, test_loader, device)

    # Plot and save results
    results['test'] = metrics
    plot_examples_with_predictions(x, y, y_hat, save_path=f'../{file_name}.png',
                                   title=f'CONV Image Enhancement')
    save_metrics_to_json(results, f'../results/{file_name}.json')

if __name__ == '__main__':
    main()


