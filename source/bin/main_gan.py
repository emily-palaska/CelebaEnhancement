import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import ImageEnhancementDataset, evaluate_image_quality, save_metrics_to_json, plot_examples_with_predictions
from train_loops import train_gan
from celeba import CelebADataset
from gan_net import Generator, Discriminator

# Main Function
def main():
    # Training Parameters
    num_samples = 75000
    num_epochs = 20
    batch_size = 64
    backbone = 'defaultdeep'
    noise = False
    g_lr = 1e-3
    d_lr = 1e-4 / 2
    file_name = 'noise' if noise else 'celeba'
    file_name += f"_gan_{backbone}_s{num_samples}_e{num_epochs}_bs{batch_size}_glr{g_lr}_dlr{d_lr}"
    print(file_name)

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
    generator = Generator(backbone=backbone).to(device)
    discriminator = Discriminator(backbone=backbone).to(device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=g_lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=d_lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # Train and evaluate the GAN
    results = train_gan(generator, discriminator, train_loader, optimizer_g, optimizer_d, criterion, device, epochs=num_epochs)
    metrics, x, y, y_hat = evaluate_image_quality(generator, test_loader, device)

    # Plot and save results
    results['test'] = metrics
    plot_examples_with_predictions(x, y, y_hat, save_path=f'../plots/{file_name}.png', title=f'GAN Image Enhancement ({backbone} backbone)')
    print("Examples saved")
    save_metrics_to_json(results, f'../results/{file_name}.json')

if __name__ == '__main__':
    main()