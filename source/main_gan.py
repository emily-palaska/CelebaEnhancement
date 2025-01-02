import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import ImageEnhancementDataset, evaluate_image_quality, save_metrics_to_json
from train_loops import train_gan
from celeba import CelebADataset
from gan_net import Generator, Discriminator

# Main Function
def main():
    # Training Parameters
    num_samples = 10000
    num_epochs = 100
    batch_size = 16
    lr = 0.0001
    file_name = f"../results/gan_s{num_samples}_e{num_epochs}_bs{batch_size}_lr{lr}.json"

    # Dataset and DatLoader
    dataset = CelebADataset(num_samples=num_samples)
    dataset.load()
    x_train, y_train, x_test, y_test = dataset.get_train_test_split()
    train_dataset = ImageEnhancementDataset(x_train, y_train, resize=True)
    test_dataset = ImageEnhancementDataset(x_test, y_test, resize=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize Models, Optimizers, and Criterion
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # Train the GAN
    results = train_gan(generator, discriminator, train_loader, optimizer_g, optimizer_d, criterion, device, epochs=num_epochs)
    results['test'] = evaluate_image_quality(generator, test_loader, device, title='Evaluation of GAN Image Enhancement')

    # Save Results
    print(results)
    save_metrics_to_json(results, file_name)

if __name__ == '__main__':
    main()