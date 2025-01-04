import time
from tqdm import tqdm
import torch

# Train Loop for conv and RBF
def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    # Initialize output file
    with open("output.txt", "w") as file:
        file.write(f"")
    loss_per_epoch = []
    duration_per_epoch = []
    model.train()
    for epoch in range(epochs):
        start_time = time.time()
        epoch_loss = 0
        for degraded, clear in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            degraded, clear = degraded.to(device), clear.to(device)

            # Forward Pass
            outputs = model(degraded)
            loss = criterion(outputs, clear)

            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        end_time = time.time()
        loss_per_epoch.append(epoch_loss / len(train_loader))
        duration_per_epoch.append(end_time - start_time)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader):.4f}, Time: {end_time - start_time : .2f}")
        with open("output.txt", "a") as file:
            file.write(
                f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader):.4f}, Time: {end_time - start_time : .2f}\n")
    return {'loss': loss_per_epoch, 'duration': duration_per_epoch}

# Training Loop for GAN
def train_gan(generator, discriminator, train_loader, optimizer_g, optimizer_d, criterion, device, epochs=10):
    g_losses = []
    d_losses = []
    duration_per_epoch = []
    for epoch in range(epochs):
        start_time = time.time()
        g_epoch_loss = 0
        d_epoch_loss = 0

        for degraded, clear in train_loader:
            degraded, clear = degraded.to(device), clear.to(device)

            # Train Discriminator
            optimizer_d.zero_grad()
            real_preds = discriminator(clear)
            fake_images = generator(degraded)
            fake_preds = discriminator(fake_images.detach())

            real_loss = criterion(real_preds, torch.ones_like(real_preds))
            fake_loss = criterion(fake_preds, torch.zeros_like(fake_preds))
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            fake_preds = discriminator(fake_images)
            fake_images = (fake_images - fake_images.min()) / (fake_images.max() - fake_images.min())
            clear = (clear - clear.min()) / (clear.max() - clear.min())
            g_loss = criterion(fake_preds, torch.ones_like(fake_preds)) + criterion(fake_images, clear)
            g_loss.backward()
            optimizer_g.step()

            g_epoch_loss += g_loss.item()
            d_epoch_loss += d_loss.item()

        end_time = time.time()
        g_losses.append(g_epoch_loss / len(train_loader))
        d_losses.append(d_epoch_loss / len(train_loader))
        duration_per_epoch.append(end_time - start_time)
        print(
            f"Epoch {epoch + 1}/{epochs}, Generator Loss: {g_epoch_loss / len(train_loader):.4f}, Discriminator Loss: {d_epoch_loss / len(train_loader):.4f}, Time: {end_time - start_time : .2f}")

    return {'g_losses': g_losses, 'd_losses': d_losses, 'duration': duration_per_epoch}
