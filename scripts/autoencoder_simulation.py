#!/usr/bin/env python
"""
autoencoder_simulation.py

This script implements a simple convolutional autoencoder using PyTorch to simulate
wireless image transmission. It adds noise to the input images (simulating channel impairments)
and trains the model to reconstruct the clean images.

Usage:
    python autoencoder_simulation.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define the convolutional autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder: reduces spatial dimensions and extracts features
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # (B, 16, 16, 16) for 32x32 input
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # (B, 32, 8, 8)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # (B, 64, 4, 4)
            nn.ReLU()
        )
        # Decoder: reconstructs the image from the compressed representation
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # (B, 32, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # (B, 16, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 3, 32, 32)
            nn.Sigmoid()  # Ensure output is between 0 and 1
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Function to simulate wireless channel noise
def add_noise(images, noise_factor=0.1):
    noisy_images = images + noise_factor * torch.randn_like(images)
    noisy_images = torch.clamp(noisy_images, 0., 1.)
    return noisy_images

def main():
    # Data preparation: load CIFAR-10 dataset
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

    # Set device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, _ in trainloader:
            inputs = inputs.to(device)
            noisy_inputs = add_noise(inputs, noise_factor=0.1)
            optimizer.zero_grad()
            outputs = model(noisy_inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(trainloader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Visualize results: original, noisy, and reconstructed images
    dataiter = iter(trainloader)
    images, _ = next(dataiter)
    noisy_images = add_noise(images, noise_factor=0.1)
    outputs = model(noisy_images.to(device))
    
    images = images.cpu().detach()
    noisy_images = noisy_images.cpu().detach()
    outputs = outputs.cpu().detach()

    fig, axs = plt.subplots(3, 8, figsize=(15, 6))
    for idx in range(8):
        axs[0, idx].imshow(images[idx].permute(1, 2, 0))
        axs[0, idx].axis('off')
        axs[0, idx].set_title("Original")
        
        axs[1, idx].imshow(noisy_images[idx].permute(1, 2, 0))
        axs[1, idx].axis('off')
        axs[1, idx].set_title("Noisy")
        
        axs[2, idx].imshow(outputs[idx].permute(1, 2, 0))
        axs[2, idx].axis('off')
        axs[2, idx].set_title("Reconstructed")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

