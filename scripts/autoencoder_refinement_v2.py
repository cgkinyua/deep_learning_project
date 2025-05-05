#!/usr/bin/env python
"""
autoencoder_refinement_v2.py

This script trains an autoencoder with an optional iterative refinement module
to simulate wireless image transmission with AI-driven error correction.
It evaluates the model using PSNR and SSIM metrics, profiles inference time,
and saves logs, metrics, and visualizations for research documentation.

Usage:
    python autoencoder_refinement_v2.py --refine --epochs 20

Arguments:
    --refine      : Enable iterative refinement (default: enabled)
    --epochs N    : Number of training epochs (default: 10)
    --refine_iter : Number of iterative refinement steps (default: 3)
    --refine_scale: Refinement scale factor (default: 0.5)
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.optim.lr_scheduler import StepLR

# Define directories
BASE_DIR = os.path.expanduser('~/deep_learning_project')
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Ensure required directories exist
for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

#############################################
# Model Definitions
#############################################

class Autoencoder(nn.Module):
    """Baseline convolutional autoencoder."""
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # (B, 16, 16, 16)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # (B, 32, 8, 8)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # (B, 64, 4, 4)
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # (B, 32, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # (B, 16, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 3, 32, 32)
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))

class RefinedAutoencoder(nn.Module):
    """Autoencoder with iterative refinement inspired by contraction mappings."""
    def __init__(self, base_autoencoder, num_iterations=3, refinement_scale=0.5):
        super(RefinedAutoencoder, self).__init__()
        self.base = base_autoencoder
        self.num_iterations = num_iterations
        self.refinement_scale = refinement_scale
        self.refinement = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        x_hat = self.base(x)
        for _ in range(self.num_iterations):
            correction = self.refinement(x_hat)
            x_hat = x_hat + self.refinement_scale * correction
        return x_hat

#############################################
# Noise Functions
#############################################

def add_noise(images, noise_factor=0.1):
    noisy_images = images + noise_factor * torch.randn_like(images)
    return torch.clamp(noisy_images, 0., 1.)

#############################################
# Evaluation Metrics
#############################################

def compute_psnr(original, reconstructed):
    original_np = original.cpu().numpy()
    reconstructed_np = reconstructed.cpu().numpy()
    psnr_vals = []
    for i in range(original_np.shape[0]):
        psnr_val = peak_signal_noise_ratio(
            original_np[i].transpose(1, 2, 0),
            reconstructed_np[i].transpose(1, 2, 0),
            data_range=1.0
        )
        psnr_vals.append(psnr_val)
    return np.mean(psnr_vals)

def compute_ssim(original, reconstructed):
    original_np = original.cpu().numpy()
    reconstructed_np = reconstructed.cpu().numpy()
    ssim_vals = []
    for i in range(original_np.shape[0]):
        ssim_val = structural_similarity(
            original_np[i].transpose(1, 2, 0),
            reconstructed_np[i].transpose(1, 2, 0),
            win_size=7,
            channel_axis=-1,
            data_range=1.0
        )
        ssim_vals.append(ssim_val)
    return np.mean(ssim_vals)

#############################################
# Training and Evaluation Functions
#############################################

def train_model(model, trainloader, valloader, device, num_epochs=10, noise_factor=0.1, log_file_path=None):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)  # reduce LR every 5 epochs
    best_val_loss = float('inf')
    best_model_state = None
    log_lines = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, _ in trainloader:
            inputs = inputs.to(device)
            noisy_inputs = add_noise(inputs, noise_factor)
            optimizer.zero_grad()
            outputs = model(noisy_inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        train_loss = running_loss / len(trainloader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, _ in valloader:
                inputs = inputs.to(device)
                noisy_inputs = add_noise(inputs, noise_factor)
                outputs = model(noisy_inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(valloader.dataset)
        
        log_line = f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        print(log_line)
        log_lines.append(log_line)
        
        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
        
        scheduler.step()
    
    if log_file_path:
        with open(log_file_path, 'w') as f:
            f.write("\n".join(log_lines))
    
    # Load best model state before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def evaluate_model(model, dataloader, device, noise_factor=0.1):
    model.eval()
    with torch.no_grad():
        dataiter = iter(dataloader)
        inputs, _ = next(dataiter)
        inputs = inputs.to(device)
        noisy_inputs = add_noise(inputs, noise_factor)
        
        start_time = time.time()
        outputs = model(noisy_inputs)
        torch.cuda.synchronize()
        end_time = time.time()
        
        inference_time = (end_time - start_time) / inputs.size(0)
        psnr_val = compute_psnr(inputs, outputs)
        ssim_val = compute_ssim(inputs, outputs)
    return psnr_val, ssim_val, inference_time, inputs, noisy_inputs, outputs

#############################################
# Main Function
#############################################

def main():
    parser = argparse.ArgumentParser(description="Train autoencoder with optional iterative refinement.")
    parser.add_argument("--refine", action="store_true", default=True,
                        help="Enable iterative refinement (default: enabled)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs (default: 10)")
    parser.add_argument("--refine_iter", type=int, default=3,
                        help="Number of iterative refinement steps (default: 3)")
    parser.add_argument("--refine_scale", type=float, default=0.5,
                        help="Refinement scale factor (default: 0.5)")
    args = parser.parse_args()

    # Data preparation: load CIFAR-10 dataset from DATA_DIR
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip()  # Additional augmentation
    ])
    full_trainset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
    train_size = int(0.9 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = torch.utils.data.random_split(full_trainset, [train_size, val_size])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    valloader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate the model
    base_model = Autoencoder()
    if args.refine:
        model = RefinedAutoencoder(base_autoencoder=base_model, num_iterations=args.refine_iter, refinement_scale=args.refine_scale)
    else:
        model = base_model
    model = model.to(device)

    # Train the model and log training progress
    log_file = os.path.join(LOGS_DIR, "training_log.txt")
    model = train_model(model, trainloader, valloader, device, num_epochs=args.epochs, noise_factor=0.1, log_file_path=log_file)
    
    # Save the best trained model
    model_file = os.path.join(MODELS_DIR, "autoencoder_model.pth")
    torch.save(model.state_dict(), model_file)
    print(f"Best model saved to {model_file}")
    
    # Evaluate the model on the validation set
    psnr_val, ssim_val, inference_time, originals, noisy_imgs, reconstructions = evaluate_model(model, valloader, device, noise_factor=0.1)
    print(f"Evaluation Metrics: PSNR = {psnr_val:.2f}, SSIM = {ssim_val:.4f}, Inference Time per image = {inference_time*1000:.2f} ms")
    
    # Save evaluation metrics to CSV file
    metrics = {"PSNR": [psnr_val], "SSIM": [ssim_val], "InferenceTime_ms": [inference_time*1000]}
    df_metrics = pd.DataFrame(metrics)
    metrics_file = os.path.join(RESULTS_DIR, "evaluation_metrics.csv")
    df_metrics.to_csv(metrics_file, index=False)
    print(f"Evaluation metrics saved to {metrics_file}")
    
    # Visualization: plot original, noisy, and reconstructed images for a batch from the validation set
    fig, axs = plt.subplots(3, 8, figsize=(15, 6))
    originals = originals.cpu().detach()
    noisy_imgs = noisy_imgs.cpu().detach()
    reconstructions = torch.clamp(reconstructions, 0., 1.)
    reconstructions = reconstructions.cpu().detach()
    for idx in range(8):
        axs[0, idx].imshow(originals[idx].permute(1, 2, 0))
        axs[0, idx].axis('off')
        axs[0, idx].set_title("Original")
        
        axs[1, idx].imshow(noisy_imgs[idx].permute(1, 2, 0))
        axs[1, idx].axis('off')
        axs[1, idx].set_title("Noisy")
        
        axs[2, idx].imshow(reconstructions[idx].permute(1, 2, 0))
        axs[2, idx].axis('off')
        axs[2, idx].set_title("Reconstructed")
    plt.tight_layout()
    visual_file = os.path.join(RESULTS_DIR, "reconstruction.png")
    plt.savefig(visual_file)
    print(f"Reconstruction visualization saved to {visual_file}")
    plt.show()

if __name__ == "__main__":
    main()

