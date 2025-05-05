#!/usr/bin/env python
"""
banach_comparison_analysis.py

This script compares two configurations of the refined autoencoder:
1. Standard refined autoencoder (without contraction regularization)
2. Refined autoencoder with contraction regularization (using a specified lambda value)

The script evaluates both models on the CIFAR-10 test set, computes PSNR, SSIM, and
inference time per image, and generates qualitative visualizations including error maps.
All metrics and visual outputs are saved for further analysis and inclusion in the research paper.

Usage:
    python banach_comparison_analysis.py --epochs 10 --lambda_contraction 0.1 --refine_iter 5 --refine_scale 0.7
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import spectral_norm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt

# Directory setup
BASE_DIR = os.path.expanduser('~/deep_learning_project')
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

#############################################
# Model Definitions with Spectral Normalization
#############################################

class Autoencoder(nn.Module):
    """Baseline convolutional autoencoder."""
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # (B,16,16,16)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # (B,32,8,8)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # (B,64,4,4)
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # (B,32,8,8)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # (B,16,16,16)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B,3,32,32)
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))

class RefinedAutoencoder(nn.Module):
    """Refined autoencoder with iterative refinement. Spectral normalization is applied
    to the refinement module to constrain its Lipschitz constant."""
    def __init__(self, base_autoencoder, num_iterations=5, refinement_scale=0.7):
        super(RefinedAutoencoder, self).__init__()
        self.base = base_autoencoder
        self.num_iterations = num_iterations
        self.refinement_scale = refinement_scale
        self.refinement = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 3, kernel_size=3, padding=1)),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(3, 3, kernel_size=3, padding=1))
        )
    
    def forward(self, x):
        x_hat = self.base(x)
        for _ in range(self.num_iterations):
            correction = self.refinement(x_hat)
            x_hat = x_hat + self.refinement_scale * correction
        return x_hat

#############################################
# Utility Functions: Noise and Evaluation Metrics
#############################################

def add_noise(images, noise_factor=0.1):
    noisy_images = images + noise_factor * torch.randn_like(images)
    return torch.clamp(noisy_images, 0., 1.)

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
# Training Functions
#############################################

def train_model_standard(model, trainloader, valloader, device, num_epochs=10, noise_factor=0.1):
    """Standard training without contraction regularization."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    best_val_loss = float('inf')
    best_model_state = None
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
        print(f"Standard Mode - Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
        scheduler.step()
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model

def train_model_contraction(model, trainloader, valloader, device, num_epochs=10, noise_factor=0.1, lambda_contraction=0.1):
    """Training with contraction regularization.
    This adds a loss term to penalize large differences between successive refinement outputs."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    best_val_loss = float('inf')
    best_model_state = None
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, _ in trainloader:
            inputs = inputs.to(device)
            noisy_inputs = add_noise(inputs, noise_factor)
            optimizer.zero_grad()
            # Forward pass with contraction regularization
            x_hat = model.base(noisy_inputs)
            contraction_loss = 0.0
            for _ in range(model.num_iterations):
                correction = model.refinement(x_hat)
                x_hat_new = x_hat + model.refinement_scale * correction
                contraction_loss += torch.mean((x_hat_new - x_hat) ** 2)
                x_hat = x_hat_new
            loss_reconstruction = criterion(x_hat, inputs)
            loss = loss_reconstruction + lambda_contraction * contraction_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        train_loss = running_loss / len(trainloader.dataset)
        
        # Validation (using standard reconstruction loss)
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
        print(f"Contraction Mode - Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
        scheduler.step()
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model

#############################################
# Evaluation Function on Test Set and Error Map Generation
#############################################

def evaluate_on_test(model, device, noise_factor=0.1, transform=None):
    if transform is None:
        transform = transforms.ToTensor()
    testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)
    model.eval()
    total_psnr, total_ssim, total_time, total_images = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for inputs, _ in testloader:
            inputs = inputs.to(device)
            noisy_inputs = add_noise(inputs, noise_factor)
            start_time = time.time()
            outputs = model(noisy_inputs)
            torch.cuda.synchronize()
            end_time = time.time()
            batch_time = end_time - start_time
            total_time += batch_time
            batch_size = inputs.size(0)
            total_images += batch_size
            total_psnr += compute_psnr(inputs, outputs) * batch_size
            total_ssim += compute_ssim(inputs, outputs) * batch_size
    avg_psnr = total_psnr / total_images
    avg_ssim = total_ssim / total_images
    avg_inference_time = (total_time / total_images) * 1000  # in ms
    return avg_psnr, avg_ssim, avg_inference_time

def generate_error_maps(model, device, noise_factor=0.1, num_images=8):
    """Generate error maps (absolute differences) for a batch of test images."""
    transform = transforms.ToTensor()
    testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=num_images, shuffle=True, num_workers=4)
    model.eval()
    with torch.no_grad():
        inputs, _ = next(iter(testloader))
        inputs = inputs.to(device)
        noisy_inputs = add_noise(inputs, noise_factor)
        outputs = model(noisy_inputs)
    error_maps = torch.abs(inputs - outputs)
    return inputs.cpu(), outputs.cpu(), error_maps.cpu()

#############################################
# Main Experiment: Compare Standard vs Contraction Modes
#############################################

def main():
    parser = argparse.ArgumentParser(description="Compare refined autoencoder training with and without contraction regularization.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--use_contraction", action="store_true", default=False, help="Enable contraction regularization")
    parser.add_argument("--lambda_contraction", type=float, default=0.1, help="Weight for contraction regularization (default: 0.1)")
    parser.add_argument("--refine_iter", type=int, default=5, help="Number of iterative refinement steps (default: 5)")
    parser.add_argument("--refine_scale", type=float, default=0.7, help="Refinement scale factor (default: 0.7)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data preparation: load CIFAR-10 with basic transformation
    transform = transforms.ToTensor()
    full_trainset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
    train_size = int(0.9 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = torch.utils.data.random_split(full_trainset, [train_size, val_size])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    valloader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=False, num_workers=4)
    
    # Instantiate the refined autoencoder model
    base_model = Autoencoder()
    model = RefinedAutoencoder(base_autoencoder=base_model, num_iterations=args.refine_iter, refinement_scale=args.refine_scale)
    model = model.to(device)
    
    # Train using the selected mode
    if args.use_contraction:
        print("Training with contraction regularization (lambda = {:.3f})".format(args.lambda_contraction))
        model = train_model_contraction(model, trainloader, valloader, device, num_epochs=args.epochs, noise_factor=0.1, lambda_contraction=args.lambda_contraction)
        mode = "contraction"
    else:
        print("Training without contraction regularization (standard iterative refinement)")
        model = train_model_standard(model, trainloader, valloader, device, num_epochs=args.epochs, noise_factor=0.1)
        mode = "standard"
    
    # Save the trained model
    model_filename = os.path.join(MODELS_DIR, f"autoencoder_refined_{mode}_lambda{args.lambda_contraction:.2f}.pth")
    torch.save(model.state_dict(), model_filename)
    print(f"Trained model saved to {model_filename}")
    
    # Evaluate on the test set
    psnr, ssim, inf_time = evaluate_on_test(model, device, noise_factor=0.1, transform=transform)
    print(f"Test Set Evaluation Metrics:\nPSNR = {psnr:.2f} dB\nSSIM = {ssim:.4f}\nInference Time per image = {inf_time:.2f} ms")
    
    # Save evaluation metrics to CSV
    results = {
        "Mode": [mode],
        "Lambda": [args.lambda_contraction],
        "PSNR": [psnr],
        "SSIM": [ssim],
        "InferenceTime_ms": [inf_time]
    }
    df = pd.DataFrame(results)
    results_file = os.path.join(RESULTS_DIR, f"banach_comparison_{mode}_lambda{args.lambda_contraction:.2f}.csv")
    df.to_csv(results_file, index=False)
    print(f"Experiment results saved to {results_file}")
    
    # Generate and save error maps for qualitative analysis
    originals, reconstructions, error_maps = generate_error_maps(model, device, noise_factor=0.1, num_images=8)
    fig, axs = plt.subplots(3, 8, figsize=(20, 8))
    for idx in range(8):
        axs[0, idx].imshow(originals[idx].permute(1,2,0))
        axs[0, idx].axis('off')
        axs[0, idx].set_title("Original")
        
        axs[1, idx].imshow(reconstructions[idx].permute(1,2,0))
        axs[1, idx].axis('off')
        axs[1, idx].set_title("Reconstructed")
        
        axs[2, idx].imshow(error_maps[idx].permute(1,2,0))
        axs[2, idx].axis('off')
        axs[2, idx].set_title("Error Map")
    plt.tight_layout()
    error_map_file = os.path.join(RESULTS_DIR, f"error_maps_{mode}_lambda{args.lambda_contraction:.2f}.png")
    plt.savefig(error_map_file)
    print(f"Error map visualization saved to {error_map_file}")
    plt.show()

if __name__ == "__main__":
    main()

