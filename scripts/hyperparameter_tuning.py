#!/usr/bin/env python
"""
hyperparameter_tuning.py

This script performs hyperparameter tuning for the iterative refinement module
of an autoencoder designed for wireless image transmission. It runs a grid search
over specified numbers of refinement iterations and refinement scale factors.
For each combination, the script trains the model for a fixed number of epochs,
evaluates it on a validation set using PSNR and SSIM metrics, and logs the results.

Usage:
    python hyperparameter_tuning.py --epochs 10

Command-line Arguments:
    --epochs       : Number of training epochs for each hyperparameter setting (default: 10)
    --refine_steps : List of refinement iteration counts to try (default: 1,3,5)
    --refine_scales: List of refinement scale factors to try (default: 0.3,0.5,0.7)
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
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Directories setup
BASE_DIR = os.path.expanduser('~/deep_learning_project')
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

#############################################
# Model Definitions (same as in autoencoder_refinement_v2.py)
#############################################

class Autoencoder(nn.Module):
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
# Training and Evaluation Functions
#############################################

def train_model(model, trainloader, valloader, device, num_epochs=10, noise_factor=0.1):
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
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
        
        scheduler.step()
    
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
    return psnr_val, ssim_val, inference_time

#############################################
# Main Hyperparameter Tuning Loop
#############################################

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for iterative refinement in autoencoder.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs per hyperparameter setting (default: 10)")
    parser.add_argument("--refine_steps", nargs='+', type=int, default=[1, 3, 5],
                        help="List of refinement iteration counts to try (default: [1,3,5])")
    parser.add_argument("--refine_scales", nargs='+', type=float, default=[0.3, 0.5, 0.7],
                        help="List of refinement scale factors to try (default: [0.3, 0.5, 0.7])")
    args = parser.parse_args()
    
    # Data preparation: load CIFAR-10 with augmentation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip()
    ])
    full_trainset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
    train_size = int(0.9 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = torch.utils.data.random_split(full_trainset, [train_size, val_size])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    valloader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=False, num_workers=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Grid search over hyperparameters
    results = []
    for steps in args.refine_steps:
        for scale in args.refine_scales:
            print(f"\n--- Tuning: Refinement Steps = {steps}, Refinement Scale = {scale} ---")
            base_model = Autoencoder()
            model = RefinedAutoencoder(base_autoencoder=base_model, num_iterations=steps, refinement_scale=scale)
            model = model.to(device)
            model = train_model(model, trainloader, valloader, device, num_epochs=args.epochs, noise_factor=0.1)
            psnr_val, ssim_val, inf_time = evaluate_model(model, valloader, device, noise_factor=0.1)
            print(f"Result for steps={steps}, scale={scale}: PSNR = {psnr_val:.2f}, SSIM = {ssim_val:.4f}, Inference Time = {inf_time*1000:.2f} ms")
            results.append({
                "Refinement_Steps": steps,
                "Refinement_Scale": scale,
                "PSNR": psnr_val,
                "SSIM": ssim_val,
                "InferenceTime_ms": inf_time*1000
            })
            # Optionally, save model for each configuration
            model_filename = os.path.join(MODELS_DIR, f"autoencoder_refine_steps{steps}_scale{scale}.pth")
            torch.save(model.state_dict(), model_filename)
            print(f"Model saved to {model_filename}")
    
    # Save hyperparameter tuning results to CSV
    df_results = pd.DataFrame(results)
    results_file = os.path.join(RESULTS_DIR, "hyperparameter_tuning_results.csv")
    df_results.to_csv(results_file, index=False)
    print(f"\nHyperparameter tuning results saved to {results_file}")

if __name__ == "__main__":
    main()

