#!/usr/bin/env python
"""
ablation_and_data_augmentation.py

This script conducts two experiments:
1. Ablation Study: Compares the baseline autoencoder (no refinement)
   with the refined autoencoder (with iterative refinement).
2. Data Augmentation Experiment: Compares a basic augmentation strategy
   with an extended augmentation strategy when training the refined model.

Evaluation metrics (PSNR, SSIM, inference time) are computed on the CIFAR-10 test set,
and all results are saved to CSV files for documentation.

Usage:
    python ablation_and_data_augmentation.py --epochs 10
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
import matplotlib.pyplot as plt

# Directories
BASE_DIR = os.path.expanduser('~/deep_learning_project')
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
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
    def __init__(self, base_autoencoder, num_iterations=3, refinement_scale=0.7):
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

#############################################
# Main Experiment: Ablation Study and Data Augmentation
#############################################

def main():
    parser = argparse.ArgumentParser(description="Ablation study and data augmentation experiments.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs per experiment (default: 10)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define two augmentation strategies:
    # 1. Basic: only convert to tensor
    basic_transform = transforms.ToTensor()
    
    # 2. Extended: random horizontal flip, random crop, and color jitter
    extended_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor()
    ])
    
    # Load CIFAR-10 training data for ablation study (using basic augmentation)
    full_trainset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=basic_transform)
    train_size = int(0.9 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = torch.utils.data.random_split(full_trainset, [train_size, val_size])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    valloader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=False, num_workers=4)
    
    # Test set loader (basic augmentation)
    test_loader_basic = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=basic_transform),
        batch_size=128, shuffle=False, num_workers=4)
    
    # Ablation study: compare baseline vs refined autoencoder (using basic augmentation)
    experiments = []
    for model_type in ["baseline", "refined"]:
        print(f"\n--- Ablation Study: Model = {model_type} ---")
        base_model = Autoencoder()
        if model_type == "refined":
            model = RefinedAutoencoder(base_autoencoder=base_model, num_iterations=5, refinement_scale=0.7)
        else:
            model = base_model
        model = model.to(device)
        model = train_model(model, trainloader, valloader, device, num_epochs=args.epochs, noise_factor=0.1)
        
        psnr, ssim, inf_time = evaluate_on_test(model, device, noise_factor=0.1, transform=basic_transform)
        print(f"Ablation Result for {model_type}: PSNR = {psnr:.2f}, SSIM = {ssim:.4f}, Inference Time = {inf_time:.2f} ms")
        experiments.append({
            "Experiment": "Ablation",
            "Model_Type": model_type,
            "Augmentation": "Basic",
            "PSNR": psnr,
            "SSIM": ssim,
            "InferenceTime_ms": inf_time
        })
        # Save the model
        model_filename = os.path.join(MODELS_DIR, f"autoencoder_{model_type}_basic.pth")
        torch.save(model.state_dict(), model_filename)
        print(f"Model saved to {model_filename}")
    
    # Data Augmentation Experiment: using refined autoencoder with best settings from ablation
    print("\n--- Data Augmentation Experiment (Refined Autoencoder) ---")
    for aug_name, transform_strategy in zip(["Basic", "Extended"], [basic_transform, extended_transform]):
        print(f"\n--- Using {aug_name} Augmentation ---")
        # Load dataset with the specified augmentation
        full_trainset_aug = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform_strategy)
        train_size = int(0.9 * len(full_trainset_aug))
        val_size = len(full_trainset_aug) - train_size
        trainset_aug, valset_aug = torch.utils.data.random_split(full_trainset_aug, [train_size, val_size])
        trainloader_aug = torch.utils.data.DataLoader(trainset_aug, batch_size=128, shuffle=True, num_workers=4)
        valloader_aug = torch.utils.data.DataLoader(valset_aug, batch_size=128, shuffle=False, num_workers=4)
        
        # Use refined autoencoder for augmentation experiments
        base_model = Autoencoder()
        model = RefinedAutoencoder(base_autoencoder=base_model, num_iterations=5, refinement_scale=0.7)
        model = model.to(device)
        model = train_model(model, trainloader_aug, valloader_aug, device, num_epochs=args.epochs, noise_factor=0.1)
        
        # Evaluate on test set with corresponding augmentation transform
        psnr, ssim, inf_time = evaluate_on_test(model, device, noise_factor=0.1, transform=transform_strategy)
        print(f"Augmentation Result ({aug_name}): PSNR = {psnr:.2f}, SSIM = {ssim:.4f}, Inference Time = {inf_time:.2f} ms")
        experiments.append({
            "Experiment": "Augmentation",
            "Model_Type": "Refined",
            "Augmentation": aug_name,
            "PSNR": psnr,
            "SSIM": ssim,
            "InferenceTime_ms": inf_time
        })
        model_filename = os.path.join(MODELS_DIR, f"autoencoder_refined_{aug_name}.pth")
        torch.save(model.state_dict(), model_filename)
        print(f"Model saved to {model_filename}")
    
    # Save all experiment results to CSV
    df_experiments = pd.DataFrame(experiments)
    results_file = os.path.join(RESULTS_DIR, "ablation_augmentation_results.csv")
    df_experiments.to_csv(results_file, index=False)
    print(f"\nAblation and augmentation experiment results saved to {results_file}")

if __name__ == "__main__":
    main()

