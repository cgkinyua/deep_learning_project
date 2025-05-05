#!/usr/bin/env python
"""
research_extended_pipeline.py

This script integrates extended experiments to deepen the integration of Banach
space ideals into our deep learning model. It includes:

1. Dynamic λ scheduling for contraction regularization.
2. Training with advanced data augmentation.
3. Extended evaluation under various noise conditions (Gaussian and burst).
4. Real-time streaming simulation.
5. Generation of qualitative visualizations (error maps).
6. Saving all experiment logs, evaluation metrics (CSV), and visual outputs.

Usage:
    python research_extended_pipeline.py --epochs 20 --base_lambda 0.01 --final_lambda 0.1 --use_dynamic_lambda --advanced_aug
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
    """Refined autoencoder with iterative refinement and spectral normalization."""
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

def add_gaussian_noise(images, noise_factor=0.1):
    noisy_images = images + noise_factor * torch.randn_like(images)
    return torch.clamp(noisy_images, 0., 1.)

def add_burst_noise(images, burst_prob=0.3, block_size=8):
    """Simulate burst noise by zeroing out a random block in each image."""
    noisy_images = images.clone()
    B, C, H, W = noisy_images.shape
    for i in range(B):
        if np.random.rand() < burst_prob:
            x = np.random.randint(0, W - block_size)
            y = np.random.randint(0, H - block_size)
            noisy_images[i, :, y:y+block_size, x:x+block_size] = 0.0
    return noisy_images

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
# Dynamic Lambda Scheduling
#############################################

def dynamic_lambda(epoch, total_epochs, base_lambda, final_lambda):
    """Linearly increase lambda from base_lambda to final_lambda over training."""
    return base_lambda + (final_lambda - base_lambda) * (epoch / total_epochs)

#############################################
# Training Functions
#############################################

def train_model_dynamic_contraction(model, trainloader, valloader, device, num_epochs=10, noise_factor=0.1, base_lambda=0.01, final_lambda=0.1):
    """Train with dynamic contraction regularization.
    The contraction loss weight is increased linearly from base_lambda to final_lambda."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        current_lambda = dynamic_lambda(epoch, num_epochs, base_lambda, final_lambda)
        model.train()
        running_loss = 0.0
        for inputs, _ in trainloader:
            inputs = inputs.to(device)
            noisy_inputs = add_gaussian_noise(inputs, noise_factor)
            optimizer.zero_grad()
            x_hat = model.base(noisy_inputs)
            contraction_loss = 0.0
            for _ in range(model.num_iterations):
                correction = model.refinement(x_hat)
                x_hat_new = x_hat + model.refinement_scale * correction
                contraction_loss += torch.mean((x_hat_new - x_hat) ** 2)
                x_hat = x_hat_new
            loss_reconstruction = criterion(x_hat, inputs)
            loss = loss_reconstruction + current_lambda * contraction_loss
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
                noisy_inputs = add_gaussian_noise(inputs, noise_factor)
                outputs = model(noisy_inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(valloader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Dynamic λ: {current_lambda:.3f}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
        scheduler.step()
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model

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
            noisy_inputs = add_gaussian_noise(inputs, noise_factor)
            optimizer.zero_grad()
            outputs = model(noisy_inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        train_loss = running_loss / len(trainloader.dataset)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, _ in valloader:
                inputs = inputs.to(device)
                noisy_inputs = add_gaussian_noise(inputs, noise_factor)
                outputs = model(noisy_inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(valloader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Standard, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
        scheduler.step()
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model

#############################################
# Evaluation Functions
#############################################

def evaluate_on_test(model, device, noise_type="gaussian", noise_param=0.1, transform=None):
    if transform is None:
        transform = transforms.ToTensor()
    testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)
    model.eval()
    total_psnr, total_ssim, total_time, total_images = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for inputs, _ in testloader:
            inputs = inputs.to(device)
            if noise_type == "gaussian":
                noisy_inputs = add_gaussian_noise(inputs, noise_factor=noise_param)
            elif noise_type == "burst":
                noisy_inputs = add_burst_noise(inputs, burst_prob=1.0, block_size=int(noise_param))
            else:
                raise ValueError("Unsupported noise type")
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
    avg_inference_time = (total_time / total_images) * 1000
    return avg_psnr, avg_ssim, avg_inference_time

def simulate_real_time_stream(model, device, transform=None):
    if transform is None:
        transform = transforms.ToTensor()
    testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)
    model.eval()
    total_time = 0.0
    total_frames = 0
    with torch.no_grad():
        for inputs, _ in testloader:
            inputs = inputs.to(device)
            start_time = time.time()
            _ = model(inputs)
            torch.cuda.synchronize()
            end_time = time.time()
            total_time += (end_time - start_time)
            total_frames += 1
    fps = total_frames / total_time
    return fps

def generate_error_maps(model, device, noise_type="gaussian", noise_param=0.1, num_images=8, transform=None):
    if transform is None:
        transform = transforms.ToTensor()
    testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=num_images, shuffle=True, num_workers=4)
    model.eval()
    with torch.no_grad():
        inputs, _ = next(iter(testloader))
        inputs = inputs.to(device)
        if noise_type == "gaussian":
            noisy_inputs = add_gaussian_noise(inputs, noise_factor=noise_param)
        elif noise_type == "burst":
            noisy_inputs = add_burst_noise(inputs, burst_prob=1.0, block_size=int(noise_param))
        else:
            raise ValueError("Unsupported noise type")
        outputs = model(noisy_inputs)
    error_maps = torch.abs(inputs - outputs)
    return inputs.cpu(), outputs.cpu(), error_maps.cpu()

#############################################
# Main Experiment: Extended Pipeline
#############################################

def main():
    parser = argparse.ArgumentParser(description="Extended Research Pipeline: Dynamic Contraction, Advanced Augmentation, and Extended Evaluation.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (default: 20)")
    parser.add_argument("--use_dynamic_lambda", action="store_true", default=False, help="Enable dynamic lambda scheduling for contraction regularization")
    parser.add_argument("--base_lambda", type=float, default=0.01, help="Base lambda for contraction regularization (default: 0.01)")
    parser.add_argument("--final_lambda", type=float, default=0.1, help="Final lambda for contraction regularization (default: 0.1)")
    parser.add_argument("--use_contraction", action="store_true", default=False, help="Enable contraction regularization (default: False)")
    parser.add_argument("--advanced_aug", action="store_true", default=False, help="Use advanced data augmentation (default: False)")
    parser.add_argument("--refine_iter", type=int, default=5, help="Number of refinement iterations (default: 5)")
    parser.add_argument("--refine_scale", type=float, default=0.7, help="Refinement scale factor (default: 0.7)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set augmentation transform
    if args.advanced_aug:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.ToTensor()
    
    # Load CIFAR-10 dataset with the chosen transform
    full_trainset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
    train_size = int(0.9 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = torch.utils.data.random_split(full_trainset, [train_size, val_size])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    valloader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=False, num_workers=4)
    
    # Instantiate model
    base_model = Autoencoder()
    model = RefinedAutoencoder(base_autoencoder=base_model, num_iterations=args.refine_iter, refinement_scale=args.refine_scale)
    model = model.to(device)
    
    # Train model with selected mode
    if args.use_contraction:
        if args.use_dynamic_lambda:
            print("Training with dynamic contraction regularization (λ from {:.3f} to {:.3f})".format(args.base_lambda, args.final_lambda))
            model = train_model_dynamic_contraction(model, trainloader, valloader, device, num_epochs=args.epochs, noise_factor=0.1, base_lambda=args.base_lambda, final_lambda=args.final_lambda)
            training_mode = "dynamic_contraction"
        else:
            print("Training with fixed contraction regularization (λ = {:.3f})".format(args.final_lambda))
            model = train_model_contraction(model, trainloader, valloader, device, num_epochs=args.epochs, noise_factor=0.1, lambda_contraction=args.final_lambda)
            training_mode = "contraction"
    else:
        print("Training without contraction regularization (standard refinement)")
        model = train_model_standard(model, trainloader, valloader, device, num_epochs=args.epochs, noise_factor=0.1)
        training_mode = "standard"
    
    # Save trained model
    model_filename = os.path.join(MODELS_DIR, f"autoencoder_extended_{training_mode}_aug{'advanced' if args.advanced_aug else 'basic'}.pth")
    torch.save(model.state_dict(), model_filename)
    print(f"Trained model saved to {model_filename}")
    
    # Extended evaluation: evaluate model on test set under multiple noise conditions
    noise_factors = [0.05, 0.1, 0.15, 0.2]
    gaussian_results = []
    for nf in noise_factors:
        psnr, ssim, inf_time = evaluate_on_test(model, device, noise_type="gaussian", noise_param=nf, transform=transform)
        print(f"Gaussian Noise (factor={nf}): PSNR={psnr:.2f} dB, SSIM={ssim:.4f}, Inference Time={inf_time:.2f} ms")
        gaussian_results.append({"Noise_Type": "Gaussian", "Noise_Param": nf, "PSNR": psnr, "SSIM": ssim, "InferenceTime_ms": inf_time})
    
    block_sizes = [4, 8, 12, 16]
    burst_results = []
    for bs in block_sizes:
        psnr, ssim, inf_time = evaluate_on_test(model, device, noise_type="burst", noise_param=bs, transform=transform)
        print(f"Burst Noise (block_size={bs}): PSNR={psnr:.2f} dB, SSIM={ssim:.4f}, Inference Time={inf_time:.2f} ms")
        burst_results.append({"Noise_Type": "Burst", "Noise_Param": bs, "PSNR": psnr, "SSIM": ssim, "InferenceTime_ms": inf_time})
    
    df_gaussian = pd.DataFrame(gaussian_results)
    df_burst = pd.DataFrame(burst_results)
    df_all = pd.concat([df_gaussian, df_burst], ignore_index=True)
    results_file = os.path.join(RESULTS_DIR, f"extended_evaluation_results_{training_mode}_aug{'advanced' if args.advanced_aug else 'basic'}.csv")
    df_all.to_csv(results_file, index=False)
    print(f"Extended evaluation results saved to {results_file}")
    
    # Real-time streaming simulation
    fps = simulate_real_time_stream(model, device, transform=transform)
    print(f"Real-time simulation: {fps:.2f} frames per second")
    
    # Generate error maps for a selected noise condition
    inputs_g, outputs_g, error_maps_g = generate_error_maps(model, device, noise_type="gaussian", noise_param=0.1, num_images=8, transform=transform)
    fig, axs = plt.subplots(3, 8, figsize=(20, 8))
    for i in range(8):
        axs[0, i].imshow(inputs_g[i].permute(1,2,0))
        axs[0, i].axis('off')
        axs[0, i].set_title("Orig (Gauss)")
        axs[1, i].imshow(outputs_g[i].permute(1,2,0))
        axs[1, i].axis('off')
        axs[1, i].set_title("Recon (Gauss)")
        axs[2, i].imshow(error_maps_g[i].permute(1,2,0))
        axs[2, i].axis('off')
        axs[2, i].set_title("Error Map")
    plt.tight_layout()
    error_map_file_g = os.path.join(RESULTS_DIR, f"error_maps_gaussian_{training_mode}_aug{'advanced' if args.advanced_aug else 'basic'}.png")
    plt.savefig(error_map_file_g)
    print(f"Gaussian error map visualization saved to {error_map_file_g}")
    plt.close(fig)
    
    inputs_b, outputs_b, error_maps_b = generate_error_maps(model, device, noise_type="burst", noise_param=8, num_images=8, transform=transform)
    fig, axs = plt.subplots(3, 8, figsize=(20, 8))
    for i in range(8):
        axs[0, i].imshow(inputs_b[i].permute(1,2,0))
        axs[0, i].axis('off')
        axs[0, i].set_title("Orig (Burst)")
        axs[1, i].imshow(outputs_b[i].permute(1,2,0))
        axs[1, i].axis('off')
        axs[1, i].set_title("Recon (Burst)")
        axs[2, i].imshow(error_maps_b[i].permute(1,2,0))
        axs[2, i].axis('off')
        axs[2, i].set_title("Error Map")
    plt.tight_layout()
    error_map_file_b = os.path.join(RESULTS_DIR, f"error_maps_burst_{training_mode}_aug{'advanced' if args.advanced_aug else 'basic'}.png")
    plt.savefig(error_map_file_b)
    print(f"Burst error map visualization saved to {error_map_file_b}")
    plt.show()

if __name__ == "__main__":
    main()

