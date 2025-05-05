#!/usr/bin/env python
"""
extended_evaluation.py

This script extends our evaluation of the refined autoencoder by:
1. Evaluating the model under multiple noise conditions:
   - Gaussian noise: varying noise factors.
   - Burst noise: simulating burst errors by zeroing out random blocks.
2. Simulating a real-time streaming scenario by processing the entire CIFAR-10 test set
   and computing throughput (frames per second).
3. Generating comprehensive evaluation metrics (PSNR, SSIM, inference time) and saving the results
   to CSV files.
4. Creating qualitative visualizations (error maps) for selected noise conditions.

Usage:
    python extended_evaluation.py --model_path /path/to/model.pth --use_contraction --refine_iter 5 --refine_scale 0.7
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from torch.nn.utils import spectral_norm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt

# Directory setup
BASE_DIR = os.path.expanduser('~/deep_learning_project')
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR, exist_ok=True)

#############################################
# Model Definitions (same as before)
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
# Noise Simulation Functions
#############################################

def add_gaussian_noise(images, noise_factor=0.1):
    """Add Gaussian noise to images."""
    noisy_images = images + noise_factor * torch.randn_like(images)
    return torch.clamp(noisy_images, 0., 1.)

def add_burst_noise(images, burst_prob=0.3, block_size=8):
    """
    Simulate burst noise by zeroing out a random block in each image.
    burst_prob: probability that an image will have burst noise applied.
    block_size: size of the square block to zero.
    """
    noisy_images = images.clone()
    B, C, H, W = noisy_images.shape
    for i in range(B):
        if np.random.rand() < burst_prob:
            # Choose random top-left corner for the block
            x = np.random.randint(0, W - block_size)
            y = np.random.randint(0, H - block_size)
            noisy_images[i, :, y:y+block_size, x:x+block_size] = 0.0
    return noisy_images

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
# Evaluation Functions
#############################################

def evaluate_model_on_noise(model, device, noise_type="gaussian", noise_param=0.1, transform=None):
    """
    Evaluate model on the CIFAR-10 test set under a specific noise condition.
    noise_type: "gaussian" or "burst"
    noise_param: for gaussian, it's the noise factor; for burst, it's the block_size.
    """
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
    avg_inference_time = (total_time / total_images) * 1000  # ms
    return avg_psnr, avg_ssim, avg_inference_time

def simulate_real_time_stream(model, device, transform=None):
    """
    Simulate real-time processing by streaming through the test set and computing throughput.
    Returns frames per second.
    """
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

#############################################
# Visualization: Generate Error Maps
#############################################

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
# Main Function: Extended Evaluation and Real-Time Simulation
#############################################

def main():
    parser = argparse.ArgumentParser(description="Extended Evaluation: Evaluate refined autoencoder under various noise conditions and simulate real-time streaming.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model (.pth file)")
    parser.add_argument("--use_contraction", action="store_true", default=False, help="Flag indicating if the model uses contraction regularization")
    parser.add_argument("--refine_iter", type=int, default=5, help="Number of refinement iterations (default: 5)")
    parser.add_argument("--refine_scale", type=float, default=0.7, help="Refinement scale factor (default: 0.7)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate model and load state
    base_model = Autoencoder()
    if args.use_contraction:
        model = RefinedAutoencoder(base_autoencoder=base_model, num_iterations=args.refine_iter, refinement_scale=args.refine_scale)
    else:
        model = RefinedAutoencoder(base_autoencoder=base_model, num_iterations=args.refine_iter, refinement_scale=args.refine_scale)
    model = model.to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Loaded model from {args.model_path}")
    
    # Evaluate under different Gaussian noise levels
    gaussian_results = []
    noise_factors = [0.05, 0.1, 0.15, 0.2]
    for nf in noise_factors:
        psnr, ssim, inf_time = evaluate_model_on_noise(model, device, noise_type="gaussian", noise_param=nf)
        print(f"Gaussian Noise (factor={nf}): PSNR={psnr:.2f} dB, SSIM={ssim:.4f}, Inference Time={inf_time:.2f} ms")
        gaussian_results.append({"Noise_Type": "Gaussian", "Noise_Param": nf, "PSNR": psnr, "SSIM": ssim, "InferenceTime_ms": inf_time})
    
    # Evaluate under different Burst noise conditions (block sizes)
    burst_results = []
    block_sizes = [4, 8, 12, 16]
    for bs in block_sizes:
        psnr, ssim, inf_time = evaluate_model_on_noise(model, device, noise_type="burst", noise_param=bs)
        print(f"Burst Noise (block_size={bs}): PSNR={psnr:.2f} dB, SSIM={ssim:.4f}, Inference Time={inf_time:.2f} ms")
        burst_results.append({"Noise_Type": "Burst", "Noise_Param": bs, "PSNR": psnr, "SSIM": ssim, "InferenceTime_ms": inf_time})
    
    # Save extended evaluation results to CSV
    df_gaussian = pd.DataFrame(gaussian_results)
    df_burst = pd.DataFrame(burst_results)
    df_all = pd.concat([df_gaussian, df_burst], ignore_index=True)
    results_file = os.path.join(RESULTS_DIR, "extended_evaluation_results.csv")
    df_all.to_csv(results_file, index=False)
    print(f"Extended evaluation results saved to {results_file}")
    
    # Simulate real-time streaming
    fps = simulate_real_time_stream(model, device)
    print(f"Real-time simulation: {fps:.2f} frames per second")
    
    # Generate error maps for one Gaussian noise condition and one Burst noise condition
    inputs_g, outputs_g, error_maps_g = generate_error_maps(model, device, noise_type="gaussian", noise_param=0.1, num_images=8)
    inputs_b, outputs_b, error_maps_b = generate_error_maps(model, device, noise_type="burst", noise_param=8, num_images=8)
    
    # Plot and save error maps
    fig, axs = plt.subplots(3, 8, figsize=(20, 8))
    for i in range(8):
        axs[0, i].imshow(inputs_g[i].permute(1, 2, 0))
        axs[0, i].axis('off')
        axs[0, i].set_title("Orig (Gaussian)")
        axs[1, i].imshow(outputs_g[i].permute(1, 2, 0))
        axs[1, i].axis('off')
        axs[1, i].set_title("Recon (Gaussian)")
        axs[2, i].imshow(error_maps_g[i].permute(1, 2, 0))
        axs[2, i].axis('off')
        axs[2, i].set_title("Error Map")
    plt.tight_layout()
    error_map_file_g = os.path.join(RESULTS_DIR, "error_maps_gaussian.png")
    plt.savefig(error_map_file_g)
    print(f"Gaussian error map visualization saved to {error_map_file_g}")
    plt.close(fig)
    
    fig, axs = plt.subplots(3, 8, figsize=(20, 8))
    for i in range(8):
        axs[0, i].imshow(inputs_b[i].permute(1, 2, 0))
        axs[0, i].axis('off')
        axs[0, i].set_title("Orig (Burst)")
        axs[1, i].imshow(outputs_b[i].permute(1, 2, 0))
        axs[1, i].axis('off')
        axs[1, i].set_title("Recon (Burst)")
        axs[2, i].imshow(error_maps_b[i].permute(1, 2, 0))
        axs[2, i].axis('off')
        axs[2, i].set_title("Error Map")
    plt.tight_layout()
    error_map_file_b = os.path.join(RESULTS_DIR, "error_maps_burst.png")
    plt.savefig(error_map_file_b)
    print(f"Burst error map visualization saved to {error_map_file_b}")
    plt.show()

if __name__ == "__main__":
    main()

