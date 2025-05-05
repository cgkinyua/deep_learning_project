#!/usr/bin/env python
"""
evaluate_test_set.py

This script evaluates the trained autoencoder model on the official CIFAR-10 test set.
It computes PSNR, SSIM, and inference time per image, and saves the evaluation metrics
to a CSV file for further analysis.

Usage:
    python evaluate_test_set.py --model_path ~/deep_learning_project/models/autoencoder_model.pth
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
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Define directories
BASE_DIR = os.path.expanduser('~/deep_learning_project')
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Ensure RESULTS_DIR exists
os.makedirs(RESULTS_DIR, exist_ok=True)

#############################################
# Model Definitions (Same as before)
#############################################

class Autoencoder(nn.Module):
    """Baseline convolutional autoencoder."""
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), 
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))

class RefinedAutoencoder(nn.Module):
    """Autoencoder with iterative refinement."""
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
# Evaluation on Test Set
#############################################

def evaluate_on_test(model, device, noise_factor=0.1):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # Load the CIFAR-10 test set
    testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)
    
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    total_images = 0
    total_inference_time = 0.0
    
    with torch.no_grad():
        for inputs, _ in testloader:
            inputs = inputs.to(device)
            noisy_inputs = add_noise(inputs, noise_factor)
            start_time = time.time()
            outputs = model(noisy_inputs)
            torch.cuda.synchronize()
            end_time = time.time()
            
            batch_time = end_time - start_time
            total_inference_time += batch_time
            total_images += inputs.size(0)
            
            total_psnr += compute_psnr(inputs, outputs) * inputs.size(0)
            total_ssim += compute_ssim(inputs, outputs) * inputs.size(0)
    
    avg_psnr = total_psnr / total_images
    avg_ssim = total_ssim / total_images
    avg_inference_time = (total_inference_time / total_images) * 1000  # in milliseconds
    
    return avg_psnr, avg_ssim, avg_inference_time

#############################################
# Main Function
#############################################

def main():
    parser = argparse.ArgumentParser(description="Evaluate autoencoder model on CIFAR-10 test set.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved model (.pth file)")
    parser.add_argument("--refine", action="store_true", default=True,
                        help="Indicate that the saved model is a refined autoencoder (default: True)")
    parser.add_argument("--refine_iter", type=int, default=3,
                        help="Number of refinement iterations used in the model (default: 3)")
    parser.add_argument("--refine_scale", type=float, default=0.7,
                        help="Refinement scale factor used in the model (default: 0.7)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate base model and wrap with refinement if needed
    base_model = Autoencoder()
    if args.refine:
        model = RefinedAutoencoder(base_autoencoder=base_model, num_iterations=args.refine_iter, refinement_scale=args.refine_scale)
    else:
        model = base_model
    model = model.to(device)
    
    # Load the saved model state
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Loaded model from {args.model_path}")
    
    # Evaluate on the test set
    psnr, ssim, inf_time = evaluate_on_test(model, device, noise_factor=0.1)
    print(f"Test Set Evaluation Metrics:\nPSNR = {psnr:.2f} dB\nSSIM = {ssim:.4f}\nInference Time per image = {inf_time:.2f} ms")
    
    # Save evaluation results to CSV
    results = {"PSNR": [psnr], "SSIM": [ssim], "InferenceTime_ms": [inf_time]}
    df = pd.DataFrame(results)
    results_file = os.path.join(RESULTS_DIR, "test_set_evaluation_metrics.csv")
    df.to_csv(results_file, index=False)
    print(f"Test set evaluation metrics saved to {results_file}")

if __name__ == "__main__":
    main()

