import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import time

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Simplified autoencoder model
class SimpleBaselineAutoencoder(nn.Module):
    def __init__(self):
        super(SimpleBaselineAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 16x16x16
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 8x8x32
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x16x16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x32x3
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Hyperparameters
batch_size = 64
learning_rate = 1e-3
epochs = 5

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.CIFAR10(root='../data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, and optimizer
model = SimpleBaselineAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
print("Training the simplified baseline model...")
for epoch in range(epochs):
    for images, _ in train_loader:
        images = images.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, images)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Save the trained model
model_path = '../models/simplified_autoencoder_baseline.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Evaluation
print("Evaluating the simplified baseline model...")
test_dataset = datasets.CIFAR10(root='../data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

psnr_values = []
ssim_values = []
inference_times = []

model.eval()
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)

        # Measure inference time
        start_time = time.time()
        outputs = model(images)
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # ms

        # Convert tensors to numpy arrays for metric calculation
        images = images.cpu().numpy()
        outputs = outputs.cpu().numpy()

        psnr = peak_signal_noise_ratio(images[0].transpose(1, 2, 0), outputs[0].transpose(1, 2, 0), data_range=1.0)
        ssim = structural_similarity(
            images[0].transpose(1, 2, 0), 
            outputs[0].transpose(1, 2, 0), 
            data_range=1.0, 
            win_size=3, 
            channel_axis=-1
        )

        psnr_values.append(psnr)
        ssim_values.append(ssim)
        inference_times.append(inference_time)

# Calculate average metrics
avg_psnr = np.mean(psnr_values)
avg_ssim = np.mean(ssim_values)
avg_inference_time = np.mean(inference_times)

# Save the metrics to a CSV file
metrics = {
    'PSNR (dB)': [avg_psnr],
    'SSIM': [avg_ssim],
    'Inference Time (ms)': [avg_inference_time]
}
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv('../results/simplified_baseline_evaluation_metrics.csv', index=False)
print("Simplified baseline evaluation metrics saved successfully!")
