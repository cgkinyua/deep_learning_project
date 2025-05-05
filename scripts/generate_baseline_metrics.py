import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import pandas as pd
import time

# Define a simple baseline autoencoder model
class BaselineAutoencoder(nn.Module):
    def __init__(self):
        super(BaselineAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Load the model
model_path = "../models/autoencoder_baseline_basic.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BaselineAutoencoder().to(device)

# Load the trained weights
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Load CIFAR-10 dataset for testing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_dataset = datasets.CIFAR10(root='../data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Evaluate the model
psnr_values = []
ssim_values = []
inference_times = []

for images, _ in test_loader:
    images = images.to(device)

    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        outputs = model(images)
    end_time = time.time()

    inference_time = (end_time - start_time) / images.size(0) * 1000  # ms per image
    inference_times.append(inference_time)

    images = images.cpu().numpy().transpose(0, 2, 3, 1)
    outputs = outputs.cpu().numpy().transpose(0, 2, 3, 1)

    for i in range(images.shape[0]):
        psnr = peak_signal_noise_ratio(images[i], outputs[i], data_range=2.0)
        ssim = structural_similarity(images[i], outputs[i], multichannel=True, win_size=3, data_range=2.0)
        psnr_values.append(psnr)
        ssim_values.append(ssim)

# Save results
avg_psnr = sum(psnr_values) / len(psnr_values)
avg_ssim = sum(ssim_values) / len(ssim_values)
avg_inference_time = sum(inference_times) / len(inference_times)

results = pd.DataFrame({
    'Noise_Type': ['Baseline'],
    'Noise_Param': [0],
    'PSNR': [avg_psnr],
    'SSIM': [avg_ssim],
    'InferenceTime_ms': [avg_inference_time]
})

results.to_csv("../results/baseline_evaluation_metrics.csv", index=False)
print("Baseline evaluation metrics saved successfully!")
