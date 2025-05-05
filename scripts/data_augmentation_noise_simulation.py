import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

# Load CIFAR-10 dataset
transform = transforms.ToTensor()
dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)

# Get a sample image
image, _ = dataset[0]

# Data augmentation transformations
horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)
rotation = transforms.RandomRotation(15)
random_crop = transforms.RandomCrop(32, padding=4)
gaussian_noise = transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))
salt_pepper_noise = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor()
])

# Apply transformations
flipped_image = horizontal_flip(image)
rotated_image = rotation(image)
cropped_image = random_crop(image)
gaussian_image = gaussian_noise(image)
salt_pepper_image = salt_pepper_noise(image)

# Plot the images
fig, axs = plt.subplots(2, 3, figsize=(10, 7))
fig.suptitle('Figure 3.3: Data Augmentation Techniques and Noise Simulation')

# Display original image with nearest interpolation
axs[0, 0].imshow(np.transpose(image.numpy(), (1, 2, 0)), interpolation='nearest')
axs[0, 0].set_title('Original')
axs[0, 0].axis('off')

# Display augmented images
axs[0, 1].imshow(np.transpose(flipped_image.numpy(), (1, 2, 0)), interpolation='nearest')
axs[0, 1].set_title('Horizontal Flip')
axs[0, 1].axis('off')

axs[0, 2].imshow(np.transpose(rotated_image.numpy(), (1, 2, 0)), interpolation='nearest')
axs[0, 2].set_title('Rotation (15 deg)')
axs[0, 2].axis('off')

axs[1, 0].imshow(np.transpose(cropped_image.numpy(), (1, 2, 0)), interpolation='nearest')
axs[1, 0].set_title('Random Crop')
axs[1, 0].axis('off')

axs[1, 1].imshow(np.transpose(gaussian_image.numpy(), (1, 2, 0)), interpolation='nearest')
axs[1, 1].set_title('Gaussian Noise')
axs[1, 1].axis('off')

axs[1, 2].imshow(np.transpose(salt_pepper_image.numpy(), (1, 2, 0)), interpolation='nearest')
axs[1, 2].set_title('Salt & Pepper Noise')
axs[1, 2].axis('off')

plt.tight_layout()
plt.show()
