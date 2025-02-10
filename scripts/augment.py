import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from scripts.utils import calculate_mean_std

dataset_path = 'data'

dataset_mean, dataset_std = calculate_mean_std(dataset_path)

# Define the augmentation pipeline
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[dataset_mean], std=[dataset_std])
])
# Function to augment an individual image
def augment_image(image, transform=transform):
    """Apply transformations to a single image."""
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)
    
    augmented_image = transform(image)
    
    # Ensure the augmented image has the correct shape [1, 64, 64]
    augmented_image = augmented_image.unsqueeze(0) if augmented_image.dim() == 3 else augmented_image
    
    return augmented_image


# Function to apply augmentation to all images in a list of tensors
def augment_images(images, transform=transform):
    """Apply augmentation to a list of images (tensors or PIL images)."""
    augmented_images = []
    for image in images:
        augmented_image = augment_image(image, transform)
        augmented_images.append(augmented_image)
    return torch.stack(augmented_images)  # Stack them into a tensor

# Function to load and augment images from a folder
def load_and_augment_images(folder_path, transform=transform):
    """Loads images from a folder and applies augmentation."""
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        image = Image.open(img_path).convert('L')  # Open and convert to grayscale ('L')
        augmented_image = transform(image)         # Apply the transformations
        images.append(augmented_image)
    return torch.stack(images)  # Stack them into a tensor

