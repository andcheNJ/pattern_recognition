import os
from PIL import Image
import torchvision.transforms as transforms
import torch

# Define the preprocessing pipeline
# Calculate mean and std of your dataset first, then:
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Example values
])

def preprocess_folder(folder_path):
    """Preprocess all images in a folder."""
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        image = Image.open(img_path)
        preprocessed_image = transform(image)
        images.append(preprocessed_image)
    return torch.stack(images)
