# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:29:34 2024

@author: Andrew
"""

import torch
from PIL import Image
from torchvision import transforms
from scripts.model import SiameseNetwork  # Update import path
import os
# Load the trained model WITH DEVICE HANDLING
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load(r'C:\Users\Andrew\OneDrive\Dokumente\Arduino\image_project\models\model_1.pth', map_location=device))
model.eval()

# Update preprocessing to MATCH TRAINING (add normalization)
preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Add your actual values here
])

def predict_pattern(new_image_path, reference_patterns):
    # Load and preprocess with device handling
    new_image = Image.open(new_image_path)
    new_image = preprocess(new_image).unsqueeze(0).to(device)  # Add device
    
    min_distance = float('inf')
    predicted_pattern = None
    
    for pattern_number, pattern_image_path in enumerate(reference_patterns):
        pattern_image = Image.open(pattern_image_path)
        pattern_image = preprocess(pattern_image).unsqueeze(0).to(device)  # Add device
        
        with torch.no_grad():
            # Use the Siamese Network properly
            output1, output2 = model(new_image, pattern_image)
            distance = torch.nn.functional.pairwise_distance(output1, output2)
            
        distance_value = distance.item()
        
        if distance_value < min_distance:
            min_distance = distance_value
            predicted_pattern = pattern_number

    return predicted_pattern, min_distance

# Example usage with error handling:
# Example usage:
reference_patterns = []
pattern_folder = r"C:\Users\Andrew\OneDrive\Dokumente\Arduino\image_project\data"
for pattern_num in range(1, 13):  # For 12 patterns
    pattern_path = os.path.join(pattern_folder, f"pattern_{pattern_num}", "img1.png")
    reference_patterns.append(pattern_path)
new_image_path = r"C:\Users\Andrew\OneDrive\Dokumente\Arduino\image_project\test_pics\test_pic_8.png"
try:
    predicted_pattern, distance = predict_pattern(new_image_path, reference_patterns)
    print(f"Predicted Pattern: {predicted_pattern}, Distance: {distance:.4f}")
except Exception as e:
    print(f"Prediction failed: {str(e)}")

