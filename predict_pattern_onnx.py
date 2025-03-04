# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 11:03:32 2025

@author: testhouse
"""

import os
import onnxruntime as ort
import numpy as np
from PIL import Image

###############################################################################
# 1. PURE-PYTHON PREPROCESS
###############################################################################
def preprocess_image(image_path, size=(64, 64)):
    """
    Mimic the same steps as:
      - transforms.Resize((64, 64))
      - transforms.Grayscale()
      - transforms.ToTensor()
      - transforms.Normalize(mean=[0.5], std=[0.5])

    Returns a NumPy array of shape (1, 1, 64, 64).
    """
    image = Image.open(image_path).convert("L")
    image = image.resize(size, Image.BILINEAR)
    image_np = np.array(image, dtype=np.float32)
    image_np /= 255.0
    image_np = (image_np - 0.5) / 0.5
    image_np = np.expand_dims(image_np, axis=0)  # channel dimension
    image_np = np.expand_dims(image_np, axis=0)  # batch dimension
    return image_np

###############################################################################
# 2. LOAD THE ONNX MODEL
###############################################################################
# Point this to your exported .onnx file
ONNX_MODEL_PATH = "siamese_network.onnx"

# Create an ONNX Runtime inference session
session = ort.InferenceSession(ONNX_MODEL_PATH)

# We can grab the input/output names from the session
input1_name = session.get_inputs()[0].name  # "input1"
input2_name = session.get_inputs()[1].name  # "input2"
output_names = [o.name for o in session.get_outputs()]  # ["output1", "output2"]

###############################################################################
# 3. INFERENCE FUNCTION
###############################################################################
def predict_pattern_onnx(new_image_path, reference_patterns_path):
    """
    Predict which reference pattern in 'reference_patterns_path' is most similar
    to the image at 'new_image_path', using an ONNX-exported Siamese model.
    
    Returns:
        predicted_pattern (int): Index (0-based) of the best-matching pattern
        min_distance (float): The best (lowest) distance
    """
    # 1) Preprocess the new image (shape: [1,1,64,64])
    new_image_numpy = preprocess_image(new_image_path)

    min_distance = float('inf')
    predicted_pattern = None

    # 2) List all reference files
    reference_files = sorted(os.listdir(reference_patterns_path))

    for pattern_number, pattern_file in enumerate(reference_files):
        # Skip non-image files
        if not pattern_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        pattern_image_path = os.path.join(reference_patterns_path, pattern_file)
        
        # 3) Preprocess the reference pattern
        pattern_image_numpy = preprocess_image(pattern_image_path)

        # 4) Run inference
        ort_inputs = {
            input1_name: new_image_numpy,
            input2_name: pattern_image_numpy
        }
        output1, output2 = session.run(output_names, ort_inputs)

        # output1, output2 are NumPy arrays, e.g. shape [1, 128]
        # 5) Calculate distance in NumPy
        distance_value = np.linalg.norm(output1 - output2, axis=1)[0]

        # 6) Track the smallest distance
        if distance_value < min_distance:
            min_distance = distance_value
            predicted_pattern = pattern_number

    return predicted_pattern, min_distance

# Optional: test the function
if __name__ == "__main__":
    test_new_image = r"C:\Users\testhouse.PZI17028\Pictures\Camera Roll\sample_6.png"
    test_reference_folder = r"D:\TestPics_PHUD\testPics\pHUD50 Eco 0x0A-0x14_1500x250\dataset_mini"
    idx, dist = predict_pattern_onnx(test_new_image, test_reference_folder)
    print(f"Best match index: {idx}, distance: {dist}")

