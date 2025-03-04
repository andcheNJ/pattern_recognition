# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 11:03:32 2025

@author: testhouse
"""

import os
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms

###############################################################################
# 1. LOAD THE ONNX MODEL
###############################################################################
# Point this to your exported .onnx file
ONNX_MODEL_PATH = r"D:\ECU-Test\Workspace_2\displays-ees25-test\DisplayFunctions\Data\PHUD_Bitmap\Model\siamese_network.onnx"

# Make sure the file actually exists
assert os.path.isfile(ONNX_MODEL_PATH), f"ONNX model not found: {ONNX_MODEL_PATH}"

session = ort.InferenceSession(ONNX_MODEL_PATH)

# Create an ONNX Runtime inference session
session = ort.InferenceSession(ONNX_MODEL_PATH)

# We can grab the input/output names from the session
input1_name = session.get_inputs()[0].name  # "input1"
input2_name = session.get_inputs()[1].name  # "input2"
output_names = [o.name for o in session.get_outputs()]  # ["output1", "output2"]

###############################################################################
# 2. PREPROCESSING (SAME AS PYTORCH TRAINING)
###############################################################################
preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

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
    # 1) Load & preprocess the new image
    new_image = Image.open(new_image_path)
    new_image_tensor = preprocess(new_image).unsqueeze(0)  # shape: [1,1,64,64]
    new_image_numpy = new_image_tensor.cpu().numpy()       # ONNX Runtime needs NumPy

    min_distance = float('inf')
    predicted_pattern = None

    # 2) List all reference files
    reference_files = sorted(os.listdir(reference_patterns_path))

    for pattern_number, pattern_file in enumerate(reference_files):
        if not pattern_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        pattern_image_path = os.path.join(reference_patterns_path, pattern_file)
        
        # 3) Preprocess the pattern image
        pattern_image = Image.open(pattern_image_path)
        pattern_image_tensor = preprocess(pattern_image).unsqueeze(0)
        pattern_image_numpy = pattern_image_tensor.cpu().numpy()

        # 4) Run inference
        #    We feed new_image_numpy as "input1" and pattern_image_numpy as "input2"
        ort_inputs = {
            input1_name: new_image_numpy,
            input2_name: pattern_image_numpy
        }
        output1, output2 = session.run(output_names, ort_inputs)

        # output1, output2 are NumPy arrays of shape [1, 128] (if your fc out is 128)
        # 5) Calculate distance in NumPy
        distance_value = np.linalg.norm(output1 - output2, axis=1)[0]

        # 6) Track the smallest distance
        if distance_value < min_distance:
            min_distance = distance_value
            predicted_pattern = pattern_number

    return predicted_pattern, min_distance


# # Optional: test the function
# if __name__ == "__main__":
#     test_new_image = r"C:\Users\testhouse.PZI17028\Pictures\Camera Roll\sample_4.png"
#     test_reference_folder = r"D:\ECU-Test\Workspace_2\displays-ees25-test\DisplayFunctions\Data\PHUD_Bitmap\Patterns"
#     idx, dist = predict_pattern_onnx(test_new_image, test_reference_folder)
#     print(f"Best match index: {idx}, distance: {dist}")
