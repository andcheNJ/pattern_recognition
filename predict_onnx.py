# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 23:35:58 2025

@author: Andrew
"""

import os
import onnxruntime as ort
import numpy as np
from PIL import Image

class ONNXSiamesePredictor:
    def __init__(self, onnx_path):
        self.session = ort.InferenceSession(onnx_path)
        
    def predict_pattern(self, new_image_path, reference_patterns_path):
        new_embedding = self._get_embedding(new_image_path)
        
        min_distance = float('inf')
        predicted_pattern = None
        
        for pattern_number, pattern_file in enumerate(sorted(os.listdir(reference_patterns_path))):
            if not self._is_image_file(pattern_file):
                continue
                
            pattern_path = os.path.join(reference_patterns_path, pattern_file)
            pattern_embedding = self._get_embedding(pattern_path)
            
            distance = np.linalg.norm(new_embedding - pattern_embedding)
            
            if distance < min_distance:
                min_distance = distance
                predicted_pattern = pattern_number
                
        return predicted_pattern, min_distance
    
    def _get_embedding(self, image_path):
        # Pure PIL/numpy preprocessing
        img = Image.open(image_path)
        img = self._preprocess_image(img)
        return self._inference(img)
    
    def _preprocess_image(self, img):
        # 1. Resize
        img = img.resize((64, 64))
        
        # 2. Convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')
            
        # 3. Convert to numpy array and normalize
        arr = np.array(img, dtype=np.float32)
        
        # 4. Normalize: (x - 127.5) / 127.5 to match [0,1] -> [-1,1]
        arr = (arr / 127.5) - 1.0
        
        # 5. Add channel and batch dimensions -> [1, 1, H, W]
        arr = arr[np.newaxis, np.newaxis, :, :]
        
        return arr
    
    def _inference(self, arr):
        inputs = {
            "input1": arr,
            "input2": arr  # Dummy input since we process one image at a time
        }
        outputs = self.session.run(None, inputs)
        return outputs[0][0]  # Return first output's first batch
        
    def _is_image_file(self, filename):
        return filename.lower().endswith(('.png', '.jpg', '.jpeg'))

# Usage remains the same
predictor = ONNXSiamesePredictor(r"C:\Users\Andrew\OneDrive\Dokumente\Arduino\image_project\models\siamese_network.onnx")
predicted_pattern, distance = predictor.predict_pattern(
    r"C:\Users\Andrew\OneDrive\Dokumente\Arduino\image_project\data\test_samples\sample_2.png",
    r"C:\Users\Andrew\OneDrive\Dokumente\Arduino\image_project\data\patterns"
)
print(f"Predicted: {predicted_pattern}, Distance: {distance:.4f}")