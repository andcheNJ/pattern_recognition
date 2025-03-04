# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 10:55:20 2025

@author: testhouse
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scripts.model import SiameseNetwork  # <-- your existing model definition
import os

def export_siamese_model_to_onnx(
    model_path="D:/pattern_recognition/models/model_3.pth",
    onnx_path="siamese_network.onnx",
    input_size=(1, 64, 64),  # (channels, height, width)
    opset_version=11
):
    """
    Load the PyTorch SiameseNetwork, then export it to an ONNX file.
    """
    # 1. Load the model
    model = SiameseNetwork()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # 2. Create dummy inputs for the 2 branches of the Siamese
    #    The shape is (batch_size, channels, height, width).
    dummy_input1 = torch.randn(1, *input_size, requires_grad=False)
    dummy_input2 = torch.randn(1, *input_size, requires_grad=False)

    # 3. Export to ONNX
    #    We'll give two inputs (input1, input2) and two outputs (output1, output2).
    torch.onnx.export(
        model,
        (dummy_input1, dummy_input2),
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input1", "input2"],
        output_names=["output1", "output2"],
        dynamic_axes=None  # If you want variable batch sizes, you can set this
    )

    print(f"Exported Siamese model to {onnx_path}")


if __name__ == "__main__":
    export_siamese_model_to_onnx()
