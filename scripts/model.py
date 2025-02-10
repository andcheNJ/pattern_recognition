import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseConvNet(nn.Module):
    def __init__(self):
        super().__init__()  # Fix: Remove the recursive BaseConvNet reference
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_network = BaseConvNet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  # Move entire model to device

    def forward(self, input1, input2):
        # Ensure inputs are on correct device
        input1 = input1.to(self.device)
        input2 = input2.to(self.device)
        return self.base_network(input1), self.base_network(input2)
# # Example: Create an instance of the Siamese network
# model = SiameseNetwork()
