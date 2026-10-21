"""Original notebook architectures; state-dict names are preserved."""
import torch.nn as nn
# ==========================================
# 1. Reusable Convolution Block
# ==========================================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x
    

# ==========================================
# 2. Complete Steel Defect CNN
# ==========================================

class SteelDefectCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Feature extraction
        self.block1 = ConvBlock(1, 16)
        self.block2 = ConvBlock(16, 32)
        self.block3 = ConvBlock(32, 64)

        # Classification head
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(40000, 6)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.flatten(x)
        x = self.classifier(x)

        return x

class SteelDefectCNN_GAP(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = ConvBlock(1, 16)
        self.block2 = ConvBlock(16, 32)
        self.block3 = ConvBlock(32, 64)

        # Replace large flattening head with GAP
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        # Define classifier with correct input size
        self.classifier = nn.Linear(64,6)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Apply GAP
        x = self.gap(x)
        # Flatten output
        x = self.flatten(x)
        # Calculate class logits
        x = self.classifier(x)
        return x
