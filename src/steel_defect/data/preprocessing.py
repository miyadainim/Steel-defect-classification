"""Notebook transforms without augmentation or interpolation changes."""
from torchvision import transforms

def resnet_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def cnn_transform():
    return transforms.ToTensor()
