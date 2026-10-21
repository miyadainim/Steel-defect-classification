
"""Inference utilities for the NEU-CLS steel defect classifier."""

from pathlib import Path

import torch
from torch import nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "best_resnet18_finetuned.pth"
)


# --------------------------------------------------
# 2. Image preprocessing
# --------------------------------------------------

# Identical to the preprocessing used during
# ResNet18 validation and testing.

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# --------------------------------------------------
# 3. Load trained ResNet18
# --------------------------------------------------

def load_model(device="cpu"):
    """
    Load the fine-tuned ResNet18 checkpoint.

    Args:
        device: "cpu", "mps", or "cuda".

    Returns:
        Model in evaluation mode.
    """

    model = resnet18(weights=None)

    # Replace classification head with six outputs.
    model.fc = nn.Linear(
        model.fc.in_features,
        len(CLASS_NAMES)
    )

    # Load trained model weights.
    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model


# --------------------------------------------------
# 4. Predict image
# --------------------------------------------------

def predict(image: Image.Image, model, device="cpu"):
    """
    Classify a single steel surface image.

    Args:
        image: PIL image.
        model: Loaded ResNet18 model.
        device: Inference device.

    Returns:
        predicted_index: Integer class index.
        probabilities: List of six softmax scores.
    """

    # Apply preprocessing.
    image_tensor = TRANSFORM(image)

    # Add batch dimension: (3,224,224) -> (1,3,224,224).
    image_tensor = image_tensor.unsqueeze(0)

    # Transfer tensor to inference device.
    image_tensor = image_tensor.to(device)

    # Disable gradient calculations for inference.
    with torch.inference_mode():
        logits = model(image_tensor)

        probabilities = torch.softmax(
            logits, dim=1
        )[0]

    predicted_index = probabilities.argmax().item()

    return predicted_index, probabilities.cpu().tolist()
