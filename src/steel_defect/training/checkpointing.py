"""Strict legacy state-dict loading and exclusive checkpoint creation."""
from pathlib import Path
import torch

def load_checkpoint(model, path):
    model.load_state_dict(torch.load(Path(path), map_location='cpu', weights_only=True))
    return model

def save_checkpoint(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        torch.save(model.state_dict(), f)
