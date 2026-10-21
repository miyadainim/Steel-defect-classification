"""Stable class ordering and repository defaults."""
from pathlib import Path
CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models/best_resnet18_finetuned.pth"
