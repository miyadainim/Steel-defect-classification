"""Real six-class inference; no random-weight or mock fallback is provided."""
import os
from pathlib import Path
from pickle import UnpicklingError

import torch
from PIL import Image

from steel_defect.constants import CLASS_NAMES, MODEL_PATH
from steel_defect.data.preprocessing import resnet_transform
from torchvision import transforms
from steel_defect.models.resnet import build_model
from steel_defect.deployment.registry import MODELS, DEFAULT_MODEL, checkpoint_path

TRANSFORM = resnet_transform()


class CheckpointError(RuntimeError):
    """A supplied checkpoint cannot be loaded as a six-class ResNet18."""


def get_checkpoint_path(model_name: str = DEFAULT_MODEL) -> Path:
    """Use the selected experiment path; the legacy override applies to the final model."""
    configured = os.environ.get('STEEL_DEFECT_CHECKPOINT') if model_name == DEFAULT_MODEL else None
    return Path(configured).expanduser() if configured else checkpoint_path(model_name)


def load_model(device: str = 'cpu', checkpoint=None, model_name: str = DEFAULT_MODEL):
    """Load a local state dict strictly; missing/invalid weights never yield a model.

    No pretrained weights are downloaded by inference. Only use checkpoints from
    trusted, authorized sources. ``weights_only=True`` limits pickle loading.
    """
    path = Path(checkpoint) if checkpoint is not None else get_checkpoint_path(model_name)
    if not path.is_file():
        raise FileNotFoundError(
            'Model checkpoint not found. Supply an authorized six-class ResNet18 '
            'state dict at models/best_resnet18_finetuned.pth or set '
            'STEEL_DEFECT_CHECKPOINT. This candidate should include the verified final weights; check your checkout.'
        )
    model = build_model(MODELS[model_name]['architecture'])
    try:
        state = torch.load(path, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, EOFError, UnpicklingError) as exc:
        raise CheckpointError(
            'Unable to load this checkpoint. Use a compatible six-class ResNet18/CNN/GAP '
            'PyTorch state dict and the documented package versions.'
        ) from exc
    model._steel_defect_architecture = MODELS[model_name]['architecture']
    return model.to(device).eval()


def predict(image: Image.Image, model, device: str = 'cpu') -> tuple[int, list[float]]:
    """Return the genuine model argmax and six softmax scores in stable class order."""
    architecture = getattr(model, '_steel_defect_architecture', 'resnet_finetuned')
    tensor = preprocess_for_architecture(image, architecture).unsqueeze(0).to(device)
    with torch.inference_mode():
        scores = model(tensor).softmax(dim=1)[0]
    return scores.argmax().item(), scores.cpu().tolist()


def preprocess_for_architecture(image: Image.Image, architecture: str):
    """Preserve the original dataset conversion; CNN/GAP never resize or normalize."""
    if architecture.startswith('resnet'):
        return TRANSFORM(image)
    if architecture not in {'cnn', 'gap'}:
        raise ValueError('Unknown inference architecture')
    if image.size != (200, 200):
        raise ValueError('CNN/GAP inputs must be 200 x 200 pixels; automatic resizing is disabled to preserve preprocessing.')
    return transforms.ToTensor()(image.convert('L'))
