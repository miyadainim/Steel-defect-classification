"""Compatibility imports for existing notebooks and clients."""
import sys
from pathlib import Path

# Keep legacy imports usable from a checkout without an editable installation.
_source_root = Path(__file__).resolve().parent / 'src'
if _source_root.is_dir() and str(_source_root) not in sys.path:
    sys.path.insert(0, str(_source_root))

from steel_defect.deployment.inference import CLASS_NAMES, MODEL_PATH, TRANSFORM, CheckpointError, get_checkpoint_path, load_model, predict
__all__ = ['CLASS_NAMES', 'MODEL_PATH', 'TRANSFORM', 'load_model', 'predict', 'CheckpointError', 'get_checkpoint_path']
