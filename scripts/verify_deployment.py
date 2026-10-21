"""Optional CPU parity using one authorized checkpoint and one supplied image.

The reference is the preserved original inference SOURCE, not a notebook or a
bundled model. This command requires private inputs and never runs a dataset split.
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from steel_defect.constants import MODEL_PATH, PROJECT_ROOT
from steel_defect.deployment.inference import TRANSFORM, load_model, predict


def verify(image: Image.Image, checkpoint: Path) -> dict:
    reference_path = PROJECT_ROOT / 'tests/fixtures/original_inference.py'
    spec = importlib.util.spec_from_file_location('original_inference_reference', reference_path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    reference.MODEL_PATH = Path(checkpoint)
    torch.testing.assert_close(TRANSFORM(image), reference.TRANSFORM(image), rtol=0, atol=0)
    expected_index, expected_scores = reference.predict(image, reference.load_model())
    index, scores = predict(image, load_model(checkpoint=checkpoint))
    np.testing.assert_allclose(scores, expected_scores, atol=1e-6, rtol=1e-5)
    assert index == expected_index
    return dict(preprocessing_exact=True, scores_compared=6,
                max_absolute_error=float(np.max(np.abs(np.array(scores) - expected_scores))),
                atol=1e-6, rtol=1e-5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=MODEL_PATH)
    parser.add_argument('--image', type=Path, required=True, help='A locally authorized non-test image')
    args = parser.parse_args()
    if not args.checkpoint.is_file() or not args.image.is_file():
        parser.error('Supply an existing authorized local checkpoint and image; neither is included in this release.')
    with Image.open(args.image) as source:
        report = verify(source.convert('RGB'), args.checkpoint)
    print(report)


if __name__ == '__main__':
    main()
