"""Use the actual bundled checkpoint on unlabelled synthetic input, never a test split."""
import hashlib
import numpy as np
import pytest
from PIL import Image
from steel_defect.constants import MODEL_PATH
from steel_defect.deployment.inference import load_model, predict

EXPECTED = '70c4e036c83d712d0cd507d6fc8a9694de280248091a57da2476e834df66e970'


def test_bundled_checkpoint_hash_and_real_inference(monkeypatch):
    if not MODEL_PATH.is_file():
        pytest.skip('Optional original checkpoint is not distributed')
    monkeypatch.delenv('STEEL_DEFECT_CHECKPOINT', raising=False)
    if hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != EXPECTED:
        pytest.skip('This optional check applies to the original research checkpoint, not user-trained weights')
    assert MODEL_PATH.stat().st_size == 44797534
    image = Image.fromarray(np.arange(40000, dtype=np.uint8).reshape(200, 200)).convert('RGB')
    index, scores = predict(image, load_model())
    assert 0 <= index < 6 and len(scores) == 6
    assert np.isfinite(scores).all() and sum(scores) == pytest.approx(1, abs=1e-6)
    from scripts.verify_deployment import verify
    report = verify(image, MODEL_PATH)
    assert report['preprocessing_exact'] and report['max_absolute_error'] <= 1e-6
