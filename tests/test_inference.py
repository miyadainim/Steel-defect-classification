"""Synthetic test-only weights validate plumbing; they are not trained predictions."""
import numpy as np
import pytest
import torch
from PIL import Image
from steel_defect.deployment.inference import CheckpointError, TRANSFORM, get_checkpoint_path, load_model, predict
from steel_defect.models.resnet import build_model
from steel_defect.training.checkpointing import save_checkpoint


@pytest.fixture
def image():
    return Image.fromarray(np.arange(40000, dtype=np.uint8).reshape(200, 200)).convert('RGB')


@pytest.fixture
def synthetic_checkpoint(tmp_path):
    # Temporary untrained state dict: never bundled or used by the public app.
    path = tmp_path / 'synthetic.pth'
    save_checkpoint(build_model('resnet_finetuned'), path)
    return path


def test_preprocessing(image):
    tensor = TRANSFORM(image)
    assert tensor.shape == (3, 224, 224) and tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()


def test_checkpoint_contract_and_softmax(synthetic_checkpoint, image):
    index, scores = predict(image, load_model(checkpoint=synthetic_checkpoint))
    assert 0 <= index < 6 and len(scores) == 6
    assert np.isfinite(scores).all() and sum(scores) == pytest.approx(1, abs=1e-6)
    with pytest.raises(FileExistsError):
        save_checkpoint(build_model('resnet_finetuned'), synthetic_checkpoint)


def test_missing_checkpoint_is_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match='STEEL_DEFECT_CHECKPOINT'):
        load_model(checkpoint=tmp_path / 'missing.pth')


def test_corrupt_checkpoint_is_actionable(tmp_path):
    path = tmp_path / 'corrupt.pth'
    path.write_bytes(b'not a model')
    with pytest.raises(CheckpointError, match='six-class ResNet18'):
        load_model(checkpoint=path)


def test_wrong_architecture_is_rejected(tmp_path):
    path = tmp_path / 'wrong.pth'
    torch.save({'unrelated': torch.zeros(1)}, path)
    with pytest.raises(CheckpointError):
        load_model(checkpoint=path)


def test_environment_checkpoint_override(tmp_path, monkeypatch):
    path = tmp_path / 'local.pth'
    monkeypatch.setenv('STEEL_DEFECT_CHECKPOINT', str(path))
    assert get_checkpoint_path() == path


def test_original_source_parity_with_test_only_weights(synthetic_checkpoint, image):
    from scripts.verify_deployment import verify
    report = verify(image, synthetic_checkpoint)
    assert report['preprocessing_exact'] and report['scores_compared'] == 6
    assert report['max_absolute_error'] <= 1e-6


def test_default_loader_uses_supplied_file_without_fallback(synthetic_checkpoint, monkeypatch):
    monkeypatch.setenv('STEEL_DEFECT_CHECKPOINT', str(synthetic_checkpoint))
    model = load_model()
    expected = torch.load(synthetic_checkpoint, map_location='cpu', weights_only=True)
    assert not model.training
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, expected[key], rtol=0, atol=0)
