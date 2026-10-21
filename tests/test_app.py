"""The actual Streamlit app must be useful without private model artifacts."""
from streamlit.testing.v1 import AppTest
from steel_defect.constants import PROJECT_ROOT, MODEL_PATH


def test_missing_model_shows_setup_and_no_predictions(tmp_path, monkeypatch):
    monkeypatch.setenv('STEEL_DEFECT_CHECKPOINT', str(tmp_path / 'absent.pth'))
    app = AppTest.from_file(str(PROJECT_ROOT / 'app.py')).run(timeout=30)
    assert not app.exception
    assert any('Model checkpoint not found' in message.value for message in app.info)
    assert any('STEEL_DEFECT_CHECKPOINT' in message.value for message in app.info)
    assert not app.success and not app.error
    assert app.get('file_uploader')[0].proto.disabled


def test_invalid_model_is_handled_without_predictions(tmp_path, monkeypatch):
    path = tmp_path / 'invalid.pth'
    path.write_bytes(b'not a checkpoint')
    monkeypatch.setenv('STEEL_DEFECT_CHECKPOINT', str(path))
    app = AppTest.from_file(str(PROJECT_ROOT / 'app.py')).run(timeout=30)
    assert not app.exception and not app.success
    assert any('checkpoint could not be loaded' in message.value for message in app.error)


def test_supplied_compatible_file_enables_upload_but_no_unsolicited_scores(tmp_path, monkeypatch):
    from steel_defect.models.resnet import build_model
    from steel_defect.training.checkpointing import save_checkpoint
    path = tmp_path / 'synthetic-contract-fixture.pth'
    # Test-only weights establish valid-file control flow, not classification quality.
    save_checkpoint(build_model('resnet_finetuned'), path)
    monkeypatch.setenv('STEEL_DEFECT_CHECKPOINT', str(path))
    app = AppTest.from_file(str(PROJECT_ROOT / 'app.py')).run(timeout=30)
    assert not app.exception and not app.error and not app.info and not app.success
    assert not app.get('file_uploader')[0].proto.disabled


import io
import numpy as np
import pytest
from PIL import Image


@pytest.mark.skipif(not MODEL_PATH.is_file(), reason='Original weights are not distributed')
@pytest.mark.parametrize('format,suffix,mime', [
    ('PNG', 'png', 'image/png'), ('JPEG', 'jpg', 'image/jpeg'),
    ('JPEG', 'jpeg', 'image/jpeg'), ('BMP', 'bmp', 'image/bmp')])
def test_actual_upload_and_bundled_model_prediction(format, suffix, mime, monkeypatch):
    from steel_defect.deployment.inference import load_model, predict
    monkeypatch.delenv('STEEL_DEFECT_CHECKPOINT', raising=False)
    image = Image.fromarray(np.arange(40000, dtype=np.uint8).reshape(200, 200)).convert('RGB')
    stream = io.BytesIO(); image.save(stream, format=format)
    raw = stream.getvalue()
    with Image.open(io.BytesIO(raw)) as decoded:
        index, scores = predict(decoded.convert('RGB'), load_model())
    labels = ['Crazing', 'Inclusion', 'Patches', 'Pitted surface', 'Rolled-in scale', 'Scratches']
    app = AppTest.from_file(str(PROJECT_ROOT / 'app.py')).run(timeout=30)
    assert not app.exception and not app.error
    app.file_uploader[0].set_value((f'synthetic.{suffix}', raw, mime)).run(timeout=30)
    assert not app.exception and not app.error
    assert app.success[0].value == f'Predicted defect: {labels[index]}'
    markdown = [element.value for element in app.markdown]
    for name, score in zip(labels, scores):
        assert f'{name}: {score:.2%}' in markdown
    assert len(app.get('progress')) == 6


@pytest.mark.skipif(not MODEL_PATH.is_file(), reason='Original weights are not distributed')
def test_invalid_upload_is_handled_with_actual_model(monkeypatch):
    monkeypatch.delenv('STEEL_DEFECT_CHECKPOINT', raising=False)
    app = AppTest.from_file(str(PROJECT_ROOT / 'app.py')).run(timeout=30)
    app.file_uploader[0].set_value(('invalid.png', b'not an image', 'image/png')).run(timeout=30)
    assert not app.exception and not app.success
    assert any('Unable to read this image' in message.value for message in app.error)
