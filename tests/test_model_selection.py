"""All released weights, preprocessing and selector paths use genuine inference."""
import hashlib
import io
import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms
from streamlit.testing.v1 import AppTest
from steel_defect.constants import PROJECT_ROOT
from steel_defect.deployment.registry import MODELS, DEFAULT_MODEL
from steel_defect.deployment.inference import load_model, predict, preprocess_for_architecture, TRANSFORM


@pytest.mark.parametrize('key', list(MODELS))
def test_each_released_checkpoint(key, monkeypatch):
    monkeypatch.delenv('STEEL_DEFECT_CHECKPOINT', raising=False)
    spec = MODELS[key]
    path = PROJECT_ROOT / 'models' / spec['filename']
    if not path.is_file():
        pytest.skip('Optional original checkpoint is not distributed')
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
        pytest.skip('Optional original-checkpoint check; user-trained state dict differs')
    model = load_model(model_name=key)
    assert sum(p.numel() for p in model.parameters()) == spec['parameters']
    image = Image.fromarray(np.arange(40000, dtype=np.uint8).reshape(200, 200)).convert('RGB')
    tensor = preprocess_for_architecture(image, spec['architecture'])
    expected_shape = (3,224,224) if spec['architecture'].startswith('resnet') else (1,200,200)
    assert tensor.shape == expected_shape
    with torch.inference_mode():
        logits = model(tensor.unsqueeze(0))
    assert logits.shape == (1,6)
    index, scores = predict(image, model)
    assert 0 <= index < 6 and len(scores) == 6
    assert np.isfinite(scores).all() and sum(scores) == pytest.approx(1, abs=1e-6)
    torch.testing.assert_close(torch.tensor(scores), logits.softmax(1)[0], atol=1e-6, rtol=1e-5)


def test_architecture_specific_preprocessing():
    image = Image.fromarray(np.arange(40000,dtype=np.uint8).reshape(200,200)).convert('RGB')
    expected = transforms.ToTensor()(image.convert('L'))
    for architecture in ['cnn','gap']:
        torch.testing.assert_close(preprocess_for_architecture(image,architecture), expected, atol=0, rtol=0)
    torch.testing.assert_close(preprocess_for_architecture(image,'resnet_finetuned'),TRANSFORM(image),atol=0,rtol=0)
    with pytest.raises(ValueError, match='200 x 200'):
        preprocess_for_architecture(image.resize((250,250)),'cnn')


def test_switch_every_model_in_actual_streamlit_app(monkeypatch):
    if any(not (PROJECT_ROOT/'models'/spec['filename']).is_file() for spec in MODELS.values()):
        pytest.skip('Optional original checkpoint collection is not distributed')
    monkeypatch.delenv('STEEL_DEFECT_CHECKPOINT', raising=False)
    app = AppTest.from_file(str(PROJECT_ROOT/'app.py')).run(timeout=30)
    assert not app.exception and not app.error
    assert app.selectbox[0].value == DEFAULT_MODEL == 'resnet_finetuned'
    image = Image.fromarray(np.arange(40000,dtype=np.uint8).reshape(200,200)).convert('RGB')
    buffer = io.BytesIO();image.save(buffer,format='PNG')
    app.file_uploader[0].set_value(('synthetic.png',buffer.getvalue(),'image/png')).run(timeout=30)
    labels=['Crazing','Inclusion','Patches','Pitted surface','Rolled-in scale','Scratches']
    for key,spec in MODELS.items():
        app.selectbox[0].set_value(key).run(timeout=30)
        assert not app.exception and not app.error
        assert any(spec['label'] in caption.value for caption in app.caption)
        index,scores=predict(image,load_model(model_name=key))
        assert app.success[0].value == f'Predicted defect: {labels[index]}'
        for label,score in zip(labels,scores):
            assert f'{label}: {score:.2%}' in [item.value for item in app.markdown]
        assert len(app.get('progress')) == 6


def test_all_selector_choices_handle_missing_local_weights(monkeypatch):
    monkeypatch.delenv('STEEL_DEFECT_CHECKPOINT', raising=False)
    for key,spec in list(MODELS.items()):
        monkeypatch.setitem(MODELS,key,{**spec,'filename':'not-distributed-'+spec['filename']})
    app=AppTest.from_file(str(PROJECT_ROOT/'app.py')).run(timeout=30)
    assert app.selectbox[0].value == DEFAULT_MODEL
    for key in MODELS:
        app.selectbox[0].set_value(key).run(timeout=30)
        assert not app.exception and not app.error and not app.success
        assert any('Model checkpoint not found' in item.value for item in app.info)
        assert app.file_uploader[0].proto.disabled
