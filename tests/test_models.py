import pytest
import torch
from steel_defect.models.resnet import build_model

@pytest.mark.parametrize('name,shape', [('cnn',(2,1,200,200)),('gap',(2,1,200,200)),('resnet_frozen',(2,3,224,224)),('resnet_finetuned',(2,3,224,224))])
def test_shapes(name, shape):
    model = build_model(name).eval()
    with torch.inference_mode():
        result = model(torch.zeros(shape))
    assert result.shape == (2,6) and torch.isfinite(result).all()
    if name.startswith('resnet'):
        trainable = [key for key, p in model.named_parameters() if p.requires_grad]
        assert all(key.startswith(('layer4.', 'fc.') if name.endswith('finetuned') else ('fc.',)) for key in trainable)
