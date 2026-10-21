import numpy as np
import torch
from PIL import Image
from steel_defect.constants import CLASS_NAMES
from steel_defect.data.dataset import SteelDefectDataset
from steel_defect.data.preprocessing import cnn_transform

def test_dataset(tmp_path):
    image = tmp_path / 'image.bmp'
    Image.fromarray(np.arange(40000, dtype=np.uint8).reshape(200, 200)).save(image)
    x, label = SteelDefectDataset([image], [5], cnn_transform())[0]
    assert x.shape == (1, 200, 200)
    assert x.dtype == torch.float32 and torch.isfinite(x).all()
    assert 0 <= x.min() <= x.max() <= 1
    assert label == 5
    assert CLASS_NAMES == ['crazing','inclusion','patches','pitted_surface','rolled-in_scale','scratches']
