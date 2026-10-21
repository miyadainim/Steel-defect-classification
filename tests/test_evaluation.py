"""Validate sample-weighted loss without optimizing or training any model."""
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from steel_defect.evaluation.metrics import evaluate


def test_sample_weighted_loss_for_uneven_batches():
    model = nn.Linear(3, 6).eval()
    images, labels = torch.randn(5, 3), torch.tensor([0, 1, 2, 3, 4])
    report = evaluate(model, DataLoader(TensorDataset(images, labels), batch_size=2))
    with torch.inference_mode():
        expected = nn.functional.cross_entropy(model(images), labels).item()
    assert abs(report['loss'] - expected) < 1e-6
    assert len(report['confusion_matrix']) == 6
