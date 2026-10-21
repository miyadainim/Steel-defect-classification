"""Explicit six-class metrics and sample-weighted loss."""
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from steel_defect.constants import CLASS_NAMES

def evaluate(model, loader, device='cpu'):
    model.eval()
    truth, predictions, total_loss = [], [], 0.0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += nn.functional.cross_entropy(logits, labels).item() * images.size(0)
            truth.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    if not truth:
        raise ValueError('Cannot evaluate an empty dataset')
    return dict(loss=total_loss / len(truth), accuracy=float(accuracy_score(truth, predictions)),
                macro_f1=float(f1_score(truth, predictions, labels=list(range(6)), average='macro', zero_division=0)),
                confusion_matrix=confusion_matrix(truth, predictions, labels=list(range(6))).tolist(),
                classification_report=classification_report(truth, predictions, labels=list(range(6)),
                    target_names=CLASS_NAMES, zero_division=0, output_dict=True),
                labels=truth, predictions=predictions)
