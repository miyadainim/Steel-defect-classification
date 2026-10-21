"""Six-class ResNet18 and the notebook's transfer-learning policy."""
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from steel_defect.models.cnn import SteelDefectCNN, SteelDefectCNN_GAP

def build_model(name, pretrained=False):
    if name == 'cnn':
        return SteelDefectCNN()
    if name == 'gap':
        return SteelDefectCNN_GAP()
    if name not in {'resnet_frozen', 'resnet_finetuned'}:
        raise ValueError(f'Unknown model: {name}')
    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 6)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    if name == 'resnet_finetuned':
        for p in model.layer4.parameters():
            p.requires_grad = True
    return model
