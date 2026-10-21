"""Sample-weighted cross entropy and validation-loss early stopping."""
import copy
import logging
import random
import numpy as np
import torch
from torch import nn
from steel_defect.training.checkpointing import save_checkpoint

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train_one_epoch(model, loader, optimizer, device='cpu', strategy='cnn'):
    model.train()
    if strategy == 'resnet_frozen':
        model.eval()  # Notebook cell 63 keeps all pretrained BN statistics fixed.
    elif strategy == 'resnet_finetuned':
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    total = 0.0
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward(); optimizer.step()
        total += loss.item() * images.size(0)
    return total / len(loader.dataset)

def fit(model, train_loader, val_loader, optimizer, output, epochs=20, patience=5,
        device='cpu', strategy='cnn'):
    from steel_defect.evaluation.metrics import evaluate
    if epochs < 1 or patience < 1:
        raise ValueError('epochs and patience must be positive')
    best_loss, best_epoch = float('inf'), 0
    best_weights = None
    if strategy == 'resnet_finetuned':
        best_loss = evaluate(model, val_loader, device)['loss']
        best_weights = copy.deepcopy(model.state_dict())
    history, stale = [], 0
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device, strategy)
        report = evaluate(model, val_loader, device)
        row = dict(epoch=epoch, train_loss=loss, val_loss=report['loss'], val_accuracy=report['accuracy'])
        history.append(row); logging.info('%s', row)
        if report['loss'] < best_loss:
            best_loss, best_epoch = report['loss'], epoch
            best_weights = copy.deepcopy(model.state_dict()); stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_weights)
    save_checkpoint(model, output)
    return dict(best_epoch=best_epoch, best_val_loss=best_loss, history=history)
