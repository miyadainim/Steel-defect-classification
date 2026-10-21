"""Train a new named run; never overwrite archived checkpoints."""
import argparse
import json
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from steel_defect.constants import PROJECT_ROOT
from steel_defect.data.dataset import SteelDefectDataset
from steel_defect.data.preprocessing import cnn_transform, resnet_transform
from steel_defect.data.splitting import validate_disjoint
from steel_defect.models.resnet import build_model
from steel_defect.training.checkpointing import load_checkpoint
from steel_defect.training.trainer import fit, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'configs/default.json')
    parser.add_argument('--experiment', required=True, choices=['cnn_sgd','cnn_adam','gap_adam','resnet_frozen','resnet_finetuned'])
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--splits', type=Path)
    parser.add_argument('--optimizer', choices=['sgd', 'adam'])
    parser.add_argument('--lr', type=float)
    parser.add_argument('--head-lr', type=float)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device')
    parser.add_argument('--init-checkpoint', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config.update(config.pop('experiments')[args.experiment])
    for key, value in vars(args).items():
        if value is not None and key in {'dataset_root','splits','optimizer','lr','head_lr','batch_size','epochs','patience','seed','device','init_checkpoint'}:
            config[key] = str(value) if isinstance(value, Path) else value
    if Path(args.run_id).name != args.run_id or args.run_id in {'.','..'}:
        parser.error('run-id must be a single directory name')
    output = PROJECT_ROOT / 'results/runs' / args.run_id
    if output.exists():
        parser.error('Run directory already exists; choose a new run-id')
    for key in ['dataset_root','splits','init_checkpoint']:
        if config.get(key):
            p = Path(config[key]); config[key] = str(p if p.is_absolute() else PROJECT_ROOT / p)
    if config['batch_size'] < 1 or config['epochs'] < 1 or config['patience'] < 1 or config['lr'] <= 0:
        parser.error('batch-size, epochs, patience and lr must be positive')
    if config['model'] == 'resnet_finetuned' and config['optimizer'] != 'adam':
        parser.error('Fine-tuning preserves the notebook Adam parameter groups')
    splits = Path(config['splits'])
    validate_disjoint([splits / f'{s}.csv' for s in ['train','val','test']])
    seed_everything(config['seed'])
    model = build_model(config['model'], pretrained=config['model'] == 'resnet_frozen' and not config.get('init_checkpoint'))
    if config.get('init_checkpoint'):
        load_checkpoint(model, config['init_checkpoint'])
    model.to(config['device'])
    # No private generator: notebook DataLoader uses the global torch RNG.
    transform = resnet_transform() if config['model'].startswith('resnet') else cnn_transform()
    loaders = [DataLoader(SteelDefectDataset.from_manifest(splits / f'{s}.csv', config['dataset_root'], transform),
                batch_size=config['batch_size'], shuffle=s == 'train') for s in ['train','val']]
    if config['model'] == 'resnet_finetuned':
        optimizer = torch.optim.Adam([{'params': model.layer4.parameters(), 'lr': config['lr']},
                                      {'params': model.fc.parameters(), 'lr': config['head_lr']}])
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = (torch.optim.SGD if config['optimizer'] == 'sgd' else torch.optim.Adam)(params, lr=config['lr'])
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(), logging.FileHandler(output / 'train.log')])
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    report = fit(model, *loaders, optimizer, output / 'best.pth', config['epochs'], config['patience'], config['device'], config['model'])
    (output / 'history.json').write_text(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
