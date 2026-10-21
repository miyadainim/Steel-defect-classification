"""Evaluate only an explicitly selected split and checkpoint."""
import argparse
import json
from pathlib import Path
from torch.utils.data import DataLoader
from steel_defect.constants import PROJECT_ROOT
from steel_defect.data.dataset import SteelDefectDataset
from steel_defect.data.preprocessing import cnn_transform, resnet_transform
from steel_defect.models.resnet import build_model
from steel_defect.training.checkpointing import load_checkpoint
from steel_defect.evaluation.metrics import evaluate
from steel_defect.evaluation.visualization import save_confusion_matrix

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--model', required=True, choices=['cnn','gap','resnet_frozen','resnet_finetuned'])
    p.add_argument('--split', required=True, choices=['train','val','test'])
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--splits', type=Path, default=PROJECT_ROOT / 'data/splits')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=32)
    a = p.parse_args()
    if a.output.exists():
        p.error('Output directory exists; choose a new report directory')
    model = load_checkpoint(build_model(a.model), a.checkpoint).to(a.device)
    transform = resnet_transform() if a.model.startswith('resnet') else cnn_transform()
    dataset = SteelDefectDataset.from_manifest(a.splits / f'{a.split}.csv', a.dataset_root, transform)
    report = evaluate(model, DataLoader(dataset, batch_size=a.batch_size), a.device)
    report.update(split=a.split, checkpoint=str(a.checkpoint.resolve()))
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'metrics.json').write_text(json.dumps(report, indent=2))
    import numpy as np
    save_confusion_matrix(np.array(report['confusion_matrix']), a.output / 'confusion_matrix.png')
    print(json.dumps({k: report[k] for k in ['split','accuracy','macro_f1','loss']}))

if __name__ == '__main__':
    main()
