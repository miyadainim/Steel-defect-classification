"""Benchmark a specified checkpoint without using test data."""
import argparse
import json
from pathlib import Path
from PIL import Image
from steel_defect.models.resnet import build_model
from steel_defect.training.checkpointing import load_checkpoint
from steel_defect.data.preprocessing import cnn_transform, resnet_transform
from steel_defect.deployment.benchmark import benchmark

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--model', choices=['cnn','gap','resnet_frozen','resnet_finetuned'], default='resnet_finetuned')
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--image', type=Path, help='Decoded-image pipeline mode; batch size must be one')
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--repetitions', type=int, default=100)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.batch_size < 1 or (a.image and a.batch_size != 1):
        p.error('Positive batch size required; image pipeline supports batch size one')
    if a.output.exists():
        p.error('Output file exists')
    model = load_checkpoint(build_model(a.model), a.checkpoint)
    resnet = a.model.startswith('resnet')
    image = None
    if a.image:
        with Image.open(a.image) as source:
            image = source.convert('RGB' if resnet else 'L')
    shape = (a.batch_size, 3, 224, 224) if resnet else (a.batch_size, 1, 200, 200)
    report = benchmark(model, a.device, shape, image, a.warmup, a.repetitions,
                       resnet_transform() if resnet else cnn_transform())
    report['checkpoint'] = str(a.checkpoint.resolve())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
