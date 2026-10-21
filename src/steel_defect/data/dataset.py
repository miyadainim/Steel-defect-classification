"""Path-based grayscale dataset, as used in notebook cell 1."""
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from steel_defect.data.splitting import read_manifest

class SteelDefectDataset(Dataset):
    def __init__(self, image_paths, labels, transform):
        if len(image_paths) != len(labels):
            raise ValueError("Image and label counts differ")
        if any(not 0 <= int(label) < 6 for label in labels):
            raise ValueError("Labels must be in [0, 5]")
        self.image_paths = [Path(p) for p in image_paths]
        self.labels = [int(label) for label in labels]
        self.transform = transform

    @classmethod
    def from_manifest(cls, manifest, dataset_root, transform):
        rows = read_manifest(manifest)
        return cls([Path(dataset_root) / row['path'] for row in rows],
                   [int(row['label']) for row in rows], transform)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        with Image.open(self.image_paths[idx]) as image:
            tensor = self.transform(image.convert("L"))
        return tensor, self.labels[idx]
