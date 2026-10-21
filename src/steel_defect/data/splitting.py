"""Explicit manifests and recovery of the notebook's ordered split recipe."""
import csv
from pathlib import Path
from steel_defect.constants import CLASS_NAMES

def read_manifest(path):
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Empty split manifest")
    seen = set()
    for row in rows:
        p = Path(row['path'])
        label = int(row['label'])
        if p.is_absolute() or '..' in p.parts or len(p.parts) != 2:
            raise ValueError(f"Unsafe dataset path: {p}")
        if not 0 <= label < 6 or p.parts[0] != CLASS_NAMES[label]:
            raise ValueError(f"Class mapping mismatch: {row}")
        if row['path'] in seen:
            raise ValueError(f"Repeated image: {p}")
        seen.add(row['path'])
    return rows

def validate_disjoint(manifests):
    sets = [set(row['path'] for row in read_manifest(p)) for p in manifests]
    for i, a in enumerate(sets):
        for b in sets[i+1:]:
            if a & b:
                raise ValueError("Split paths overlap")
    # Notebook filenames are globally unique; check this as well.
    names = [{Path(p).name for p in s} for s in sets]
    for i, a in enumerate(names):
        for b in names[i+1:]:
            if a & b:
                raise ValueError("Split filenames overlap")

def recover_notebook_splits(dataset_root, output_dir):
    """Reconstruct recipe, not a claim of a saved historical manifest."""
    from sklearn.model_selection import train_test_split
    root, out = Path(dataset_root), Path(output_dir)
    paths, labels = [], []
    for label, name in enumerate(CLASS_NAMES):
        for p in sorted((root / name).iterdir()):
            if p.suffix.lower() in {'.bmp', '.jpg', '.jpeg', '.png'} and p.name != 'Pa_105.bmp':
                paths.append(p.relative_to(root).as_posix())
                labels.append(label)
    if len(paths) != 1799:
        raise ValueError(f"Expected original inventory of 1799, got {len(paths)}")
    train, temp, yl, yt = train_test_split(paths, labels, test_size=.3, stratify=labels, random_state=42)
    val, test, yv, ys = train_test_split(temp, yt, test_size=.5, stratify=yt, random_state=42)
    out.mkdir(parents=True, exist_ok=True)
    for name, x, y in [('train', train, yl), ('val', val, yv), ('test', test, ys)]:
        target = out / f'{name}.csv'
        with target.open('x', newline='') as f:
            writer = csv.writer(f); writer.writerow(['path', 'label'])
            writer.writerows(zip(x, y))
    validate_disjoint([out / f'{s}.csv' for s in ['train', 'val', 'test']])
