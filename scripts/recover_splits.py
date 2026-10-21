"""Recover the archived split recipe into new manifests (never overwrite)."""
import argparse
from pathlib import Path
from steel_defect.data.splitting import recover_notebook_splits
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    recover_notebook_splits(a.dataset_root, a.output)
