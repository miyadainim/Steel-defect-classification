"""Manifest contracts are tested using synthetic metadata, never private splits."""
import csv
import pytest
from steel_defect.data.splitting import read_manifest, validate_disjoint


def manifest(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['path', 'label'])
        writer.writerows(rows)
    return path


def test_disjoint_synthetic_manifests(tmp_path):
    paths = [manifest(tmp_path / f'{name}.csv', [(f'crazing/{name}.bmp', 0)]) for name in ['train', 'val', 'test']]
    validate_disjoint(paths)
    assert read_manifest(paths[0]) == [{'path': 'crazing/train.bmp', 'label': '0'}]


def test_overlap_is_rejected(tmp_path):
    paths = [manifest(tmp_path / f'{i}.csv', [('inclusion/shared.bmp', 1)]) for i in range(2)]
    with pytest.raises(ValueError, match='overlap'):
        validate_disjoint(paths)


@pytest.mark.parametrize('path,label', [('../escape.bmp', 0), ('/outside.bmp', 0), ('inclusion/wrong.bmp', 0)])
def test_unsafe_or_inconsistent_path_is_rejected(tmp_path, path, label):
    with pytest.raises(ValueError):
        read_manifest(manifest(tmp_path / 'split.csv', [(path, label)]))
