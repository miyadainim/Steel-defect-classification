"""Assert the checked-out source release has no dependency on the private project."""
from pathlib import Path
from steel_defect.constants import PROJECT_ROOT, MODEL_PATH
import steel_defect


def test_package_resolves_inside_public_checkout():
    assert Path(steel_defect.__file__).resolve().is_relative_to(PROJECT_ROOT)
    assert (PROJECT_ROOT / 'LICENSE').is_file()
    assert (PROJECT_ROOT / 'README.md').is_file()


def test_private_assets_are_excluded_by_publication_policy():
    # Local authorized assets may be supplied after checkout; they must stay ignored.
    rules = set((PROJECT_ROOT / '.gitignore').read_text().splitlines())
    assert {'*.ipynb', '*.pth', '*.pt', '*.ckpt', '*.onnx', '*.png', '*.bmp', '*.rar', '*.zip'} <= rules
    assert not any(line.startswith('!') and line.endswith('.pth') for line in rules)
    assert '/data/*' in rules and '/models/*' in rules
