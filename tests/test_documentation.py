"""Public instructions must not link to omitted private documents or artifacts."""
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
from steel_defect.constants import PROJECT_ROOT


def test_public_documentation_local_links_resolve():
    failures = []
    for document in PROJECT_ROOT.rglob('*.md'):
        if any(part.endswith('.egg-info') or part.startswith('.venv') for part in document.parts):
            continue
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', document.read_text()):
            target = target.split()[0].strip('<>')
            if urlsplit(target).scheme or target.startswith('#'):
                continue
            destination = document.parent / unquote(target.split('#', 1)[0])
            if not destination.exists():
                failures.append((str(document.relative_to(PROJECT_ROOT)), target))
    assert not failures, failures
