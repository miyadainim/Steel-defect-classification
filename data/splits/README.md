# Local split manifests

No dataset images or private research manifests are bundled. Obtain NEU-CLS
from the original maintainer linked in the root README. Use `scripts/recover_splits.py`
with an authorized local original inventory to recover the recorded ordered recipe.
Generated CSVs contain image paths relative to your dataset root plus integer labels.
They are ignored by Git. Recovered identities must not be called independently
verified historical identities; the original full runtime path lists were not saved.
