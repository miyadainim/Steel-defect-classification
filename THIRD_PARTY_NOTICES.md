# Third-party notices and excluded assets

This project imports dependencies; it does not bundle their Python packages or
binary libraries. Installed packages retain all their own license/NOTICE files.
The project MIT license cannot replace their terms. If later distributing a
container, executable, wheels or copied upstream source, carry the applicable
notices and comply with those artifacts' licenses, including bundled components.

## Main runtime dependencies

- PyTorch 2.14.1: BSD-style project terms plus bundled third-party terms. A copy of
  the installed upstream project license is retained in [torch-LICENSE](licenses/torch-LICENSE.txt).
- TorchVision 0.29.1: BSD-3-Clause software license, retained in
  [torchvision-LICENSE](licenses/torchvision-LICENSE.txt). Retain copyright/conditions/
  disclaimers when redistributing upstream code; do not imply upstream endorsement.
- Streamlit 1.66.0: Apache-2.0, retained in
  [streamlit-LICENSE](licenses/streamlit-LICENSE.txt). Preserve required notices and
  identify changes to upstream material if you distribute modified upstream code.
- Pillow, NumPy and other direct/transitive dependencies remain separately licensed.
  The pinned dependency files describe installation requirements, not a grant to
  relicense upstream packages. This candidate contains no vendored dependency packages.

These convenience license copies are not a complete license bill of materials
for all components in any future bundled binary distribution.

## NEU-CLS attribution and rights

Dataset creators: Kechen Song and Yunhui Yan, Northeastern University.
Original source: [NEU surface defect database](https://faculty.neu.edu.cn/songkc/en/zdylm/263270/list/index.htm).
The source requests citation of K. Song and Y. Yan, “A noise robust method based
on completed local binary patterns for hot-rolled steel strip surface defects,”
Applied Surface Science 285 (2013), 858–864.

The reviewed source page provides download/citation information but no explicit
redistribution grant was located. Raw images, pixel-bearing figures, archives and
private manifests are intentionally absent. Obtain authorized data separately;
this release does not treat public download availability as permission to redistribute.

## Pretrained and fine-tuned model rights

The research model was initialized from ImageNet-trained ResNet18. TorchVision's
[model documentation](https://docs.pytorch.org/vision/stable/models.html#general-information-on-pre-trained-weights)
warns pretrained models may have separate dataset-related terms. ImageNet's
[database access terms](https://www.image-net.org/download.php) constrain database
access/use; they do not by themselves resolve downstream checkpoint redistribution.
No checkpoints are distributed in this code-only release. Local weights are not
asserted to have redistribution clearance. They are NOT MIT licensed; no checkpoint
license is granted. Do not publish model weights until redistribution rights are confirmed
or the owner has separately completed a documented rights review. Moving
weights to a release asset, Git LFS or another service would not resolve rights.

## Reference source

`tests/fixtures/original_inference.py` preserves original project inference source
for numerical comparison. It contains no model weights or dataset pixels and is
covered as original project code. Private notebook outputs are not included.

The custom CNN/GAP weights derive from NEU-CLS; the two ResNet18 stages also
derive from ImageNet-pretrained initialization. Rights remain unresolved for
all five. Excluding raw images does not establish permission to publish weights.
