# Historical research summary — reported, not reproduced here

The private research project compared CNN optimization, a smaller GAP head and
transfer learning on NEU-CLS. The original notebook, two redundant checkpoints, raw data, visual outputs and private split/evaluation
artifacts remain excluded. Model weights are excluded from the code-only publication.

| Model | Historical validation accuracy | Historical validation loss |
|---|---:|---:|
| Original CNN + SGD | 68.52% | .9862 |
| Original CNN + Adam | 94.81% | .1520 |
| GAP CNN + Adam | 80.74% | .5266 |
| Frozen ResNet18 | 99.63% | .0283 |
| Fine-tuned ResNet18 | 100.00% | approximately .0005 |

Fine-tuned ResNet18 was selected by validation loss. Saved private held-out records
report 99.26% (268/270) accuracy, .9926 macro-F1 and .0264 cross-entropy loss; the
two observed errors were scratches predicted as inclusion. These are historical
claims checked against private records earlier, not independently demonstrated by
this candidate verification. No test evaluation was run to prepare this export.

Historical batch-one Apple Silicon MPS **forward-only** median/P95 was .794/1.049 ms
for CNN (1×1×200×200) and 5.171/6.354 ms for ResNet18 (1×3×224×224), with 20 warmups
and 100 repetitions. CPU ResNet18 **decoded-image preprocessing + forward + softmax/
CPU scores** was 37.942/102.275 ms, with 10 warmups and 100 repetitions. Exact
historical chip/CPU/thread metadata was not saved. These are different pipelines,
not a controlled CPU/GPU comparison, and exclude browser/network and file decoding.
No performance measurements were generated during this source export.

Earlier private CPU verification found all six refactored scores exactly matched
the original implementation on a fixed synthetic image and one corroborated
training image. That trained checkpoint is now included locally; the private real-image
parity evidence is not included. The public synthetic parity test validates code plumbing only.

The recorded dataset audit excluded one exact duplicate, recovering 1,799 images
and a 1,259/270/270 split recipe. Forty saved memberships were corroborated; the
other identities remain recipe-inferred. pHash/SSIM screening cannot establish
independence of source steel surfaces. No industrial deployment validation,
external-domain validity, calibrated confidence or defect-free detection is claimed.

Future work: authorized surface-group-aware external evaluation, calibration and
rejection studies on separate data, model/data provenance, monitoring and actual
end-to-end application latency measurement. None is claimed complete.
