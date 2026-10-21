# Add your local checkpoints here

No model weights are published. To enable predictions, train your own model or
obtain compatible weights you are authorized to use, then put its state dict in
this folder using the matching filename:

- `best_resnet18_finetuned.pth`: default Fine-tuned ResNet18.
- `best_sgd.pth`: CNN + SGD.
- `best_adam.pth`: CNN + Adam.
- `best_gap.pth`: GAP CNN + Adam.
- `best_resnet18_frozen.pth`: Frozen ResNet18.

Only the checkpoint for your selected model is needed. Training saves a new
`results/runs/<run-id>/best.pth`; copy it here under the appropriate filename.
All `.pth` files remain ignored by Git. Do not force-add or redistribute weights
without the necessary permissions. Missing files produce setup guidance, not predictions.
