"""Verified checkpoint registry; default and class order are stable."""
from steel_defect.constants import PROJECT_ROOT

DEFAULT_MODEL = 'resnet_finetuned'
MODELS = {'resnet_finetuned': {'label': 'Fine-tuned ResNet18', 'architecture': 'resnet_finetuned', 'filename': 'best_resnet18_finetuned.pth', 'sha256': '70c4e036c83d712d0cd507d6fc8a9694de280248091a57da2476e834df66e970', 'parameters': 11179590}, 'cnn_sgd': {'label': 'CNN + SGD', 'architecture': 'cnn', 'filename': 'best_sgd.pth', 'sha256': 'bfc34f4abfe38c5a16748e8176f7d08277d729e352e05c470bf74c3a8684c3d7', 'parameters': 263302}, 'cnn_adam': {'label': 'CNN + Adam', 'architecture': 'cnn', 'filename': 'best_adam.pth', 'sha256': 'ae91f52a2db1a8f440b65fdedcd397260524c22d1432994675807ffff6651cc6', 'parameters': 263302}, 'gap_adam': {'label': 'GAP CNN + Adam', 'architecture': 'gap', 'filename': 'best_gap.pth', 'sha256': '9cd531286765e8cca16537dfc1cd81b385e7055e4d42b57db154b2fc5c632802', 'parameters': 23686}, 'resnet_frozen': {'label': 'Frozen ResNet18', 'architecture': 'resnet_frozen', 'filename': 'best_resnet18_frozen.pth', 'sha256': '91443866b57221ef927a4af5f40a62e5afc81c705252458b498af27298bc05ac', 'parameters': 11179590}}


def checkpoint_path(model_name):
    return PROJECT_ROOT / 'models' / MODELS[model_name]['filename']
