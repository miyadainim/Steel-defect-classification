"""Save actual evaluation confusion matrices using a noninteractive backend."""
def save_confusion_matrix(matrix, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    from steel_defect.constants import CLASS_NAMES
    fig, ax = plt.subplots(figsize=(9, 7))
    ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(ax=ax, cmap='Blues', xticks_rotation=45)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
