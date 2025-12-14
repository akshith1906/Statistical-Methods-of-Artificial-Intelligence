import os
import numpy as np

def binary_accuracy(y_pred, y_true, thresh=0.5):
    """y_pred expected in (0,1), y_true in {0,1}."""
    if y_true.ndim == 1: y_true = y_true[:, None]
    preds = (y_pred >= thresh).astype(np.float32)
    return float((preds == y_true).mean())

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
