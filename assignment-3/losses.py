import numpy as np

class MSELoss:
    def forward(self, y_pred, y_true):
        if y_true.ndim == 1: y_true = y_true[:, None]
        diff = y_pred - y_true
        return float(0.5 * np.mean(diff * diff))
    def backward(self, y_pred, y_true):
        if y_true.ndim == 1: y_true = y_true[:, None]
        return (y_pred - y_true) / max(y_pred.shape[0], 1)

class BCELoss:
    def __init__(self, eps=1e-7): self.eps = eps
    def forward(self, y_pred, y_true):
        if y_true.ndim == 1: y_true = y_true[:, None]
        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        return float(-np.mean(y_true*np.log(y_pred) + (1 - y_true)*np.log(1 - y_pred)))
    def backward(self, y_pred, y_true):
        if y_true.ndim == 1: y_true = y_true[:, None]
        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        return (y_pred - y_true) / (y_pred * (1 - y_pred)) / max(y_pred.shape[0], 1)
