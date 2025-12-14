import numpy as np
from .losses import MSELoss, BCELoss

class Model:
    def __init__(self, layers, loss="bce", lr=1e-2):
        self.layers = layers
        self.lr = lr
        self.loss_fn = MSELoss() if loss.lower() == "mse" else BCELoss()

    def forward(self, x):
        out = x
        for L in self.layers:
            out = L.forward(out)
        return out

    def backward(self, dloss):
        grad = dloss
        for L in reversed(self.layers):
            grad = L.backward(grad)

    def train_step(self, xb, yb):
        ypred = self.forward(xb)
        loss = self.loss_fn.forward(ypred, yb)
        dloss = self.loss_fn.backward(ypred, yb)
        self.backward(dloss)  # accumulates grads inside layers
        return loss, ypred

    def zero_grad(self):
        for L in self.layers:
            L.zero_grad()

    def update(self, grad_accum_scale=1.0):
        for L in self.layers:
            L.update(self.lr, scale=grad_accum_scale)

    def predict(self, x):
        return self.forward(x)

    def save_to(self, path_npz):
        payload = {}
        for i, L in enumerate(self.layers):
            payload[f"W{i}"] = L.W
            payload[f"b{i}"] = L.b
        np.savez(path_npz, **payload)

    def load_from(self, path_npz):
        data = np.load(path_npz)
        # shape check
        for i, L in enumerate(self.layers):
            Wk, bk = f"W{i}", f"b{i}"
            if Wk not in data or bk not in data:
                raise ValueError(f"Missing {Wk}/{bk} in saved file")
            if data[Wk].shape != L.W.shape or data[bk].shape != L.b.shape:
                raise ValueError(
                    f"Shape mismatch at layer {i}: "
                    f"expected {L.W.shape}/{L.b.shape}, "
                    f"got {data[Wk].shape}/{data[bk].shape}"
                )
        # load
        for i, L in enumerate(self.layers):
            L.W[:] = data[f"W{i}"]
            L.b[:] = data[f"b{i}"]
