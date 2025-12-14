import numpy as np

class Linear:
    def __init__(self, in_dim, out_dim, activation, scale=0.1, seed=None):
        rng = np.random.default_rng(seed)
        self.W = (rng.standard_normal((in_dim, out_dim)) * scale).astype(np.float32)
        self.b = np.zeros((out_dim,), dtype=np.float32)
        self.act = activation

        self.x = None  # cache

        # cumulative grads
        self.dW_acc = np.zeros_like(self.W)
        self.db_acc = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        z = x @ self.W + self.b
        return self.act.forward(z)

    def backward(self, dy):
        dz = self.act.backward(dy)
        B = max(self.x.shape[0], 1)
        dW = (self.x.T @ dz) / B
        db = dz.sum(axis=0) / B

        self.dW_acc += dW
        self.db_acc += db

        dx = dz @ self.W.T
        return dx

    def zero_grad(self):
        self.dW_acc.fill(0.0)
        self.db_acc.fill(0.0)

    def update(self, lr, scale=1.0):
        self.W -= lr * (self.dW_acc * scale)
        self.b -= lr * (self.db_acc * scale)
        self.zero_grad()
