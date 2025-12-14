import numpy as np

class ReLU:
    def __init__(self): self.mask = None
    def forward(self, x):
        self.mask = (x > 0).astype(x.dtype)
        return x * self.mask
    def backward(self, dy): 
        return dy * self.mask

class Tanh:
    def __init__(self): self.y = None
    def forward(self, x):
        self.y = np.tanh(x); return self.y
    def backward(self, dy):
        return dy * (1.0 - self.y**2)

class Sigmoid:
    def __init__(self): self.y = None
    def forward(self, x):
        self.y = 1.0 / (1.0 + np.exp(-x)); return self.y
    def backward(self, dy):
        return dy * (self.y * (1.0 - self.y))

class Identity:
    def forward(self, x): 
        return x
    def backward(self, dy): 
        return dy
