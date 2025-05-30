import numpy as np

class SGD:
    """Stochastic Gradient Descent optimizer."""
    def __init__(self, lr=0.1):
        self.lr = lr

    def step(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.lr * g 