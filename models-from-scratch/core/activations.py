import numpy as np

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    """Derivative of the sigmoid function."""
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)

def relu_deriv(x):
    """Derivative of the ReLU function."""
    return (x > 0).astype(float) 